import cv2
import os
import time
import numpy as np
from datetime import timedelta
from collections import deque
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional, Callable
from pydantic import BaseModel
import asyncio
from PIL import Image
import torch
import hashlib

# Import the new MediaPipe detector
try:
    from .mediapipe_detector import MediaPipeFaceDetector, FaceQuality
except ImportError:
    # Fallback when running as script
    from mediapipe_detector import MediaPipeFaceDetector, FaceQuality

# Import model cache
try:
    from .model_cache import model_cache
except ImportError:
    # Fallback when running as script
    from model_cache import model_cache

# Import memory pool
try:
    from .memory_pool import memory_pool
except ImportError:
    # Fallback when running as script
    from memory_pool import memory_pool

# Conditional import for websocket manager
try:
    from .websocket_manager import manager
except ImportError:
    # Fallback when running as script
    manager = None

class RoiPosition(BaseModel):
    top: float
    bottom: float
    left: float
    right: float

@dataclass
class ProcessingConfig:
    """Configuration settings for video processing"""
    emotion_sensitivity: int
    roi_position: RoiPosition
    min_emotion_duration: float = 0.5
    debug: bool = False
    frame_buffer_size: int = 2
    target_emotions: List[str] = None
    image_format: str = "png"
    image_quality: int = 6
    focus_threshold: float = 0.75  # Optimized for YouTube thumbnail sharpness
    glasses_threshold: float = 0.3
    glasses_reduction_factor: float = 0.6
    glasses_aware_processing: bool = True  # Enable glasses-aware face detection
    batch_size: int = 4
    enable_batch_processing: bool = True
    # New deduplication settings
    enable_deduplication: bool = True
    deduplication_window: float = 1.0  # Time window in seconds to group similar emotions
    max_frames_per_emotion: int = 2    # Maximum frames to save per emotion per window
    min_score_improvement: float = 0.05  # Minimum score improvement to save additional frame
    # Performance optimization settings
    emotion_detection_skip_frames: int = 15  # Process every 15th frame (1 out of 15) for maximum speed
    emotion_cache_size: int = 2000  # Large cache size for maximum hit rates and performance
    emotion_input_size: int = 224  # ViT model input size requirement (224x224 pixels)
    enable_emotion_caching: bool = True  # Enable caching of emotion detection results for performance
    enable_batch_emotion_detection: bool = False  # Disable batch processing to minimize overhead

class EmotionEvent:
    """Represents a detected emotion event with temporal information"""
    def __init__(self, emotion: str, score: float, frame: np.ndarray, frame_number: int, timestamp: float):
        self.emotion = emotion
        self.score = score
        self.frame = frame
        self.frame_number = frame_number
        self.timestamp = timestamp
        self.face_rect = None  # Will be set when processing

class TemporalDeduplicator:
    """Temporal deduplication to reduce redundant emotion frames"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.emotion_windows = {}  # emotion -> deque of recent events
        self.saved_events = []
        self.logger = logging.getLogger(__name__)
    
    def should_save_emotion(self, emotion: str, score: float, timestamp: float, frame: np.ndarray, frame_number: int) -> bool:
        """Determine if this emotion detection should be saved based on temporal deduplication"""
        if not self.config.enable_deduplication:
            return True
        
        # Initialize window for this emotion if it doesn't exist
        if emotion not in self.emotion_windows:
            self.emotion_windows[emotion] = deque()
        
        window = self.emotion_windows[emotion]
        
        # Remove old events outside the time window
        while window and (timestamp - window[0].timestamp) > self.config.deduplication_window:
            window.popleft()
        
        # Check if we should save this emotion
        should_save = True
        
        if len(window) >= self.config.max_frames_per_emotion:
            # Check if this score is significantly better than existing ones
            best_score = max(event.score for event in window)
            if score <= (best_score + self.config.min_score_improvement):
                should_save = False
        
        if should_save:
            # Create and store the event
            event = EmotionEvent(emotion, score, frame.copy(), frame_number, timestamp)
            window.append(event)
            self.saved_events.append(event)
            
            # Keep only the most recent events in the window
            while len(window) > self.config.max_frames_per_emotion:
                window.popleft()
        
        return should_save
    
    def get_saved_events(self) -> List[EmotionEvent]:
        """Get all saved emotion events"""
        return self.saved_events
    
    def clear(self):
        """Clear all stored events (call before processing new video)"""
        self.emotion_windows.clear()
        self.saved_events.clear()

class ProgressTracker:
    """Tracks and reports processing progress"""
    def __init__(self, total_frames: int, update_interval: float = 1.0):  # Progress update interval in seconds
        self.total_frames = total_frames
        self.last_update = 0
        self.last_time = time.time()
        self.update_interval = update_interval
        self.start_time = self.last_time
        self.smoothed_fps = 0
        self.alpha = 0.1  # Smoothing factor for FPS calculation (smaller = smoother)
        self.last_eta = None

    def update(self, frame_count: int) -> Optional[dict]:
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        # Calculate current FPS based on processed frames since last update
        frames_since_last = frame_count - self.last_update
        time_since_last = current_time - self.last_time
        current_fps = frames_since_last / time_since_last if time_since_last > 0 else 0
        
        # Apply exponential moving average smoothing to FPS
        self.smoothed_fps = (self.alpha * current_fps) + ((1 - self.alpha) * self.smoothed_fps)
        
        # Calculate progress percentage based on frame count
        progress = min((frame_count / self.total_frames) * 100, 100)
        
        # Calculate estimated time remaining based on current processing speed
        if self.smoothed_fps > 0:
            frames_remaining = self.total_frames - frame_count
            current_eta = frames_remaining / self.smoothed_fps
            
            # Apply smoothing to ETA to reduce fluctuations
            if self.last_eta is None:
                self.last_eta = current_eta
            else:
                # Use weighted average for ETA smoothing (30% new, 70% previous)
                current_eta = (0.3 * current_eta) + (0.7 * self.last_eta)
            self.last_eta = current_eta
        else:
            current_eta = 0
        
        if current_time - self.last_time >= self.update_interval:
            self.last_update = frame_count
            self.last_time = current_time
            
            return {
                "type": "progress",
                "progress": round(progress, 1),
                "fps": round(self.smoothed_fps, 1),
                "elapsed": time.strftime('%H:%M:%S', time.gmtime(elapsed)),
                "eta": time.strftime('%H:%M:%S', time.gmtime(current_eta))
            }
        return None

class EmotionDetector:
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.face_cache = {}
        self.last_face_location = None
        self._setup_logging()
        self._initialize_model()
        self._cached_roi = None
        self._cached_dimensions = None
        
        # Initialize MediaPipe face detector
        self.mediapipe_detector = MediaPipeFaceDetector(config)
        
        # Initialize temporal deduplicator
        self.deduplicator = TemporalDeduplicator(config)
        
        # Get emotion labels from model cache
        self.emotion_labels = model_cache.get_emotion_labels()
        
        # Set target emotions (default to all if not specified)
        if self.config.target_emotions is None:
            self.config.target_emotions = self.emotion_labels
        
        # Initialize emotion detection optimizations
        self._emotion_cache = {}
        self._frame_counter = 0
        self._emotion_batch_buffer = []
        self._last_emotion_detection_time = 0
        
        # Performance monitoring
        self._emotion_detection_times = deque(maxlen=100)
        self._total_emotion_detections = 0
        self._cached_emotion_hits = 0

    def _initialize_model(self):
        """Initialize the HuggingFace emotion detection model from cache, using MPS (Apple Silicon GPU) if available"""
        try:
            # Get cached models
            self.processor, self.model, device = model_cache.get_models()
            import torch
            
            # Force MPS usage if available (Apple Silicon GPU)
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device('mps')
                self.logger.info("Using Apple Silicon GPU (MPS) for emotion detection model")
                # Move model to MPS device
                self.model = self.model.to(self.device)
                # Force model to eval mode for inference
                self.model.eval()
                # Test MPS is working
                try:
                    test_tensor = torch.randn(1, 3, 224, 224).to(self.device)
                    with torch.no_grad():
                        _ = self.model(test_tensor)
                    self.logger.info("MPS device test successful - GPU acceleration active")
                except Exception as e:
                    self.logger.warning(f"MPS test failed, falling back to CPU: {e}")
                    self.device = torch.device('cpu')
                    self.model = self.model.to(self.device)
            else:
                self.device = torch.device('cpu')
                self.logger.info("Using CPU for emotion detection model (MPS not available)")
            
            # Log device info for debugging
            device_info = model_cache.get_device_info()
            self.logger.info(f"Device info: {device_info}")
            self.logger.info(f"Actual device being used: {self.device}")
            
        except Exception as e:
            raise ValueError(f"Error loading emotion detection model: {e}")

    def _setup_logging(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG if self.config.debug else logging.INFO)

    def _get_face_hash(self, face_roi: np.ndarray) -> str:
        """Generate a hash for the face ROI to use as cache key - ULTRA-STABLE VERSION"""
        try:
            # Extract center region of the face (more stable than full bounding box)
            h, w = face_roi.shape[:2]
            center_h, center_w = h // 2, w // 2
            
            # Use 40% of the face area from the center (focus on core facial features)
            crop_size = min(h, w) * 0.4
            crop_h = int(crop_size)
            crop_w = int(crop_size)
            
            # Calculate crop coordinates (center crop)
            y1 = max(0, center_h - crop_h // 2)
            y2 = min(h, center_h + crop_h // 2)
            x1 = max(0, center_w - crop_w // 2)
            x2 = min(w, center_w + crop_w // 2)
            
            # Crop to center region
            center_face = face_roi[y1:y2, x1:x2]
            
            # Resize to very small size for hashing
            small_face = cv2.resize(center_face, (8, 8))
            
            # Convert to grayscale
            gray_face = cv2.cvtColor(small_face, cv2.COLOR_BGR2GRAY)
            
            # Apply very strong Gaussian blur to reduce noise
            blurred_face = cv2.GaussianBlur(gray_face, (3, 3), 0)
            
            # Ultra-aggressive quantization (only 2 levels: 0 or 1)
            # Threshold at the mean value
            mean_val = np.mean(blurred_face)
            quantized = (blurred_face > mean_val).astype(np.uint8)
            
            # Generate hash from quantized image
            return hashlib.md5(quantized.tobytes()).hexdigest()
            
        except Exception as e:
            # Fallback to original method if there's an error
            small_face = cv2.resize(face_roi, (32, 32))
            gray_face = cv2.cvtColor(small_face, cv2.COLOR_BGR2GRAY)
            return hashlib.md5(gray_face.tobytes()).hexdigest()

    def _get_face_hash_debug(self, face_roi: np.ndarray) -> Dict:
        """Debug version that returns hash components for analysis"""
        try:
            # Extract center region of the face
            h, w = face_roi.shape[:2]
            center_h, center_w = h // 2, w // 2
            
            # Use 40% of the face area from the center
            crop_size = min(h, w) * 0.4
            crop_h = int(crop_size)
            crop_w = int(crop_size)
            
            # Calculate crop coordinates (center crop)
            y1 = max(0, center_h - crop_h // 2)
            y2 = min(h, center_h + crop_h // 2)
            x1 = max(0, center_w - crop_w // 2)
            x2 = min(w, center_w + crop_w // 2)
            
            # Crop to center region
            center_face = face_roi[y1:y2, x1:x2]
            
            # Resize to very small size for hashing
            small_face = cv2.resize(center_face, (8, 8))
            
            # Convert to grayscale
            gray_face = cv2.cvtColor(small_face, cv2.COLOR_BGR2GRAY)
            
            # Apply very strong Gaussian blur to reduce noise
            blurred_face = cv2.GaussianBlur(gray_face, (3, 3), 0)
            
            # Ultra-aggressive quantization (only 2 levels: 0 or 1)
            mean_val = np.mean(blurred_face)
            quantized = (blurred_face > mean_val).astype(np.uint8)
            
            # Generate hash from quantized image
            hash_value = hashlib.md5(quantized.tobytes()).hexdigest()
            
            return {
                'hash': hash_value,
                'original_face_size': face_roi.shape,
                'center_face_size': center_face.shape,
                'small_face_size': small_face.shape,
                'crop_coords': (x1, y1, x2, y2),
                'gray_mean': np.mean(gray_face),
                'blurred_mean': np.mean(blurred_face),
                'quantized_mean': np.mean(quantized),
                'quantized_std': np.std(quantized),
                'quantized_unique': len(np.unique(quantized)),
                'threshold_value': mean_val
            }
            
        except Exception as e:
            return {'error': str(e)}

    def _should_skip_emotion_detection(self) -> bool:
        """Determine if we should skip emotion detection based on frame skipping settings"""
        if self.config.emotion_detection_skip_frames <= 1:
            return False
        return self._frame_counter % self.config.emotion_detection_skip_frames != 0

    def _process_emotion_batch(self) -> List[Dict[str, float]]:
        """Process the current emotion detection batch"""
        if not self._emotion_batch_buffer:
            return []
        
        try:
            # Convert all face images to PIL and preprocess
            pil_images = []
            for face_roi in self._emotion_batch_buffer:
                # Convert BGR to RGB
                face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(face_rgb)
                pil_images.append(pil_image)
            
            # Preprocess batch of images with custom size
            inputs = self.processor(
                images=pil_images, 
                return_tensors="pt",
                size={"height": self.config.emotion_input_size, "width": self.config.emotion_input_size}
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get predictions for entire batch
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1)
            
            # Convert batch results to list of dictionaries
            batch_results = []
            for i in range(probabilities.shape[0]):
                emotion_scores = {}
                for j, emotion in enumerate(self.emotion_labels):
                    emotion_scores[emotion] = probabilities[i][j].item()
                batch_results.append(emotion_scores)
            
            return batch_results
            
        except Exception as e:
            self.logger.error(f"Error in batch emotion detection: {e}")
            # Return empty results for failed batch
            return [{emotion: 0.0 for emotion in self.emotion_labels} for _ in self._emotion_batch_buffer]

    def detect_emotions(self, frame: np.ndarray, face_rect: Tuple[int, int, int, int]) -> Dict[str, float]:
        """Detect emotions in a face using the HuggingFace model - OPTIMIZED VERSION"""
        x, y, w, h = face_rect
        
        # Extract face region using memory pool
        face_roi = memory_pool.get_face_roi_buffer((h, w))
        face_roi[:] = frame[y:y + h, x:x + w]
        
        # Check if we should skip emotion detection
        if self._should_skip_emotion_detection():
            # Return cached result or default
            if self.config.enable_emotion_caching:
                face_hash = self._get_face_hash(face_roi)
                if face_hash in self._emotion_cache:
                    self._cached_emotion_hits += 1
                    memory_pool.return_image_buffer(face_roi)
                    return self._emotion_cache[face_hash]
            memory_pool.return_image_buffer(face_roi)
            return {emotion: 0.0 for emotion in self.emotion_labels}
        
        # Check cache first
        if self.config.enable_emotion_caching:
            face_hash = self._get_face_hash(face_roi)
            if face_hash in self._emotion_cache:
                self._cached_emotion_hits += 1
                memory_pool.return_image_buffer(face_roi)
                return self._emotion_cache[face_hash]
        
        # Single emotion detection (optimized - no batch overhead)
        try:
            start_time = time.perf_counter()
            
            # Convert BGR to RGB
            face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(face_rgb)
            
            # Preprocess image for the model with custom size
            inputs = self.processor(
                images=pil_image, 
                return_tensors="pt",
                size={"height": self.config.emotion_input_size, "width": self.config.emotion_input_size}
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get predictions with optimized settings
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1)
            
            # Convert to dictionary
            emotion_scores = {}
            for i, emotion in enumerate(self.emotion_labels):
                emotion_scores[emotion] = probabilities[0][i].item()
            
            # Cache result with more aggressive caching
            if self.config.enable_emotion_caching:
                face_hash = self._get_face_hash(face_roi)
                # Always cache (don't check size limit for maximum performance)
                self._emotion_cache[face_hash] = emotion_scores
                # Only clean cache if it gets too large
                if len(self._emotion_cache) > self.config.emotion_cache_size * 2:
                    # Remove oldest entries
                    keys_to_remove = list(self._emotion_cache.keys())[:len(self._emotion_cache) // 2]
                    for key in keys_to_remove:
                        del self._emotion_cache[key]
            
            # Record timing
            detection_time = (time.perf_counter() - start_time) * 1000
            self._emotion_detection_times.append(detection_time)
            self._total_emotion_detections += 1
            
            # Return buffer to memory pool
            memory_pool.return_image_buffer(face_roi)
            
            return emotion_scores
            
        except Exception as e:
            self.logger.error(f"Error detecting emotions: {e}")
            # Return buffer to memory pool even on error
            memory_pool.return_image_buffer(face_roi)
            return {emotion: 0.0 for emotion in self.emotion_labels}

    def get_emotion_detection_stats(self) -> Dict:
        """Get statistics about emotion detection performance"""
        avg_time = np.mean(self._emotion_detection_times) if self._emotion_detection_times else 0
        cache_hit_rate = self._cached_emotion_hits / max(1, self._total_emotion_detections)
        
        return {
            'avg_emotion_detection_time_ms': avg_time,
            'total_detections': self._total_emotion_detections,
            'cached_hits': self._cached_emotion_hits,
            'cache_hit_rate': cache_hit_rate,
            'cache_size': len(self._emotion_cache),
            'batch_buffer_size': len(self._emotion_batch_buffer)
        }

    def calculate_roi(self, width: int, height: int) -> Tuple[int, int, int, int]:
        """Calculate ROI coordinates based on configuration"""
        # Cache key for optimization
        cache_key = (width, height, self.config.roi_position.top, self.config.roi_position.bottom, 
                    self.config.roi_position.left, self.config.roi_position.right)
        
        # Check cache
        if self._cached_roi and self._cached_dimensions == cache_key:
            return self._cached_roi
        
        # Calculate ROI coordinates
        x1 = int(width * self.config.roi_position.left / 100)
        y1 = int(height * self.config.roi_position.top / 100)
        x2 = int(width * self.config.roi_position.right / 100)
        y2 = int(height * self.config.roi_position.bottom / 100)
        
        # Cache the result
        self._cached_roi = (x1, y1, x2, y2)
        self._cached_dimensions = cache_key
        return self._cached_roi

    def detect_emotions_batch(self, face_images: List[np.ndarray]) -> List[Dict[str, float]]:
        """Detect emotions in multiple faces using batch processing (legacy method - not used in current implementation)"""
        if not face_images:
            return []
        
        try:
            # Convert all face images to PIL and preprocess
            pil_images = []
            for face_roi in face_images:
                # Convert BGR to RGB
                face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(face_rgb)
                pil_images.append(pil_image)
            
            # Preprocess batch of images
            inputs = self.processor(images=pil_images, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get predictions for entire batch
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1)
            
            # Convert batch results to list of dictionaries
            batch_results = []
            for i in range(probabilities.shape[0]):
                emotion_scores = {}
                for j, emotion in enumerate(self.emotion_labels):
                    emotion_scores[emotion] = probabilities[i][j].item()
                batch_results.append(emotion_scores)
            
            return batch_results
            
        except Exception as e:
            self.logger.error(f"Error in batch emotion detection: {e}")
            # Return empty results for failed batch
            return [{emotion: 0.0 for emotion in self.emotion_labels} for _ in face_images]

    def process_frames_batch(self, frames: List[np.ndarray]) -> List[Dict]:
        """Process multiple frames in batch (legacy method - not used in current implementation)"""
        if not frames:
            return []
        
        batch_results = []
        
        # Process each frame in the batch
        for frame in frames:
            height, width = frame.shape[:2]
            roi = self.calculate_roi(width, height)
            
            # Use MediaPipe to detect faces with quality assessment
            faces_with_quality = self.mediapipe_detector.detect_faces_with_quality(frame, roi)
            
            frame_result = {}
            
            if faces_with_quality:
                # Get the best quality face
                best_face_result = self.mediapipe_detector.get_best_face(faces_with_quality)
                
                if best_face_result:
                    face_rect, quality = best_face_result
                    
                    # Check if face quality is good enough for processing
                    if self.mediapipe_detector.should_process_face(quality, self.config.focus_threshold):
                        # Detect emotions (no memory pooling)
                        emotion_scores = self.detect_emotions(frame, face_rect)
                        
                        # Filter emotions based on sensitivity and target emotions
                        filtered_emotions = {}
                        for emotion, score in emotion_scores.items():
                            if emotion in self.config.target_emotions:
                                # Convert sensitivity (1-5) to threshold (0.5-0.9)
                                # 1=50%, 2=60%, 3=70%, 4=80%, 5=90%
                                threshold = 0.4 + (self.config.emotion_sensitivity * 0.1)
                                if score > threshold:
                                    filtered_emotions[emotion] = score
                        
                        if filtered_emotions:
                            frame_result[face_rect] = {
                                'emotions': filtered_emotions,
                                'quality': quality
                            }
            
            batch_results.append(frame_result)
        
        return batch_results

    def process_frame(self, frame):
        """Process a single frame using MediaPipe for fast face detection - ULTRA OPTIMIZED"""
        # Increment frame counter for emotion detection skipping
        self._frame_counter += 1
        
        # ULTRA OPTIMIZATION: Skip face detection entirely if we're skipping emotion detection
        if self._should_skip_emotion_detection():
            return {}  # Return empty dict immediately - no processing needed
        
        height, width = frame.shape[:2]
        roi = self.calculate_roi(width, height)
        
        emotions_dict = {}
        
        # Use MediaPipe to detect faces with quality assessment
        faces_with_quality = self.mediapipe_detector.detect_faces_with_quality(frame, roi)
        
        if faces_with_quality:
            # Get the best quality face
            best_face_result = self.mediapipe_detector.get_best_face(faces_with_quality)
            
            if best_face_result:
                face_rect, quality = best_face_result
                
                # Simplified quality check - just check confidence
                if quality.confidence > 0.7:  # Lower threshold for speed
                    # Detect emotions
                    emotion_scores = self.detect_emotions(frame, face_rect)
                    
                    # Filter emotions based on sensitivity and target emotions
                    filtered_emotions = {}
                    for emotion, score in emotion_scores.items():
                        if emotion in self.config.target_emotions:
                            # Convert sensitivity (1-5) to threshold (0.5-0.9)
                            # 1=50%, 2=60%, 3=70%, 4=80%, 5=90%
                            threshold = 0.4 + (self.config.emotion_sensitivity * 0.1)
                            if score > threshold:
                                filtered_emotions[emotion] = score
                    
                    if filtered_emotions:
                        emotions_dict[face_rect] = {
                            'emotions': filtered_emotions,
                            'quality': quality
                        }
        
        return emotions_dict

    def _draw_debug(self, frame: np.ndarray, faces: List[Tuple[int, int, int, int]], 
                   emotions_dict: Dict, roi: Tuple[int, int, int, int]) -> np.ndarray:
        """Draw debug information on frame"""
        debug_frame = frame.copy()
        
        # Draw ROI
        x1, y1, x2, y2 = roi
        cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(debug_frame, "ROI", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Draw detected faces and emotions
        for face_rect, face_data in emotions_dict.items():
            x, y, w, h = face_rect
            cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Draw emotion scores
            y_offset = y + h + 20
            for emotion, score in face_data['emotions'].items():
                text = f"{emotion}: {score:.2f}"
                cv2.putText(debug_frame, text, (x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                y_offset += 15
        
        return debug_frame

    def extract_emotions_from_video(self, video_path: Path, output_dir: Path, 
                                  should_stop: Callable[[], bool] = None,
                                  progress_callback: Callable[[dict], None] = None) -> int:
        """Extract emotion frames from video (legacy method - not used in current implementation)"""
        
        # Note: This legacy method is not used in the current implementation
        # The app uses extract_emotions_best_frames() instead for better performance and thumbnail selection
        self.logger.info(f"Using legacy sequential processing (not recommended)")
        return self._extract_emotions_sequential(video_path, output_dir, should_stop, progress_callback)

    def _extract_emotions_sequential(self, video_path: Path, output_dir: Path, 
                                   should_stop: Callable[[], bool] = None,
                                   progress_callback: Callable[[dict], None] = None) -> int:
        """Ultra-high-performance sequential processing method"""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        progress_tracker = ProgressTracker(total_frames, update_interval=2.0)  # Very infrequent updates
        
        frame_count = 0
        processed_count = 0
        saved_count = 0
        
        # Clear deduplicator for new video
        self.deduplicator.clear()
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Processing video: {video_path.name} (Ultra-high-performance sequential mode)")
        self.logger.info(f"Total frames: {total_frames}, FPS: {fps}")
        self.logger.info(f"Emotion sensitivity: {self.config.emotion_sensitivity}/5 (threshold: {(0.4 + self.config.emotion_sensitivity * 0.1) * 100:.0f}%)")
        self.logger.info(f"Target emotions: {self.config.target_emotions}")
        if self.config.enable_deduplication:
            self.logger.info(f"Deduplication: {self.config.deduplication_window}s window, max {self.config.max_frames_per_emotion} frames per emotion")
        
        try:
            while True:
                if should_stop and should_stop():
                    self.logger.info("Processing stopped by user")
                    break
                
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                timestamp = frame_count / fps
                
                # Process frame directly (no batching overhead)
                emotions_dict = self.process_frame(frame)
                
                if emotions_dict:
                    processed_count += 1
                    
                    # Check each emotion for deduplication
                    for face_rect, face_data in emotions_dict.items():
                        for emotion, score in face_data['emotions'].items():
                            # Check if we should save this emotion detection
                            if self.deduplicator.should_save_emotion(emotion, score, timestamp, frame, frame_count):
                                # Save the frame
                                emotions_str = f"{emotion}_{score:.2f}_{frame_count:06d}"
                                filename = f"{emotions_str}.{self.config.image_format}"
                                filepath = output_dir / filename
                                
                                # Save frame with format-specific settings
                                if self.config.image_format.lower() == "png":
                                    cv2.imwrite(str(filepath), frame, 
                                              [cv2.IMWRITE_PNG_COMPRESSION, self.config.image_quality])
                                elif self.config.image_format.lower() == "jpg":
                                    cv2.imwrite(str(filepath), frame, 
                                              [cv2.IMWRITE_JPEG_QUALITY, self.config.image_quality])
                                elif self.config.image_format.lower() == "tiff":
                                    cv2.imwrite(str(filepath), frame, 
                                              [cv2.IMWRITE_TIFF_COMPRESSION, self.config.image_quality])
                                elif self.config.image_format.lower() == "webp":
                                    cv2.imwrite(str(filepath), frame, 
                                              [cv2.IMWRITE_WEBP_QUALITY, self.config.image_quality])
                                else:
                                    cv2.imwrite(str(filepath), frame, 
                                              [cv2.IMWRITE_PNG_COMPRESSION, 6])
                                saved_count += 1
                                
                                self.logger.info(f"Saved {emotion} ({score:.2f}) at frame {frame_count}")
                                if progress_callback:
                                    progress_callback({
                                        "type": "log",
                                        "timestamp": time.strftime('%H:%M:%S'),
                                        "message": f"Saved {emotion} ({score:.2f}) at frame {frame_count}"
                                    })
                
                # Update progress very infrequently
                if progress_callback and frame_count % 120 == 0:  # Update every 120 frames (much less frequent)
                    progress_data = progress_tracker.update(frame_count)
                    if progress_data:
                        self.logger.debug(f"Sending progress update: {progress_data}")
                        progress_callback(progress_data)
        
        finally:
            cap.release()
        
        self.logger.info(f"Ultra-high-performance processing complete. Processed {processed_count} frames with emotions, saved {saved_count} unique emotion frames.")
        if self.config.enable_deduplication:
            self.logger.info(f"Deduplication reduced output by {processed_count - saved_count} frames ({((processed_count - saved_count) / max(processed_count, 1) * 100):.1f}% reduction)")
        else:
            self.logger.info("Deduplication disabled - all frames saved for maximum speed")
        if progress_callback:
            progress_callback({
                "type": "log",
                "timestamp": time.strftime('%H:%M:%S'),
                "message": f"Summary: Processed {processed_count} frames, saved {saved_count} unique emotion frames from '{video_path.name}' using ultra-high-performance processing."
            })
            progress_callback({"type": "complete"})
        return saved_count

    def extract_emotions_best_frames(self, video_path: Path, output_dir: Path, 
                                   should_stop: Callable[[], bool] = None,
                                   progress_callback: Callable[[dict], None] = None,
                                   skip_interval: int = 25, emotion_threshold: float = 0.8,
                                   search_forward: int = 5, search_backward: int = 5) -> int:
        """Extract emotion frames using 'skip N, find best' approach for thumbnail selection"""
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Processing video with best-frame approach: {video_path.name}")
        self.logger.info(f"Total frames: {total_frames}, FPS: {fps}")
        self.logger.info(f"Skip interval: {skip_interval}, Emotion threshold: {emotion_threshold}")
        self.logger.info(f"Search window: {search_backward} frames back, {search_forward} frames forward")
        self.logger.info(f"Target emotions: {self.config.target_emotions}")
        
        # Initialize best frame finder
        best_finder = BestFrameFinder(self.config, self)
        
        # Initialize timing for progress tracking
        self._start_time = time.time()
        
        saved_count = 0
        processed_windows = 0
        
        try:
            # Process every Nth frame
            for frame_num in range(0, total_frames, skip_interval):
                if should_stop and should_stop():
                    self.logger.info("Processing stopped by user")
                    break
                
                # Find best frame in window around this frame
                best_result = best_finder.find_best_frame_in_window(
                    video_path, frame_num, emotion_threshold, search_forward, search_backward, should_stop
                )
                
                # Check stop condition after finding best frame
                if should_stop and should_stop():
                    self.logger.info("Processing stopped by user")
                    break
                
                if best_result:
                    best_frame_num, best_score, best_emotion = best_result
                    
                    # Load the best frame
                    cap.set(cv2.CAP_PROP_POS_FRAMES, best_frame_num)
                    ret, frame = cap.read()
                    
                    if ret:
                        # Save the best frame
                        emotions_str = f"{best_emotion}_{best_score:.2f}_{best_frame_num:06d}"
                        filename = f"{emotions_str}.{self.config.image_format}"
                        filepath = output_dir / filename
                        
                        # Save frame with format-specific settings
                        if self.config.image_format.lower() == "png":
                            cv2.imwrite(str(filepath), frame, 
                                      [cv2.IMWRITE_PNG_COMPRESSION, self.config.image_quality])
                        elif self.config.image_format.lower() == "jpg":
                            cv2.imwrite(str(filepath), frame, 
                                      [cv2.IMWRITE_JPEG_QUALITY, self.config.image_quality])
                        elif self.config.image_format.lower() == "tiff":
                            cv2.imwrite(str(filepath), frame, 
                                      [cv2.IMWRITE_TIFF_COMPRESSION, self.config.image_quality])
                        elif self.config.image_format.lower() == "webp":
                            cv2.imwrite(str(filepath), frame, 
                                      [cv2.IMWRITE_WEBP_QUALITY, self.config.image_quality])
                        else:
                            cv2.imwrite(str(filepath), frame, 
                                      [cv2.IMWRITE_PNG_COMPRESSION, 6])
                        
                        saved_count += 1
                        self.logger.info(f"Saved best frame {best_frame_num}: {best_emotion} ({best_score:.3f})")
                        
                        if progress_callback:
                            progress_callback({
                                "type": "log",
                                "timestamp": time.strftime('%H:%M:%S'),
                                "message": f"Saved best frame {best_frame_num}: {best_emotion} ({best_score:.3f})"
                            })
                
                processed_windows += 1
                
                # Progress updates
                if progress_callback and processed_windows % 10 == 0:  # Update every 10 windows
                    # Calculate progress percentage
                    progress_percentage = (frame_num / total_frames) * 100
                    
                    # Calculate estimated time based on processing speed
                    current_time = time.time()
                    if not hasattr(self, '_start_time'):
                        self._start_time = current_time
                    
                    elapsed = current_time - self._start_time
                    elapsed_str = time.strftime('%H:%M:%S', time.gmtime(elapsed))
                    
                    # Estimate FPS based on windows processed
                    if elapsed > 0:
                        windows_per_second = processed_windows / elapsed
                        fps = windows_per_second * skip_interval  # Convert to frame FPS
                    else:
                        fps = 0
                    
                    # Calculate ETA
                    if fps > 0:
                        remaining_frames = total_frames - frame_num
                        eta_seconds = remaining_frames / fps
                        eta_str = time.strftime('%H:%M:%S', time.gmtime(eta_seconds))
                    else:
                        eta_str = "00:00:00"
                    
                    progress_callback({
                        "type": "progress",
                        "progress": round(progress_percentage, 1),
                        "fps": round(fps, 1),
                        "elapsed": elapsed_str,
                        "eta": eta_str
                    })
        
        finally:
            cap.release()
        
        self.logger.info(f"Best-frame processing complete. Processed {processed_windows} windows, saved {saved_count} best emotion frames.")
        
        if progress_callback:
            progress_callback({
                "type": "log",
                "timestamp": time.strftime('%H:%M:%S'),
                "message": f"Summary: Processed {processed_windows} windows, saved {saved_count} best emotion frames from '{video_path.name}' using best-frame approach."
            })
            progress_callback({"type": "complete"})
        
        return saved_count

    @classmethod
    def create_optimized_config(cls, emotion_sensitivity: int, roi_position: RoiPosition) -> 'ProcessingConfig':
        """Create an optimized configuration for production use"""
        return ProcessingConfig(
            emotion_sensitivity=emotion_sensitivity,
            roi_position=roi_position,
            # Performance optimizations
                            emotion_detection_skip_frames=15,  # Process every 15th frame (1 out of 15) for maximum speed
                emotion_cache_size=2000,  # Large cache size for maximum hit rates and performance
                emotion_input_size=224,  # ViT model input size requirement (224x224 pixels)
            enable_emotion_caching=True,
            enable_batch_emotion_detection=False,  # Disable batch processing to avoid overhead
            # Standard settings
            enable_deduplication=True,
            deduplication_window=1.0,
            max_frames_per_emotion=2,
            min_score_improvement=0.05,
            focus_threshold=0.75,  # Optimized for YouTube thumbnail sharpness
            enable_batch_processing=False
        )

    @classmethod
    def create_ultra_optimized_config(cls, emotion_sensitivity: int, roi_position: RoiPosition) -> 'ProcessingConfig':
        """Create an ultra-optimized configuration for maximum speed"""
        return ProcessingConfig(
            emotion_sensitivity=emotion_sensitivity,
            roi_position=roi_position,
            # ULTRA Performance optimizations
            emotion_detection_skip_frames=20,  # Process every 20th frame (1 out of 20) for maximum speed
            emotion_cache_size=3000,  # Large cache size for maximum hit rates and performance
            emotion_input_size=224,  # ViT model input size requirement (224x224 pixels)
            enable_emotion_caching=True,
            enable_batch_emotion_detection=False,  # Disable batch processing to avoid overhead
            # Standard settings
            enable_deduplication=True,
            deduplication_window=1.0,
            max_frames_per_emotion=2,
            min_score_improvement=0.05,
            focus_threshold=0.75,  # Optimized for YouTube thumbnail sharpness
            enable_batch_processing=False
        )

class BestFrameFinder:
    """Find the best emotion frame in each window by checking forward/backward only if initial frame qualifies"""
    
    def __init__(self, config: ProcessingConfig, emotion_detector: 'EmotionDetector'):
        self.config = config
        self.emotion_detector = emotion_detector
        self.logger = logging.getLogger(__name__)
        self.saved_frames = []
    
    def find_best_frame_in_window(self, video_path: Path, start_frame: int, 
                                 emotion_threshold: float = 0.8, 
                                 search_forward: int = 5, 
                                 search_backward: int = 5,
                                 should_stop: Callable[[], bool] = None) -> Optional[Tuple[int, float, str]]:
        """
        Only scrub forward/backward if the initial frame has a qualifying emotion.
        Returns (frame_number, score, emotion) or None if no qualifying emotion found in initial frame.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 1. Check initial frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return None
        
        # Check stop condition before processing initial frame
        if should_stop and should_stop():
            cap.release()
            return None
            
        emotions_dict = self.emotion_detector.process_frame(frame)
        best_score = 0.0
        best_frame = None
        best_emotion = None
        # Find best qualifying emotion in initial frame
        for face_rect, face_data in emotions_dict.items():
            for emotion, score in face_data['emotions'].items():
                if score > emotion_threshold and score > best_score:
                    best_score = score
                    best_frame = start_frame
                    best_emotion = emotion
        # If no qualifying emotion, skip window
        if best_frame is None:
            cap.release()
            return None
        # 2. Scrub forward/backward for better score of the same emotion
        search_start = max(0, start_frame - search_backward)
        search_end = min(total_frames - 1, start_frame + search_forward)
        for frame_num in range(search_start, search_end + 1):
            # Check stop condition at the beginning of each frame processing
            if should_stop and should_stop():
                cap.release()
                return None
                
            if frame_num == start_frame:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret:
                continue
            emotions_dict = self.emotion_detector.process_frame(frame)
            for face_rect, face_data in emotions_dict.items():
                for emotion, score in face_data['emotions'].items():
                    if emotion == best_emotion and score > best_score:
                        best_score = score
                        best_frame = frame_num
        cap.release()
        return (best_frame, best_score, best_emotion) 