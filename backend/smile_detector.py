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

class RoiPosition(BaseModel):
    top: float
    bottom: float
    left: float
    right: float

@dataclass
class ProcessingConfig:
    """Configuration settings for smile detection processing"""
    skip_frames: int
    min_smile_duration: float
    debug: bool
    frame_buffer_size: int
    smile_sensitivity: int
    roi_position: RoiPosition
    cache_size: int = 100
    target_width_4k: int = 1920
    target_width_hd: int = 1280
    compression_params: List[int] = None

    def __post_init__(self):
        if self.compression_params is None:
            self.compression_params = [
                cv2.IMWRITE_PNG_COMPRESSION, 9,
                cv2.IMWRITE_PNG_STRATEGY, cv2.IMWRITE_PNG_STRATEGY_DEFAULT
            ]

class ProgressTracker:
    """Tracks and reports processing progress with performance metrics"""
    def __init__(self, total_frames: int, update_interval: float = 5):
        self.total_frames = total_frames
        self.last_update = 0
        self.last_time = time.time()
        self.update_interval = update_interval
        self.start_time = self.last_time

    def update(self, frame_count: int) -> Optional[str]:
        current_time = time.time()
        if current_time - self.last_time >= self.update_interval:
            progress = (frame_count / self.total_frames) * 100
            fps = (frame_count - self.last_update) / (current_time - self.last_time)
            elapsed = current_time - self.start_time
            eta = (self.total_frames - frame_count) / (frame_count / elapsed) if frame_count > 0 else 0
            
            status = (
                f"Progress: {progress:.1f}% "
                f"({frame_count}/{self.total_frames} frames) - "
                f"{fps:.1f} fps - "
                f"Elapsed: {timedelta(seconds=int(elapsed))} - "
                f"ETA: {timedelta(seconds=int(eta))}"
            )
            
            self.last_update = frame_count
            self.last_time = current_time
            return status
        return None

class SmileDetector:
    """Advanced smile detection with optimization and caching"""
    
    def __init__(self, config: ProcessingConfig):
        """Initialize the smile detector with given configuration"""
        self.config = config
        self.face_cache = {}
        self._initialize_cascades()
        self._setup_logging()

    def _initialize_cascades(self):
        """Initialize and verify cascade classifiers"""
        cascade_path = cv2.data.haarcascades
        self.face_cascade = cv2.CascadeClassifier(cascade_path + 'haarcascade_frontalface_alt2.xml')
        self.smile_cascade = cv2.CascadeClassifier(cascade_path + 'haarcascade_smile.xml')
        
        if self.face_cascade.empty() or self.smile_cascade.empty():
            raise ValueError("Error loading cascade classifiers. Check OpenCV installation.")

    def _setup_logging(self):
        """Configure logging with appropriate level and format"""
        logging.basicConfig(
            level=logging.DEBUG if self.config.debug else logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Enhance frame for better detection"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        # Adjust contrast and brightness
        gray = cv2.convertScaleAbs(gray, alpha=1.2, beta=10)
        
        # Apply bilateral filter to reduce noise while preserving edges
        return cv2.bilateralFilter(gray, 9, 75, 75)

    def calculate_target_dimensions(self, frame: np.ndarray) -> Tuple[int, int]:
        """Calculate target dimensions based on input resolution"""
        height, width = frame.shape[:2]
        target_width = self.config.target_width_4k if width > 3000 else self.config.target_width_hd
        scale = target_width / width
        return target_width, int(height * scale)

    def calculate_roi(self, width: int, height: int) -> Tuple[int, int, int, int]:
        """Calculate Region of Interest based on configured position"""
        roi = self.config.roi_position
        x1 = int(width * roi.left / 100)
        y1 = int(height * roi.top / 100)
        x2 = int(width * roi.right / 100)
        y2 = int(height * roi.bottom / 100)
        return (x1, y1, x2, y2)

    def detect_faces(self, processed_frame: np.ndarray, roi: Tuple[int, int, int, int]) -> List[Tuple[int, int, int, int]]:
        """Detect faces in the frame with caching"""
        x1, y1, x2, y2 = roi
        roi_region = processed_frame[y1:y2, x1:x2]
        
        # Create cache key from frame data and ROI
        cache_key = hash(roi_region.tobytes() + str(roi).encode())
        
        # Check cache first
        if cache_key in self.face_cache:
            return self.face_cache[cache_key]
        
        # Detect faces with optimized parameters
        faces = self.face_cascade.detectMultiScale(
            roi_region,
            scaleFactor=1.15,
            minNeighbors=3,
            minSize=(45, 45),
            maxSize=(500, 500)
        )
        
        # Adjust coordinates relative to ROI
        all_faces = [(x + x1, y + y1, w, h) for (x, y, w, h) in faces]
        
        # Keep only the largest face if any faces are found
        result = [max(all_faces, key=lambda rect: rect[2] * rect[3])] if all_faces else []
        
        # Update cache with size limit
        if len(self.face_cache) > self.config.cache_size:
            self.face_cache.clear()
        self.face_cache[cache_key] = result
        
        return result

    def detect_smile(self, processed_frame: np.ndarray, face_rect: Tuple[int, int, int, int]) -> List[Tuple[int, int, int, int]]:
        """Detect smiles within a face region"""
        x, y, w, h = face_rect
        face_region = processed_frame[y:y + h, x:x + w]
        
        # Focus on lower half of face
        lower_half_y = int(h * 0.50)
        lower_face = face_region[lower_half_y:, :]
        
        # Calculate smile detection parameters based on face size
        smile_min_size = (int(w*0.30), int(h*0.17))
        smile_max_size = (int(w*0.85), int(h*0.50))
        
        # Detect smiles with configured sensitivity
        smiles = self.smile_cascade.detectMultiScale(
            lower_face,
            scaleFactor=1.12,
            minNeighbors=self.config.smile_sensitivity,
            minSize=smile_min_size,
            maxSize=smile_max_size
        )
        
        # Adjust y coordinates relative to face position
        return [(sx, sy + lower_half_y, sw, sh) for (sx, sy, sw, sh) in smiles]

    def draw_debug_overlay(self, frame: np.ndarray, faces: List[Tuple[int, int, int, int]], 
                         smiles_dict: Dict, roi: Tuple[int, int, int, int]) -> np.ndarray:
        """Draw debug visualization"""
        debug_frame = frame.copy()
        
        # Draw ROI
        x1, y1, x2, y2 = roi
        cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw faces and smiles
        for face_rect in faces:
            x, y, w, h = face_rect
            cv2.rectangle(debug_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Draw detected smiles
            if tuple(face_rect) in smiles_dict:
                for (sx, sy, sw, sh) in smiles_dict[tuple(face_rect)]:
                    cv2.rectangle(debug_frame, 
                                (x + sx, y + sy),
                                (x + sx + sw, y + sy + sh),
                                (0, 0, 255), 2)
        
        # Add face count
        cv2.putText(debug_frame, f"Faces: {len(faces)}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return debug_frame

    def process_frame(self, frame: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
        """Process a single frame"""
        # Scale frame to target dimensions
        target_width, target_height = self.calculate_target_dimensions(frame)
        detection_frame = cv2.resize(frame, (target_width, target_height),
                                   interpolation=cv2.INTER_AREA)
        
        # Calculate ROI
        roi = self.calculate_roi(target_width, target_height)
        
        # Process frame
        processed = self.preprocess_frame(detection_frame)
        faces = self.detect_faces(processed, roi)
        
        smiles_dict = {}
        has_smile = False
        
        # Detect smiles if faces found
        if faces:
            face_rect = faces[0]
            smiles = self.detect_smile(processed, face_rect)
            if smiles:
                has_smile = True
                smiles_dict[tuple(face_rect)] = smiles
        
        # Create debug frame if needed
        debug_frame = (self.draw_debug_overlay(detection_frame, faces, smiles_dict, roi)
                      if self.config.debug else None)
        
        return has_smile, debug_frame

    def extract_smiles_from_video(self, video_path: Path, output_folder: Path, 
                                should_stop_check: Optional[Callable[[], bool]] = None) -> int:
        """Extract smiles from video"""
        should_stop_check = should_stop_check or (lambda: False)
        
        self.logger.info("Starting video processing...")
        output_folder.mkdir(parents=True, exist_ok=True)
        debug_folder = output_folder / 'debug' if self.config.debug else None
        
        if debug_folder:
            debug_folder.mkdir(exist_ok=True)
        
        video = cv2.VideoCapture(str(video_path))
        if not video.isOpened():
            raise ValueError(f"Error opening video file: {video_path}")
        
        try:
            # Get video properties
            fps = video.get(cv2.CAP_PROP_FPS)
            total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            min_smile_frames = int(self.config.min_smile_duration * fps)
            
            # Initialize buffers
            frame_buffer = deque(maxlen=self.config.frame_buffer_size * 2 + 1)
            processed_buffer = deque(maxlen=self.config.frame_buffer_size * 2 + 1)
            
            # Initialize counters
            frame_count = smile_count = consecutive_smiles = 0
            last_smile_frame = None
            
            # Initialize progress tracker
            progress = ProgressTracker(total_frames)
            
            while True:
                # Check for stop signal
                if should_stop_check():
                    self.logger.info("Processing stopped by user")
                    return smile_count
                
                # Read frame
                ret, frame = video.read()
                if not ret:
                    break
                
                frame_buffer.append(frame.copy())
                
                # Process frame if needed
                if frame_count % self.config.skip_frames == 0:
                    # Process frame
                    is_smiling, debug_frame = self.process_frame(frame)
                    processed_buffer.append((is_smiling, debug_frame))
                    
                    # Handle smile detection
                    if is_smiling:
                        consecutive_smiles += 1
                        if self._should_save_smile(consecutive_smiles, min_smile_frames,
                                                 last_smile_frame, frame_count, fps):
                            smile_count = self._save_smile_sequence(
                                smile_count, frame_count, fps,
                                frame_buffer, processed_buffer,
                                output_folder, debug_folder
                            )
                            last_smile_frame = frame_count
                    else:
                        consecutive_smiles = 0
                    
                    # Update progress
                    if status := progress.update(frame_count):
                        self.logger.info(status)
                
                frame_count += 1
            
            return smile_count
            
        finally:
            video.release()

    def _should_save_smile(self, consecutive_smiles: int, min_smile_frames: int,
                          last_smile_frame: Optional[int], frame_count: int, fps: float) -> bool:
        """Determine if current smile should be saved"""
        return (consecutive_smiles >= min_smile_frames and
                (last_smile_frame is None or
                 frame_count - last_smile_frame > fps * 2))

    def _save_smile_sequence(self, smile_count: int, frame_count: int, fps: float,
                           frame_buffer: deque, processed_buffer: deque,
                           output_folder: Path, debug_folder: Optional[Path]) -> int:
        """Save smile sequence with optimized file handling"""
        smile_count += 1
        timestamp = timedelta(seconds=frame_count/fps)
        
        for i, buf_frame in enumerate(frame_buffer):
            relative_pos = i - self.config.frame_buffer_size
            
            # Save main frame
            filename = f'smile_{smile_count:03d}_frame_{relative_pos:+d}_time_{timestamp}.png'
            cv2.imwrite(str(output_folder / filename), buf_frame,
                       self.config.compression_params)
            
            # Save debug frame if needed
            if self.config.debug and i < len(processed_buffer):
                self._save_debug_smile_frame(debug_folder, smile_count,
                                          relative_pos, timestamp,
                                          processed_buffer[i][1])
        
        self.logger.info(f"Saved smile sequence {smile_count} at {timestamp}")
        return smile_count

    def _save_debug_frame(self, debug_folder: Path, frame_count: int,
                         is_smiling: bool, frame: np.ndarray):
        """Save debug frame with optimized compression"""
        debug_path = debug_folder / f'debug_frame_{frame_count}_smile_{is_smiling}.png'
        cv2.imwrite(str(debug_path), frame, self.config.compression_params)

    def _save_debug_smile_frame(self, debug_folder: Path, smile_count: int,
                              relative_pos: int, timestamp: timedelta,
                              frame: np.ndarray):
        """Save debug smile frame with optimized compression"""
        debug_filename = f'debug_smile_{smile_count:03d}_frame_{relative_pos:+d}_time_{timestamp}.png'
        cv2.imwrite(str(debug_folder / debug_filename), frame,
                   self.config.compression_params)

def process_videos(base_dir: Optional[Path] = None):
    """Process multiple videos with parallel execution"""
    base_dir = base_dir or Path.home() / 'Desktop/Smile Youre On Candid Camera'
    video_dir = base_dir / '1 VIDEO'
    output_dir = base_dir / '2 SMILES'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    video_files = list(video_dir.glob('*.[mM][pP]4'))
    video_files.extend(video_dir.glob('*.[mM][oO][vV]'))
    video_files.extend(video_dir.glob('*.[aA][vV][iI]'))
    
    if not video_files:
        print(f"No video files found in {video_dir}")
        return
    
    config = ProcessingConfig(
        skip_frames=8,            # Reduced from 4 to catch more potential smiles
        min_smile_duration=0.5,   # Reduced from 0.4 to allow slightly shorter smiles
        debug=False,              # Set to True temporarily to see what's being detected
        frame_buffer_size=2
    )
    
    print(f"Found {len(video_files)} video files to process:")
    for i, video_file in enumerate(video_files, 1):
        print(f"{i}. {video_file.name}")
    print("\nStarting processing...")
    
    detector = SmileDetector(config)
    total_smiles = 0
    
    for i, video_file in enumerate(video_files, 1):
        print(f"\n{'='*50}")
        print(f"Processing video {i}/{len(video_files)}: {video_file.name}")
        video_output_dir = output_dir / video_file.stem
        print(f"Output directory: {video_output_dir}")
        print(f"{'='*50}\n")
        
        try:
            num_smiles = detector.extract_smiles_from_video(video_file, video_output_dir)
            total_smiles += num_smiles
            print(f"\nCompleted {video_file.name}: Found {num_smiles} smiles.")
        except Exception as e:
            print(f"Error processing {video_file.name}: {str(e)}")
            logging.error(f"Error processing video: {str(e)}", exc_info=True)
    
    print(f"\n{'='*50}")
    print(f"Processing complete!")
    print(f"Processed {len(video_files)} videos")
    print(f"Found {total_smiles} total smiles")
    print(f"{'='*50}")

if __name__ == "__main__":
    try:
        process_videos()
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        logging.error("Unexpected error", exc_info=True)