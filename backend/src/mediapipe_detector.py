import mediapipe as mp
import cv2
import numpy as np
import logging
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

@dataclass
class FaceQuality:
    """Face quality assessment results"""
    confidence: float  # Face detection confidence (0-1)
    blur_score: float  # Blur assessment (higher = sharper)
    occlusion_score: float  # Occlusion assessment (higher = less occluded)
    lighting_score: float  # Lighting assessment (higher = better lighting)
    overall_quality: float  # Combined quality score (0-1)

class MediaPipeFaceDetector:
    """Fast face detection and quality assessment using MediaPipe"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Configure face detection with lower confidence for glasses compatibility
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,  # 0=short range, 1=full range
            min_detection_confidence=0.3  # Lower threshold to better detect faces with glasses
        )
        
        self.logger.info("MediaPipe Face Detection initialized")
    
    def detect_faces_with_quality(self, frame: np.ndarray, roi: Tuple[int, int, int, int]) -> List[Tuple[Tuple[int, int, int, int], FaceQuality]]:
        """
        Detect faces and assess quality in a single pass
        
        Returns:
            List of (face_rect, quality) tuples
        """
        x1, y1, x2, y2 = roi
        roi_frame = frame[y1:y2, x1:x2]
        
        # Convert BGR to RGB (MediaPipe requires RGB)
        rgb_frame = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        results = self.face_detection.process(rgb_frame)
        
        faces_with_quality = []
        
        if results.detections:
            for detection in results.detections:
                # Get face bounding box
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = roi_frame.shape
                
                # Convert relative coordinates to absolute
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                # Adjust coordinates back to original frame space
                face_rect = (x + x1, y + y1, width, height)
                
                # Assess face quality
                quality = self._assess_face_quality(roi_frame, detection)
                
                faces_with_quality.append((face_rect, quality))
        
        return faces_with_quality
    
    def _assess_face_quality(self, face_frame: np.ndarray, detection) -> FaceQuality:
        """Assess the quality of a detected face"""
        
        # Get detection confidence
        confidence = detection.score[0]
        
        # Calculate blur score using Laplacian variance (optimized)
        blur_score = self._calculate_blur_score(face_frame)
        
        # Calculate occlusion score (simplified)
        occlusion_score = self._calculate_occlusion_score(face_frame)
        
        # Calculate lighting score
        lighting_score = self._calculate_lighting_score(face_frame)
        
        # Combine scores into overall quality (reduced occlusion weight for glasses)
        overall_quality = (
            confidence * 0.5 +      # Increased confidence weight
            blur_score * 0.3 +      # Keep blur weight
            occlusion_score * 0.1 + # Reduced occlusion weight (was 0.2)
            lighting_score * 0.1    # Keep lighting weight
        )
        
        return FaceQuality(
            confidence=confidence,
            blur_score=blur_score,
            occlusion_score=occlusion_score,
            lighting_score=lighting_score,
            overall_quality=overall_quality
        )
    
    def _calculate_blur_score(self, face_frame: np.ndarray) -> float:
        """Calculate blur score using optimized Laplacian variance"""
        # Downsample for speed
        small_face = cv2.resize(face_frame, (64, 64))
        gray = cv2.cvtColor(small_face, cv2.COLOR_BGR2GRAY)
        
        # Calculate Laplacian variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        
        # Normalize to 0-1 range (typical values: 0-1000)
        normalized_score = min(variance / 500.0, 1.0)
        
        return normalized_score
    
    def _calculate_occlusion_score(self, face_frame: np.ndarray) -> float:
        """Calculate occlusion score with glasses-aware detection"""
        gray = cv2.cvtColor(face_frame, cv2.COLOR_BGR2GRAY)
        
        # Use adaptive edge detection to better handle glasses
        # Lower thresholds to detect more subtle occlusions while avoiding glasses
        edges = cv2.Canny(gray, 30, 100)
        
        # Focus on center region where most facial occlusions occur
        h, w = edges.shape
        center_h_start, center_h_end = int(h * 0.2), int(h * 0.8)
        center_w_start, center_w_end = int(w * 0.2), int(w * 0.8)
        center_region = edges[center_h_start:center_h_end, center_w_start:center_w_end]
        
        # Calculate edge density in center region only
        center_edge_density = np.sum(center_region > 0) / (center_region.shape[0] * center_region.shape[1])
        
        # Check for glasses patterns (horizontal lines in upper region)
        upper_region = edges[:int(h * 0.4), :]
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
        horizontal_lines = cv2.morphologyEx(upper_region, cv2.MORPH_OPEN, horizontal_kernel)
        glasses_indicator = np.sum(horizontal_lines > 0) / (upper_region.shape[0] * upper_region.shape[1])
        
        # If glasses are detected, reduce occlusion penalty
        if glasses_indicator > 0.02:  # Threshold for glasses detection
            # Reduce occlusion score for faces with glasses
            normalized_score = min(center_edge_density / 0.08, 0.7)  # Cap at 0.7 for glasses
        else:
            # Normal occlusion detection for faces without glasses
            normalized_score = min(center_edge_density / 0.05, 1.0)
        
        return normalized_score
    
    def _calculate_lighting_score(self, face_frame: np.ndarray) -> float:
        """Calculate lighting quality score"""
        # Convert to LAB color space
        lab = cv2.cvtColor(face_frame, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        
        # Calculate lighting statistics
        mean_brightness = np.mean(l_channel)
        std_brightness = np.std(l_channel)
        
        # Good lighting: mean around 128, low std (even lighting)
        brightness_score = 1.0 - abs(mean_brightness - 128) / 128
        evenness_score = 1.0 - min(std_brightness / 50.0, 1.0)
        
        # Combine scores
        lighting_score = (brightness_score * 0.7 + evenness_score * 0.3)
        
        return max(0.0, min(1.0, lighting_score))
    
    def should_process_face(self, quality: FaceQuality, focus_threshold: float = 0.5) -> bool:
        """Determine if a face should be processed based on quality"""
        
        # Check overall quality threshold
        if quality.overall_quality < focus_threshold:
            if self.config.debug:
                self.logger.debug(f"Face rejected - quality: {quality.overall_quality:.2f}, "
                                 f"threshold: {focus_threshold:.2f}")
            return False
        
        # Additional checks for very poor conditions
        if quality.blur_score < 0.1:  # Very blurry
            if self.config.debug:
                self.logger.debug(f"Face rejected - too blurry: {quality.blur_score:.2f}")
            return False
        
        if quality.lighting_score < 0.2:  # Very poor lighting
            if self.config.debug:
                self.logger.debug(f"Face rejected - poor lighting: {quality.lighting_score:.2f}")
            return False
        
        return True
    
    def get_best_face(self, faces_with_quality: List[Tuple[Tuple[int, int, int, int], FaceQuality]]) -> Optional[Tuple[Tuple[int, int, int, int], FaceQuality]]:
        """Get the best quality face from the list"""
        if not faces_with_quality:
            return None
        
        # Sort by overall quality (highest first)
        sorted_faces = sorted(faces_with_quality, key=lambda x: x[1].overall_quality, reverse=True)
        
        return sorted_faces[0]
    
    def draw_debug_info(self, frame: np.ndarray, faces_with_quality: List[Tuple[Tuple[int, int, int, int], FaceQuality]]) -> np.ndarray:
        """Draw debug information on frame"""
        debug_frame = frame.copy()
        
        for face_rect, quality in faces_with_quality:
            x, y, w, h = face_rect
            
            # Draw face rectangle
            color = (0, 255, 0) if quality.overall_quality > 0.5 else (0, 0, 255)
            cv2.rectangle(debug_frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw quality info
            text_y = y - 10
            cv2.putText(debug_frame, f"Quality: {quality.overall_quality:.2f}", (x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            text_y -= 15
            cv2.putText(debug_frame, f"Blur: {quality.blur_score:.2f}", (x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            text_y -= 15
            cv2.putText(debug_frame, f"Conf: {quality.confidence:.2f}", (x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        
        return debug_frame
    
    def __del__(self):
        """Cleanup MediaPipe resources"""
        if hasattr(self, 'face_detection'):
            self.face_detection.close() 