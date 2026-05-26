import cv2
import os
from pathlib import Path
import base64
import numpy as np
import logging
from typing import Optional, Tuple

# Constants
VIDEO_EXTENSIONS = [
    '.[mM][pP]4',
    '.[mM][oO][vV]',
    '.[aA][vV][iI]',
    '.[mM][kK][vV]',
    '.[wW][eE][bB][mM]',
]

logger = logging.getLogger(__name__)

class VideoUtils:
    @staticmethod
    def encode_frame_to_base64(frame):
        _, buffer = cv2.imencode('.jpg', frame)
        return base64.b64encode(buffer).decode('utf-8')

    @staticmethod
    def find_video_files(directory: Path) -> list:
        video_files = []
        for ext in VIDEO_EXTENSIONS:
            video_files.extend(directory.glob(f"*{ext}"))
        return video_files

    @staticmethod
    def resize_frame(frame, max_dimension):
        height, width = frame.shape[:2]
        if width > max_dimension or height > max_dimension:
            scale = max_dimension / max(width, height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            return cv2.resize(frame, (new_width, new_height))
        return frame

    @staticmethod
    def get_video_info(video_path: Path):
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                print(f"Failed to open video: {video_path}")
                return None
                
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # Round duration to nearest second
            duration = round(frame_count / fps) if fps > 0 else 0
            size_mb = os.path.getsize(video_path) / (1024 * 1024)
            
            cap.release()
            
            return {
                "name": video_path.name,
                "path": str(video_path),
                "duration": duration,  # Send raw duration in seconds
                "size": f"{size_mb:.1f}MB"
            }
        except Exception as e:
            print(f"Error getting video info for {video_path}: {e}")
            return None 