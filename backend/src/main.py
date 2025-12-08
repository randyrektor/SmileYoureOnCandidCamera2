from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from pydantic import BaseModel
from typing import Optional, Set, List
import cv2
from pathlib import Path
from .emotion_detector import EmotionDetector, ProcessingConfig, RoiPosition
from .mediapipe_detector import MediaPipeFaceDetector
from .utils import VideoUtils, VIDEO_EXTENSIONS
from .websocket_manager import manager
from .model_cache import model_cache
from .memory_pool import memory_pool
from concurrent.futures import ThreadPoolExecutor
import threading
import logging
from datetime import datetime
import os

# Constants
MAX_IMAGE_DIMENSION = 1280

# Create the FastAPI app instance
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class ProcessingParams(BaseModel):
    debug: bool = False
    emotionSensitivity: int

    roiPosition: RoiPosition
    videoPath: Optional[str] = None
    targetEmotions: Optional[List[str]] = None
    # Best-frame approach settings (now the default)
    useBestFrameApproach: bool = True  # Default to best-frame approach
    skipInterval: int = 25
    searchForward: int = 5
    searchBackward: int = 5


class ProcessingState:
    def __init__(self):
        self.is_processing = False
        self.active_task = None
        self.processing_event = threading.Event()
        self.executor = None
        self.detector = None
    
    def reset(self):
        self.is_processing = False
        self.active_task = None
        self.processing_event.clear()
        self.detector = None

state = ProcessingState()  # Create state instance here
processor = None

class VideoProcessor:
    def __init__(self, state: ProcessingState):
        self.state = state

    async def process_video(self, params: ProcessingParams, manager):
        try:
            # Clear the event when starting processing
            self.state.processing_event.clear()
            
            video_path = Path(params.videoPath)
            output_dir = Path.home() / 'Desktop' / 'Smile Youre On Candid Camera' / '2 EMOTIONS' / video_path.stem
            output_dir.mkdir(parents=True, exist_ok=True)
            
            config = ProcessingConfig(
                min_emotion_duration=0.5,
                debug=params.debug,
                frame_buffer_size=2,
                emotion_sensitivity=params.emotionSensitivity,
                roi_position=params.roiPosition,
                target_emotions=params.targetEmotions,
                batch_size=2,  # Small batch size for frequent progress updates
                enable_batch_processing=False,  # Disabled for maximum processing speed

                # Performance optimizations for best-frame approach
                emotion_detection_skip_frames=15,  # Process every 15th frame for best-frame approach
                emotion_cache_size=3000,  # Large cache size for maximum performance
                emotion_input_size=224,  # ViT model input size requirement (224x224 pixels)
                enable_emotion_caching=True,
                enable_batch_emotion_detection=False  # Disable batch processing to minimize overhead
            )
            
            self.state.detector = EmotionDetector(config)
            loop = asyncio.get_running_loop()
            
            def process_in_thread():
                try:
                    def send_progress(data):
                        if not self.state.processing_event.is_set():  # Check if we should still send updates
                            if data:
                                loop.call_soon_threadsafe(
                                    lambda: asyncio.create_task(
                                        manager.broadcast(data)
                                    )
                                )
                    
                    # Use best-frame approach by default (optimized for thumbnail selection)
                    # Calculate emotion threshold from sensitivity: base 0.4 + sensitivity * 0.1
                    emotion_threshold = 0.4 + (params.emotionSensitivity * 0.1)
                    logging.info(f"Using best-frame approach: skip={params.skipInterval}, "
                               f"search={params.searchBackward}+{params.searchForward}, "
                               f"threshold={emotion_threshold:.2f} (from sensitivity {params.emotionSensitivity})")
                    return self.state.detector.extract_emotions_best_frames(
                        video_path,
                        output_dir,
                        lambda: self.state.processing_event.is_set(),
                        send_progress,
                        params.skipInterval,
                        emotion_threshold,
                        params.searchForward,
                        params.searchBackward
                    )
                except Exception as e:
                    logging.error(f"Thread error: {str(e)}")
                    raise
            
            return await loop.run_in_executor(None, process_in_thread)
            
        except Exception as e:
            logging.error(f"Processing error: {str(e)}")
            raise

# Set up logging
class LogHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.formatter = logging.Formatter('%(asctime)s - %(message)s', '%H:%M:%S')

    def emit(self, record):
        try:
            asyncio.create_task(manager.broadcast({
                "type": "log",
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "message": record.getMessage()
            }))
        except Exception as e:
            print(f"Error in log handler: {e}")

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)
logger.addHandler(LogHandler())

def ensure_directories():
    base_dir = Path.home() / 'Desktop' / 'Smile Youre On Candid Camera'
    video_dir = base_dir / '1 VIDEO'
    output_dir = base_dir / '2 EMOTIONS'
    
    video_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Video directory: {video_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    return video_dir, output_dir

@app.get("/api/available-videos")
async def get_available_videos():
    video_dir, _ = ensure_directories()
    videos = []
    
    video_files = VideoUtils.find_video_files(video_dir)
    for video_path in video_files:
        video_info = VideoUtils.get_video_info(video_path)
        if video_info:
            videos.append(video_info)
    
    return {"videos": videos}

@app.websocket("/ws/logs")
async def websocket_logs_endpoint(websocket: WebSocket):
    print("WebSocket connection attempt to /ws/logs")
    await manager.connect(websocket)
    try:
        while True:
            # Keep the connection alive
            data = await websocket.receive_text()
            print(f"Received message: {data}")
    except WebSocketDisconnect:
        print("WebSocket disconnected from /ws/logs")
        manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket error in /ws/logs: {e}")
        manager.disconnect(websocket)

@app.post("/api/start-processing")
async def start_processing(request: Request):
    try:
        body = await request.json()
        print("Received /api/start-processing payload:", body)
    except Exception as e:
        print("Error reading request body:", e)
        raise HTTPException(status_code=400, detail="Invalid JSON")

    try:
        params = ProcessingParams(**body)
    except Exception as e:
        print("Validation error in ProcessingParams:", e)
        raise HTTPException(status_code=422, detail=f"Validation error: {e}")

    try:
        if state.is_processing:
            await stop_processing()
            
        # Create video processor instance
        video_processor = VideoProcessor(state)
        
        # Start processing
        state.is_processing = True
        state.processing_event.clear()
        
        loop = asyncio.get_running_loop()
        state.active_task = loop.create_task(
            video_processor.process_video(params, manager)
        )
        
        return {"status": "success", "message": "Processing started"}
        
    except Exception as e:
        state.reset()
        logger.error(f"Processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/stop-processing")
async def stop_processing():
    if not state.is_processing:
        return {"status": "success", "message": "No processing in progress"}
    
    try:
        # Set the stop event first
        state.processing_event.set()
        
        # Wait briefly for the process to stop gracefully
        await asyncio.sleep(0.5)
        
        # Force cancel if still running
        if state.active_task and not state.active_task.done():
            state.active_task.cancel()
            try:
                await state.active_task
            except asyncio.CancelledError:
                pass
        
        # Reset state
        state.reset()
        state.is_processing = False
        
        return {"status": "success", "message": "Processing stopped"}
    except Exception as e:
        logging.error(f"Error during stop: {e}")
        state.reset()
        return {"status": "error", "message": str(e)}

@app.get("/api/model-status")
async def get_model_status():
    """Get the status of cached ML models"""
    try:
        device_info = model_cache.get_device_info()
        return {
            "status": "success",
            "models_loaded": device_info["models_loaded"],
            "device": device_info["device"],
            "cuda_available": device_info["cuda_available"],
            "cuda_device_count": device_info["cuda_device_count"]
        }
    except Exception as e:
        logger.error(f"Error getting model status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/memory-pool-stats")
async def get_memory_pool_stats():
    """Get memory pool statistics and performance metrics"""
    try:
        stats = memory_pool.get_stats()
        return {
            "status": "success",
            "data": stats
        }
    except Exception as e:
        logger.error(f"Error getting memory pool stats: {e}")
        return {
            "status": "error",
            "message": str(e)
        }

@app.post("/api/test-memory-pool")
async def test_memory_pool():
    """Test endpoint to trigger memory pool activity"""
    try:
        import numpy as np
        
        # Test memory pool activity
        buffer1 = memory_pool.get_image_buffer(100, 100, 3)
        buffer1.fill(255)
        memory_pool.return_image_buffer(buffer1)
        
        buffer2 = memory_pool.get_image_buffer(100, 100, 3)
        memory_pool.return_image_buffer(buffer2)
        
        stats = memory_pool.get_stats()
        return {
            "status": "success",
            "message": "Memory pool test completed",
            "data": stats
        }
    except Exception as e:
        logger.error(f"Error testing memory pool: {e}")
        return {
            "status": "error",
            "message": str(e)
        }

@app.get("/api/preview-frame")
async def get_preview_frame(video_path: Optional[str] = None):
    if not video_path:
        video_dir = Path.home() / 'Desktop' / 'Smile Youre On Candid Camera' / '1 VIDEO'
        video_files = VideoUtils.find_video_files(video_dir)
        
        if not video_files:
            raise HTTPException(status_code=404, detail="No video files found")
        
        video_path = str(video_files[0])
    
    if not Path(video_path).exists():
        raise HTTPException(status_code=404, detail="Video file not found")
    
    cap = cv2.VideoCapture(video_path)
    
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        middle_frame = total_frames // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
        
        ret, frame = cap.read()
        if not ret:
            raise HTTPException(status_code=500, detail="Could not read frame from video")
        
        frame = VideoUtils.resize_frame(frame, MAX_IMAGE_DIMENSION)
        frame_base64 = VideoUtils.encode_frame_to_base64(frame)
        
        return {
            "frame": frame_base64,
            "dimensions": {"width": frame.shape[1], "height": frame.shape[0]}
        }
    finally:
        cap.release()

class DetectRoiParams(BaseModel):
    video_path: str
    num_samples: int = 8

@app.post("/api/detect-roi")
async def detect_roi(params: DetectRoiParams):
    """
    Automatically detect optimal ROI by sampling frames throughout the video
    and analyzing face positions
    
    Args:
        params: DetectRoiParams with video_path and optional num_samples
    
    Returns:
        Optimal ROI position as percentages
    """
    if not Path(params.video_path).exists():
        raise HTTPException(status_code=404, detail="Video file not found")
    
    try:
        # Run detection in executor to avoid blocking
        loop = asyncio.get_running_loop()
        roi_position = await loop.run_in_executor(
            None, 
            _detect_roi_from_video, 
            params.video_path, 
            params.num_samples
        )
        
        return {
            "status": "success",
            "roi": roi_position,
            "message": f"ROI detected from {params.num_samples} samples"
        }
    except Exception as e:
        logger.error(f"Error detecting ROI: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def _detect_roi_from_video(video_path: str, num_samples: int = 8) -> dict:
    """
    Internal function to detect ROI from video frames
    Samples frames evenly throughout the video and finds the optimal bounding box
    """
    cap = cv2.VideoCapture(video_path)
    
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if total_frames < 1:
            raise ValueError("Video has no frames")
        
        # Calculate sample positions (evenly distributed)
        # Skip first and last 10% to avoid intro/outro
        start_frame = int(total_frames * 0.1)
        end_frame = int(total_frames * 0.9)
        sample_interval = max(1, (end_frame - start_frame) // num_samples)
        sample_positions = [start_frame + i * sample_interval for i in range(num_samples)]
        
        # Initialize face detector with default config
        default_roi = RoiPosition(top=0, bottom=100, left=0, right=100)
        config = ProcessingConfig(
            min_emotion_duration=0.5,
            debug=False,
            frame_buffer_size=2,
            emotion_sensitivity=4,
            roi_position=default_roi,
            target_emotions=['happy']
        )
        detector = MediaPipeFaceDetector(config)
        
        # Collect face positions from all samples
        face_boxes = []
        
        logger.info(f"Sampling {num_samples} frames from video for ROI detection")
        
        for frame_pos in sample_positions:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            # Detect faces in the full frame (using full frame as ROI)
            roi = (0, 0, frame_width, frame_height)
            faces_with_quality = detector.detect_faces_with_quality(frame, roi)
            
            # Get the best quality face
            if faces_with_quality:
                best_face = detector.get_best_face(faces_with_quality)
                if best_face:
                    face_rect, quality = best_face
                    # Only use high-quality detections
                    if quality.overall_quality > 0.3:
                        face_boxes.append(face_rect)
                        logger.debug(f"Frame {frame_pos}: Found face at {face_rect}, quality: {quality.overall_quality:.2f}")
        
        if not face_boxes:
            logger.warning("No faces detected in sampled frames, using default ROI")
            return {
                "top": 20,
                "bottom": 65,
                "left": 25,
                "right": 75
            }
        
        # Calculate bounding box that encompasses all detected faces
        # Add padding around the faces for context
        min_x = min(box[0] for box in face_boxes)
        min_y = min(box[1] for box in face_boxes)
        max_x = max(box[0] + box[2] for box in face_boxes)
        max_y = max(box[1] + box[3] for box in face_boxes)
        
        # Add padding (30% on each side)
        padding_x = (max_x - min_x) * 0.3
        padding_y = (max_y - min_y) * 0.3
        
        min_x = max(0, min_x - padding_x)
        min_y = max(0, min_y - padding_y)
        max_x = min(frame_width, max_x + padding_x)
        max_y = min(frame_height, max_y + padding_y)
        
        # Convert to percentages
        roi_left = (min_x / frame_width) * 100
        roi_right = (max_x / frame_width) * 100
        roi_top = (min_y / frame_height) * 100
        roi_bottom = (max_y / frame_height) * 100
        
        # Ensure minimum size (at least 20% of frame)
        roi_width = roi_right - roi_left
        roi_height = roi_bottom - roi_top
        
        if roi_width < 20:
            center_x = (roi_left + roi_right) / 2
            roi_left = max(0, center_x - 10)
            roi_right = min(100, center_x + 10)
        
        if roi_height < 20:
            center_y = (roi_top + roi_bottom) / 2
            roi_top = max(0, center_y - 10)
            roi_bottom = min(100, center_y + 10)
        
        logger.info(f"Detected ROI from {len(face_boxes)} face samples: "
                   f"L:{roi_left:.1f}% R:{roi_right:.1f}% T:{roi_top:.1f}% B:{roi_bottom:.1f}%")
        
        return {
            "top": round(roi_top, 1),
            "bottom": round(roi_bottom, 1),
            "left": round(roi_left, 1),
            "right": round(roi_right, 1)
        }
        
    finally:
        cap.release()

@app.on_event("startup")
async def startup_event():
    video_dir, output_dir = ensure_directories()
    global state, processor
    
    # Initialize state
    state = ProcessingState()
    state.executor = ThreadPoolExecutor(max_workers=1)
    
    # Initialize processor with default config for emotion detection
    default_roi = RoiPosition(top=20, bottom=65, left=25, right=75)
    config = ProcessingConfig(
        min_emotion_duration=0.5,
        debug=False,
        frame_buffer_size=2,
        emotion_sensitivity=4,  # 80% threshold (0.4 + 4 * 0.1 = 0.8)
        focus_threshold=0.75,  # Optimized for YouTube thumbnail sharpness
        roi_position=default_roi,
        target_emotions=['happy', 'surprise', 'angry', 'sad', 'fear', 'disgust', 'neutral']
    )
    
    processor = VideoProcessor(state)
    state.processing_event = threading.Event()
    
    # Preload models in background
    logger.info("Preloading ML models...")
    try:
        # This will load models in a background thread
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, model_cache.get_models)
        logger.info("ML models preloaded successfully")
    except Exception as e:
        logger.error(f"Failed to preload models: {e}")
        # Don't fail startup - models will be loaded on first use
    
    logger.info("Application started with emotion detection")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)