from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from pydantic import BaseModel
from typing import Optional, Set, List
import cv2
from pathlib import Path
from .emotion_detector import EmotionDetector, ProcessingConfig, RoiPosition
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
            output_dir = Path.home() / 'Desktop' / 'Emotion_Thumbnails' / video_path.stem
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
    base_dir = Path.home() / 'Desktop' / 'Emotion_Thumbnails'
    video_dir = base_dir / 'Videos'
    output_dir = base_dir / 'Results'
    
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
        video_dir = Path.home() / 'Desktop' / 'Emotion_Thumbnails' / 'Videos'
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