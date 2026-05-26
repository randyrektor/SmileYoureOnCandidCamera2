from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from pydantic import BaseModel
from typing import Optional, List
import cv2
from pathlib import Path
from dotenv import load_dotenv
from .emotion_detector import EmotionDetector, ProcessingConfig, RoiPosition
from .face_analyzer import FaceAnalyzer
from .utils import VideoUtils
from .paths import get_video_dir, get_output_dir
from .websocket_manager import manager
from concurrent.futures import ThreadPoolExecutor
import threading
import logging
from datetime import datetime
import os

load_dotenv()

# Constants
MAX_IMAGE_DIMENSION = 1280

_log_level = getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO)
logging.basicConfig(
    level=_log_level,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class ProcessingParams(BaseModel):
    debug: bool = False
    emotionSensitivity: int
    roiPosition: RoiPosition
    videoPath: Optional[str] = None
    targetEmotions: Optional[List[str]] = None
    # Analyze every Nth frame. Clamped to 1..10 server-side.
    skipInterval: int = 3


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
            output_dir = get_output_dir() / video_path.stem
            output_dir.mkdir(parents=True, exist_ok=True)
            
            config = ProcessingConfig(
                debug=params.debug,
                emotion_sensitivity=params.emotionSensitivity,
                roi_position=params.roiPosition,
                target_emotions=params.targetEmotions,
                analysis_skip_frames=max(1, min(params.skipInterval, 10)),
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
                    emotion_threshold = 0.4 + (params.emotionSensitivity * 0.1)
                    logging.info(
                        f"Processing skip={params.skipInterval}, "
                        f"threshold={emotion_threshold:.2f} "
                        f"(from sensitivity {params.emotionSensitivity})"
                    )
                    return self.state.detector.extract_emotions_best_frames(
                        video_path,
                        output_dir,
                        lambda: self.state.processing_event.is_set(),
                        send_progress,
                        params.skipInterval,
                        emotion_threshold,
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


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global state, processor

    ensure_directories()
    state = ProcessingState()
    state.executor = ThreadPoolExecutor(max_workers=1)
    state.processing_event = threading.Event()
    processor = VideoProcessor(state)

    # Eagerly download the MediaPipe model bundle so the first run is snappy.
    try:
        from .face_analyzer import _ensure_model
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _ensure_model)
    except Exception as e:
        logger.error(f"Failed to prefetch face landmarker model: {e}")

    logger.info("Application started with emotion detection")
    yield

    if state.executor:
        state.executor.shutdown(wait=False)


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def ensure_directories():
    video_dir = get_video_dir()
    output_dir = get_output_dir()

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


@app.get("/api/config")
async def get_config():
    video_dir, output_dir = ensure_directories()
    return {
        "video_dir": str(video_dir),
        "output_dir": str(output_dir),
    }


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

@app.get("/api/preview-frame")
async def get_preview_frame(video_path: Optional[str] = None):
    if not video_path:
        video_dir = get_video_dir()
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
    """Detect a tight ROI by sampling faces across the video."""
    cap = cv2.VideoCapture(video_path)
    analyzer = None

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if total_frames < 1:
            raise ValueError("Video has no frames")

        start_frame = int(total_frames * 0.1)
        end_frame = int(total_frames * 0.9)
        sample_interval = max(1, (end_frame - start_frame) // num_samples)
        sample_positions = [start_frame + i * sample_interval for i in range(num_samples)]

        analyzer = FaceAnalyzer(min_detection_confidence=0.3)
        face_boxes = []

        logger.info(f"Sampling {num_samples} frames from video for ROI detection")
        full_roi = (0, 0, frame_width, frame_height)

        for frame_pos in sample_positions:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            ret, frame = cap.read()
            if not ret:
                continue
            analysis = analyzer.analyze(frame, full_roi)
            if analysis is None:
                continue
            face_boxes.append(analysis.bbox)
            logger.debug(f"Frame {frame_pos}: face at {analysis.bbox}")

        if not face_boxes:
            logger.warning("No faces detected in sampled frames, using default ROI")
            return {"top": 20, "bottom": 65, "left": 25, "right": 75}
        
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
        if analyzer is not None:
            analyzer.close()
        cap.release()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)