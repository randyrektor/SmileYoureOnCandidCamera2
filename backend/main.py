from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from pydantic import BaseModel
from typing import Optional
import cv2
import numpy as np
from pathlib import Path
import base64
from smile_detector import SmileDetector, ProcessingConfig, RoiPosition
from concurrent.futures import ThreadPoolExecutor
import threading

# Constants
VIDEO_EXTENSIONS = ['.[mM][pP]4', '.[mM][oO][vV]', '.[aA][vV][iI]']
MAX_IMAGE_DIMENSION = 1280

class ProcessingParams(BaseModel):
    debug: bool
    skipFrames: int
    smileSensitivity: int
    roiPosition: RoiPosition

class ProcessingState:
    def __init__(self):
        self.detector = None
        self.is_processing = False
        self.active_task = None
        self.processing_event = threading.Event()
        self.executor = ThreadPoolExecutor(max_workers=1)

    def reset(self):
        self.is_processing = False
        self.processing_event.clear()
        self.active_task = None

class VideoUtils:
    @staticmethod
    def encode_frame_to_base64(frame):
        _, buffer = cv2.imencode('.jpg', frame)
        return base64.b64encode(buffer).decode('utf-8')

    @staticmethod
    def find_video_files(directory: Path):
        video_files = []
        for ext in VIDEO_EXTENSIONS:
            video_files.extend(directory.glob(f'*{ext}'))
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

class VideoProcessor:
    def __init__(self, state: ProcessingState):
        self.state = state

    async def process_video(self, params: ProcessingParams):
        try:
            self.state.processing_event.clear()
            config = ProcessingConfig(
                skip_frames=params.skipFrames,
                min_smile_duration=0.5,
                debug=params.debug,
                frame_buffer_size=2,
                smile_sensitivity=params.smileSensitivity,
                roi_position=params.roiPosition
            )
            
            self.state.detector = SmileDetector(config)
            video_dir = Path.home() / 'Desktop/Smile Youre On Candid Camera/1 VIDEO'
            output_dir = Path.home() / 'Desktop/Smile Youre On Candid Camera/2 SMILES'
            
            video_files = VideoUtils.find_video_files(video_dir)
            if not video_files:
                raise ValueError("No video files found")
            
            video_file = video_files[0]
            video_output_dir = output_dir / video_file.stem
            
            loop = asyncio.get_running_loop()
            num_smiles = await loop.run_in_executor(
                self.state.executor,
                self.state.detector.extract_smiles_from_video,
                video_file,
                video_output_dir,
                self.state.processing_event.is_set
            )
            return num_smiles
            
        except Exception as e:
            print(f"Error during processing: {str(e)}")
            raise e
        finally:
            print("Cleaning up processing state")
            self.state.reset()

def create_app():
    app = FastAPI()
    state = ProcessingState()
    processor = VideoProcessor(state)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/preview-frame")
    async def get_preview_frame():
        video_dir = Path.home() / 'Desktop/Smile Youre On Candid Camera/1 VIDEO'
        video_files = VideoUtils.find_video_files(video_dir)
        
        if not video_files:
            raise HTTPException(status_code=404, detail="No video files found")
        
        video_path = str(video_files[0])
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

    @app.post("/api/start-processing")
    async def start_processing(params: ProcessingParams):
        if state.is_processing:
            raise HTTPException(status_code=400, detail="Processing already in progress")
        
        state.is_processing = True
        state.active_task = asyncio.create_task(processor.process_video(params))
        
        try:
            num_smiles = await state.active_task
            return {
                "status": "success",
                "message": f"Processing complete! Found {num_smiles} smiles",
                "smiles_found": num_smiles
            }
        except Exception as e:
            state.is_processing = False
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/stop-processing")
    async def stop_processing():
        if not state.is_processing:
            return {"status": "success", "message": "No processing in progress"}
        
        state.processing_event.set()
        state.is_processing = False
        
        if state.active_task:
            try:
                await state.active_task
            except Exception as e:
                print(f"Error during stop: {e}")
        
        return {"status": "success", "message": "Processing stopped"}

    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)