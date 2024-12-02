from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from pydantic import BaseModel
from typing import Optional, Dict
import cv2
import numpy as np
from pathlib import Path
import io
import base64
import logging
import json
from smile_detector import SmileDetector, ProcessingConfig, RoiPosition
from concurrent.futures import ThreadPoolExecutor
import threading

app = FastAPI()
executor = ThreadPoolExecutor(max_workers=1)
processing_event = threading.Event()

# WebSocket clients
connected_clients = set()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
detector = None
is_processing = False
active_task = None

class ProcessingParams(BaseModel):
    debug: bool
    skipFrames: int
    smileSensitivity: int
    roiPosition: RoiPosition

# Custom WebSocket log handler
class WebSocketLogHandler(logging.Handler):
    def emit(self, record):
        try:
            # Check if we have structured data
            structured_data = getattr(record, 'structured', None)
            if structured_data:
                message = structured_data
            else:
                message = {
                    "type": "log",
                    "data": {
                        "timestamp": record.asctime,
                        "level": record.levelname,
                        "message": record.getMessage()
                    }
                }
            
            # Send to all connected clients
            for client in connected_clients:
                asyncio.create_task(client.send_json(message))
        except Exception as e:
            print(f"Error sending log to WebSocket: {e}")

# Initialize WebSocket handler
websocket_handler = WebSocketLogHandler()
logging.getLogger().addHandler(websocket_handler)

def encode_frame_to_base64(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.add(websocket)
    try:
        while True:
            await websocket.receive_text()
    except Exception:
        connected_clients.remove(websocket)

@app.get("/api/preview-frame")
async def get_preview_frame():
    video_dir = Path.home() / 'Desktop/Smile Youre On Candid Camera/1 VIDEO'
    video_files = list(video_dir.glob('*.[mM][pP]4'))
    video_files.extend(video_dir.glob('*.[mM][oO][vV]'))
    video_files.extend(video_dir.glob('*.[aA][vV][iI]'))
    
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
        
        height, width = frame.shape[:2]
        max_dimension = 1280
        if width > max_dimension or height > max_dimension:
            scale = max_dimension / max(width, height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height))
        
        frame_base64 = encode_frame_to_base64(frame)
        return {"frame": frame_base64, "dimensions": {"width": frame.shape[1], "height": frame.shape[0]}}
    finally:
        cap.release()

async def process_video(params: ProcessingParams):
    global detector, is_processing, active_task, processing_event
    
    try:
        processing_event.clear()  # Clear any existing stop signal
        config = ProcessingConfig(
            skip_frames=params.skipFrames,
            min_smile_duration=0.5,
            debug=params.debug,
            frame_buffer_size=2,
            smile_sensitivity=params.smileSensitivity,
            roi_position=params.roiPosition
        )
        
        detector = SmileDetector(config)
        video_dir = Path.home() / 'Desktop/Smile Youre On Candid Camera/1 VIDEO'
        output_dir = Path.home() / 'Desktop/Smile Youre On Candid Camera/2 SMILES'
        
        video_files = list(video_dir.glob('*.[mM][pP]4'))
        video_files.extend(video_dir.glob('*.[mM][oO][vV]'))
        video_files.extend(video_dir.glob('*.[aA][vV][iI]'))
        
        if not video_files:
            raise ValueError("No video files found")
        
        video_file = video_files[0]
        video_output_dir = output_dir / video_file.stem
        
        def check_should_stop():
            return processing_event.is_set()
        
        loop = asyncio.get_running_loop()
        num_smiles = await loop.run_in_executor(
            executor,
            detector.extract_smiles_from_video,
            video_file,
            video_output_dir,
            check_should_stop
        )
        return num_smiles
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        raise e
    finally:
        print("Cleaning up processing state")
        is_processing = False
        processing_event.clear()
        active_task = None

@app.post("/api/start-processing")
async def start_processing(params: ProcessingParams):
    global is_processing, active_task
    
    if is_processing:
        raise HTTPException(status_code=400, detail="Processing already in progress")
    
    is_processing = True
    
    active_task = asyncio.create_task(process_video(params))
    try:
        num_smiles = await active_task
        return {
            "status": "success",
            "message": f"Processing complete! Found {num_smiles} smiles",
            "smiles_found": num_smiles
        }
    except Exception as e:
        is_processing = False
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/stop-processing")
async def stop_processing():
    global is_processing, processing_event, active_task
    
    print("Stop processing requested")
    
    if not is_processing:
        return {"status": "success", "message": "No processing in progress"}
    
    processing_event.set()
    is_processing = False
    
    if active_task:
        try:
            await active_task
        except Exception as e:
            print(f"Error during stop: {e}")
    
    return {"status": "success", "message": "Processing stopped"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)