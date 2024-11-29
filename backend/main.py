from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from pydantic import BaseModel
from typing import Optional, Dict
import cv2
import numpy as np
from pathlib import Path
import io
import base64
from smile_detector import SmileDetector, ProcessingConfig, RoiPosition

app = FastAPI()

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
should_stop = False
active_task = None

class ProcessingParams(BaseModel):
    debug: bool
    skipFrames: int
    smileSensitivity: int
    roiPosition: RoiPosition

def encode_frame_to_base64(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')

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
    global detector, is_processing, should_stop, active_task
    
    try:
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
            return should_stop
        
        num_smiles = await asyncio.to_thread(
            detector.extract_smiles_from_video,
            video_file,
            video_output_dir,
            check_should_stop
        )
        return num_smiles
        
    except asyncio.CancelledError:
        print("Processing cancelled")
        return 0
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        raise e
    finally:
        print("Cleaning up processing state")
        is_processing = False
        should_stop = False
        active_task = None

@app.post("/api/start-processing")
async def start_processing(params: ProcessingParams):
    global is_processing, should_stop, active_task
    
    if is_processing:
        raise HTTPException(status_code=400, detail="Processing already in progress")
    
    is_processing = True
    should_stop = False
    
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
    global is_processing, should_stop, active_task
    
    print("Stop processing requested")
    
    if not is_processing:
        return {"status": "success", "message": "No processing in progress"}
    
    should_stop = True
    is_processing = False
    
    if active_task and not active_task.done():
        print("Cancelling active task")
        active_task.cancel()
        try:
            await active_task
        except asyncio.CancelledError:
            print("Task successfully cancelled")
        except Exception as e:
            print(f"Error during cancellation: {e}")
    
    return {"status": "success", "message": "Processing stopped"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)