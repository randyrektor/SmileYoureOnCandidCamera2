from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import cv2
import numpy as np
from pathlib import Path
import io
import base64
from PIL import Image
from pydantic import BaseModel
from typing import Optional, Dict
from smile_detector import SmileDetector, ProcessingConfig, RoiPosition  # Updated import

app = FastAPI()

# Add CORS middleware
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

class RoiPosition(BaseModel):
    top: float
    bottom: float
    left: float
    right: float

class ProcessingParams(BaseModel):
    debug: bool
    skipFrames: int
    smileSensitivity: int  # Added
    roiPosition: RoiPosition

def encode_frame_to_base64(frame):
    """Convert OpenCV frame to base64 string"""
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')

@app.get("/api/preview-frame")
async def get_preview_frame():
    video_dir = Path.home() / 'Desktop/Smile Youre On Candid Camera/1 VIDEO'
    
    # Get first video file
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
            
        # Resize frame for preview
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

@app.post("/api/start-processing")
async def start_processing(params: ProcessingParams):
    try:
        global detector, is_processing, should_stop
        
        print(f"Start processing request received. Current state - is_processing: {is_processing}, should_stop: {should_stop}")
        
        if is_processing:
            raise HTTPException(status_code=400, detail="Processing already in progress")
        
        should_stop = False  # Reset at start
        is_processing = True
        
        print(f"Processing started. Updated state - is_processing: {is_processing}, should_stop: {should_stop}")
        
        config = ProcessingConfig(
            skip_frames=params.skipFrames,
            min_smile_duration=0.5,
            debug=params.debug,
            frame_buffer_size=2,
            smile_sensitivity=params.smileSensitivity,  # Added
            roi_position=params.roiPosition  # Added
        )
        
        detector = SmileDetector(config)
        
        # Process videos
        video_dir = Path.home() / 'Desktop/Smile Youre On Candid Camera/1 VIDEO'
        output_dir = Path.home() / 'Desktop/Smile Youre On Candid Camera/2 SMILES'
        
        video_files = list(video_dir.glob('*.[mM][pP]4'))
        video_files.extend(video_dir.glob('*.[mM][oO][vV]'))
        video_files.extend(video_dir.glob('*.[aA][vV][iI]'))
        
        if not video_files:
            raise HTTPException(status_code=404, detail="No video files found")
        
        # Process first video file
        video_file = video_files[0]
        video_output_dir = output_dir / video_file.stem
        print(f"Starting processing of {video_file.name}")
        
        def check_should_stop():
            print(f"Checking stop status: {should_stop}")
            return should_stop
        
        num_smiles = detector.extract_smiles_from_video(
            video_file, 
            video_output_dir,
            should_stop_check=check_should_stop
        )
        
        print(f"Processing complete with {num_smiles} smiles found")
        is_processing = False
        
        return {
            "status": "success",
            "message": f"Processing complete! Found {num_smiles} smiles",
            "smiles_found": num_smiles
        }
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        is_processing = False
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/stop-processing")
async def stop_processing():
    try:
        global is_processing, should_stop
        print(f"Stop requested - current should_stop value: {should_stop}")
        print(f"Current is_processing value: {is_processing}")
        
        should_stop = True
        is_processing = False
        
        print(f"Updated should_stop value: {should_stop}")
        print(f"Updated is_processing value: {is_processing}")
        
        return {"status": "success", "message": "Processing stopped"}
    except Exception as e:
        print(f"Error in stop_processing: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)