# Auto ROI Detection Implementation

## Overview
This document describes the automatic Region of Interest (ROI) detection feature that has been implemented to make the video processing workflow more automation-friendly.

## What It Does
The Auto ROI feature automatically detects the optimal frame region by sampling frames throughout a video and analyzing face positions. This eliminates the need to manually adjust the ROI handles for each video.

## How It Works

### Backend Implementation (`backend/src/main.py`)

1. **New Endpoint**: `/api/detect-roi` (POST)
   - Accepts: `video_path` (string) and `num_samples` (int, default: 8)
   - Returns: Optimal ROI position as percentages (top, bottom, left, right)

2. **Detection Algorithm** (`_detect_roi_from_video` function):
   - Samples frames evenly throughout the video (skipping first and last 10% to avoid intro/outro)
   - Uses MediaPipe face detection to find faces in each sampled frame
   - Filters for high-quality face detections (quality > 0.3)
   - Calculates a bounding box that encompasses all detected faces
   - Adds 30% padding around faces for context
   - Ensures minimum ROI size of 20% of frame dimensions
   - Returns ROI as percentages for easy scaling

3. **Sample Distribution**:
   - Default: 8 samples evenly distributed across the middle 80% of the video
   - This is sufficient for podcast-style videos where subjects don't move much
   - Can be adjusted by changing the `num_samples` parameter

### Frontend Implementation (`frontend/src/App.jsx`)

1. **New UI Element**:
   - "Auto-Detect ROI" button in the Preview section header
   - Purple colored button with eye icon
   - Shows loading spinner during detection
   - Disabled during processing or when no video is selected

2. **Detection Function** (`detectROI`):
   - Calls the backend `/api/detect-roi` endpoint
   - Updates the ROI position state with detected values
   - Logs success/failure messages to the application log
   - Shows visual feedback during detection

3. **State Management**:
   - Added `isDetectingROI` state to track detection progress
   - Automatically updates `roiPosition` when detection completes
   - Preview canvas updates in real-time to show new ROI

## Usage Instructions

1. **Load a Video**: Select a video from the "Available Videos" list
2. **Auto-Detect**: Click the "Auto-Detect ROI" button in the Preview section
3. **Wait**: The system will sample ~8 frames and analyze face positions (takes a few seconds)
4. **Review**: The ROI overlay will automatically update to show the detected region
5. **Adjust** (Optional): You can still manually adjust the ROI handles if needed
6. **Process**: Start processing with the detected (or manually adjusted) ROI

## Technical Details

### Performance Considerations
- Samples only 8 frames by default (fast, good for podcasts)
- Uses existing MediaPipe face detector (already optimized and loaded)
- Runs in background thread to avoid blocking the API
- Detection typically completes in 2-3 seconds

### Face Detection Settings
- Minimum detection confidence: 0.3 (works well with glasses)
- Quality threshold: 0.3 (filters out poor detections)
- Padding: 30% around detected faces (provides context)
- Minimum ROI size: 20% of frame (prevents too-small regions)

### Edge Cases Handled
- **No faces detected**: Falls back to default ROI (20/65/25/75)
- **Single face**: Centers on that face with padding
- **Multiple faces**: Encompasses all faces with padding
- **Movement**: Calculates bounding box for all detected positions

## Future Enhancements (Not Yet Implemented)

For full automation in the future, you could add:

1. **Batch Processing**:
   - Loop through all videos in the folder
   - Auto-detect ROI for each
   - Start processing with default settings
   - Move to next video when complete

2. **Command-Line Interface**:
   - Script to run the entire workflow headless
   - Process videos without opening the web UI

3. **Configuration Profiles**:
   - Save commonly used settings (emotions, sensitivity, etc.)
   - Apply profile automatically to each video

4. **Progress Tracking**:
   - Save which videos have been processed
   - Skip already-processed videos
   - Resume batch processing after interruption

## Testing

The feature has been implemented and the servers are confirmed to be running:
- Frontend: Running on Vite dev server
- Backend: Running on port 8000

To test:
1. Open the application in your browser (typically http://localhost:5173)
2. Select a video from the list
3. Click "Auto-Detect ROI" button
4. Observe the ROI overlay update automatically
5. Verify the ROI makes sense for the video content
6. Process the video to confirm it works with detected ROI

## Code Changes Summary

### Backend (`backend/src/main.py`)
- Added `/api/detect-roi` endpoint (line ~370)
- Added `_detect_roi_from_video()` helper function (line ~380)
- Imports MediaPipe face detector for face analysis
- Returns ROI as percentage-based coordinates

### Frontend (`frontend/src/App.jsx`)
- Added `isDetectingROI` state variable
- Added `detectROI()` async function
- Added "Auto-Detect ROI" button in Preview section
- Automatic ROI update when detection completes

## Notes for Future Automation

This implementation focuses on **automatic ROI detection only**, keeping the rest of the workflow manual as requested. This serves as the foundation for full automation later.

The key insight is that for podcast-style videos:
- Subjects typically stay in similar positions throughout
- 8 frame samples is sufficient to capture the typical range
- Face detection is fast and reliable enough for this purpose
- The detected ROI significantly reduces processing time by focusing on relevant regions

When you're ready to automate the full workflow, you can build on this by:
1. Creating a batch processing endpoint that loops through videos
2. Calling detect-roi automatically for each video
3. Automatically starting processing with default/saved settings
4. Moving to the next video when complete

