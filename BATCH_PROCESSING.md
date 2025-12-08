# Batch Processing Feature

## Overview
The batch processing feature allows you to automatically process multiple videos in sequence, with each video getting its own auto-detected ROI. This makes the workflow fully automated - just load your videos and click one button!

## How It Works

### Workflow for Each Video
For each video in the batch, the system:
1. **Auto-detects ROI** - Samples 8 frames throughout the video to find optimal face positions
2. **Applies ROI** - Uses the detected ROI for that specific video
3. **Processes video** - Extracts emotion frames using your current settings
4. **Moves to next** - Automatically continues to the next video

### Key Features

**Smart Error Handling**
- If a video fails (corrupted file, no faces detected, etc.), the error is logged
- Processing continues with the next video automatically
- Final summary shows how many videos succeeded vs. failed

**Single Notification**
- Audio chime plays only when **all videos** are complete
- No interruptions during batch processing
- Browser notification also fires only at the end

**Visual Feedback**
- Shows which video is currently being processed (e.g., "Processing video 2 of 3")
- Displays current video name
- Progress bar updates in real-time for each video
- Batch completion message shows success count

**Settings**
- Uses the same emotion/sensitivity settings for all videos
- Each video gets its own unique auto-detected ROI
- Settings are typically the defaults loaded on startup

## User Interface

### Process All Videos Button
- **Location**: Below the "Start Processing" button in the Controls section
- **Color**: Green (vs. blue for single video)
- **Label**: "Process All Videos (3)" - shows count
- **States**:
  - Disabled when: No videos loaded, no emotions selected, or single processing active
  - During batch: Changes to red "Stop Batch Processing" button

### Batch Progress Display
When batch processing is active, a status panel appears showing:
```
Processing video 2 of 3
Current: riverside_scott_raw-synced-video-cfr_syntax.fm_1386.mp4
```

### Batch Completion Display
When complete, shows:
```
✓ Batch processing complete! (1 error)
```

## Logs

The batch processing logs provide detailed information:

```
[09:45:30] Starting batch processing of 3 videos...
[09:45:31] [1/3] Processing: riverside_matt_brown_raw...mp4
[09:45:35]   ✓ ROI detected for riverside_matt_brown_raw...mp4
[09:52:10]   ✓ Completed: riverside_matt_brown_raw...mp4
[09:52:11] [2/3] Processing: riverside_scott_raw...mp4
[09:52:15]   ✓ ROI detected for riverside_scott_raw...mp4
[09:58:45]   ✓ Completed: riverside_scott_raw...mp4
[09:58:46] [3/3] Processing: riverside_wes_bos_raw...mp4
[09:58:50]   ✗ Error processing riverside_wes_bos_raw...mp4: Failed to detect ROI
[09:58:51] Batch complete! Processed 2 of 3 videos successfully.
```

## Technical Details

### State Management
New state variables added:
- `isBatchProcessing`: Boolean tracking if batch is active
- `batchProgress`: Object with `current`, `total`, and `currentVideo` name
- `batchErrors`: Array of errors encountered during batch
- `batchComplete`: Boolean indicating batch completion

### Processing Logic

**ROI Per Video**
Each video gets a fresh ROI detection:
```javascript
1. Detect ROI for video N
2. Apply detected ROI to state
3. Update preview to show ROI
4. Start processing with that ROI
5. Wait for completion
6. Move to video N+1
```

**Completion Detection**
The system monitors progress via WebSocket messages:
- Polls every 5 seconds to check progress
- Detects completion when progress reaches 100%
- Has safety timeouts (2 hours per video max)
- Detects if processing is stuck (5 minutes without progress)

**Error Recovery**
When a video fails:
```javascript
1. Log the error with timestamp
2. Add to batchErrors array
3. Stop any active processing
4. Reset processing state
5. Continue to next video
```

### UI Behavior During Batch

**Disabled Features**
- Video selection (prevents changing video mid-batch)
- Single video "Start Processing" button
- ROI manual adjustment (each video gets auto-ROI)

**Active Features**
- Stop Batch Processing button (cancels entire batch)
- Real-time progress updates
- Log streaming
- Preview updates for each video

## Usage Instructions

### Basic Batch Processing

1. **Load Videos**
   - Place videos in the `~/Desktop/Smile Youre On Candid Camera/1 VIDEO/` directory
   - Videos will appear in the "Available Videos" list

2. **Configure Settings** (optional)
   - Select target emotions (default: all emotions)
   - Adjust emotion accuracy slider (default: 4/5 = 80%)
   - These settings apply to all videos in the batch

3. **Start Batch**
   - Click "Process All Videos (N)" button
   - System begins processing all videos automatically

4. **Monitor Progress**
   - Watch the batch progress indicator
   - Check logs for detailed progress
   - Each video's ROI detection and processing is logged

5. **Completion**
   - Audio chime plays when all videos are done
   - Batch summary shows in logs
   - Check for any errors in the final summary

### Stopping Batch Processing

If you need to cancel:
1. Click "Stop Batch Processing" button
2. Current video processing will stop
3. Remaining videos won't be processed
4. Partial results are saved for completed videos

## Error Handling Examples

### No Faces Detected
```
[10:15:30] [2/3] Processing: video_without_face.mp4
[10:15:34]   ✗ Error processing video_without_face.mp4: Failed to detect ROI
```
**Result**: Continues to next video, uses default ROI

### Processing Timeout
```
[10:20:00] [3/3] Processing: very_long_video.mp4
[12:20:01]   ✗ Error processing very_long_video.mp4: Processing timeout
```
**Result**: Stops after 2 hours, continues to next video

### Corrupted Video
```
[10:25:00] [1/3] Processing: corrupted.mp4
[10:25:01]   ✗ Error processing corrupted.mp4: Failed to start processing
```
**Result**: Skips video, continues to next

## Performance Considerations

### Processing Time
- Each video takes approximately 5-15 minutes depending on:
  - Video length (60 minutes = ~5-10 min processing)
  - ROI size (smaller = faster)
  - Emotion sensitivity (higher = more selective)
  - Number of target emotions

### Resource Usage
- One video processed at a time (sequential, not parallel)
- Memory is managed efficiently with pooling
- System can be left unattended for hours
- Background tabs work (WebSocket keeps connection alive)

### Recommendations
For best results:
- Process during off-hours (long batch jobs)
- Keep browser tab active (prevents throttling)
- Monitor first video to ensure settings are correct
- Check disk space before starting (output can be large)

## Output Structure

Each video creates its own output directory:
```
~/Desktop/Smile Youre On Candid Camera/2 EMOTIONS/
├── riverside_matt_brown_raw.../
│   ├── happy_frame_001.png
│   ├── surprise_frame_045.png
│   └── ...
├── riverside_scott_raw.../
│   ├── happy_frame_012.png
│   └── ...
└── riverside_wes_bos_raw.../
    └── ...
```

## Future Enhancements

Potential improvements for later:
1. **Parallel Processing**: Process multiple videos simultaneously
2. **Priority Queue**: Reorder videos during batch
3. **Resume Capability**: Resume interrupted batches
4. **Scheduled Processing**: Start batch at specific time
5. **Advanced Settings**: Per-video emotion/sensitivity overrides
6. **Export Settings**: Save and load batch configurations
7. **Email Notification**: Get notified when batch completes
8. **Progress Estimation**: Show estimated time remaining for batch
9. **Video Preview**: Show thumbnails during batch processing
10. **Batch Templates**: Save common batch configurations

## Troubleshooting

### Batch Won't Start
**Check:**
- At least one video is loaded
- At least one emotion is selected
- No single video processing is active

### Batch Stuck on One Video
**Solutions:**
- Wait for timeout (2 hours max)
- Click "Stop Batch Processing"
- Check backend logs for errors
- Verify video file isn't corrupted

### No Sound at End
**Check:**
- Audio was initialized (click anywhere on page first)
- Volume is up
- Browser allows audio
- Batch actually completed (check logs)

### Some Videos Failed
**Review:**
- Check logs for specific error messages
- Verify video files are valid
- Check if videos have visible faces
- Ensure videos are in supported formats (MP4, MOV, AVI)

## Code Changes Summary

### Frontend (`frontend/src/App.jsx`)

**New State Variables** (lines ~60-63):
```javascript
const [isBatchProcessing, setIsBatchProcessing] = useState(false)
const [batchProgress, setBatchProgress] = useState({ current: 0, total: 0, currentVideo: null })
const [batchErrors, setBatchErrors] = useState([])
const [batchComplete, setBatchComplete] = useState(false)
```

**New Functions**:
- `processBatchVideos()` - Main batch processing orchestrator
- `stopBatchProcessing()` - Cancel batch operation

**Modified Functions**:
- WebSocket handler - Suppress notifications during batch
- `handleVideoSelect()` - Disabled during batch

**New UI Components**:
- "Process All Videos" button (green)
- "Stop Batch Processing" button (red, shown during batch)
- Batch progress indicator
- Batch completion message
- Video selection disabled state

### No Backend Changes Required
The batch processing feature uses existing backend endpoints:
- `/api/detect-roi` - For auto-ROI detection
- `/api/start-processing` - For processing each video
- `/api/stop-processing` - For stopping when needed

## Testing Checklist

Before deployment, verify:
- ✅ Batch processes all videos successfully
- ✅ Each video gets unique auto-detected ROI
- ✅ Progress updates correctly for each video
- ✅ Errors are logged but don't stop batch
- ✅ Single notification only at end
- ✅ Can stop batch mid-processing
- ✅ Video selection disabled during batch
- ✅ Settings applied consistently across batch
- ✅ Logs show detailed progress
- ✅ Completion summary is accurate

## Conclusion

The batch processing feature transforms the application from a single-video tool into a true automation solution. Users can now:
- ✅ Load multiple videos
- ✅ Click one button
- ✅ Walk away
- ✅ Get notified when complete
- ✅ Review results for all videos

This is a major step toward full automation and significantly reduces the time and effort required to process multiple podcast videos!

