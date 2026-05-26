# React Baby (Thumbnail Faces)

Local tool for scanning reaction videos and exporting the best still frames per
emotion, sorted per person, for podcast/YouTube thumbnails.

## Quick start

```bash
./start-app.sh
```

- **UI:** http://localhost:5173
- **API:** http://localhost:8000 (docs at `/docs`)

`stop-app.sh` shuts everything down. The Vite dev server proxies `/api` and
`/ws` to the backend, so the frontend uses relative URLs.

### First-time backend setup

Requires **Python 3.12+** (numpy 1.26 segfaults on macOS Tahoe's Accelerate
framework; numpy 2.x fixes it but is Python 3.10+ only).

```bash
cd backend
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

On first run the backend downloads the MediaPipe Face Landmarker model
(~3.7 MB) into `~/.cache/react-baby/`.

## How it works

For each video, the pipeline:

1. **Detect a face ROI** automatically from 8 sampled frames so the analyzer
   only looks where the face actually is.
2. Run **MediaPipe Face Landmarker** on every Nth frame within the ROI,
   producing 478 landmarks and 52 **blendshapes** (eyeBlink, mouthSmile,
   jawOpen, browInnerUp, …) per face.
3. Map blendshapes to per-emotion scores via deterministic recipes — no ML
   softmax noise. `happy` is mostly mouthSmile + cheekSquint; `surprise` is
   browInnerUp + eyeWide + jawOpen; `angry` requires brow-down **and**
   lower-face tension together.
4. **Hard reject** frames that aren't thumbnail-worthy:
   - Either eye blinking
   - Below the absolute sharpness floor (motion blur)
   - Head off-axis (yaw > 30° or pitch > 25°) — MediaPipe blendshapes get
     unreliable past that
   - Negative emotions (`angry` / `sad` / `disgust` / `fear`) on a clearly
     smiling face — you cannot be both
5. Group surviving high-scoring frames into **reaction events** and pick the
   top-K per event by a **composite quality score**:
   `emotion × eyes-open × sharpness × head-pose × face-size`. Picks within an
   event must be ≥0.5s apart so we don't save two near-duplicate frames.
6. Save frames to `<output_dir>/<video_name>/<emotion>_<score>_<frame>.png`,
   one folder per video.

Because each video contains one person and is named after them, you get
per-person output folders for free.

## Folder layout (defaults)

| Role | Path |
|------|------|
| Input videos | `~/Desktop/Smile Youre On Candid Camera/1 VIDEO` |
| Emotion exports | `~/Desktop/Smile Youre On Candid Camera/2 EMOTIONS/<video-name>/` |

Override with environment variables (see `backend/.env.example`):

- `REACT_BABY_BASE_DIR` — parent folder (sets default video + output subfolders)
- `REACT_BABY_VIDEO_DIR` — input videos only
- `REACT_BABY_OUTPUT_DIR` — export root only
- `LOG_LEVEL` — e.g. `INFO`, `WARNING`

Copy `backend/.env.example` to `backend/.env` and edit as needed.

## Features

- Auto-detected ROI per video, shown in the preview as soon as the video loads
- Target-emotion filters and a 1–5 strictness slider
- Single-video and batch processing across the whole input folder
- Live FPS / elapsed / ETA over WebSocket
- Optional desktop notifications when a job finishes

## Supported video formats

`.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`
