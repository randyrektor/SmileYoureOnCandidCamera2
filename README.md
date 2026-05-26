# React Baby (Thumbnail Faces)

Local tool for scanning reaction videos, setting a face ROI, and exporting still frames tagged by emotion for YouTube thumbnails.

## Quick start

```bash
./start_dev.sh
```

- **UI:** http://localhost:5173  
- **API:** http://localhost:8000 (docs at `/docs`)

The Vite dev server proxies `/api` and `/ws` to the backend, so the frontend uses relative URLs.

### First-time backend setup

Requires **Python 3.12+** (numpy 1.26 segfaults on macOS Tahoe's Accelerate
framework; numpy 2.x fixes it but is Python 3.10+ only).

```bash
cd backend
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

On first run the backend will download the MediaPipe Face Landmarker model
(~3.7MB) into `~/.cache/react-baby/`. Delete that folder with
`python clear_cache.py` to force a re-download.

## How it works

For each video, the pipeline:

1. Runs **MediaPipe Face Landmarker** on sampled frames in the ROI, getting
   478 facial landmarks and 52 **blendshapes** (eyeBlink, mouthSmile, jawOpen,
   browInnerUp, etc.) per face.
2. Maps blendshapes to emotion scores via deterministic recipes — no ML
   softmax noise. `happy` is mostly mouthSmile + cheekSquint; `surprise` is
   browInnerUp + eyeWide + jawOpen.
3. Groups contiguous high-emotion frames into **reaction events**, then picks
   the top-K frames per event by a **composite quality score**:
   `emotion × eyes-open × sharpness × head-pose × face-size`.
4. Hard-rejects any frame where either eye is blinking or absolute sharpness
   is below a motion-blur floor.
5. Saves frames to `<output_dir>/<video_name>/<emotion>_<score>_<frame>.png`,
   one folder per video.

Because each video contains one person and is named after them, you get
per-person output folders for free.

## Folder layout (defaults)

By default, videos and exports use:

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

- ROI editor with drag handles and auto-detect
- Emotion filters and sensitivity
- Single-video and batch processing
- Live progress and logs over WebSocket
- Optional desktop notifications when a job finishes

## Other scripts

- `start-app.sh` / `start-react-baby.sh` — older launchers with health checks and logs
- `stop_dev.sh` / `stop-app.sh` — stop dev processes

## Supported video formats

`.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`
