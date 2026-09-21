# React Baby (Thumbnail Faces)

Local tool for scanning reaction videos and exporting the best still frames per
emotion, sorted per person, for podcast/YouTube thumbnails.

## Quick start

```bash
./start-app.sh
```

- **UI:** http://localhost:5173
- **API:** http://localhost:8000 (docs at `/docs`)

`stop-app.sh` shuts everything down (it only touches the processes
`start-app.sh` started). The Vite dev server proxies `/api` and `/ws` to the
backend, so the frontend uses relative URLs.

### First-time setup

Backend needs **Python 3.12+**. Frontend needs **Node 24 LTS** (anything
≥ 22.12 works; see `.nvmrc`).

```bash
cd backend
python3.12 -m venv venv
venv/bin/pip install -e ".[dev]"

cd ../frontend
npm install
```

On first run the backend downloads the MediaPipe Face Landmarker model
(~3.7 MB) into `~/.cache/react-baby/`.

> **Why is MediaPipe pinned to 0.10.35?** MediaPipe 1.0.x aborts the whole
> process on macOS arm64 as soon as any Tasks vision graph opens
> ([google-ai-edge/mediapipe#6356](https://github.com/google-ai-edge/mediapipe/issues/6356)).
> Bump the pin in `backend/pyproject.toml` once that is fixed upstream.

## How it works

For each video, a worker process:

1. **Detects a face ROI** automatically from 8 sampled frames so the analyzer
   only looks where the face actually is.
2. **Decodes once, hardware accelerated** (VideoToolbox on macOS via FFmpeg).
   Skipped frames are grabbed but never decoded to pixels.
3. Runs **MediaPipe Face Landmarker in VIDEO mode** on every Nth frame,
   tracking landmarks between frames instead of re-detecting, producing 478
   landmarks and 52 **blendshapes** (eyeBlink, mouthSmile, jawOpen, …).
4. Maps blendshapes to per-emotion scores via deterministic recipes — no ML
   softmax noise. `happy` is mostly mouthSmile + cheekSquint; `surprise` is
   browInnerUp + eyeWide + jawOpen; `angry` requires brow-down **and**
   lower-face tension together.
5. **Hard rejects** frames that aren't thumbnail-worthy: either eye blinking,
   noticeably blurrier than the video's own rolling sharpness baseline (motion
   blur; the gate is relative because absolute sharpness depends on the
   camera and codec), head off-axis (yaw > 30° or pitch > 25°), or a negative
   emotion on a clearly smiling face.
6. Groups surviving frames into **reaction events** and keeps a small in-memory
   reservoir of the best frames per event, scored by
   `emotion × eyes-open × sharpness × head-pose × face-size`. The top two per
   event, at least 0.5 s apart, are written to disk. Because the pixels are
   kept from the analysis pass, the saved frame is exactly the frame that was
   scored; there is no second decode or seek.
7. Saves to `<output_dir>/<video_name>/<emotion>_<score>_<frame>.png`.

Because each video contains one person and is named after them, you get
per-person output folders for free.

**Process All** runs videos in parallel across a process pool (default: one
worker per four cores, max six; override with `REACT_BABY_WORKERS`). The job
runs on the backend, so closing the browser tab does not stop it; reopening
the UI reattaches to the running job.

## Folder layout (defaults)

| Role | Path |
|------|------|
| Input videos | `~/Desktop/Smile Youre On Candid Camera/1 VIDEO` |
| Emotion exports | `~/Desktop/Smile Youre On Candid Camera/2 EMOTIONS/<video-name>/` |

Override with environment variables (see `backend/.env.example`):

- `REACT_BABY_BASE_DIR` — parent folder (sets default video + output subfolders)
- `REACT_BABY_VIDEO_DIR` — input videos only
- `REACT_BABY_OUTPUT_DIR` — export root only
- `REACT_BABY_WORKERS` — parallel videos in a batch
- `REACT_BABY_HOST` / `REACT_BABY_PORT` — bind address (default `127.0.0.1:8000`)
- `LOG_LEVEL` — e.g. `INFO`, `WARNING`

Copy `backend/.env.example` to `backend/.env` and edit as needed.

The API only serves files under the input folder and binds to loopback by
default, so nothing else on your network can read frames off your disk.

## Development

```bash
# backend
cd backend && venv/bin/ruff check . && venv/bin/pytest

# frontend
cd frontend && npm run lint && npm run build
```

CI runs the same checks on every push (`.github/workflows/ci.yml`).

## Layout

```
backend/
  pyproject.toml          deps + ruff/pytest config (pip install -e ".[dev]")
  react_baby/
    main.py               FastAPI app, REST + WebSocket endpoints
    jobs.py               process pool, progress relay, job lifecycle
    worker.py             per-video job run inside a worker process
    emotion_detector.py   blendshape recipes, reaction events, frame selection
    face_analyzer.py      MediaPipe Face Landmarker wrapper (image/video modes)
    roi.py                face-area auto-detection
    video_io.py           OpenCV helpers (hw decode, discovery, JPEG)
    paths.py              folder config + path validation
  tests/
frontend/
  src/App.jsx             state + orchestration
  src/components/         VideoList, Preview, Controls, JobStatus, LogPanel
  src/hooks/useEvents.js  reconnecting WebSocket
  src/state/job.js        reducer for job lifecycle messages
```

## Supported video formats

`.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`
