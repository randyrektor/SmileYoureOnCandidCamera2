import { useCallback, useEffect, useReducer, useRef, useState } from 'react'
import { api } from './api/client'
import { Controls } from './components/Controls'
import { JobStatus } from './components/JobStatus'
import { LogPanel } from './components/LogPanel'
import { Preview } from './components/Preview'
import { TooltipProvider } from './components/Tooltip'
import { VideoList } from './components/VideoList'
import { useEvents } from './hooks/useEvents'
import { EMOTIONS } from './lib/emotions'
import { now } from './lib/format'
import { initialJob, jobReducer } from './state/job'

// Used only when auto-detection fails outright. Roughly a 16:9 talking head.
const ROI_FALLBACK = { top: 20, bottom: 65, left: 25, right: 75 }
const MAX_LOGS = 500
const SETTINGS_KEY = 'react-baby.settings'

const defaultSettings = {
  sensitivity: 4,
  targetEmotions: EMOTIONS.filter((e) => e !== 'neutral'),
  skipInterval: 3,
  sound: true,
}

function loadSettings() {
  try {
    return { ...defaultSettings, ...JSON.parse(localStorage.getItem(SETTINGS_KEY) ?? '{}') }
  } catch {
    return defaultSettings
  }
}

export default function App() {
  const [videos, setVideos] = useState([])
  const [selected, setSelected] = useState(null)
  const [roi, setRoi] = useState(null)
  const [detectingRoi, setDetectingRoi] = useState(false)
  const [settings, setSettings] = useState(loadSettings)
  const [job, dispatch] = useReducer(jobReducer, initialJob)
  const [logs, setLogs] = useState([])
  const [config, setConfig] = useState(null)
  const [backendOk, setBackendOk] = useState(true)
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [notificationPermission, setNotificationPermission] = useState(() =>
    'Notification' in window ? Notification.permission : 'unsupported',
  )
  const selectedRef = useRef(null)
  const settingsRef = useRef(settings)

  useEffect(() => {
    settingsRef.current = settings
    try {
      localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings))
    } catch {
      /* private mode */
    }
  }, [settings])

  const log = useCallback((message, timestamp = now()) => {
    setLogs((prev) => [...prev.slice(-(MAX_LOGS - 1)), { id: crypto.randomUUID(), timestamp, message }])
  }, [])

  const notifyComplete = useCallback((result) => {
    const body = result.stopped
      ? 'Job stopped.'
      : `Saved ${result.saved} frame${result.saved === 1 ? '' : 's'} from ${result.videos} video${
          result.videos === 1 ? '' : 's'
        }.`
    if ('Notification' in window && Notification.permission === 'granted') {
      try {
        const n = new Notification('Thumbnail Faces', { body, tag: 'job-complete' })
        n.onclick = () => {
          window.focus()
          n.close()
        }
        setTimeout(() => n.close(), 10000)
      } catch (err) {
        console.warn('Could not show notification:', err)
      }
    }
    if (settingsRef.current.sound && !result.stopped) {
      new Audio('/completion-chime.wav').play().catch(() => {})
    }
  }, [])

  const handleEvent = useCallback(
    (msg) => {
      switch (msg.type) {
        case 'log':
          log(msg.video ? `[${msg.video}] ${msg.message}` : msg.message, msg.timestamp)
          return
        case 'roi':
          if (msg.video === selectedRef.current?.name) setRoi(msg.roi)
          break
        case 'video_complete':
          log(
            `${msg.video}: ${
              msg.error ? `error: ${msg.error}` : msg.stopped ? 'stopped' : `saved ${msg.saved} frame(s)`
            }`,
          )
          break
        case 'complete':
          log(
            msg.stopped
              ? 'Job stopped.'
              : `Job complete: ${msg.saved} frame(s) from ${msg.videos} video(s) in ${msg.elapsed}s.`,
          )
          notifyComplete(msg)
          break
        default:
          break
      }
      dispatch(msg)
    },
    [log, notifyComplete],
  )
  const connected = useEvents(handleEvent)

  const selectVideo = useCallback(
    async (video) => {
      setSelected(video)
      selectedRef.current = video
      setRoi(null)
      if (!video) return
      setDetectingRoi(true)
      const stillCurrent = () => selectedRef.current?.path === video.path
      try {
        const { roi } = await api.detectRoi(video.path)
        if (stillCurrent()) setRoi(roi)
      } catch (err) {
        if (stillCurrent()) {
          setRoi(ROI_FALLBACK)
          log(`ROI auto-detect failed (${err.message}); using default area`)
        }
      } finally {
        if (stillCurrent()) setDetectingRoi(false)
      }
    },
    [log],
  )

  const refreshVideos = useCallback(async () => {
    const { videos } = await api.videos()
    setVideos(videos)
    return videos
  }, [])

  useEffect(() => {
    let cancelled = false
    ;(async () => {
      try {
        const [cfg, list, current] = await Promise.all([api.config(), refreshVideos(), api.currentJob()])
        if (cancelled) return
        setConfig(cfg)
        setBackendOk(true)
        if (current.running) dispatch({ type: 'sync', ...current.job })
        if (list.length > 0 && !selectedRef.current) selectVideo(list[0])
      } catch (err) {
        if (!cancelled) {
          console.error('Backend unreachable:', err)
          setBackendOk(false)
        }
      }
    })()
    return () => {
      cancelled = true
    }
  }, [refreshVideos, selectVideo])

  const startJob = async (videoPaths, roiPosition) => {
    try {
      await api.startJob({
        videoPaths,
        roiPosition,
        emotionSensitivity: settings.sensitivity,
        targetEmotions: settings.targetEmotions,
        skipInterval: settings.skipInterval,
      })
      log(`Starting ${videoPaths.length} video${videoPaths.length === 1 ? '' : 's'}…`)
    } catch (err) {
      log(`Could not start: ${err.message}`)
    }
  }

  const stopJob = async () => {
    try {
      await api.stopJob()
    } catch (err) {
      log(`Could not stop: ${err.message}`)
    }
  }

  const requestNotifications = async () => {
    if (!('Notification' in window)) return
    setNotificationPermission(await Notification.requestPermission())
  }

  return (
    <TooltipProvider>
      <div className="min-h-screen bg-gray-900">
        <div className="container mx-auto px-4 py-8 max-w-6xl">
          <h1 className="text-3xl font-bold mb-4">Thumbnail Faces</h1>

          {(!backendOk || !connected) && (
            <div className="mb-6 p-4 rounded-lg bg-red-900/40 border border-red-700 text-red-100 text-sm">
              Cannot reach the backend. Run{' '}
              <code className="px-1 py-0.5 rounded bg-gray-800">./start-app.sh</code> from the project
              root (API on port 8000, UI on port 5173).
            </div>
          )}

          <div className="space-y-6">
            <VideoList
              videos={videos}
              selected={selected}
              onSelect={selectVideo}
              onRefresh={() => refreshVideos().catch((err) => log(`Refresh failed: ${err.message}`))}
              jobVideos={job.videos}
              disabled={job.running}
            />

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <Preview videoPath={selected?.path} roi={roi} detecting={detectingRoi} />
              <Controls
                settings={settings}
                onSettings={setSettings}
                job={job}
                selected={selected}
                roiReady={!!roi && !detectingRoi}
                videoCount={videos.length}
                onProcessOne={() => startJob([selected.path], roi)}
                onProcessAll={() => startJob(videos.map((v) => v.path), null)}
                onStop={stopJob}
                notificationPermission={notificationPermission}
                onEnableNotifications={requestNotifications}
                config={config}
                showAdvanced={showAdvanced}
                onToggleAdvanced={() => setShowAdvanced((v) => !v)}
              />
            </div>

            <JobStatus job={job} />
            <LogPanel logs={logs} />
          </div>
        </div>
      </div>
    </TooltipProvider>
  )
}
