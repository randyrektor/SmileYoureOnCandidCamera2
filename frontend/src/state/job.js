/** Reducer for the job lifecycle messages the backend sends over the WebSocket. */

export const initialJob = { running: false, order: [], videos: {}, result: null, started: null }

const queued = () => ({ status: 'queued', progress: 0 })
const entry = (state, name) => state.videos[name] ?? queued()
const withVideo = (state, name, patch) => ({
  ...state,
  videos: { ...state.videos, [name]: { ...entry(state, name), ...patch } },
})

export function jobReducer(state, msg) {
  switch (msg.type) {
    case 'job_started':
    case 'sync':
      return {
        running: true,
        started: msg.started,
        result: null,
        order: msg.videos,
        videos: Object.fromEntries(msg.videos.map((name) => [name, queued()])),
      }
    case 'progress':
      return withVideo(state, msg.video, {
        status: 'running',
        progress: msg.progress,
        fps: msg.fps,
        elapsed: msg.elapsed,
        eta: msg.eta,
      })
    case 'roi':
      return withVideo(state, msg.video, { roi: msg.roi })
    case 'video_complete': {
      const status = msg.error ? 'error' : msg.stopped ? 'stopped' : 'done'
      return withVideo(state, msg.video, {
        status,
        progress: status === 'done' ? 100 : entry(state, msg.video).progress,
        saved: msg.saved,
        error: msg.error,
      })
    }
    case 'complete':
      return { ...state, running: false, result: msg }
    case 'reset':
      return initialJob
    default:
      return state
  }
}

export function overallProgress(job) {
  if (!job.order.length) return 0
  const sum = job.order.reduce((acc, name) => acc + (job.videos[name]?.progress ?? 0), 0)
  return sum / job.order.length
}
