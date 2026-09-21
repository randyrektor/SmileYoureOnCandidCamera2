/** Relative URLs: the Vite dev server proxies /api and /ws to the backend. */

async function apiJson(path, options = {}) {
  const res = await fetch(path, {
    ...options,
    headers: { 'Content-Type': 'application/json', ...(options.headers ?? {}) },
  })
  if (!res.ok) {
    let detail = `HTTP ${res.status}`
    try {
      const body = await res.json()
      if (body.detail) detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail)
    } catch {
      /* not JSON */
    }
    throw new Error(detail)
  }
  return res.json()
}

const post = (path, body) => apiJson(path, { method: 'POST', body: body ? JSON.stringify(body) : undefined })

export const api = {
  health: () => apiJson('/api/health'),
  config: () => apiJson('/api/config'),
  videos: () => apiJson('/api/videos'),
  currentJob: () => apiJson('/api/jobs/current'),
  detectRoi: (videoPath) => post('/api/detect-roi', { video_path: videoPath, num_samples: 8 }),
  startJob: (params) => post('/api/jobs', params),
  stopJob: () => post('/api/jobs/stop'),
}

export function previewUrl(videoPath) {
  return `/api/preview-frame?video_path=${encodeURIComponent(videoPath)}`
}

export function wsEventsUrl() {
  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  return `${proto}//${window.location.host}/ws/events`
}
