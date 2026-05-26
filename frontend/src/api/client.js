/** Relative URLs — Vite dev server proxies /api and /ws to the backend. */

export function apiUrl(path) {
  return path.startsWith('/') ? path : `/${path}`
}

export async function apiFetch(path, options = {}) {
  return fetch(apiUrl(path), options)
}

export function wsLogsUrl() {
  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  return `${proto}//${window.location.host}/ws/logs`
}
