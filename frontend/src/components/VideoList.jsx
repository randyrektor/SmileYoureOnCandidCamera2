import { formatDuration } from '../lib/format'

const STATUS_STYLES = {
  queued: 'text-gray-400',
  running: 'text-blue-300',
  done: 'text-green-400',
  error: 'text-red-400',
  stopped: 'text-yellow-400',
}

function statusLabel(v) {
  if (!v) return null
  if (v.status === 'running') return `${v.progress.toFixed(0)}%`
  if (v.status === 'done') return `✓ ${v.saved ?? 0} saved`
  if (v.status === 'error') return '✗ error'
  if (v.status === 'stopped') return 'stopped'
  return 'queued'
}

export function VideoList({ videos, selected, onSelect, onRefresh, jobVideos, disabled }) {
  return (
    <section className="bg-gray-800 rounded-lg p-4">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-xl font-semibold">Videos</h2>
        <button
          onClick={onRefresh}
          className="p-2 rounded-lg bg-gray-700 hover:bg-gray-600 transition-colors"
          title="Refresh video list"
          aria-label="Refresh video list"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
            />
          </svg>
        </button>
      </div>
      {videos.length === 0 && (
        <p className="text-sm text-gray-400">No videos found. Drop files into the input folder and refresh.</p>
      )}
      <div className="grid grid-cols-1 gap-1.5">
        {videos.map((video) => {
          const jv = jobVideos[video.name]
          const isSelected = selected?.path === video.path
          return (
            <button
              key={video.path}
              onClick={() => onSelect(video)}
              disabled={disabled}
              className={`relative overflow-hidden p-2 rounded-lg text-left flex justify-between items-center gap-3 ${
                isSelected ? 'bg-blue-600' : 'bg-gray-700 hover:bg-gray-600'
              } disabled:opacity-60 disabled:cursor-not-allowed`}
            >
              {jv?.status === 'running' && (
                <div
                  className="absolute inset-y-0 left-0 bg-white/10 transition-[width] duration-500"
                  style={{ width: `${jv.progress}%` }}
                />
              )}
              <span className="relative font-medium text-sm truncate">{video.name}</span>
              <span className="relative text-xs flex items-center gap-3 shrink-0">
                {jv && <span className={STATUS_STYLES[jv.status]}>{statusLabel(jv)}</span>}
                <span className="text-gray-300">{formatDuration(video.duration)}</span>
              </span>
            </button>
          )
        })}
      </div>
    </section>
  )
}
