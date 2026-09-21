import { formatDuration } from '../lib/format'
import { overallProgress } from '../state/job'

function Summary({ result }) {
  const failed = result.errors.length
  const tone = result.stopped ? 'text-yellow-400' : failed ? 'text-yellow-400' : 'text-green-400'
  return (
    <div className={`text-sm font-semibold ${tone}`}>
      {result.stopped ? 'Stopped. ' : 'Done. '}
      Saved {result.saved} frame{result.saved === 1 ? '' : 's'} from {result.videos} video
      {result.videos === 1 ? '' : 's'} in {formatDuration(result.elapsed)}
      {failed > 0 && ` (${failed} failed)`}
    </div>
  )
}

export function JobStatus({ job }) {
  const progress = job.running ? overallProgress(job) : job.result ? 100 : 0
  const done = job.order.filter((n) => ['done', 'error', 'stopped'].includes(job.videos[n]?.status)).length
  const running = job.order.filter((n) => job.videos[n]?.status === 'running')

  return (
    <section className="bg-gray-800 rounded-lg p-4">
      <h2 className="text-xl font-semibold mb-4">Status</h2>
      <div className="space-y-3">
        <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
          <div
            className="h-full bg-blue-600 transition-[width] duration-300"
            style={{ width: `${progress}%` }}
          />
        </div>
        <div className="text-sm text-gray-400">
          {job.running ? (
            <div className="space-y-2">
              <div>
                {done} of {job.order.length} finished
                {running.length > 0 && ` · ${running.length} in progress`}
              </div>
              {running.map((name) => {
                const v = job.videos[name]
                return (
                  <div key={name} className="flex justify-between gap-4 text-xs font-mono">
                    <span className="truncate">{name}</span>
                    <span className="shrink-0">
                      {v.progress.toFixed(0)}% · {v.fps?.toFixed(0) ?? '–'} fps · ETA{' '}
                      {formatDuration(v.eta)}
                    </span>
                  </div>
                )
              })}
            </div>
          ) : job.result ? (
            <>
              <Summary result={job.result} />
              {job.result.errors.map((e) => (
                <div key={e.video} className="text-xs text-red-400 mt-1">
                  {e.video}: {e.error}
                </div>
              ))}
            </>
          ) : (
            <div>Idle</div>
          )}
        </div>
      </div>
    </section>
  )
}
