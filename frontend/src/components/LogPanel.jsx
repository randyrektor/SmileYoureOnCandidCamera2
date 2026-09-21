import { useEffect, useRef } from 'react'

export function LogPanel({ logs }) {
  const bottomRef = useRef(null)
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ block: 'nearest' })
  }, [logs])

  return (
    <section className="bg-gray-800 rounded-lg p-4">
      <h2 className="text-xl font-semibold mb-4">Logs for Nerds</h2>
      <div className="space-y-1 text-sm font-mono text-gray-400 max-h-48 overflow-y-auto">
        {logs.map((log) => (
          <div key={log.id} className="flex gap-2">
            <span className="text-gray-500 shrink-0">[{log.timestamp}]</span>
            <span className="break-all">{log.message}</span>
          </div>
        ))}
        <div ref={bottomRef} />
      </div>
    </section>
  )
}
