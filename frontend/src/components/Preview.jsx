import { useState } from 'react'
import { previewUrl } from '../api/client'

function Spinner() {
  return (
    <svg className="animate-spin w-3 h-3 text-blue-400" viewBox="0 0 24 24" fill="none">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z" />
    </svg>
  )
}

/**
 * Middle frame of the selected video with the detected face area overlaid.
 * The overlay is positioned in percentages of the image itself, so it stays
 * correct for any aspect ratio without a canvas.
 */
export function Preview({ videoPath, roi, detecting }) {
  const [loadedPath, setLoadedPath] = useState(null)
  const loaded = loadedPath === videoPath

  return (
    <section className="lg:col-span-2 bg-gray-800 rounded-lg p-4">
      <h2 className="text-xl font-semibold mb-4">Preview</h2>
      <div className="relative overflow-hidden rounded-lg bg-black">
        {videoPath ? (
          <img
            key={videoPath}
            src={previewUrl(videoPath)}
            onLoad={() => setLoadedPath(videoPath)}
            alt=""
            className="block w-full h-auto"
          />
        ) : (
          <div className="aspect-video flex items-center justify-center text-gray-500">No video selected</div>
        )}
        {loaded && roi && (
          <div
            className="absolute pointer-events-none rounded-xl border-4 border-green-500 shadow-[0_0_0_9999px_rgba(0,0,0,0.6)]"
            style={{
              left: `${roi.left}%`,
              top: `${roi.top}%`,
              width: `${roi.right - roi.left}%`,
              height: `${roi.bottom - roi.top}%`,
            }}
          />
        )}
        {videoPath && detecting && (
          <div className="absolute top-3 right-3 px-3 py-1.5 rounded-md bg-gray-900/80 border border-gray-700 text-xs text-gray-200 flex items-center gap-2">
            <Spinner /> Detecting face area…
          </div>
        )}
      </div>
    </section>
  )
}
