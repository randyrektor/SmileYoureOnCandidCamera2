import * as Slider from '@radix-ui/react-slider'
import { Tooltip } from './Tooltip'

import { EMOTIONS } from '../lib/emotions'

function LabeledSlider({ label, tooltip, value, min, max, onChange, footer, small }) {
  return (
    <div className="space-y-2">
      <Tooltip text={tooltip}>
        <span className={small ? 'text-xs text-gray-400' : 'text-sm'}>{label}</span>
      </Tooltip>
      <Slider.Root
        className="relative flex items-center select-none touch-none h-5"
        value={[value]}
        min={min}
        max={max}
        step={1}
        onValueChange={([v]) => onChange(v)}
      >
        <Slider.Track className="bg-gray-700 relative grow rounded-full h-1">
          <Slider.Range className="absolute bg-blue-500 rounded-full h-full" />
        </Slider.Track>
        <Slider.Thumb
          className={`block bg-white border-2 border-blue-500 rounded-full shadow focus:outline-none ${
            small ? 'w-3 h-3' : 'w-5 h-5'
          }`}
          aria-label={label}
        />
      </Slider.Root>
      {footer}
    </div>
  )
}

export function Controls({
  settings,
  onSettings,
  job,
  selected,
  roiReady,
  videoCount,
  onProcessOne,
  onProcessAll,
  onStop,
  notificationPermission,
  onEnableNotifications,
  config,
  showAdvanced,
  onToggleAdvanced,
}) {
  const { sensitivity, targetEmotions, skipInterval, sound } = settings
  const set = (patch) => onSettings({ ...settings, ...patch })
  const noEmotions = targetEmotions.length === 0
  const canProcessOne = !job.running && !!selected && roiReady && !noEmotions
  const canProcessAll = !job.running && videoCount > 0 && !noEmotions

  return (
    <section className="bg-gray-800 rounded-lg p-4">
      <h2 className="text-xl font-semibold mb-4">Controls</h2>
      <div className="space-y-6">
        <div className="space-y-2">
          <Tooltip text="Select which emotions to look for">
            <span className="text-sm">Target Emotions</span>
          </Tooltip>
          <div className="grid grid-cols-2 gap-2">
            {EMOTIONS.map((emotion) => (
              <label key={emotion} className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  checked={targetEmotions.includes(emotion)}
                  onChange={(e) =>
                    set({
                      targetEmotions: e.target.checked
                        ? [...targetEmotions, emotion]
                        : targetEmotions.filter((x) => x !== emotion),
                    })
                  }
                  className="rounded border-gray-600 bg-gray-700 accent-blue-600"
                />
                <span className="text-sm capitalize">{emotion}</span>
              </label>
            ))}
          </div>
        </div>

        <LabeledSlider
          label={`Emotion Strictness: ${sensitivity} / 5`}
          tooltip="How strict to be when judging expressions. Higher = fewer but more clearly expressive frames. Lower = more frames, including subtler reactions."
          value={sensitivity}
          min={1}
          max={5}
          onChange={(v) => set({ sensitivity: v })}
          footer={
            <div className="flex justify-between text-xs text-gray-500">
              <span>Loose</span>
              <span>Strict</span>
            </div>
          }
        />

        {notificationPermission === 'default' && (
          <button
            type="button"
            onClick={onEnableNotifications}
            className="w-full px-4 py-2 rounded-lg text-sm bg-gray-700 hover:bg-gray-600"
          >
            Enable desktop notifications when a job finishes
          </button>
        )}

        <div className="space-y-2">
          {job.running ? (
            <button
              onClick={onStop}
              className="w-full px-6 py-2 rounded-lg font-medium bg-red-600 hover:bg-red-500"
            >
              Stop
            </button>
          ) : (
            <>
              <button
                onClick={onProcessOne}
                disabled={!canProcessOne}
                className="w-full px-6 py-2 rounded-lg font-medium bg-blue-600 hover:bg-blue-500 disabled:bg-gray-700 disabled:cursor-not-allowed"
              >
                {selected && !roiReady ? 'Detecting face area…' : 'Process Selected'}
              </button>
              <button
                onClick={onProcessAll}
                disabled={!canProcessAll}
                className="w-full px-6 py-2 rounded-lg font-medium bg-green-600 hover:bg-green-500 disabled:bg-gray-700 disabled:cursor-not-allowed"
              >
                Process All ({videoCount})
              </button>
            </>
          )}
          {noEmotions && (
            <p className="text-xs text-red-400 text-center">Select at least one target emotion</p>
          )}
        </div>

        <div>
          <button
            className="w-full text-left px-3 py-2 rounded bg-gray-700 hover:bg-gray-600 text-sm font-semibold text-gray-300"
            onClick={onToggleAdvanced}
          >
            {showAdvanced ? '▼' : '►'} Advanced
          </button>
          {showAdvanced && (
            <div className="mt-3 space-y-4">
              <div className="p-3 rounded-lg bg-gray-700">
                <LabeledSlider
                  small
                  label={`Analyze every ${skipInterval} frame${skipInterval === 1 ? '' : 's'}`}
                  tooltip="Lower = catches shorter reactions but slower. Higher = faster but may miss split-second expressions."
                  value={skipInterval}
                  min={1}
                  max={10}
                  onChange={(v) => set({ skipInterval: v })}
                />
              </div>
              <label className="flex items-center space-x-2 text-sm text-gray-300">
                <input
                  type="checkbox"
                  checked={sound}
                  onChange={(e) => set({ sound: e.target.checked })}
                  className="accent-blue-600"
                />
                <span>Play a chime when a job finishes</span>
              </label>
              {config && (
                <div className="p-3 rounded-lg bg-gray-700 text-xs text-gray-400 space-y-1 break-all">
                  <div>
                    <span className="text-gray-500">Videos:</span> {config.video_dir}
                  </div>
                  <div>
                    <span className="text-gray-500">Output:</span> {config.output_dir}
                  </div>
                  <div>
                    <span className="text-gray-500">Parallel workers:</span> {config.workers}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </section>
  )
}
