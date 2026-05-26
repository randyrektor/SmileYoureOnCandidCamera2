import { useState, useRef, useEffect } from 'react'
import * as Slider from '@radix-ui/react-slider'
import { Tooltip } from "./components/Tooltip"
import { apiFetch, wsLogsUrl } from './api/client'

const formatDuration = (seconds) => {
  if (isNaN(seconds)) return '00:00'
  
  // Handle minutes greater than 59
  const hours = Math.floor(seconds / 3600)
  const minutes = Math.floor((seconds % 3600) / 60)
  const remainingSeconds = Math.floor(seconds % 60)
  
  if (hours > 0) {
    return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${remainingSeconds.toString().padStart(2, '0')}`
  }
  return `${minutes.toString().padStart(2, '0')}:${remainingSeconds.toString().padStart(2, '0')}`
}

const formatTime = (timeStr) => {
  if (!timeStr) return '00:00:00'
  return timeStr
}

function EmotionDetectorApp() {
  const [availableVideos, setAvailableVideos] = useState([])
  const [selectedVideo, setSelectedVideo] = useState(null)
  const [previewImage, setPreviewImage] = useState(null)
  const [isProcessing, setIsProcessing] = useState(false)
  const [progress, setProgress] = useState(0)
  const [processingStats, setProcessingStats] = useState(null)
  const [emotionSensitivity, setEmotionSensitivity] = useState(4)

  const [targetEmotions, setTargetEmotions] = useState(['happy', 'surprise', 'angry', 'sad', 'fear', 'disgust'])
  // null until auto-detection finishes on the current preview video. Drawing
  // a wrong default rectangle before detection is more confusing than no rect.
  const [roiPosition, setRoiPosition] = useState(null)
  const [logs, setLogs] = useState([])
  const [isComplete, setIsComplete] = useState(false)
  const [showAdvanced, setShowAdvanced] = useState(false)
  // How many frames to skip between analyses. Lower = denser sampling
  // (slower, catches shorter reactions), higher = faster.
  const [skipInterval, setSkipInterval] = useState(3)
  // Auto ROI detection state
  const [isDetectingROI, setIsDetectingROI] = useState(false)
  // Batch processing state
  const [isBatchProcessing, setIsBatchProcessing] = useState(false)
  const [batchProgress, setBatchProgress] = useState({ current: 0, total: 0, currentVideo: null })
  const [batchErrors, setBatchErrors] = useState([])
  const [batchComplete, setBatchComplete] = useState(false)
  const [backendConnected, setBackendConnected] = useState(true)
  const [appConfig, setAppConfig] = useState(null)
  const [notificationPermission, setNotificationPermission] = useState(
    typeof Notification !== 'undefined' ? Notification.permission : 'unsupported'
  )

  const processingCompleteRef = useRef(false)
  const currentProgressRef = useRef(0)
  const isBatchProcessingRef = useRef(false)
  const isProcessingRef = useRef(false)

  // Browser notification (optional - only if user has enabled it)
  const showCompletionNotification = () => {
    if ('Notification' in window && Notification.permission === 'granted') {
      try {
        const notification = new Notification('Processing Complete! 🎉', {
          body: 'Your video processing has finished successfully.',
          tag: 'processing-complete',
          requireInteraction: false,
          silent: false
        })
        
        // Auto-close notification after 10 seconds
        setTimeout(() => notification.close(), 10000)
        
        // Focus window when notification is clicked
        notification.onclick = () => {
          window.focus()
          notification.close()
        }
      } catch (error) {
        console.warn('Could not show notification:', error)
      }
    }
  }

  const canvasRef = useRef(null)
  const ws = useRef(null)
  // Tracks the most recent video we kicked off ROI detection for, so a fast
  // user clicking through several videos doesn't end up with the wrong ROI
  // landing last.
  const roiRequestPathRef = useRef(null)

  // Fallback ROI used only when auto-detection fails outright (network error,
  // no face found, etc.). Roughly centered on a 16:9 talking-head shot.
  const ROI_FALLBACK = { top: 20, bottom: 65, left: 25, right: 75 }

  // Run auto ROI detection for a given video and update state. Idempotent: if
  // the user clicks another video mid-flight we discard stale results.
  const autoDetectROI = async (video) => {
    if (!video) return
    roiRequestPathRef.current = video.path
    setIsDetectingROI(true)
    try {
      const response = await apiFetch('/api/detect-roi', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ video_path: video.path, num_samples: 8 }),
      })
      if (!response.ok) throw new Error(`HTTP ${response.status}`)
      const data = await response.json()
      if (roiRequestPathRef.current !== video.path) return
      if (data.status === 'success' && data.roi) {
        setRoiPosition({
          top: data.roi.top,
          bottom: data.roi.bottom,
          left: data.roi.left,
          right: data.roi.right,
        })
      } else {
        setRoiPosition(ROI_FALLBACK)
      }
    } catch (error) {
      console.warn('Auto ROI detection failed; using fallback:', error)
      if (roiRequestPathRef.current === video.path) {
        setRoiPosition(ROI_FALLBACK)
        setLogs(prevLogs => [...prevLogs, {
          timestamp: new Date().toLocaleTimeString(),
          message: `ROI auto-detect failed (${error.message}); using default ROI`,
        }])
      }
    } finally {
      if (roiRequestPathRef.current === video.path) {
        setIsDetectingROI(false)
      }
    }
  }

  useEffect(() => {
    isBatchProcessingRef.current = isBatchProcessing
  }, [isBatchProcessing])

  useEffect(() => {
    isProcessingRef.current = isProcessing
  }, [isProcessing])

  useEffect(() => {
    const fetchAndSelectFirstVideo = async () => {
      try {
        setBackendConnected(true)

        try {
          const configResponse = await apiFetch('/api/config')
          if (configResponse.ok) {
            setAppConfig(await configResponse.json())
          }
        } catch (error) {
          console.warn('Could not fetch config:', error)
        }

        const response = await apiFetch('/api/available-videos')
        if (!response.ok) throw new Error('Failed to fetch videos')
        const data = await response.json()
        
        // Format video durations and prepare for display
        const videosWithFormattedDurations = data.videos.map(video => ({
          ...video,
          duration: parseFloat(video.duration),  // Convert duration to number
          formattedDuration: formatDuration(parseFloat(video.duration))  // Format for display
        }))
        
        setAvailableVideos(videosWithFormattedDurations)
        
        // Automatically select and preview the first available video
        if (data.videos.length > 0) {
          const firstVideo = data.videos[0]
          setSelectedVideo(firstVideo)

          const previewResponse = await apiFetch(
            `/api/preview-frame?video_path=${encodeURIComponent(firstVideo.path)}`
          )
          if (!previewResponse.ok) throw new Error('Failed to fetch preview frame')

          const previewData = await previewResponse.json()
          const img = new Image()
          img.onload = () => {
            setPreviewImage(img)
          }
          img.src = `data:image/jpeg;base64,${previewData.frame}`

          // Kick off ROI detection in parallel so the preview overlay snaps
          // to the right area as soon as both the image and ROI arrive.
          autoDetectROI(firstVideo)
        }
      } catch (error) {
        console.error('Error fetching videos:', error)
        setBackendConnected(false)
      }
    }

    fetchAndSelectFirstVideo()
  }, [])

  useEffect(() => {
    let reconnectTimeout = null;
    let wsInstance = null;
    let isUnmounted = false;

    const connectWebSocket = () => {
      if (wsInstance) {
        wsInstance.onopen = null;
        wsInstance.onclose = null;
        wsInstance.onerror = null;
        wsInstance.onmessage = null;
        try { wsInstance.close(); } catch {}
        wsInstance = null;
      }
      wsInstance = new window.WebSocket(wsLogsUrl());
      ws.current = wsInstance;

      wsInstance.onopen = () => {
        if (isUnmounted) return;
        console.log('WebSocket connected successfully');
      };
      wsInstance.onclose = (event) => {
        ws.current = null;
        wsInstance = null;
        if (!isUnmounted) {
          reconnectTimeout = setTimeout(connectWebSocket, 3000);
        }
      };
      wsInstance.onerror = (error) => {
        console.error('WebSocket error:', error);
      };
      wsInstance.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === "progress") {
            setProgress(data.progress);
            currentProgressRef.current = data.progress; // Update ref for batch processing
            setProcessingStats({
              fps: data.fps,
              elapsed: data.elapsed,
              eta: data.eta
            });
          } else if (data.type === "log") {
            setLogs(prevLogs => {
              // Deduplicate logs by timestamp and message to avoid duplicates
              if (prevLogs.some(l => l.timestamp === data.timestamp && l.message === data.message)) {
                return prevLogs;
              }
              return [...prevLogs, { timestamp: data.timestamp, message: data.message }];
            });
          } else if (data.type === "complete") {
            setIsProcessing(false);
            setProgress(100);
            currentProgressRef.current = 100; // Update ref for batch processing
            setProcessingStats(null);
            setIsComplete(true);
            processingCompleteRef.current = true; // Set ref for batch processing
            
            if (!isBatchProcessingRef.current && Notification.permission === 'granted') {
              showCompletionNotification();
            }
            
            // Add completion summary log with elapsed time
            setLogs(prevLogs => {
              let elapsed = null;
              if (data.elapsed) {
                elapsed = data.elapsed;
              } else if (processingStats && processingStats.elapsed) {
                elapsed = processingStats.elapsed;
              }
              let summary = 'Processing complete.';
              if (elapsed) {
                summary += ` Total time: ${elapsed}`;
              }
              const now = new Date();
              const timestamp = now.toLocaleTimeString();
              return [...prevLogs, { timestamp, message: summary }];
            });
          }
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };
    };

    connectWebSocket();

    return () => {
      isUnmounted = true;
      if (reconnectTimeout) clearTimeout(reconnectTimeout);
      if (wsInstance) {
        wsInstance.onopen = null;
        wsInstance.onclose = null;
        wsInstance.onerror = null;
        wsInstance.onmessage = null;
        try { wsInstance.close(); } catch {}
        wsInstance = null;
      }
      ws.current = null;
    };
  }, []);

  const handleVideoSelect = async (video) => {
    setSelectedVideo(video)
    // Hide any previous video's ROI overlay immediately. The new one is
    // re-detected asynchronously and will appear when ready.
    setRoiPosition(null)

    try {
      const response = await apiFetch(
        `/api/preview-frame?video_path=${encodeURIComponent(video.path)}`
      )
      if (!response.ok) throw new Error('Failed to fetch preview frame')

      const data = await response.json()
      const img = new Image()
      img.onload = () => {
        setPreviewImage(img)
      }
      img.src = `data:image/jpeg;base64,${data.frame}`

      autoDetectROI(video)
    } catch (error) {
      console.error('Error grabbing frame:', error)
    }
  }

  const drawPreviewWithOverlay = (img) => {
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')

    ctx.clearRect(0, 0, canvas.width, canvas.height)
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height)

    // Don't render any rectangle until ROI is known. Showing a default that
    // doesn't match the actual face area is more misleading than no overlay.
    if (!roiPosition) return

    const cornerRadius = 12;
    
    ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
    ctx.beginPath();
    // Draw outer rectangle
    ctx.rect(0, 0, canvas.width, canvas.height);
    
    // Draw inner rounded rectangle (ROI cutout)
    const x = canvas.width * (roiPosition.left / 100);
    const y = canvas.height * (roiPosition.top / 100);
    const width = canvas.width * ((roiPosition.right - roiPosition.left) / 100);
    const height = canvas.height * ((roiPosition.bottom - roiPosition.top) / 100);
    
    ctx.moveTo(x + cornerRadius, y);
    ctx.lineTo(x + width - cornerRadius, y);
    ctx.arcTo(x + width, y, x + width, y + cornerRadius, cornerRadius);
    ctx.lineTo(x + width, y + height - cornerRadius);
    ctx.arcTo(x + width, y + height, x + width - cornerRadius, y + height, cornerRadius);
    ctx.lineTo(x + cornerRadius, y + height);
    ctx.arcTo(x, y + height, x, y + height - cornerRadius, cornerRadius);
    ctx.lineTo(x, y + cornerRadius);
    ctx.arcTo(x, y, x + cornerRadius, y, cornerRadius);
    
    ctx.fill('evenodd');
    
    // Draw green border around ROI with rounded corners
    ctx.strokeStyle = '#22c55e';  // Tailwind green-500
    ctx.lineWidth = 5;
    ctx.beginPath();
    ctx.moveTo(x + cornerRadius, y);
    ctx.lineTo(x + width - cornerRadius, y);
    ctx.arcTo(x + width, y, x + width, y + cornerRadius, cornerRadius);
    ctx.lineTo(x + width, y + height - cornerRadius);
    ctx.arcTo(x + width, y + height, x + width - cornerRadius, y + height, cornerRadius);
    ctx.lineTo(x + cornerRadius, y + height);
    ctx.arcTo(x, y + height, x, y + height - cornerRadius, cornerRadius);
    ctx.lineTo(x, y + cornerRadius);
    ctx.arcTo(x, y, x + cornerRadius, y, cornerRadius);
    ctx.stroke();
  }

  const startProcessing = async () => {
    if (!selectedVideo) return

    // ROI is detected automatically on video select. If detection hasn't
    // finished yet (rare; ~400ms), wait for it rather than processing with
    // a stale value.
    if (!roiPosition || isDetectingROI) {
      setLogs(prevLogs => [...prevLogs, {
        timestamp: new Date().toLocaleTimeString(),
        message: 'Waiting for ROI auto-detection to finish before processing...',
      }])
      return
    }

    try {
      setIsProcessing(true)
      setIsComplete(false)
      setProgress(0)
      setProcessingStats(null)

      const response = await apiFetch('/api/start-processing', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          debug: false,
          emotionSensitivity: emotionSensitivity,
          roiPosition: roiPosition,
          videoPath: selectedVideo.path,
          targetEmotions: targetEmotions,
          skipInterval: skipInterval,
        }),
      })

      if (!response.ok) {
        throw new Error('Failed to start processing')
      }

      const result = await response.json()
      console.log('Processing started:', result)
    } catch (error) {
      console.error('Error starting processing:', error)
      setIsProcessing(false)
      setProcessingStats(null)
    }
  }

  const stopProcessing = async () => {
    try {
      const response = await apiFetch('/api/stop-processing', {
        method: 'POST'
      })
      
      if (response.ok) {
        setIsProcessing(false)
        setIsComplete(true)
        setProcessingStats(null)
      }
    } catch (error) {
      console.error('Error stopping processing:', error)
    }
  }

  const refreshVideos = async () => {
    try {
      const response = await apiFetch('/api/available-videos')
      if (!response.ok) throw new Error('Failed to fetch videos')
      const data = await response.json()
      
      const videosWithFormattedDurations = data.videos.map(video => ({
        ...video,
        duration: parseFloat(video.duration),
        formattedDuration: formatDuration(parseFloat(video.duration))
      }))
      
      setAvailableVideos(videosWithFormattedDurations)
    } catch (error) {
      console.error('Error refreshing videos:', error)
    }
  }

  const processBatchVideos = async () => {
    if (availableVideos.length === 0) return
    
    setIsBatchProcessing(true)
    setBatchComplete(false)
    setBatchErrors([])
    setBatchProgress({ current: 0, total: availableVideos.length, currentVideo: null })
    
    const now = new Date()
    const timestamp = now.toLocaleTimeString()
    setLogs(prevLogs => [...prevLogs, { 
      timestamp, 
      message: `Starting batch processing of ${availableVideos.length} videos...` 
    }])
    
    const batchErrorsCollected = []

    for (let i = 0; i < availableVideos.length; i++) {
      const video = availableVideos[i]
      
      try {
        // Update batch progress
        setBatchProgress({ 
          current: i + 1, 
          total: availableVideos.length, 
          currentVideo: video.name 
        })
        
        // Select the video (for visual feedback)
        setSelectedVideo(video)
        
        // Log start of video processing
        const startTime = new Date()
        const startTimestamp = startTime.toLocaleTimeString()
        setLogs(prevLogs => [...prevLogs, { 
          timestamp: startTimestamp, 
          message: `[${i + 1}/${availableVideos.length}] Processing: ${video.name}` 
        }])
        
        // Step 1: Auto-detect ROI for this video
        const roiResponse = await apiFetch('/api/detect-roi', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            video_path: video.path,
            num_samples: 8
          })
        })
        
        if (!roiResponse.ok) throw new Error('Failed to detect ROI')
        
        const roiData = await roiResponse.json()
        
        if (roiData.status === 'success' && roiData.roi) {
          // Update ROI for this video
          const detectedROI = {
            top: roiData.roi.top,
            bottom: roiData.roi.bottom,
            left: roiData.roi.left,
            right: roiData.roi.right
          }
          setRoiPosition(detectedROI)
          
          // Fetch and update preview for this video
          try {
            const previewResponse = await apiFetch(
              `/api/preview-frame?video_path=${encodeURIComponent(video.path)}`
            )
            if (previewResponse.ok) {
              const previewData = await previewResponse.json()
              const img = new Image()
              await new Promise((resolve) => {
                img.onload = () => {
                  setPreviewImage(img)
                  resolve()
                }
                img.src = `data:image/jpeg;base64,${previewData.frame}`
              })
            }
          } catch (err) {
            console.warn('Failed to load preview:', err)
          }
          
          const roiTimestamp = new Date().toLocaleTimeString()
          setLogs(prevLogs => [...prevLogs, { 
            timestamp: roiTimestamp, 
            message: `  ✓ ROI detected for ${video.name}` 
          }])
          
          // Wait a moment for state to update
          await new Promise(resolve => setTimeout(resolve, 300))
          
          // Step 2: Start processing with detected ROI
          setIsProcessing(true)
          setIsComplete(false)
          setProgress(0)
          setProcessingStats(null)
          processingCompleteRef.current = false  // Reset completion flag
          currentProgressRef.current = 0  // Reset progress
          
          const processResponse = await apiFetch('/api/start-processing', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              debug: false,
              emotionSensitivity: emotionSensitivity,
              roiPosition: detectedROI,
              videoPath: video.path,
              targetEmotions: targetEmotions,
              skipInterval: skipInterval,
            })
          })
          
          if (!processResponse.ok) {
            throw new Error('Failed to start processing')
          }
          
          // Wait for processing to complete by monitoring the ref
          await new Promise((resolve, reject) => {
            let checksWithoutProgress = 0
            const maxChecksWithoutProgress = 60 // 5 minutes without progress
            let lastSeenProgress = 0
            
            const checkInterval = setInterval(() => {
              const currentProgress = currentProgressRef.current
              const isCompleteNow = processingCompleteRef.current
              
              // Check for completion
              if (isCompleteNow || currentProgress >= 100) {
                clearInterval(checkInterval)
                setIsProcessing(false)
                setIsComplete(false)
                setProgress(0)
                processingCompleteRef.current = false // Reset for next video
                currentProgressRef.current = 0 // Reset for next video
                resolve()
                return
              }
              
              // Check if progress is stuck
              if (currentProgress === lastSeenProgress) {
                checksWithoutProgress++
                if (checksWithoutProgress >= maxChecksWithoutProgress) {
                  clearInterval(checkInterval)
                  reject(new Error('Processing appears stuck (no progress for 5 minutes)'))
                  return
                }
              } else {
                checksWithoutProgress = 0
                lastSeenProgress = currentProgress
              }
            }, 5000) // Check every 5 seconds
            
            // Safety timeout (2 hours max per video)
            setTimeout(() => {
              clearInterval(checkInterval)
              reject(new Error('Processing timeout (2 hours)'))
            }, 2 * 60 * 60 * 1000)
          })
          
          const completeTimestamp = new Date().toLocaleTimeString()
          setLogs(prevLogs => [...prevLogs, { 
            timestamp: completeTimestamp, 
            message: `  ✓ Completed: ${video.name}` 
          }])
          
        } else {
          throw new Error('Invalid ROI detection response')
        }
        
      } catch (error) {
        // Log error but continue with next video
        console.error(`Error processing ${video.name}:`, error)
        const errorTimestamp = new Date().toLocaleTimeString()
        setLogs(prevLogs => [...prevLogs, { 
          timestamp: errorTimestamp, 
          message: `  ✗ Error processing ${video.name}: ${error.message}` 
        }])
        batchErrorsCollected.push({ video: video.name, error: error.message })
        setBatchErrors([...batchErrorsCollected])
        
        // Make sure processing is stopped before continuing
        try {
          await apiFetch('/api/stop-processing', { method: 'POST' })
        } catch {}
        setIsProcessing(false)
        setProgress(0)
      }
    }
    
    // All videos processed
    setIsBatchProcessing(false)
    setBatchComplete(true)
    
    const finalTimestamp = new Date().toLocaleTimeString()
    const successCount = availableVideos.length - batchErrorsCollected.length
    setBatchErrors(batchErrorsCollected)
    setLogs(prevLogs => [...prevLogs, { 
      timestamp: finalTimestamp, 
      message: `Batch complete! Processed ${successCount} of ${availableVideos.length} videos successfully.` 
    }])

    if (Notification.permission === 'granted') {
      showCompletionNotification()
    }
  }

  const requestNotificationPermission = async () => {
    if (typeof Notification === 'undefined') return
    const permission = await Notification.requestPermission()
    setNotificationPermission(permission)
  }

  const stopBatchProcessing = async () => {
    setIsBatchProcessing(false)
    await stopProcessing()
    
    const timestamp = new Date().toLocaleTimeString()
    setLogs(prevLogs => [...prevLogs, { 
      timestamp, 
      message: 'Batch processing stopped by user.' 
    }])
  }

  useEffect(() => {
    if (previewImage) {
      drawPreviewWithOverlay(previewImage)
    }
  }, [roiPosition, previewImage])

  useEffect(() => {
    return () => {
      if (isProcessingRef.current || isBatchProcessingRef.current) {
        apiFetch('/api/stop-processing', { method: 'POST' }).catch((error) => {
          console.error('Error cleaning up:', error)
        })
      }
    }
  }, []);

  return (
    <div className="min-h-screen bg-gray-900">
      <div className="container mx-auto px-4 py-8">
        <h1 className="text-3xl font-bold mb-4">Thumbnail Faces</h1>

        {!backendConnected && (
          <div className="mb-6 p-4 rounded-lg bg-red-900/40 border border-red-700 text-red-100 text-sm">
            Cannot reach the backend. Run <code className="px-1 py-0.5 rounded bg-gray-800">./start_dev.sh</code> from the project root (API on port 8000, UI on port 5173).
          </div>
        )}

        <div className="space-y-6">
          <div className="bg-gray-800 rounded-lg p-4">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-xl font-semibold">Available Videos</h2>
              <button
                onClick={refreshVideos}
                className="p-2 rounded-lg bg-gray-700 hover:bg-gray-600 transition-colors"
                title="Refresh video list"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                </svg>
              </button>
            </div>
            <div className="grid grid-cols-1 gap-1.5">
              {availableVideos.map((video) => (
                <button
                  key={video.path}
                  onClick={() => handleVideoSelect(video)}
                  disabled={isBatchProcessing}
                  className={`p-2 rounded-lg text-left flex justify-between items-center ${
                    selectedVideo?.path === video.path
                      ? 'bg-blue-600'
                      : isBatchProcessing
                      ? 'bg-gray-700 cursor-not-allowed opacity-50'
                      : 'bg-gray-700 hover:bg-gray-600'
                  }`}
                >
                  <div className="font-medium text-sm">{video.name}</div>
                  <div className="text-sm text-gray-400">{formatDuration(video.duration)}</div>
                </button>
              ))}
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2 bg-gray-800 rounded-lg p-4">
              <div className="flex justify-between items-center mb-4">
                <h2 className="text-xl font-semibold">Preview</h2>
              </div>
              <div className="relative aspect-video bg-black rounded-lg overflow-hidden">
                <canvas
                  ref={canvasRef}
                  className="w-full h-full"
                  width={1280}
                  height={720}
                />
                {!previewImage && (
                  <div className="absolute inset-0 flex items-center justify-center text-gray-500">
                    No videos available
                  </div>
                )}
                {previewImage && isDetectingROI && (
                  <div className="absolute top-3 right-3 px-3 py-1.5 rounded-md bg-gray-900/80 border border-gray-700 text-xs text-gray-200 flex items-center gap-2">
                    <svg
                      className="animate-spin w-3 h-3 text-blue-400"
                      viewBox="0 0 24 24"
                      fill="none"
                    >
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z" />
                    </svg>
                    Detecting face area…
                  </div>
                )}
              </div>
            </div>

            <div className="bg-gray-800 rounded-lg p-4">
              <h2 className="text-xl font-semibold mb-4">Controls</h2>
              
              <div className="space-y-6">
                {/* Target Emotions FIRST */}
                <div className="space-y-2">
                  <div className="space-y-2">
                    <Tooltip text="Select which emotions to detect in the video">
                      <span className="text-sm">Target Emotions</span>
                    </Tooltip>
                  </div>
                  <div className="grid grid-cols-2 gap-2">
                    {['happy', 'surprise', 'angry', 'sad', 'fear', 'disgust', 'neutral'].map((emotion) => (
                      <label key={emotion} className="flex items-center space-x-2">
                        <input
                          type="checkbox"
                          checked={targetEmotions.includes(emotion)}
                          onChange={(e) => {
                            if (e.target.checked) {
                              setTargetEmotions(prev => [...prev, emotion])
                            } else {
                              setTargetEmotions(prev => prev.filter(e => e !== emotion))
                            }
                          }}
                          className="rounded border-gray-600 bg-gray-700 text-blue-600 focus:ring-blue-500"
                        />
                        <span className="text-sm capitalize">{emotion}</span>
                      </label>
                    ))}
                  </div>
                </div>

                <div className="space-y-2">
                  <Tooltip text="How strict to be when judging expressions. Higher = fewer but more clearly expressive frames. Lower = more frames, including subtler reactions.">
                    <span className="text-sm">Emotion Strictness: {emotionSensitivity} / 5</span>
                  </Tooltip>
                  <Slider.Root
                    className="relative flex items-center select-none touch-none h-5"
                    value={[emotionSensitivity]}
                    min={1}
                    max={5}
                    step={1}
                    onValueChange={([value]) => setEmotionSensitivity(value)}
                  >
                    <Slider.Track className="bg-gray-700 relative grow rounded-full h-1">
                      <Slider.Range className="absolute bg-blue-500 rounded-full h-full" />
                    </Slider.Track>
                    <Slider.Thumb className="block w-5 h-5 bg-white border-2 border-blue-500 rounded-full shadow focus:outline-none" />
                  </Slider.Root>
                  <div className="flex justify-between text-xs text-gray-500">
                    <span>Loose</span>
                    <span>Strict</span>
                  </div>
                </div>

                {notificationPermission !== 'granted' && notificationPermission !== 'unsupported' && (
                  <button
                    type="button"
                    onClick={requestNotificationPermission}
                    className="w-full px-4 py-2 rounded-lg text-sm bg-gray-700 hover:bg-gray-600"
                  >
                    Enable desktop notifications when processing finishes
                  </button>
                )}

                <div className="space-y-2">
                  <button
                    onClick={isProcessing ? stopProcessing : startProcessing}
                    disabled={
                      !selectedVideo
                      || targetEmotions.length === 0
                      || isBatchProcessing
                      || (!isProcessing && (isDetectingROI || !roiPosition))
                    }
                    className={`w-full px-6 py-2 rounded-lg font-medium ${
                      !selectedVideo || targetEmotions.length === 0 || isBatchProcessing
                      || (!isProcessing && (isDetectingROI || !roiPosition))
                        ? 'bg-gray-700 cursor-not-allowed'
                        : isProcessing
                        ? 'bg-red-600 hover:bg-red-500'
                        : 'bg-blue-600 hover:bg-blue-500'
                    }`}
                  >
                    {isProcessing
                      ? 'Stop Processing'
                      : (isDetectingROI ? 'Detecting face area...' : 'Process One')}
                  </button>
                  
                  <button
                    onClick={isBatchProcessing ? stopBatchProcessing : processBatchVideos}
                    disabled={availableVideos.length === 0 || targetEmotions.length === 0 || isProcessing}
                    className={`w-full px-6 py-2 rounded-lg font-medium ${
                      availableVideos.length === 0 || targetEmotions.length === 0 || isProcessing
                        ? 'bg-gray-700 cursor-not-allowed'
                        : isBatchProcessing
                        ? 'bg-red-600 hover:bg-red-500'
                        : 'bg-green-500 hover:bg-green-600'
                    }`}
                  >
                    {isBatchProcessing ? 'Stop Batch Processing' : 'Process All'}
                  </button>
                  
                  {isBatchProcessing && (
                    <div className="text-sm text-gray-400 mt-2 p-3 bg-gray-700 rounded-lg">
                      <div className="font-semibold text-green-500 mb-1">
                        Processing video {batchProgress.current} of {batchProgress.total}
                      </div>
                      {batchProgress.currentVideo && (
                        <div className="text-xs truncate">
                          Current: {batchProgress.currentVideo}
                        </div>
                      )}
                    </div>
                  )}
                  
                  {batchComplete && (
                    <div className="text-sm text-green-500 mt-2 p-3 bg-gray-700 rounded-lg">
                      ✓ Batch processing complete! 
                      {batchErrors.length > 0 && (
                        <span className="text-yellow-400">
                          {' '}({batchErrors.length} error{batchErrors.length > 1 ? 's' : ''})
                        </span>
                      )}
                    </div>
                  )}
                  
                  {targetEmotions.length === 0 && (
                    <div className="text-xs text-red-400 mt-2 text-center">
                      Please select at least one target emotion
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>

          <div className="bg-gray-800 rounded-lg p-4">
            <h2 className="text-xl font-semibold mb-4">Processing Status</h2>
            <div className="space-y-4">
              <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
                <div
                  className="h-full bg-blue-600 transition-all duration-300"
                  style={{ width: `${isProcessing ? progress : isComplete ? 100 : 0}%` }}
                />
              </div>
              <div className="text-sm text-gray-400">
                {isProcessing ? (
                  processingStats ? (
                    <div className="flex justify-between items-center">
                      <div>FPS: {processingStats.fps ? processingStats.fps.toFixed(1) : 'N/A'}</div>
                      <div>Elapsed: {processingStats.elapsed ? formatTime(processingStats.elapsed) : 'N/A'}</div>
                      <div>ETA: {processingStats.eta ? formatTime(processingStats.eta) : 'N/A'}</div>
                    </div>
                  ) : (
                    <div>Processing started...</div>
                  )
                ) : isComplete ? (
                  <div className="text-green-500 font-semibold">Task complete!</div>
                ) : (
                  <div>Waiting to start processing...</div>
                )}
              </div>
            </div>
          </div>

          <div className="bg-gray-800 rounded-lg p-4">
            <h2 className="text-xl font-semibold mb-4">Logs for Nerds</h2>
            <div className="space-y-1 text-sm font-mono text-gray-400 max-h-48 overflow-y-auto">
              {[...new Map(logs.map(log => [log.timestamp + log.message, log])).values()].map((log, index) => (
                <div key={log.timestamp + log.message + index} className="flex gap-2">
                  <span className="text-gray-500">[{log.timestamp}]</span>
                  <span>{log.message}</span>
                </div>
              ))}
            </div>
            {/* Collapsible Advanced Section */}
            <div className="mt-6">
              <button
                className="w-full text-left px-3 py-2 rounded bg-gray-700 hover:bg-gray-600 text-sm font-semibold text-gray-300 focus:outline-none"
                onClick={() => setShowAdvanced(v => !v)}
              >
                {showAdvanced ? '▼' : '►'} Advanced
              </button>
              {showAdvanced && (
                <div className="mt-3 space-y-4">
                  {appConfig && (
                    <div className="p-3 rounded-lg bg-gray-700 text-xs text-gray-400 space-y-1">
                      <div><span className="text-gray-500">Videos:</span> {appConfig.video_dir}</div>
                      <div><span className="text-gray-500">Output:</span> {appConfig.output_dir}</div>
                    </div>
                  )}
                  {/* Sampling rate */}
                  <div className="p-3 rounded-lg bg-gray-700">
                    <div className="mb-3">
                      <span className="text-sm font-medium">Sampling</span>
                    </div>
                    <div className="space-y-1">
                      <Tooltip text="Analyze every Nth frame. Lower = catches shorter reactions but slower. Higher = faster but may miss split-second expressions.">
                        <span className="text-xs text-gray-400">Analyze every {skipInterval} frame{skipInterval === 1 ? '' : 's'}</span>
                      </Tooltip>
                      <Slider.Root
                        className="relative flex items-center select-none touch-none h-4"
                        value={[skipInterval]}
                        min={1}
                        max={10}
                        step={1}
                        onValueChange={([value]) => setSkipInterval(value)}
                      >
                        <Slider.Track className="bg-gray-600 relative grow rounded-full h-1">
                          <Slider.Range className="absolute bg-blue-500 rounded-full h-full" />
                        </Slider.Track>
                        <Slider.Thumb className="block w-3 h-3 bg-white border border-blue-500 rounded-full shadow focus:outline-none" />
                      </Slider.Root>
                    </div>
                  </div>
                  
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default EmotionDetectorApp