import { useState, useRef, useEffect } from 'react'
import { FolderOpen } from 'lucide-react'
import * as Slider from '@radix-ui/react-slider'
import { Tooltip } from "./components/Tooltip"
import OptimizedProcessingTime from './optimizedProcessingTime'
import { useState as useToggleState } from 'react'

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
  const [wsConnected, setWsConnected] = useState(false)
  const [emotionSensitivity, setEmotionSensitivity] = useState(4)

  const [targetEmotions, setTargetEmotions] = useState(['happy', 'surprise', 'angry', 'sad', 'fear', 'disgust', 'neutral'])
  const [roiPosition, setRoiPosition] = useState({
    top: 20,
    bottom: 65,
    left: 25,
    right: 75
  })
  const [logs, setLogs] = useState([])
  const [isComplete, setIsComplete] = useState(false)
  const [dragHandle, setDragHandle] = useState(null)
  const [estimatedTime, setEstimatedTime] = useState('N/A')
  const [modelStatus, setModelStatus] = useState(null)
  const [memoryPoolStats, setMemoryPoolStats] = useState(null)
  const [memoryPoolPollingInterval, setMemoryPoolPollingInterval] = useState(null)
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [notificationBannerVisible, setNotificationBannerVisible] = useState(true)
  // Best-frame approach settings (default processing method)
  const [useBestFrameApproach, setUseBestFrameApproach] = useState(true)
  const [skipInterval, setSkipInterval] = useState(25)
  const [searchForward, setSearchForward] = useState(5)
  const [searchBackward, setSearchBackward] = useState(5)

  // Audio context for completion notification
  const audioContextRef = useRef(null)
  const audioInitializedRef = useRef(false)
  const notificationPermissionRequestedRef = useRef(false)
  const pendingCompletionRef = useRef(false)

  // Initialize audio context on first user interaction
  const initializeAudio = () => {
    if (audioInitializedRef.current) return
    
    try {
      // Create and initialize audio context
      audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)()
      audioInitializedRef.current = true
      console.log('Audio context initialized')
    } catch (error) {
      console.warn('Could not initialize audio context:', error)
    }
  }

  // Request notification permission on first user interaction
  const requestNotificationPermission = async () => {
    if (notificationPermissionRequestedRef.current) return
    
    if ('Notification' in window && Notification.permission === 'default') {
      try {
        const permission = await Notification.requestPermission()
        console.log('Notification permission:', permission)
        notificationPermissionRequestedRef.current = true
      } catch (error) {
        console.warn('Could not request notification permission:', error)
      }
    }
  }

  // Function to play completion notification sound
  const playCompletionChime = async () => {
    try {
      // Ensure audio context is initialized
      if (!audioInitializedRef.current) {
        initializeAudio()
      }

      // Resume audio context if suspended (required for background tabs)
      if (audioContextRef.current && audioContextRef.current.state === 'suspended') {
        await audioContextRef.current.resume()
      }

      // Create audio element and play the completion sound
      const audio = new Audio('/completion-chime.wav')
      audio.volume = 0.5 // Set volume to 50%
      
      const playPromise = audio.play()
      if (playPromise !== undefined) {
        await playPromise
        console.log('Completion chime played successfully')
      }
    } catch (error) {
      console.warn('Could not play completion chime:', error)
    }
  }

  // Browser notification (primary alert for background tabs)
  const showCompletionNotification = () => {
    if ('Notification' in window && Notification.permission === 'granted') {
      try {
        const notification = new Notification('Processing Complete! 🎉', {
          body: 'Your video processing has finished successfully.',
          icon: '/vite.svg',
          badge: '/vite.svg',
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

  const getHandleAtPosition = (x, y) => {
    const canvas = canvasRef.current;
    if (!canvas) return null;
    
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    
    // Convert to canvas coordinates
    const canvasX = (x - rect.left) * scaleX;
    const canvasY = (y - rect.top) * scaleY;

    // Check each handle
    const handlePositions = [
      {
        id: 'left',
        x: canvas.width * (roiPosition.left / 100),
        y: canvas.height * ((roiPosition.top + roiPosition.bottom) / 200)
      },
      {
        id: 'right',
        x: canvas.width * (roiPosition.right / 100),
        y: canvas.height * ((roiPosition.top + roiPosition.bottom) / 200)
      },
      {
        id: 'top',
        x: canvas.width * ((roiPosition.left + roiPosition.right) / 200),
        y: canvas.height * (roiPosition.top / 100)
      },
      {
        id: 'bottom',
        x: canvas.width * ((roiPosition.left + roiPosition.right) / 200),
        y: canvas.height * (roiPosition.bottom / 100)
      }
    ];

    // Find handle within 30px radius of click
    return handlePositions.find(handle => 
      Math.hypot(canvasX - handle.x, canvasY - handle.y) < 30
    )?.id;
  };

  // Initialize audio and request notification permission on first user interaction
  useEffect(() => {
    const handleUserInteraction = () => {
      initializeAudio()
      requestNotificationPermission()
      // Remove event listeners after first interaction
      document.removeEventListener('click', handleUserInteraction)
      document.removeEventListener('keydown', handleUserInteraction)
      document.removeEventListener('touchstart', handleUserInteraction)
    }

    // Add event listeners for user interaction
    document.addEventListener('click', handleUserInteraction)
    document.addEventListener('keydown', handleUserInteraction)
    document.addEventListener('touchstart', handleUserInteraction)

    return () => {
      document.removeEventListener('click', handleUserInteraction)
      document.removeEventListener('keydown', handleUserInteraction)
      document.removeEventListener('touchstart', handleUserInteraction)
    }
  }, [])

  // Handle completion when tab becomes visible (for background processing)
  useEffect(() => {
    const handleVisibilityChange = () => {
      // When tab becomes visible, check if there's a pending completion
      if (!document.hidden && pendingCompletionRef.current) {
        console.log('Tab became visible, triggering pending completion')
        playCompletionChime()
        pendingCompletionRef.current = false
      }
    }

    document.addEventListener('visibilitychange', handleVisibilityChange)

    return () => {
      document.removeEventListener('visibilitychange', handleVisibilityChange)
    }
  }, [])

  useEffect(() => {
    const fetchAndSelectFirstVideo = async () => {
      try {
        // Fetch model status and loading progress
        try {
          const modelResponse = await fetch('http://localhost:8000/api/model-status')
          if (modelResponse.ok) {
            const modelData = await modelResponse.json()
            setModelStatus(modelData)
            console.log("Model status:", modelData)
          }
        } catch (error) {
          console.warn('Could not fetch model status:', error)
        }

        // Fetch memory pool statistics and performance data
        try {
          const memoryResponse = await fetch('http://localhost:8000/api/memory-pool-stats')
          if (memoryResponse.ok) {
            const memoryData = await memoryResponse.json()
            setMemoryPoolStats(memoryData.data)
            console.log("Memory pool stats:", memoryData.data)
          }
        } catch (error) {
          console.warn('Could not fetch memory pool stats:', error)
        }

        const response = await fetch('http://localhost:8000/api/available-videos')
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
          
          // Fetch preview frame for the selected video
          const previewResponse = await fetch(
            `http://localhost:8000/api/preview-frame?video_path=${encodeURIComponent(firstVideo.path)}`
          )
          if (!previewResponse.ok) throw new Error('Failed to fetch preview frame')
          
          const previewData = await previewResponse.json()
          const img = new Image()
          img.onload = () => {
            setPreviewImage(img)
            drawPreviewWithOverlay(img)
          }
          img.src = `data:image/jpeg;base64,${previewData.frame}`
        }
      } catch (error) {
        console.error('Error fetching videos:', error)
      }
    }

    fetchAndSelectFirstVideo()
    
    // Cleanup function to stop memory pool polling when component unmounts
    return () => {
      stopMemoryPoolPolling()
    }
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
      wsInstance = new window.WebSocket('ws://localhost:8000/ws/logs');
      ws.current = wsInstance;

      wsInstance.onopen = () => {
        if (isUnmounted) return;
        setWsConnected(true);
        console.log('WebSocket connected successfully');
      };
      wsInstance.onclose = (event) => {
        setWsConnected(false);
        ws.current = null;
        wsInstance = null;
        if (!isUnmounted) {
          reconnectTimeout = setTimeout(connectWebSocket, 3000);
        }
      };
      wsInstance.onerror = (error) => {
        setWsConnected(false);
        console.error('WebSocket error:', error);
      };
      wsInstance.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === "progress") {
            setProgress(data.progress);
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
            setProgress(100);  // Force progress to 100%
            setProcessingStats(null);
            setIsComplete(true);
            
            // Show browser notification FIRST (works even when tab is inactive)
            showCompletionNotification();
            
            // Play audio chime
            if (document.hidden) {
              // If tab is not visible, mark completion as pending
              // Audio will play when user switches back to tab
              pendingCompletionRef.current = true;
              console.log('Tab is hidden, marked completion as pending')
            } else {
              // Tab is visible, play audio immediately
              playCompletionChime();
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
    
    try {
      const response = await fetch(
        `http://localhost:8000/api/preview-frame?video_path=${encodeURIComponent(video.path)}`
      )
      if (!response.ok) throw new Error('Failed to fetch preview frame')
      
      const data = await response.json()
      const img = new Image()
      img.onload = () => {
        setPreviewImage(img)
        drawPreviewWithOverlay(img)
      }
      img.src = `data:image/jpeg;base64,${data.frame}`
    } catch (error) {
      console.error('Error grabbing frame:', error)
    }
  }

  const drawPreviewWithOverlay = (img) => {
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    
    // Draw image
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height)
    
    // Draw ROI overlay with rounded corners
    const cornerRadius = 12;  // Adjust this value to change the roundness
    
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

    // Draw handles
    const handlePositions = [
      // Left handle
      {
        x: canvas.width * (roiPosition.left / 100),
        y: canvas.height * ((roiPosition.top + roiPosition.bottom) / 200)
      },
      // Right handle
      {
        x: canvas.width * (roiPosition.right / 100),
        y: canvas.height * ((roiPosition.top + roiPosition.bottom) / 200)
      },
      // Top handle
      {
        x: canvas.width * ((roiPosition.left + roiPosition.right) / 200),
        y: canvas.height * (roiPosition.top / 100)
      },
      // Bottom handle
      {
        x: canvas.width * ((roiPosition.left + roiPosition.right) / 200),
        y: canvas.height * (roiPosition.bottom / 100)
      }
    ];

    // Draw white circles with shadow
    ctx.shadowColor = 'rgba(0, 0, 0, 0.5)';
    ctx.shadowBlur = 4;
    ctx.fillStyle = 'white';
    ctx.lineWidth = 2;
    
    handlePositions.forEach(pos => {
      ctx.beginPath();
      ctx.arc(pos.x, pos.y, 12, 0, Math.PI * 2);
      ctx.fill();
    });

    // Reset shadow
    ctx.shadowColor = 'transparent';
    ctx.shadowBlur = 0;
  }

  const startProcessing = async () => {
    if (!selectedVideo) return
    
    try {
      setIsProcessing(true)
      setIsComplete(false)
      setProgress(0)
      setProcessingStats(null)
      // Start polling memory pool statistics during processing
      startMemoryPoolPolling()
      
      const response = await fetch('http://localhost:8000/api/start-processing', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          debug: false,
          emotionSensitivity: emotionSensitivity,
          roiPosition: roiPosition,
          videoPath: selectedVideo.path,
          targetEmotions: targetEmotions,
          // Best-frame approach settings
          useBestFrameApproach: useBestFrameApproach,
          skipInterval: skipInterval,
          searchForward: searchForward,
          searchBackward: searchBackward,
        })
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
      // Stop polling on error
      stopMemoryPoolPolling()
    }
  }

  const stopProcessing = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/stop-processing', {
        method: 'POST'
      })
      
      if (response.ok) {
        setIsProcessing(false)
        setIsComplete(true)
        setProcessingStats(null)
        // Stop polling when processing ends
        stopMemoryPoolPolling()
      }
    } catch (error) {
      console.error('Error stopping processing:', error)
    }
  }

  const startMemoryPoolPolling = () => {
    // Poll memory pool stats every 2 seconds during processing
    const interval = setInterval(async () => {
      try {
        const response = await fetch('http://localhost:8000/api/memory-pool-stats')
        if (response.ok) {
          const data = await response.json()
          setMemoryPoolStats(data.data)
        }
      } catch (error) {
        console.warn('Could not fetch memory pool stats:', error)
      }
    }, 2000)
    
    setMemoryPoolPollingInterval(interval)
  }

  const stopMemoryPoolPolling = () => {
    if (memoryPoolPollingInterval) {
      clearInterval(memoryPoolPollingInterval)
      setMemoryPoolPollingInterval(null)
    }
  }

  const refreshVideos = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/available-videos')
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

  const calculateEstimatedTime = () => {
    if (!selectedVideo) return 'N/A'
    try {
      const optimizer = new OptimizedProcessingTime()
      const result = optimizer.calculateOptimizedTime(roiPosition, selectedVideo.duration)
      return result.displayTime
    } catch (error) {
      console.error('Error calculating estimated time:', error)
      return 'N/A'
    }
  }

  useEffect(() => {
    if (previewImage) {
      drawPreviewWithOverlay(previewImage)
    }
  }, [roiPosition, previewImage])

  useEffect(() => {
    return () => {
      // Cleanup on unmount
      fetch('http://localhost:8000/api/stop-processing', { 
        method: 'POST' 
      }).catch(error => {
        console.error('Error cleaning up:', error);
      });
    };
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const handleMouseDown = (e) => {
      const handle = getHandleAtPosition(e.clientX, e.clientY);
      if (handle) {
        setDragHandle(handle);
        e.preventDefault(); // Prevent text selection while dragging
      }
    };

    const handleMouseMove = (e) => {
      if (!dragHandle || !previewImage) return;

      const rect = canvas.getBoundingClientRect();
      const scaleX = canvas.width / rect.width;
      const scaleY = canvas.height / rect.height;
      
      // Convert mouse position to percentage
      const x = ((e.clientX - rect.left) * scaleX / canvas.width) * 100;
      const y = ((e.clientY - rect.top) * scaleY / canvas.height) * 100;

      // Update ROI position based on which handle is being dragged
      setRoiPosition(prev => {
        const updated = { ...prev };
        const minSize = 10; // Minimum 10% size

        switch (dragHandle) {
          case 'left':
            updated.left = Math.min(Math.max(0, x), prev.right - minSize);
            break;
          case 'right':
            updated.right = Math.max(Math.min(100, x), prev.left + minSize);
            break;
          case 'top':
            updated.top = Math.min(Math.max(0, y), prev.bottom - minSize);
            break;
          case 'bottom':
            updated.bottom = Math.max(Math.min(100, y), prev.top + minSize);
            break;
        }
        return updated;
      });
      setTimeout(() => setEstimatedTime(calculateEstimatedTime()), 0);
    };

    const handleMouseUp = () => {
      setDragHandle(null);
    };

    // Add event listeners
    canvas.addEventListener('mousedown', handleMouseDown);
    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);

    // Cleanup
    return () => {
      canvas.removeEventListener('mousedown', handleMouseDown);
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, [dragHandle, previewImage, roiPosition]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const updateCursor = (e) => {
      const handle = getHandleAtPosition(e.clientX, e.clientY);
      if (handle) {
        switch (handle) {
          case 'left':
          case 'right':
            canvas.style.cursor = 'ew-resize';
            break;
          case 'top':
          case 'bottom':
            canvas.style.cursor = 'ns-resize';
            break;
        }
      } else {
        canvas.style.cursor = 'default';
      }
    };

    canvas.addEventListener('mousemove', updateCursor);
    return () => canvas.removeEventListener('mousemove', updateCursor);
  }, [roiPosition]);

  useEffect(() => {
    setEstimatedTime(calculateEstimatedTime())
  }, [emotionSensitivity, roiPosition, selectedVideo])

  return (
    <div className="min-h-screen bg-gray-900">
      {/* Notification Permission Banner */}
      {notificationBannerVisible && 'Notification' in window && Notification.permission !== 'granted' && (
        <div className="bg-blue-600 text-white px-4 py-3 relative">
          <div className="container mx-auto flex items-center justify-between">
            <div className="flex items-center gap-2">
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9" />
              </svg>
              <span className="text-sm font-medium">
                Enable notifications to get alerts when processing completes, even when this tab is in the background
              </span>
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={async () => {
                  await requestNotificationPermission()
                  setNotificationBannerVisible(false)
                }}
                className="px-3 py-1 bg-white text-blue-600 rounded font-medium text-sm hover:bg-blue-50"
              >
                Enable
              </button>
              <button
                onClick={() => setNotificationBannerVisible(false)}
                className="px-2 py-1 text-white hover:text-blue-100"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
          </div>
        </div>
      )}
      
      <div className="container mx-auto px-4 py-8">
        <h1 className="text-3xl font-bold mb-8">Thumbnail Faces</h1>
        
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
                  className={`p-2 rounded-lg text-left flex justify-between items-center ${
                    selectedVideo?.path === video.path
                      ? 'bg-blue-600'
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

                {/* Emotion Accuracy SECOND */}
                <div className="space-y-2">
                  <Tooltip text="Higher = Only save faces with more confident emotion predictions. Lower = Save more faces, but may include uncertain predictions.">
                    <span className="text-sm">Emotion Accuracy: {emotionSensitivity}/5 (above {50 + (emotionSensitivity - 1) * 10}%)</span>
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
                </div>

                <div className="text-sm text-gray-400 mb-4">
                  Estimated processing time: {estimatedTime}
                </div>

                <div>
                  <button
                    onClick={isProcessing ? stopProcessing : startProcessing}
                    disabled={!selectedVideo || targetEmotions.length === 0}
                    className={`w-full px-6 py-2 rounded-lg font-medium ${
                      !selectedVideo || targetEmotions.length === 0
                        ? 'bg-gray-700 cursor-not-allowed'
                        : isProcessing
                        ? 'bg-red-600 hover:bg-red-500'
                        : 'bg-blue-600 hover:bg-blue-500'
                    }`}
                  >
                    {isProcessing ? 'Stop Processing' : isComplete ? 'Start Processing' : 'Start Processing'}
                  </button>
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
                  <div className="text-green-400 font-semibold">Task complete!</div>
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
                  {/* Best-Frame Approach Settings */}
                  <div className="p-3 rounded-lg bg-gray-700">
                    <div className="flex items-center justify-between mb-3">
                      <span className="text-sm font-medium">Best-Frame Approach (Recommended)</span>
                      <label className="flex items-center space-x-2">
                        <input
                          type="checkbox"
                          checked={useBestFrameApproach}
                          onChange={(e) => setUseBestFrameApproach(e.target.checked)}
                          className="rounded border-gray-600 bg-gray-700 text-blue-600 focus:ring-blue-500"
                        />
                        <span className="text-xs text-gray-400">Enable</span>
                      </label>
                    </div>
                    {useBestFrameApproach && (
                      <div className="space-y-3">
                        <div className="space-y-1">
                          <Tooltip text="Skip every N frames to check for emotions">
                            <span className="text-xs text-gray-400">Skip Interval: {skipInterval} frames</span>
                          </Tooltip>
                          <Slider.Root
                            className="relative flex items-center select-none touch-none h-4"
                            value={[skipInterval]}
                            min={10}
                            max={50}
                            step={5}
                            onValueChange={([value]) => setSkipInterval(value)}
                          >
                            <Slider.Track className="bg-gray-600 relative grow rounded-full h-1">
                              <Slider.Range className="absolute bg-blue-500 rounded-full h-full" />
                            </Slider.Track>
                            <Slider.Thumb className="block w-3 h-3 bg-white border border-blue-500 rounded-full shadow focus:outline-none" />
                          </Slider.Root>
                        </div>
                        <div className="space-y-1">
                          <Tooltip text="Search N frames forward for better emotion score">
                            <span className="text-xs text-gray-400">Search Forward: {searchForward} frames</span>
                          </Tooltip>
                          <Slider.Root
                            className="relative flex items-center select-none touch-none h-4"
                            value={[searchForward]}
                            min={2}
                            max={10}
                            step={1}
                            onValueChange={([value]) => setSearchForward(value)}
                          >
                            <Slider.Track className="bg-gray-600 relative grow rounded-full h-1">
                              <Slider.Range className="absolute bg-blue-500 rounded-full h-full" />
                            </Slider.Track>
                            <Slider.Thumb className="block w-3 h-3 bg-white border border-blue-500 rounded-full shadow focus:outline-none" />
                          </Slider.Root>
                        </div>
                        <div className="space-y-1">
                          <Tooltip text="Search N frames backward for better emotion score">
                            <span className="text-xs text-gray-400">Search Backward: {searchBackward} frames</span>
                          </Tooltip>
                          <Slider.Root
                            className="relative flex items-center select-none touch-none h-4"
                            value={[searchBackward]}
                            min={2}
                            max={10}
                            step={1}
                            onValueChange={([value]) => setSearchBackward(value)}
                          >
                            <Slider.Track className="bg-gray-600 relative grow rounded-full h-1">
                              <Slider.Range className="absolute bg-blue-500 rounded-full h-full" />
                            </Slider.Track>
                            <Slider.Thumb className="block w-3 h-3 bg-white border border-blue-500 rounded-full shadow focus:outline-none" />
                          </Slider.Root>
                        </div>
                      </div>
                    )}
                  </div>
                  
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {/* Model Status Indicator */}
                    {modelStatus && (
                      <div className="p-3 rounded-lg bg-gray-700 h-full">
                        <div className="flex items-center justify-between">
                          <span className="text-sm font-medium">ML Models</span>
                          <div className="flex items-center gap-2">
                            <div className={`w-2 h-2 rounded-full ${modelStatus.models_loaded ? 'bg-green-500' : 'bg-red-500'}`}></div>
                            <span className="text-xs text-gray-400">
                              {modelStatus.models_loaded ? 'Ready' : 'Loading...'}
                            </span>
                          </div>
                        </div>
                        {modelStatus.device && (
                          <div className="text-xs text-gray-500 mt-1">
                            Device: {modelStatus.device}
                            {modelStatus.cuda_available && ` (${modelStatus.cuda_device_count} GPU${modelStatus.cuda_device_count > 1 ? 's' : ''})`}
                          </div>
                        )}
                      </div>
                    )}
                    {/* Memory Pool Stats */}
                    {memoryPoolStats && (
                      <div className="p-3 rounded-lg bg-gray-700 h-full">
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-sm font-medium">Memory Pool</span>
                          <div className="flex items-center gap-2">
                            <div className={`w-2 h-2 rounded-full ${memoryPoolStats.hit_rate && memoryPoolStats.hit_rate > 0.5 ? 'bg-green-500' : 'bg-yellow-500'}`}></div>
                            <span className="text-xs text-gray-400">
                              {memoryPoolStats.hit_rate ? Math.round(memoryPoolStats.hit_rate * 100) : 0}% hit rate
                            </span>
                          </div>
                        </div>
                        <div className="grid grid-cols-2 gap-2 text-xs text-gray-500">
                          <div>Allocations: {memoryPoolStats.allocations || 0}</div>
                          <div>Reuses: {memoryPoolStats.reuses || 0}</div>
                          <div>Image Buffers: {memoryPoolStats.total_image_buffers || 0}</div>
                          <div>Tensor Buffers: {memoryPoolStats.total_tensor_buffers || 0}</div>
                        </div>
                      </div>
                    )}
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
