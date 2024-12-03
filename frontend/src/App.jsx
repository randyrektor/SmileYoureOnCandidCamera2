import React, { useState, useRef, useEffect } from 'react';
import { Terminal, Settings, Play, Pause, FolderOpen } from 'lucide-react';

const SmileDetectorApp = () => {
  const [isProcessing, setIsProcessing] = useState(false);
  const [debug, setDebug] = useState(false);
  const [skipFrames, setSkipFrames] = useState(8);
  const [smileSensitivity, setSmileSensitivity] = useState(62);
  const [previewImage, setPreviewImage] = useState(null);
  const [logs, setLogs] = useState([]);
  const [roiPosition, setRoiPosition] = useState({
    top: 20,
    bottom: 65,
    left: 25,
    right: 75
  });
  
  const canvasRef = useRef(null);
  const logsEndRef = useRef(null);

  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

  const addLog = (message) => {
    setLogs(prev => [...prev, {
      timestamp: new Date().toLocaleTimeString(),
      message
    }]);
  };

  const handleGrabNewFrame = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/preview-frame');
      if (!response.ok) {
        throw new Error('Failed to fetch preview frame');
      }
      const data = await response.json();
      const img = new Image();
      img.onload = () => {
        setPreviewImage(img);
        drawPreviewWithOverlay(img);
      };
      img.src = `data:image/jpeg;base64,${data.frame}`;
    } catch (error) {
      addLog(`Error: ${error.message}`);
    }
  };

  const drawPreviewWithOverlay = (image) => {
    if (canvasRef.current && image) {
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
      
      const x = (canvas.width * roiPosition.left) / 100;
      const y = (canvas.height * roiPosition.top) / 100;
      const width = (canvas.width * (roiPosition.right - roiPosition.left)) / 100;
      const height = (canvas.height * (roiPosition.bottom - roiPosition.top)) / 100;
      
      ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
      ctx.fillRect(0, 0, canvas.width, y);
      ctx.fillRect(0, y + height, canvas.width, canvas.height - (y + height));
      ctx.fillRect(0, y, x, height);
      ctx.fillRect(x + width, y, canvas.width - (x + width), height);
      
      ctx.strokeStyle = '#00ff00';
      ctx.lineWidth = 2;
      ctx.strokeRect(x, y, width, height);
    }
  };

  useEffect(() => {
    if (previewImage) {
      drawPreviewWithOverlay(previewImage);
    }
  }, [roiPosition, previewImage]);
  
  const handleStartProcessing = async () => {
    try {
      if (isProcessing) {
        addLog('Stopping processing...');
        const response = await fetch('http://localhost:8000/api/stop-processing', {
          method: 'POST'
        });
        
        if (!response.ok) {
          throw new Error('Failed to stop processing');
        }
        
        const data = await response.json();
        console.log('Stop response:', data);
        setIsProcessing(false);
        addLog('Processing stopped');
        
      } else {
        setIsProcessing(true);
        const params = {
          debug,
          skipFrames,
          smileSensitivity,
          roiPosition
        };
        
        addLog('Starting processing...');
        const response = await fetch('http://localhost:8000/api/start-processing', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(params),
        });
        
        if (!response.ok) {
          throw new Error('Failed to start processing');
        }
   
        const result = await response.json();
        addLog(result.message);
        setIsProcessing(false);
      }
    } catch (error) {
      console.error('Error:', error);
      addLog(`Error: ${error.message}`);
      setIsProcessing(false);
    }
  };
  
  return (
    <div className="min-h-screen bg-gray-900 text-gray-100 p-6">
      <div className="max-w-6xl mx-auto space-y-6">
        <div className="flex items-center justify-between">
          <h1 className="text-2xl font-bold">Smile Detector Control Panel</h1>
          <div className="space-x-4">
            <button
              onClick={handleStartProcessing}
              className={`px-4 py-2 rounded-lg font-medium ${
                isProcessing 
                  ? 'bg-red-600 hover:bg-red-700' 
                  : 'bg-blue-600 hover:bg-blue-700'
              }`}
            >
              {isProcessing ? 'Stop' : 'Start'}
            </button>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 bg-gray-800 rounded-lg p-4">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-xl font-semibold">Preview</h2>
              <button
                onClick={handleGrabNewFrame}
                className="px-4 py-2 bg-gray-700 rounded-lg hover:bg-gray-600"
              >
                Grab New Frame
              </button>
            </div>
            <div className="relative aspect-video bg-black rounded-lg overflow-hidden">
              <canvas
                ref={canvasRef}
                className="w-full h-full"
                width={1280}
                height={720}
              />
            </div>
          </div>

          <div className="bg-gray-800 rounded-lg p-4">
            <h2 className="text-xl font-semibold mb-4">Controls</h2>
            <div className="space-y-6">
              <div className="flex items-center justify-between">
                <label>Debug Mode</label>
                <button
                  onClick={() => setDebug(!debug)}
                  className={`w-12 h-6 rounded-full transition-colors ${
                    debug ? 'bg-blue-600' : 'bg-gray-600'
                  }`}
                >
                  <div className={`w-4 h-4 bg-white rounded-full transition-transform transform mx-1 ${
                    debug ? 'translate-x-6' : ''
                  }`} />
                </button>
              </div>

              <div className="space-y-2">
                <label>Processing Speed (Skip Frames)</label>
                <input
                  type="range"
                  min="1"
                  max="15"
                  value={skipFrames}
                  onChange={(e) => setSkipFrames(Number(e.target.value))}
                  className="w-full"
                />
                <div className="text-sm text-gray-400">
                  Current: {skipFrames} frames (Higher = Faster)
                </div>
              </div>

              <div className="space-y-2">
                <label>Smile Sensitivity</label>
                <input
                  type="range"
                  min="40"
                  max="80"
                  value={smileSensitivity}
                  onChange={(e) => setSmileSensitivity(Number(e.target.value))}
                  className="w-full"
                />
                <div className="text-sm text-gray-400">
                  Higher = Larger Smile Required
                </div>
              </div>

              <div className="space-y-4">
                <h3 className="font-medium">Detection Region</h3>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="text-sm text-gray-400">Top (%)</label>
                    <input
                      type="range"
                      min="0"
                      max="40"
                      value={roiPosition.top}
                      onChange={(e) => setRoiPosition(prev => ({...prev, top: Number(e.target.value)}))}
                      className="w-full"
                    />
                  </div>
                  <div>
                    <label className="text-sm text-gray-400">Bottom (%)</label>
                    <input
                      type="range"
                      min="60"
                      max="100"
                      value={roiPosition.bottom}
                      onChange={(e) => setRoiPosition(prev => ({...prev, bottom: Number(e.target.value)}))}
                      className="w-full"
                    />
                  </div>
                  <div>
                    <label className="text-sm text-gray-400">Left (%)</label>
                    <input
                      type="range"
                      min="0"
                      max="40"
                      value={roiPosition.left}
                      onChange={(e) => setRoiPosition(prev => ({...prev, left: Number(e.target.value)}))}
                      className="w-full"
                    />
                  </div>
                  <div>
                    <label className="text-sm text-gray-400">Right (%)</label>
                    <input
                      type="range"
                      min="60"
                      max="100"
                      value={roiPosition.right}
                      onChange={(e) => setRoiPosition(prev => ({...prev, right: Number(e.target.value)}))}
                      className="w-full"
                    />
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="lg:col-span-3 bg-gray-800 rounded-lg p-4">
            <h2 className="text-xl font-semibold mb-4">Processing Logs</h2>
            <div className="bg-gray-950 rounded-lg p-4 h-[200px] overflow-y-auto font-mono text-sm">
              {logs.map((log, i) => (
                <div key={i} className="text-gray-300">
                  <span className="text-gray-500">[{log.timestamp}]</span> {log.message}
                </div>
              ))}
              <div ref={logsEndRef} />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SmileDetectorApp;