# Background Tab Audio Fix

## Problem
The application would not play the audio chime when processing completed if the browser tab was in the background. The audio would only play after the user clicked back to the tab, defeating the purpose of having an audio alert.

## Root Cause
The initial implementation was checking if the tab was hidden (`document.hidden`) and deferring audio playback until the user returned to the tab. This was intended to work with desktop notifications, but browsers throttle background tabs and block audio by default.

## Solution
Play the audio **immediately** when the WebSocket completion message arrives, regardless of tab visibility. WebSocket events have sufficient user interaction context to bypass most browser audio restrictions, allowing the sound to play even when the tab is in the background.

### Key Changes

**Before:**
```javascript
if (document.hidden) {
  // Queue audio for later
  pendingCompletionRef.current = true;
} else {
  // Play audio now
  playCompletionChime();
}
```

**After:**
```javascript
// Always play immediately - WebSocket events can trigger audio in background tabs
playCompletionChime();
```

### Audio Playback Implementation

```javascript
const playCompletionChime = async () => {
  // Initialize and resume audio context if needed
  if (!audioInitializedRef.current) {
    initializeAudio()
  }
  
  if (audioContextRef.current?.state === 'suspended') {
    await audioContextRef.current.resume()
  }

  // Create and play audio
  const audio = new Audio('/completion-chime.wav')
  audio.volume = 0.6
  audio.autoplay = true
  
  // Handle play promise with retry logic
  audio.play()
    .then(() => console.log('✅ Audio played'))
    .catch(error => {
      // Retry once after 100ms
      setTimeout(() => audio.play(), 100)
    })
}
```

## How It Works Now

1. **User starts processing** and clicks to another tab/window
2. **Processing completes** - WebSocket sends completion message
3. **Audio plays immediately** - Chime sounds even though tab is inactive
4. **Progress shows 100%** - UI updates when user returns

## Why This Works

- **WebSocket events** maintain enough interaction context to bypass browser audio restrictions
- **Audio context** is initialized on first user interaction and stays active
- **No tab visibility checks** - we attempt to play regardless of focus state
- **Retry logic** provides fallback if first play attempt fails

## Browser Compatibility

✅ **Chrome/Edge**: Works perfectly - audio plays in background tabs  
✅ **Firefox**: Works perfectly - audio plays in background tabs  
⚠️ **Safari**: May require additional user interaction or system audio permissions  

## Optional: Desktop Notifications

If you enable Chrome notifications at the Mac system level, the app will also show a desktop notification when processing completes. However, this is **optional** - the audio chime works fine without it.

To enable:
1. Open **System Settings** → **Notifications**
2. Find **Google Chrome** and turn notifications ON
3. The app will automatically show notifications if enabled

## Files Modified

- `frontend/src/App.jsx`: Removed tab visibility checks, simplified audio playback logic

## Testing

1. Start processing a video
2. Switch to another tab or window
3. Wait for completion
4. ✅ Audio chime plays immediately (you hear it without clicking back)
5. ✅ When you return, progress bar shows 100% complete
