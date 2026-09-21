import { useEffect, useRef, useState } from 'react'
import { wsEventsUrl } from '../api/client'

/**
 * Keep a WebSocket to the backend open, reconnecting when it drops.
 * `onMessage` can change on every render; the latest one is always used.
 */
export function useEvents(onMessage) {
  const handlerRef = useRef(onMessage)
  useEffect(() => {
    handlerRef.current = onMessage
  })

  const [connected, setConnected] = useState(false)

  useEffect(() => {
    let socket = null
    let reconnectTimer = null
    let pingTimer = null
    let closed = false

    const connect = () => {
      socket = new WebSocket(wsEventsUrl())
      socket.onopen = () => {
        setConnected(true)
        pingTimer = setInterval(() => {
          if (socket?.readyState === WebSocket.OPEN) socket.send('ping')
        }, 15000)
      }
      socket.onmessage = (event) => {
        try {
          handlerRef.current(JSON.parse(event.data))
        } catch (err) {
          console.error('Bad event payload', err)
        }
      }
      socket.onclose = () => {
        setConnected(false)
        clearInterval(pingTimer)
        if (!closed) reconnectTimer = setTimeout(connect, 2000)
      }
      socket.onerror = () => socket.close()
    }

    connect()
    return () => {
      closed = true
      clearTimeout(reconnectTimer)
      clearInterval(pingTimer)
      socket?.close()
    }
  }, [])

  return connected
}
