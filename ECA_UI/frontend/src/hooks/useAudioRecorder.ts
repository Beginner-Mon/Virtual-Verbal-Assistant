import { useCallback, useEffect, useRef, useState } from 'react'

function pickMime(): string {
  if (typeof MediaRecorder === 'undefined') return ''
  if (MediaRecorder.isTypeSupported('audio/webm;codecs=opus')) return 'audio/webm;codecs=opus'
  if (MediaRecorder.isTypeSupported('audio/webm')) return 'audio/webm'
  if (MediaRecorder.isTypeSupported('audio/mp4')) return 'audio/mp4'
  return ''
}

export function useAudioRecorder() {
  const [isRecording, setIsRecording] = useState(false)
  const [duration, setDuration] = useState(0)
  const [audioUrl, setAudioUrl] = useState<string | null>(null)
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null)
  const [error, setError] = useState<string | null>(null)

  const recorderRef = useRef<MediaRecorder | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const chunksRef = useRef<Blob[]>([])
  const timerRef = useRef<number | null>(null)
  const startTimeRef = useRef<number>(0)

  const clearPreview = useCallback(() => {
    if (audioUrl) URL.revokeObjectURL(audioUrl)
    setAudioUrl(null)
    setAudioBlob(null)
  }, [audioUrl])

  useEffect(() => {
    return () => {
      if (audioUrl) URL.revokeObjectURL(audioUrl)
      if (timerRef.current) window.clearInterval(timerRef.current)
      streamRef.current?.getTracks().forEach((t) => t.stop())
    }
  }, [audioUrl])

  const start = useCallback(async () => {
    setError(null)
    clearPreview()
    chunksRef.current = []

    if (typeof navigator === 'undefined' || !navigator.mediaDevices?.getUserMedia) {
      setError('Microphone not supported in this browser')
      return
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      streamRef.current = stream

      const mime = pickMime()
      const recorder = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined)
      recorderRef.current = recorder

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data)
      }

      recorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: recorder.mimeType || 'audio/webm' })
        const url = URL.createObjectURL(blob)
        setAudioBlob(blob)
        setAudioUrl(url)
        stream.getTracks().forEach((t) => t.stop())
        streamRef.current = null
      }

      recorder.onerror = () => setError('Recording failed')

      recorder.start(100)
      setIsRecording(true)
      startTimeRef.current = Date.now()
      setDuration(0)
      timerRef.current = window.setInterval(() => {
        setDuration(Math.floor((Date.now() - startTimeRef.current) / 1000))
      }, 200)
    } catch (e) {
      const name = (e as Error)?.name
      if (name === 'NotAllowedError') setError('Microphone permission denied')
      else if (name === 'NotFoundError') setError('No microphone found')
      else setError((e as Error)?.message || 'Failed to start recording')
    }
  }, [clearPreview])

  const stop = useCallback(() => {
    if (!isRecording) return
    setIsRecording(false)
    if (timerRef.current) {
      window.clearInterval(timerRef.current)
      timerRef.current = null
    }
    const rec = recorderRef.current
    if (rec && rec.state !== 'inactive') {
      try {
        rec.stop()
      } catch {
        // ignore
      }
    } else {
      streamRef.current?.getTracks().forEach((t) => t.stop())
      streamRef.current = null
    }
    recorderRef.current = null
  }, [isRecording])

  const cancel = useCallback(() => {
    if (isRecording) {
      setIsRecording(false)
      if (timerRef.current) {
        window.clearInterval(timerRef.current)
        timerRef.current = null
      }
      const rec = recorderRef.current
      if (rec && rec.state !== 'inactive') {
        try {
          rec.stop()
        } catch {
          // ignore
        }
      }
      // discard chunks
      chunksRef.current = []
      streamRef.current?.getTracks().forEach((t) => t.stop())
      streamRef.current = null
      recorderRef.current = null
      // onstop will still fire and create preview — clear it next tick
      setTimeout(() => clearPreview(), 0)
    } else {
      clearPreview()
    }
    setError(null)
    setDuration(0)
  }, [isRecording, clearPreview])

  return { isRecording, duration, audioUrl, audioBlob, error, start, stop, cancel, clearPreview, setError }
}
