import { useEffect, useRef, useState } from 'react'
import { useMotion } from '../contexts/MotionContext'
import { CANONICAL_EMOTIONS, type CanonicalEmotion } from './AvatarProfile'
import { ensureAudioContext, playSyntheticSpeech, type SyntheticSpeech } from './lipSyncAudio'

/**
 * Dev-only facial-animation test panel (facial-animation-plan.md §10 Phase A).
 * Drives the active AvatarController imperatively via the MotionContext ref —
 * no backend needed. Mounted only under import.meta.env.DEV by CharacterViewer.
 */
export default function AvatarDevPanel() {
  const { avatarRef } = useMotion()
  const [intensity, setIntensity] = useState(0.8)
  const [durationMs, setDurationMs] = useState(500)
  const [last, setLast] = useState<string>('—')
  const [mode, setMode] = useState<string>('—')
  const [speaking, setSpeaking] = useState(false)
  const speechRef = useRef<SyntheticSpeech | null>(null)

  // Expose the controller handle for e2e verification (dev bundle only).
  useEffect(() => {
    ;(window as unknown as { __avatar?: () => unknown }).__avatar = () => avatarRef.current
  }, [avatarRef])

  // Poll the engaged/idle mode for display.
  useEffect(() => {
    const id = setInterval(() => setMode(avatarRef.current?.mode ?? '—'), 400)
    return () => clearInterval(id)
  }, [avatarRef])

  const speak = () => {
    const controller = avatarRef.current
    if (!controller || speaking) return
    ensureAudioContext() // this click is the required user gesture
    const synth = playSyntheticSpeech(3000)
    speechRef.current = synth
    controller.startLipSync(synth.analyser)
    setSpeaking(true)
    window.setTimeout(() => {
      controller.stopLipSync()
      synth.stop()
      speechRef.current = null
      setSpeaking(false)
    }, 3000)
  }

  const trigger = (emotion: CanonicalEmotion) => {
    const controller = avatarRef.current
    if (!controller) {
      setLast('no avatar attached')
      return
    }
    controller.setEmotion(emotion, intensity, durationMs)
    setLast(
      `${emotion} @ ${intensity.toFixed(2)} / ${durationMs}ms` +
        (controller.hasCapability ? '' : ' (model has no expressions)'),
    )
  }

  return (
    <div
      style={{
        position: 'absolute',
        top: 12,
        left: 12,
        zIndex: 50,
        display: 'flex',
        flexDirection: 'column',
        gap: 6,
        padding: 10,
        borderRadius: 10,
        background: 'rgba(20,10,40,0.72)',
        backdropFilter: 'blur(6px)',
        color: '#ede9fe',
        font: '11px/1.4 ui-monospace, monospace',
        maxWidth: 220,
        userSelect: 'none',
      }}
    >
      <strong style={{ fontSize: 11, letterSpacing: 0.4 }}>AVATAR DEV — emotions</strong>

      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
        {CANONICAL_EMOTIONS.map((emotion) => (
          <button
            key={emotion}
            onClick={() => trigger(emotion)}
            style={{
              cursor: 'pointer',
              padding: '3px 7px',
              borderRadius: 6,
              border: '1px solid rgba(167,139,250,0.4)',
              background: 'rgba(124,58,237,0.25)',
              color: '#ede9fe',
              font: 'inherit',
            }}
          >
            {emotion}
          </button>
        ))}
      </div>

      <label style={{ display: 'flex', justifyContent: 'space-between', gap: 6 }}>
        intensity
        <input
          type="range"
          min={0}
          max={1}
          step={0.05}
          value={intensity}
          onChange={(e) => setIntensity(Number(e.target.value))}
        />
        <span style={{ width: 28, textAlign: 'right' }}>{intensity.toFixed(2)}</span>
      </label>

      <label style={{ display: 'flex', justifyContent: 'space-between', gap: 6 }}>
        duration
        <input
          type="range"
          min={0}
          max={2000}
          step={50}
          value={durationMs}
          onChange={(e) => setDurationMs(Number(e.target.value))}
        />
        <span style={{ width: 40, textAlign: 'right' }}>{durationMs}ms</span>
      </label>

      <button
        onClick={speak}
        disabled={speaking}
        style={{
          cursor: speaking ? 'default' : 'pointer',
          padding: '4px 7px',
          borderRadius: 6,
          border: '1px solid rgba(52,211,153,0.5)',
          background: speaking ? 'rgba(52,211,153,0.15)' : 'rgba(16,185,129,0.28)',
          color: '#d1fae5',
          font: 'inherit',
        }}
      >
        {speaking ? 'speaking…' : 'Speak (test lip-sync)'}
      </button>

      <span style={{ opacity: 0.7 }}>mode: {mode}</span>
      <span style={{ opacity: 0.7 }}>last: {last}</span>
    </div>
  )
}
