import { useEffect, useRef, useState, useMemo } from 'react'
import { useMotion } from '../hooks/useMotion'
import { CANONICAL_EMOTIONS, type CanonicalEmotion } from './AvatarProfile'
import { getManifest } from './vrmManifest'
import { ensureAudioContext, playSyntheticSpeech, type SyntheticSpeech } from './lipSyncAudio'

const PRESET_TO_CANONICAL: Record<string, CanonicalEmotion> = {
  neutral: 'neutral',
  joy: 'happy',
  angry: 'angry',
  sorrow: 'sad',
  fun: 'relaxed',
  surprised: 'surprised',
}

export default function AvatarDevPanel() {
  const { avatarRef, selectedVrmId, vrmOptions } = useMotion()

  const modelId = useMemo(() => {
    const selected = vrmOptions.find((o) => o.id === selectedVrmId)
    return (selected?.label ?? 'bronya.vrm')
      .replace(/\.vrm$/i, '')
      .replace(/^.*\//, '')
      .toLowerCase()
  }, [selectedVrmId, vrmOptions])

  const manifest = useMemo(() => getManifest(modelId), [modelId])

  const emotionButtons = useMemo(() => {
    const all = [...(manifest?.emotions ?? []), ...(manifest?.customs ?? [])]
    if (all.length === 0) {
      return CANONICAL_EMOTIONS.map((e) => ({ label: e, emotion: e }))
    }
    return all.map((e) => ({
      label: e.name,
      // `presetName` is null for custom blendShapes — fall through to `emit`.
      emotion: PRESET_TO_CANONICAL[e.presetName ?? ''] ?? PRESET_TO_CANONICAL[e.emit] ?? 'neutral',
    }))
  }, [manifest])
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
        {emotionButtons.map(({ label, emotion }) => (
          <button
            key={label}
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
            {label}
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

