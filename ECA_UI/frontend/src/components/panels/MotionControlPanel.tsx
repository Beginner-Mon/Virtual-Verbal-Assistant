import { Play, Pause, RotateCcw, Activity, Sliders, Camera, Sparkles, Smile } from 'lucide-react'
import { useEffect, useRef, useState, useMemo } from 'react'
import { ScrollArea } from '../ui/scroll-area'
import { useMotion } from '../../contexts/MotionContext'
import type { CharState } from '../../lib/AnimationStates'
import { CANONICAL_EMOTIONS, type CanonicalEmotion } from '../../avatar/AvatarProfile'
import { getManifest } from '../../avatar/vrmManifest'
import { ensureAudioContext, playSyntheticSpeech, playWavSpeech, type SyntheticSpeech } from '../../avatar/lipSyncAudio'
import testWavUrl from '../../asset/audio/test.wav'

const PRESET_TO_CANONICAL: Record<string, CanonicalEmotion> = {
  neutral: 'neutral',
  joy: 'happy',
  angry: 'angry',
  sorrow: 'sad',
  fun: 'relaxed',
  surprised: 'surprised',
}

export default function MotionControlPanel() {
  const {
    cameraMode,
    setCameraMode,
    currentState,
    transitionTo,
    stateOptions,
    playMotionFile,
    motionFileOptions,
    isPlaying,
    setIsPlaying,
    speed,
    setSpeed,
    clipInfo,
    handleReset,
    avatarRef,
    selectedVrmId,
    vrmOptions,
  } = useMotion()

  // Derive modelId exactly the same way CharacterViewer does.
  const modelId = useMemo(() => {
    const selected = vrmOptions.find((o) => o.id === selectedVrmId)
    return (selected?.label ?? 'seele.vrm')
      .replace(/\.vrm$/i, '')
      .replace(/^.*\//, '')
      .toLowerCase()
  }, [selectedVrmId, vrmOptions])

  const manifest = useMemo(() => getManifest(modelId), [modelId])

  // Merge emotions + emotion-like customs (e.g. bronya's "Surprised").
  const emotionButtons = useMemo(() => {
    const all = [...(manifest?.emotions ?? []), ...(manifest?.customs ?? [])]
    if (all.length === 0) {
      return CANONICAL_EMOTIONS.map((e) => ({ label: e, emotion: e }))
    }
    return all.map((e) => ({
      label: e.name,
      emotion: PRESET_TO_CANONICAL[e.presetName] ?? PRESET_TO_CANONICAL[e.emit] ?? 'neutral',
    }))
  }, [manifest])

  const [emotionIntensity, setEmotionIntensity] = useState(0.8)
  const [emotionDurationMs, setEmotionDurationMs] = useState(500)
  const [lastEmotion, setLastEmotion] = useState<string>('—')
  const [avatarMode, setAvatarMode] = useState<string>('—')
  const [speaking, setSpeaking] = useState(false)
  const [speakingWav, setSpeakingWav] = useState(false)
  const speechRef = useRef<SyntheticSpeech | null>(null)

  useEffect(() => {
    ;(window as unknown as { __avatar?: () => unknown }).__avatar = () => avatarRef.current
  }, [avatarRef])

  useEffect(() => {
    const id = setInterval(() => setAvatarMode(avatarRef.current?.mode ?? '—'), 400)
    return () => clearInterval(id)
  }, [avatarRef])

  const triggerEmotion = (emotion: CanonicalEmotion) => {
    const controller = avatarRef.current
    if (!controller) { setLastEmotion('no avatar attached'); return }
    controller.setEmotion(emotion, emotionIntensity, emotionDurationMs)
    setLastEmotion(
      `${emotion} @ ${emotionIntensity.toFixed(2)} / ${emotionDurationMs}ms` +
        (controller.hasCapability ? '' : ' (no expressions)'),
    )
  }

  const speak = () => {
    const controller = avatarRef.current
    if (!controller || speaking) return
    ensureAudioContext()
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

  const speakWav = () => {
    const controller = avatarRef.current
    if (!controller || speakingWav) return
    ensureAudioContext()
    setSpeakingWav(true)
    playWavSpeech(testWavUrl)
      .then((synth) => {
        controller.startLipSync(synth.analyser)
        window.setTimeout(() => {
          controller.stopLipSync()
          synth.stop()
          setSpeakingWav(false)
        }, synth.durationMs + 500)
      })
      .catch(() => {
        setSpeakingWav(false)
      })
  }

  return (
    <div className="flex flex-col h-full">
      <div className="px-4 py-3 border-b border-border/40 shrink-0">
        <h2 className="text-sm font-semibold text-foreground tracking-tight flex items-center gap-2">
          <Sliders className="w-4 h-4 text-primary" />
          Motion Controls
        </h2>
        <p className="text-[11px] text-muted-foreground mt-0.5">Animation source & playback</p>
      </div>

      <ScrollArea className="flex-1 min-h-0 p-4">
        <div className="flex flex-col gap-4">
          <div className="flex flex-col gap-1.5 p-3 rounded-xl bg-secondary/20 border border-border/10">
            <span className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground flex items-center gap-1.5">
              <Camera className="w-3 h-3" />
              Camera mode
            </span>
            <select
              value={cameraMode}
              onChange={(e) => setCameraMode(e.target.value as 'head' | 'hips')}
              className="w-full bg-transparent text-xs text-foreground font-medium border-none outline-none cursor-pointer mt-0.5"
            >
              <option value="head" className="bg-card text-foreground">Head - close face view</option>
              <option value="hips" className="bg-card text-foreground">Hips - wider body view</option>
            </select>
          </div>

          {/* (1) FSM state selector — the primary control. Contents are derived
              from STATES, so a new animation with a `debugLabel` shows up here
              with no edit to this file (plan §3.0 / §11). */}
          <div className="flex flex-col gap-1.5 p-3 rounded-xl bg-secondary/20 border border-border/10">
            <span className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground flex items-center gap-1.5">
              <Activity className="w-3 h-3" />
              Character state
            </span>
            <select
              value={stateOptions.some((o) => o.id === currentState) ? currentState : ''}
              onChange={(e) => void transitionTo(e.target.value as CharState)}
              className="w-full bg-transparent text-xs text-foreground font-medium border-none outline-none cursor-pointer mt-0.5"
            >
              {/* Sequence/dynamic states (thinking_loop, exercise…) are not
                  manually selectable, so show the live state as a read-only row. */}
              {!stateOptions.some((o) => o.id === currentState) && (
                <option value="" className="bg-card text-muted-foreground">
                  {currentState} (auto)
                </option>
              )}
              {stateOptions.map((option) => (
                <option key={option.id} value={option.id} className="bg-card text-foreground">
                  {option.label}
                </option>
              ))}
            </select>
          </div>

          {/* (2) Motion FILE selector — DEBUG. Plays any BVH/FBX through the
              `exercise` state. This is the only way to verify the Kimodo
              NPZ→BVH retarget without a backend or GPU (plan §4.3, test #4). */}
          <div className="flex flex-col gap-1.5 p-3 rounded-xl bg-secondary/20 border border-border/10">
            <span className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground flex items-center gap-1.5">
              <Activity className="w-3 h-3" />
              Motion file (debug)
            </span>
            <select
              defaultValue=""
              onChange={(e) => {
                if (e.target.value) void playMotionFile(e.target.value)
              }}
              className="w-full bg-transparent text-xs text-foreground font-medium border-none outline-none cursor-pointer mt-0.5"
            >
              <option value="" className="bg-card text-muted-foreground">Pick a file to play…</option>
              {motionFileOptions.map((option) => (
                <option key={option.label} value={option.url} className="bg-card text-foreground">
                  {option.label}
                </option>
              ))}
            </select>
          </div>

          <div className="flex items-center justify-between gap-2 p-3 rounded-xl bg-secondary/20 border border-border/10">
            <div className="flex items-center gap-2">
              <button
                onClick={() => setIsPlaying(!isPlaying)}
                className="w-9 h-9 rounded-lg flex items-center justify-center bg-primary text-primary-foreground shadow-lg hover:bg-primary/95 transition-all active:scale-95"
              >
                {isPlaying ? (
                  <Pause className="w-4 h-4 fill-current" />
                ) : (
                  <Play className="w-4 h-4 fill-current ml-0.5" />
                )}
              </button>
              <button
                onClick={handleReset}
                title="Reset animation"
                className="w-9 h-9 rounded-lg flex items-center justify-center border border-border/40 hover:bg-secondary/40 text-foreground transition-all active:scale-95"
              >
                <RotateCcw className="w-4 h-4" />
              </button>
            </div>

            <div className="flex items-center gap-1.5 bg-secondary/30 border border-border/20 px-2 py-1 rounded-lg">
              <Sliders className="w-3 h-3 text-muted-foreground" />
              <select
                value={speed}
                onChange={(e) => setSpeed(parseFloat(e.target.value))}
                className="bg-transparent text-xs text-foreground font-medium border-none outline-none cursor-pointer"
              >
                <option value="0.5" className="bg-card text-foreground">0.5x</option>
                <option value="1.0" className="bg-card text-foreground">1.0x</option>
                <option value="1.5" className="bg-card text-foreground">1.5x</option>
                <option value="2.0" className="bg-card text-foreground">2.0x</option>
              </select>
            </div>
          </div>

          {clipInfo && (
            <div className="flex flex-col gap-1 p-3 rounded-xl bg-secondary/20 border border-border/10">
              <span className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground flex items-center gap-1.5">
                <Sparkles className="w-3 h-3 text-purple-400" />
                Clip Info
              </span>
              <div className="flex justify-between text-[11px] text-muted-foreground mt-0.5">
                <span>Bones: {clipInfo.tracks} tracks</span>
                <span>Duration: {clipInfo.duration.toFixed(2)}s</span>
              </div>
            </div>
          )}

          {import.meta.env.DEV && (
            <div className="flex flex-col gap-2 p-3 rounded-xl bg-secondary/20 border border-border/10">
              <span className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground flex items-center gap-1.5">
                <Smile className="w-3 h-3 text-amber-400" />
                Expressions (dev)
              </span>

              <div className="flex flex-wrap gap-1">
                {emotionButtons.map(({ label, emotion }) => (
                  <button
                    key={label}
                    onClick={() => triggerEmotion(emotion)}
                    className="px-2 py-0.5 text-[11px] rounded-md border border-primary/30 bg-primary/10 text-foreground hover:bg-primary/20 transition-colors cursor-pointer"
                  >
                    {label}
                  </button>
                ))}
              </div>

              <label className="flex items-center justify-between gap-2 text-[11px] text-muted-foreground">
                intensity
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.05}
                  value={emotionIntensity}
                  onChange={(e) => setEmotionIntensity(Number(e.target.value))}
                  className="flex-1 h-1 accent-primary"
                />
                <span className="w-8 text-right tabular-nums">{emotionIntensity.toFixed(2)}</span>
              </label>

              <label className="flex items-center justify-between gap-2 text-[11px] text-muted-foreground">
                duration
                <input
                  type="range"
                  min={0}
                  max={2000}
                  step={50}
                  value={emotionDurationMs}
                  onChange={(e) => setEmotionDurationMs(Number(e.target.value))}
                  className="flex-1 h-1 accent-primary"
                />
                <span className="w-10 text-right tabular-nums">{emotionDurationMs}ms</span>
              </label>

              <button
                onClick={speak}
                disabled={speaking}
                className={`px-2 py-1 text-[11px] rounded-md border transition-colors cursor-pointer ${
                  speaking
                    ? 'border-emerald-500/30 bg-emerald-500/10 text-emerald-400 cursor-default'
                    : 'border-emerald-500/40 bg-emerald-500/15 text-emerald-300 hover:bg-emerald-500/25'
                }`}
              >
                {speaking ? 'speaking…' : 'Speak (test lip-sync)'}
              </button>

              <button
                onClick={speakWav}
                disabled={speakingWav}
                className={`px-2 py-1 text-[11px] rounded-md border transition-colors cursor-pointer ${
                  speakingWav
                    ? 'border-cyan-500/30 bg-cyan-500/10 text-cyan-400 cursor-default'
                    : 'border-cyan-500/40 bg-cyan-500/15 text-cyan-300 hover:bg-cyan-500/25'
                }`}
              >
                {speakingWav ? 'playing test.wav…' : 'Test WAV lip-sync'}
              </button>

              <div className="flex justify-between text-[10px] text-muted-foreground/60">
                <span>mode: {avatarMode}</span>
                <span className="text-right max-w-[60%] truncate">last: {lastEmotion}</span>
              </div>
            </div>
          )}
        </div>
      </ScrollArea>
    </div>
  )
}
