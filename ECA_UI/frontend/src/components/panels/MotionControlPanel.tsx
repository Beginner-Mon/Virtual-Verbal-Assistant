import { Activity, Sliders, Camera, Smile, Lock } from 'lucide-react'
import { useEffect, useState, useMemo } from 'react'
import { ScrollArea } from '../ui/scroll-area'
import { useMotion } from '../../hooks/useMotion'
import type { CharState } from '../../lib/AnimationStates'
import { CANONICAL_EMOTIONS, type CanonicalEmotion } from '../../avatar/AvatarProfile'
import { getManifest } from '../../avatar/vrmManifest'
import { DEFAULT_CAMERA_CONFIG } from '../../lib/CameraConfig'
import { fetchMotionStatus } from '../../lib/api'

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
    sessionMotions,
    avatarRef,
    selectedVrmId,
    vrmOptions,
    cameraConfig,
    setCameraConfig,
  } = useMotion()

  // Derive modelId exactly the same way CharacterViewer does.
  const modelId = useMemo(() => {
    const selected = vrmOptions.find((o) => o.id === selectedVrmId)
    return (selected?.label ?? 'bronya.vrm')
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
      // `presetName` is null for custom blendShapes — fall through to `emit`.
      emotion: PRESET_TO_CANONICAL[e.presetName ?? ''] ?? PRESET_TO_CANONICAL[e.emit] ?? 'neutral',
    }))
  }, [manifest])

  const [emotionIntensity, setEmotionIntensity] = useState(0.8)
  const [emotionDurationMs, setEmotionDurationMs] = useState(500)
  const [lastEmotion, setLastEmotion] = useState<string>('—')
  const [avatarMode, setAvatarMode] = useState<string>('—')

  // Filter motion files so Character state actions don't leak into the debug picker.
  // The picker used to list bundled sample .bvh files under asset/motions/
  // generated/. Those were fixtures for verifying the Kimodo NPZ→BVH pipeline
  // offline, before a render could actually be requested; they are gone, and
  // the list is now the motions the backend rendered this session.
  //
  // motionFileOptions still exists for AnimationRegistry, which resolves the
  // FSM's static clips (Standard Idle, action_greeting, random_Bored,
  // Thinking) out of the same index — those stay bundled and are not pickable.

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
            <div className="flex items-center justify-between">
              <span className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground flex items-center gap-1.5">
                <Camera className="w-3 h-3" />
                Camera config
              </span>
              <button
                onClick={() => setCameraConfig(DEFAULT_CAMERA_CONFIG)}
                className="text-[9px] text-muted-foreground hover:text-foreground transition-colors cursor-pointer"
              >
                Reset
              </button>
            </div>
            
            <label className="flex items-center justify-between text-[11px] text-foreground mt-1 cursor-pointer">
              <span>Follow Target (lookAt)</span>
              <input
                type="checkbox"
                checked={cameraConfig.followTarget}
                onChange={(e) => setCameraConfig({ ...cameraConfig, followTarget: e.target.checked })}
                className="accent-primary"
              />
            </label>
            
            <label className="flex items-center justify-between text-[11px] text-foreground cursor-pointer">
              <span>Enable Pan</span>
              <input
                type="checkbox"
                checked={cameraConfig.enablePan}
                onChange={(e) => setCameraConfig({ ...cameraConfig, enablePan: e.target.checked })}
                className="accent-primary"
              />
            </label>

            <div className="flex flex-col gap-1.5 mt-2">
              <span className="text-[10px] text-muted-foreground flex items-center gap-1">
                <Lock className="w-3 h-3" />
                Lock Axes (right-drag)
              </span>
              <div className="flex gap-1.5">
                {(['X', 'Y', 'Z'] as const).map((axis) => {
                  const key = `lock${axis}` as const
                  const locked = cameraConfig[key]
                  return (
                    <button
                      key={axis}
                      onClick={() => setCameraConfig({ ...cameraConfig, [key]: !locked })}
                      className={`flex-1 py-1.5 text-[11px] font-medium rounded-md border transition-colors cursor-pointer flex items-center justify-center gap-1 ${
                        locked
                          ? 'bg-primary text-primary-foreground border-primary shadow-sm'
                          : 'bg-secondary/40 text-muted-foreground border-border/20 hover:bg-secondary/60 hover:text-foreground'
                      }`}
                      title={locked ? `Lock ${axis} â€” drag won't change ${axis}` : `Unlock ${axis}`}
                    >
                      <Lock className="w-3 h-3" />
                      {axis}
                    </button>
                  )
                })}
              </div>
            </div>

            <select
              value={cameraMode}
              onChange={(e) => setCameraMode(e.target.value as 'head' | 'hips')}
              className="w-full bg-transparent text-xs text-foreground font-medium border-none outline-none cursor-pointer mt-1.5 pt-1.5 border-t border-border/10"
            >
              <option value="head" className="bg-card text-foreground">Target: Head</option>
              <option value="hips" className="bg-card text-foreground">Target: Hips</option>
            </select>
          </div>

          {/* (1) FSM state selector — dev-only. Contents derived from STATES debugLabel. */}
          {import.meta.env.DEV && (
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
          )}

          {/* (2) Replay a motion the assistant rendered.
              NOT dev-gated, unlike the two blocks around it: `exercise` is a
              one-shot that returns to idle, so a user who looked away has lost
              it, and this is the only way back. The blocks either side stay
              dev-only because they drive the FSM and blend shapes directly â€”
              those are for us, this is for the user. */}
          <div className="flex flex-col gap-1.5 p-3 rounded-xl bg-secondary/20 border border-border/10">
              <span className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground flex items-center gap-1.5">
                <Activity className="w-3 h-3" />
                Xem lại động tác
              </span>
              <select
                // Uncontrolled with a reset: picking the SAME motion twice must
                // fire again, and a controlled value would make the second pick
                // a no-op — replaying is the whole point of this list.
                value=""
                disabled={sessionMotions.length === 0}
                onChange={(e) => {
                  const picked = sessionMotions.find((m) => m.jobId === e.target.value)
                  if (!picked) return
                  void (async () => {
                    // A motion played earlier this session already has a URL,
                    // and the clip is cached under its job_id — that replay
                    // touches nothing and works even after the URL's
                    // five-minute signature has expired.
                    //
                    // One restored from conversation history has neither: fresh
                    // page, empty cache, and no URL worth storing. Ask for a
                    // signed one now, at the moment it is about to be used.
                    let url = picked.url
                    if (!url) {
                      const status = await fetchMotionStatus(picked.jobId)
                      if (status.status !== 'done' || !status.url) {
                        console.warn('[motion] replay unavailable:', status)
                        return
                      }
                      url = status.url
                    }
                    await playMotionFile(url, picked.jobId, picked.label)
                  })()
                }}
                className="w-full bg-transparent text-xs text-foreground font-medium border-none outline-none cursor-pointer mt-0.5 disabled:opacity-50"
              >
                <option value="" className="bg-card text-muted-foreground">
                  {sessionMotions.length === 0
                    ? 'Chưa có động tác nào — hãy hỏi để xem một động tác'
                    : 'Chọn để xem lại…'}
                </option>
                {sessionMotions.map((m) => (
                  <option key={m.jobId} value={m.jobId} className="bg-card text-foreground">
                    {m.label}
                  </option>
                ))}
              </select>
          </div>

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

