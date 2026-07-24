import type { VRM } from '@pixiv/three-vrm'
import type { AvatarProfile, CanonicalEmotion } from './AvatarProfile'
import { AnimationState } from './AnimationState'
import { VRMExpressionAdapter } from './VRMExpressionAdapter'
import { ExpressionMixer, type ExpressionContributor } from './ExpressionMixer'
import { ExpressionController } from './ExpressionController'
import { BlinkController } from './BlinkController'
import { EyeController } from './EyeController'
import { IdleBehaviorController } from './IdleBehaviorController'
import { LipSyncController } from './LipSyncController'

const DEFAULT_EMOTION_DURATION_MS = 500
const EVENT_GRACE_MS = 3000
const TTS_GRACE_MS = 1500
const IDLE_NEUTRAL_FADE_MS = 800

/**
 * Single facade over the avatar animation stack (facial-animation-plan.md §3).
 * Business logic (DevPanel now, SSE handler later) talks ONLY to this class in
 * canonical emotion names; everything below is private.
 *
 * ENGAGED vs IDLE (§1) is derived here from an engagement deadline: external
 * events (setEmotion, TTS playing) push the deadline forward; when it passes,
 * the client-autonomous idle behavior takes over. Blink runs in both modes.
 */
export class AvatarController {
  readonly profile: AvatarProfile
  private readonly state = new AnimationState()
  private readonly adapter: VRMExpressionAdapter
  private readonly mixer = new ExpressionMixer()
  private readonly expression: ExpressionController
  private readonly blink: BlinkController
  private readonly eye: EyeController
  private readonly idle: IdleBehaviorController
  private readonly lipSync: LipSyncController
  private readonly contributors: readonly ExpressionContributor[]

  constructor(vrm: VRM, profile: AvatarProfile) {
    this.profile = profile
    this.adapter = new VRMExpressionAdapter(vrm, profile)
    this.expression = new ExpressionController(profile)
    this.blink = new BlinkController(profile)
    this.eye = new EyeController(vrm)
    this.lipSync = new LipSyncController(profile)
    this.idle = new IdleBehaviorController(this.expression, this.eye)
    // Order = layering (§5). Emotion first; lip-sync overrides the mouth; blink last.
    this.contributors = [this.expression, this.lipSync, this.blink]
  }

  // ── External commands (refresh engagement -> ENGAGED) ────────────────────

  /** Transition to a canonical emotion. Marks the avatar engaged. */
  setEmotion(
    emotion: CanonicalEmotion,
    intensity = 1,
    durationMs = DEFAULT_EMOTION_DURATION_MS,
  ): void {
    this.expression.setEmotion(emotion, intensity, durationMs)
    this.notifyEngaged(durationMs)
  }

  /** Refresh the engagement deadline (call on any backend avatar event). */
  notifyEngaged(durationMs = 0): void {
    const until = now() + Math.max(EVENT_GRACE_MS, durationMs)
    if (until > this.state.engagedUntil) this.state.engagedUntil = until
  }

  /** Start lip sync from an analyser (a TTS clip / synthetic speech). Engages. */
  startLipSync(analyser: AnalyserNode): void {
    this.lipSync.start(analyser)
    this.notifyEngaged(0)
  }

  stopLipSync(): void {
    this.lipSync.stop()
    // Give a short grace so we don't snap to idle the instant audio ends (§1.2).
    this.state.engagedUntil = Math.max(this.state.engagedUntil, now() + TTS_GRACE_MS)
  }

  /** Feed normalized mouse gaze in [-1..1] (x right+, y up+). */
  setMouse(nx: number, ny: number): void {
    this.eye.setMouse(nx, ny, now())
  }

  // ── Frame update ─────────────────────────────────────────────────────────

  /**
   * Advance one frame. MUST run between mixer.update and vrm.update in the host
   * useFrame (§8 rule 1) so setValue lands before vrm.update applies it.
   */
  tick(delta: number): void {
    const t = now()

    // TTS keeps us engaged while it plays.
    if (this.lipSync.isPlaying) this.notifyEngaged(0)

    const wasEngaged = this.state.mode === 'engaged'
    const engaged = this.state.isEngaged(t)
    this.state.mode = engaged ? 'engaged' : 'idle'

    if (wasEngaged && !engaged) {
      // Entering idle: cross-fade current emotion to neutral before the
      // wanderer takes over (§1.3), and restart wander timers.
      this.expression.setEmotion('neutral', 1, IDLE_NEUTRAL_FADE_MS)
      this.idle.reset()
    }

    if (!engaged) this.idle.tick(delta)

    this.expression.tick(delta)
    this.blink.tick(delta)
    this.lipSync.tick(delta)
    this.eye.tick(delta, t)

    this.mixer.compose(this.state.frame, this.contributors)
    this.adapter.write(this.state.frame)
  }

  // ── Introspection ────────────────────────────────────────────────────────

  get hasCapability(): boolean {
    return this.adapter.hasCapability
  }

  get mode(): string {
    return this.state.mode
  }

  /** Read the last-composed weight of a channel (debug/verification only). */
  debugChannelWeight(channel: string): number {
    return this.state.frame.get(channel) ?? 0
  }

  /** Read smoothed gaze angles (debug/verification only). */
  debugEye(): { yaw: number; pitch: number } {
    return this.eye.debugAngles()
  }

  /** Release: zero every managed channel + restore gaze so nothing is left applied. */
  detach(): void {
    this.lipSync.detach()
    this.eye.detach()
    this.adapter.reset()
    this.state.frame.clear()
  }
}

function now(): number {
  return typeof performance !== 'undefined' ? performance.now() : Date.now()
}
