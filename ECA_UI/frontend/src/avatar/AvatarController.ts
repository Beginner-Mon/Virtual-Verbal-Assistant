import type { VRM } from '@pixiv/three-vrm'
import type { AvatarProfile, CanonicalEmotion } from './AvatarProfile'
import { AnimationState } from './AnimationState'
import { VRMExpressionAdapter } from './VRMExpressionAdapter'
import { ExpressionMixer, type ExpressionContributor } from './ExpressionMixer'
import { ExpressionController } from './ExpressionController'
import { BlinkController } from './BlinkController'

const DEFAULT_EMOTION_DURATION_MS = 500

/**
 * Single facade over the avatar animation stack (facial-animation-plan.md §3).
 * Business logic (DevPanel now, SSE handler later) talks ONLY to this class in
 * canonical emotion names; everything below is private.
 *
 * One imperative object, driven by a single tick() from the existing useFrame —
 * never React state (§8 rule 3).
 */
export class AvatarController {
  readonly profile: AvatarProfile
  private readonly state = new AnimationState()
  private readonly adapter: VRMExpressionAdapter
  private readonly mixer = new ExpressionMixer()
  private readonly expression: ExpressionController
  private readonly blink: BlinkController
  private readonly contributors: readonly ExpressionContributor[]

  constructor(vrm: VRM, profile: AvatarProfile) {
    this.profile = profile
    this.adapter = new VRMExpressionAdapter(vrm, profile)
    this.expression = new ExpressionController(profile)
    this.blink = new BlinkController(profile)
    // Order = layering. Emotion first, blink second (disjoint in Phase A).
    // Phase C's lip-sync contributor appends after emotion to override visemes.
    this.contributors = [this.expression, this.blink]
  }

  /** Transition to a canonical emotion. Safe to call every turn / mid-fade. */
  setEmotion(
    emotion: CanonicalEmotion,
    intensity = 1,
    durationMs = DEFAULT_EMOTION_DURATION_MS,
  ): void {
    this.expression.setEmotion(emotion, intensity, durationMs)
  }

  /**
   * Advance one frame. MUST be called between mixer.update and vrm.update in the
   * host useFrame (§8 rule 1) so setValue lands before vrm.update applies it.
   */
  tick(delta: number): void {
    this.expression.tick(delta)
    this.blink.tick(delta)
    this.mixer.compose(this.state.frame, this.contributors)
    this.adapter.write(this.state.frame)
  }

  /** True when the attached model can render at least one expression channel. */
  get hasCapability(): boolean {
    return this.adapter.hasCapability
  }

  /** Read the last-composed weight of a channel (debug/verification only). */
  debugChannelWeight(channel: string): number {
    return this.state.frame.get(channel) ?? 0
  }

  /** Release: zero every managed channel so nothing is left applied. */
  detach(): void {
    this.adapter.reset()
    this.state.frame.clear()
  }
}
