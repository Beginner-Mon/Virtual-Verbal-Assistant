/**
 * Single source of truth for avatar animation, shared across controllers.
 * Plain object — deliberately NOT React state (see facial-animation-plan.md §8
 * rule 3: frame data must never flow through React).
 */

export type AvatarMode = 'engaged' | 'idle'

export class AnimationState {
  /** ENGAGED = backend-driven; IDLE = client-autonomous. Phase B drives this. */
  mode: AvatarMode = 'idle'

  /** Wall-clock ms until which we stay ENGAGED. Phase B refreshes on events. */
  engagedUntil = 0

  /**
   * Reusable per-frame channel buffer (channel name -> weight). Cleared and
   * refilled every tick by the mixer to avoid per-frame allocation (§8 rule 6).
   * Map.clear() preserves capacity.
   */
  readonly frame: Map<string, number> = new Map()

  isEngaged(now: number): boolean {
    return now < this.engagedUntil
  }
}
