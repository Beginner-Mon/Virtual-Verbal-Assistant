/**
 * Channel-based expression mixer (facial-animation-plan.md §5).
 *
 * Replaces the proposal's per-controller "priority 1/2/3/4". Ownership is static
 * per output channel: each contributor writes only the channels it owns, in a
 * defined order, into one shared frame buffer. Conflicts are impossible because
 * owners are disjoint by construction (emotion never writes `blink`; blink only
 * writes `blink`). Phase C's lip-sync contributor runs LAST so it can override
 * the mouth viseme channels while emotion keeps the rest.
 */

export interface ExpressionContributor {
  /** Write owned channels into the shared frame. Later contributors win per channel. */
  contribute(frame: Map<string, number>): void
}

export class ExpressionMixer {
  /**
   * Clear the frame and let each contributor write its owned channels, in order.
   * The frame is a reused buffer (AnimationState.frame) — cleared, not realloc'd.
   */
  compose(frame: Map<string, number>, contributors: readonly ExpressionContributor[]): void {
    frame.clear()
    for (const contributor of contributors) {
      contributor.contribute(frame)
    }
  }
}
