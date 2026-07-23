import type { ExpressionContributor } from './ExpressionMixer'
import type { AvatarProfile } from './AvatarProfile'

/**
 * Auto-blink. Owns the `blink` channel only. Runs in BOTH modes — blinking is
 * physiological, not an expression (facial-animation-plan.md §1.3). three-vrm's
 * own overrideBlink multiplies this onto any eye-open of the active emotion, so
 * we simply drive the blink weight and let the library handle the interaction.
 *
 * Delta-time based: interval and cycle are in seconds, correct at any FPS.
 */
const BLINK_DURATION_SEC = 0.15
const INTERVAL_MIN_SEC = 2
const INTERVAL_MAX_SEC = 6
const DOUBLE_BLINK_PROB = 0.12
const DOUBLE_BLINK_GAP_SEC = 0.12

export class BlinkController implements ExpressionContributor {
  private readonly channel: string

  private waiting = true
  private timeUntilNext = randomInterval()
  private cycleElapsed = 0
  private doubleQueued = false
  private gapRemaining = 0
  private weight = 0

  constructor(profile: AvatarProfile) {
    this.channel = profile.blinkChannel
  }

  tick(delta: number): void {
    if (this.waiting) {
      // Short gap between the two halves of a double-blink.
      if (this.gapRemaining > 0) {
        this.gapRemaining -= delta
        if (this.gapRemaining <= 0) this.startBlink(false)
        return
      }
      this.timeUntilNext -= delta
      if (this.timeUntilNext <= 0) {
        this.startBlink(Math.random() < DOUBLE_BLINK_PROB)
      }
      return
    }

    // Blinking: half-sine 0 -> 1 -> 0 across BLINK_DURATION_SEC.
    this.cycleElapsed += delta
    const t = this.cycleElapsed / BLINK_DURATION_SEC
    if (t >= 1) {
      this.weight = 0
      this.waiting = true
      if (this.doubleQueued) {
        this.doubleQueued = false
        this.gapRemaining = DOUBLE_BLINK_GAP_SEC
      } else {
        this.timeUntilNext = randomInterval()
      }
      return
    }
    this.weight = Math.sin(Math.PI * t)
  }

  contribute(frame: Map<string, number>): void {
    frame.set(this.channel, this.weight)
  }

  private startBlink(queueDouble: boolean): void {
    this.waiting = false
    this.cycleElapsed = 0
    this.doubleQueued = queueDouble
    this.weight = 0
  }
}

function randomInterval(): number {
  return INTERVAL_MIN_SEC + Math.random() * (INTERVAL_MAX_SEC - INTERVAL_MIN_SEC)
}
