import type { CanonicalEmotion } from './AvatarProfile'
import type { ExpressionController } from './ExpressionController'
import type { EyeController } from './EyeController'

/**
 * Client-autonomous idle behavior (facial-animation-plan.md §4). Active only in
 * IDLE — the AvatarController gates it. Drives the expression + eye controllers
 * DIRECTLY (not through AvatarController.setEmotion) so it never refreshes the
 * engagement timer.
 *
 * Emotion wanderer alternates rest <-> a weighted micro-emotion so it always
 * passes through neutral (never jumps between two expressions). Gaze wanderer
 * nudges a look target; EyeController still lets a live mouse win.
 */
const EMOTION_INTERVAL_SEC: [number, number] = [4, 9]
const MICRO_INTENSITY: [number, number] = [0.15, 0.4]
const EMOTION_TRANSITION_MS = 800
const GAZE_INTERVAL_SEC: [number, number] = [2, 5]
const GAZE_RANGE = 0.3

// Weighted micro-emotions (bias to neutral). Only calm expressions in idle.
const MICRO_WEIGHTS: Array<{ emotion: CanonicalEmotion; weight: number }> = [
  { emotion: 'happy', weight: 0.62 },
  { emotion: 'relaxed', weight: 0.38 },
]

export class IdleBehaviorController {
  private readonly expression: ExpressionController
  private readonly eye: EyeController

  private emotionTimer = randomRange(EMOTION_INTERVAL_SEC)
  private gazeTimer = randomRange(GAZE_INTERVAL_SEC)
  private atRest = true

  constructor(expression: ExpressionController, eye: EyeController) {
    this.expression = expression
    this.eye = eye
  }

  /** Called when the avatar (re)enters IDLE — restart wander timers. */
  reset(): void {
    this.emotionTimer = randomRange(EMOTION_INTERVAL_SEC)
    this.gazeTimer = randomRange(GAZE_INTERVAL_SEC)
    this.atRest = true
  }

  tick(delta: number): void {
    // Emotion wanderer: alternate rest <-> micro-emotion (always via neutral).
    this.emotionTimer -= delta
    if (this.emotionTimer <= 0) {
      if (this.atRest) {
        const emotion = pickWeighted()
        const intensity = randomRange(MICRO_INTENSITY)
        this.expression.setEmotion(emotion, intensity, EMOTION_TRANSITION_MS)
        this.atRest = false
      } else {
        this.expression.setEmotion('neutral', 1, EMOTION_TRANSITION_MS)
        this.atRest = true
      }
      this.emotionTimer = randomRange(EMOTION_INTERVAL_SEC)
    }

    // Gaze wanderer: occasional saccade to a nearby point, sometimes back to center.
    this.gazeTimer -= delta
    if (this.gazeTimer <= 0) {
      if (Math.random() < 0.35) {
        this.eye.setWander(0, 0)
      } else {
        this.eye.setWander(randomSigned(GAZE_RANGE), randomSigned(GAZE_RANGE))
      }
      this.gazeTimer = randomRange(GAZE_INTERVAL_SEC)
    }
  }
}

function pickWeighted(): CanonicalEmotion {
  const total = MICRO_WEIGHTS.reduce((s, w) => s + w.weight, 0)
  let r = Math.random() * total
  for (const { emotion, weight } of MICRO_WEIGHTS) {
    r -= weight
    if (r <= 0) return emotion
  }
  return 'neutral'
}

function randomRange([min, max]: [number, number]): number {
  return min + Math.random() * (max - min)
}

function randomSigned(magnitude: number): number {
  return (Math.random() * 2 - 1) * magnitude
}
