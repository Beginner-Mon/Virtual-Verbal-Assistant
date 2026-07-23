import type { ExpressionContributor } from './ExpressionMixer'
import type { AvatarProfile, CanonicalEmotion, EmotionRecipe } from './AvatarProfile'

/**
 * Owns emotion channels. Cross-fades between recipes using delta-time
 * interpolation so a 500ms transition looks identical at 30fps or 144Hz
 * (facial-animation-plan.md §8 rule 5).
 *
 * Interrupt handling: calling setEmotion mid-transition snapshots the CURRENT
 * interpolated weights as the new `from`, so it fades from where it is — never
 * snaps back to 0 first.
 */
export class ExpressionController implements ExpressionContributor {
  private readonly profile: AvatarProfile

  private from: Map<string, number> = new Map()
  private to: Map<string, number> = new Map()
  private progress = 1 // 1 = settled
  private durationSec = 0.001

  // Reused scratch to dedupe channel union without per-frame allocation.
  private readonly seen: Set<string> = new Set()

  constructor(profile: AvatarProfile) {
    this.profile = profile
  }

  /**
   * Transition to a canonical emotion at `intensity` (0..1) over `durationMs`.
   * Unknown emotions are ignored (validated upstream, but guard anyway).
   */
  setEmotion(emotion: CanonicalEmotion, intensity: number, durationMs: number): void {
    const recipe: EmotionRecipe | undefined = this.profile.recipes[emotion]
    if (!recipe) {
      console.warn(`[avatar] unknown emotion "${emotion}" ignored`)
      return
    }

    // Snapshot current interpolated weights -> new `from` (smooth interrupt).
    const e = easeInOutCubic(this.progress)
    const snapshot = new Map<string, number>()
    this.seen.clear()
    for (const [ch, toW] of this.to) {
      const fromW = this.from.get(ch) ?? 0
      snapshot.set(ch, lerp(fromW, toW, e))
      this.seen.add(ch)
    }
    for (const [ch, fromW] of this.from) {
      if (this.seen.has(ch)) continue
      snapshot.set(ch, lerp(fromW, 0, e))
    }

    const clampedIntensity = clamp01(intensity)
    const target = new Map<string, number>()
    for (const [ch, w] of Object.entries(recipe)) {
      target.set(ch, w * clampedIntensity)
    }

    this.from = snapshot
    this.to = target
    this.progress = 0
    this.durationSec = Math.max(durationMs, 1) / 1000
  }

  tick(delta: number): void {
    if (this.progress >= 1) return
    this.progress = Math.min(1, this.progress + delta / this.durationSec)
    if (this.progress >= 1) {
      // Settled: from-only channels are at rest, drop them to bound the maps.
      this.from.clear()
    }
  }

  contribute(frame: Map<string, number>): void {
    const e = easeInOutCubic(this.progress)
    this.seen.clear()
    for (const [ch, toW] of this.to) {
      const fromW = this.from.get(ch) ?? 0
      frame.set(ch, lerp(fromW, toW, e))
      this.seen.add(ch)
    }
    for (const [ch, fromW] of this.from) {
      if (this.seen.has(ch)) continue
      frame.set(ch, lerp(fromW, 0, e))
    }
  }
}

function easeInOutCubic(t: number): number {
  return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t
}

function clamp01(v: number): number {
  if (v < 0) return 0
  if (v > 1) return 1
  return v
}
