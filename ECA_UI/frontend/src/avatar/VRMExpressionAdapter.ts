import type { VRM } from '@pixiv/three-vrm'
import type { AvatarProfile } from './AvatarProfile'

/**
 * The ONLY class allowed to touch the VRM (facial-animation-plan.md §2). Resolves
 * profile channels to three-vrm expression weights and writes them each frame.
 *
 * Capability detection: at attach time it enumerates which of the profile's
 * channels actually exist on this model (getExpression != null). Missing channels
 * become safe no-ops — this is what lets a stripped model like bronya_long.vrm
 * (0 blendshape groups) run without crashing.
 */
export class VRMExpressionAdapter {
  private readonly vrm: VRM
  /** Channels present on this model that the profile can drive. */
  private readonly managed: string[]
  private readonly available: Set<string>

  constructor(vrm: VRM, profile: AvatarProfile) {
    this.vrm = vrm

    // Collect every channel the profile might drive: emotion recipes + blink + visemes.
    const wanted = new Set<string>()
    for (const recipe of Object.values(profile.recipes)) {
      for (const channel of Object.keys(recipe)) wanted.add(channel)
    }
    wanted.add(profile.blinkChannel)
    for (const channel of Object.values(profile.visemes)) wanted.add(channel)

    const manager = vrm.expressionManager
    this.available = new Set<string>()
    if (manager) {
      for (const channel of wanted) {
        if (manager.getExpression(channel) != null) this.available.add(channel)
      }
    }
    this.managed = [...this.available]

    const missing = [...wanted].filter((c) => !this.available.has(c))
    if (!manager) {
      console.warn(
        `[avatar] model "${profile.modelId}" has no expressionManager — all facial output disabled`,
      )
    } else if (missing.length > 0) {
      console.warn(
        `[avatar] model "${profile.modelId}" missing ${missing.length} expression channel(s), will no-op: ${missing.join(', ')}`,
      )
    }
  }

  /** True when the model can render at least one managed channel. */
  get hasCapability(): boolean {
    return this.managed.length > 0
  }

  /**
   * Write a full frame. Every MANAGED channel is written (defaulting to 0 when
   * absent from `frame`) so a channel that dropped out this tick relaxes to rest
   * instead of sticking at its last weight.
   */
  write(frame: Map<string, number>): void {
    const manager = this.vrm.expressionManager
    if (!manager) return
    for (const channel of this.managed) {
      const weight = frame.get(channel) ?? 0
      manager.setValue(channel, clamp01(weight))
    }
  }

  /** Zero every managed channel — used on detach so no weight is left applied. */
  reset(): void {
    const manager = this.vrm.expressionManager
    if (!manager) return
    for (const channel of this.managed) manager.setValue(channel, 0)
  }
}

function clamp01(v: number): number {
  if (v < 0) return 0
  if (v > 1) return 1
  return v
}
