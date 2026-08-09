/**
 * Accumulates root motion from one-shot animations so the character stays at
 * its final position when transitioning back to a looping state (e.g. idle).
 *
 * The idle clip has no hips position track, so the mixer restores the rest
 * pose — which teleports the character back to origin. This class captures
 * the hips displacement at the end of a one-shot and applies it as a
 * persistent offset on the model group, so the idle clip runs at local
 * origin while the character visually stays put.
 *
 * CRITICAL: the offset is ramped in over the crossfade duration, NOT applied
 * instantly. During the crossfade, the outgoing action still contributes its
 * hips displacement at a weight that decreases linearly from 1 to 0. If we
 * applied the full group offset at once, the displacement would be DOUBLED
 * (group + clip) and then gradually settle. By ramping group offset from 0
 * to full in sync with the clip weight going from 1 to 0, the two cancel
 * out perfectly and the character stays motionless at its final position.
 *
 * Works with the existing GroundClamp: both write to `target.position`, but
 * on different axes — this class writes XY (horizontal displacement in the
 * Z-up world), and GroundClamp writes Z (vertical lift). They compose
 * cleanly because GroundClamp captures `baseZ` once and adds its lift on
 * top.
 */

import * as THREE from 'three'
import type { VRM } from '@pixiv/three-vrm'

export class RootMotionAccumulator {
  private readonly target: THREE.Object3D
  /** Accumulated world-space offset from all completed blends. */
  private readonly offset = new THREE.Vector3()
  /** Scratch vector for getWorldPosition — avoids per-frame allocation. */
  private readonly scratch = new THREE.Vector3()
  /** Hips world position at the START of the current one-shot (or null). */
  private startHipsWorld: THREE.Vector3 | null = null
  /** The authored (initial) position of the target group. */
  private readonly basePosition: THREE.Vector3

  /** Offset being ramped in during the current crossfade. */
  private readonly pendingOffset = new THREE.Vector3()
  /** Seconds elapsed since the blend started. */
  private blendElapsed = 0
  /** Total crossfade duration for the current blend. */
  private blendDuration = 0
  /** Whether a blend-in is in progress. */
  private blending = false

  constructor(target: THREE.Object3D) {
    this.target = target
    this.basePosition = target.position.clone()
  }

  /**
   * Call when a one-shot animation starts playing.
   * Captures the hips world position as the reference for displacement.
   */
  beginOneShot(vrm: VRM): void {
    const hips = vrm.humanoid?.getNormalizedBoneNode('hips' as any)
    if (!hips) return
    this.startHipsWorld = new THREE.Vector3()
    hips.getWorldPosition(this.startHipsWorld)
  }

  /**
   * Call when a one-shot animation finishes, BEFORE the transition to the
   * next state. Records the hips displacement and begins ramping it into the
   * group position over `crossfadeSec` seconds — synchronized with the
   * mixer's crossfade so the visual position stays perfectly constant.
   */
  commitOneShot(vrm: VRM, crossfadeSec: number): void {
    if (!this.startHipsWorld) return
    const hips = vrm.humanoid?.getNormalizedBoneNode('hips' as any)
    if (!hips) { this.startHipsWorld = null; return }

    hips.getWorldPosition(this.scratch)
    // Store the displacement as a PENDING offset — it will be blended in
    // over the crossfade duration by `update()`.
    this.pendingOffset.set(
      this.scratch.x - this.startHipsWorld.x,
      this.scratch.y - this.startHipsWorld.y,
      0, // Z is NOT accumulated — GroundClamp owns vertical positioning.
    )
    this.blendElapsed = 0
    this.blendDuration = Math.max(crossfadeSec, 0.001) // avoid div-by-zero
    this.blending = true
    this.startHipsWorld = null
  }

  /**
   * Call every frame, AFTER the mixer update. Ramps the pending offset in
   * sync with the crossfade so the character stays visually stationary.
   */
  update(delta: number): void {
    if (!this.blending) return

    this.blendElapsed += delta
    const t = Math.min(this.blendElapsed / this.blendDuration, 1)

    // Exercise weight goes linearly 1→0 during fadeOut, so we ramp 0→1.
    this.target.position.x = this.basePosition.x + this.offset.x + this.pendingOffset.x * t
    this.target.position.y = this.basePosition.y + this.offset.y + this.pendingOffset.y * t

    if (t >= 1) {
      // Blend complete — fold pending into the permanent offset.
      this.offset.x += this.pendingOffset.x
      this.offset.y += this.pendingOffset.y
      this.pendingOffset.set(0, 0, 0)
      this.blending = false
    }
  }

  /** Cancel tracking without accumulating (e.g. animation was interrupted). */
  cancelOneShot(): void {
    this.startHipsWorld = null
  }

  /** Whether a one-shot is currently being tracked. */
  get isTracking(): boolean {
    return this.startHipsWorld !== null
  }

  /** Current accumulated offset (read-only). */
  get currentOffset(): Readonly<THREE.Vector3> {
    return this.offset
  }
}
