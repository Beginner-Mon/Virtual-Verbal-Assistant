import * as THREE from 'three'
import { VRMHumanBoneName } from '@pixiv/three-vrm'
import type { VRM } from '@pixiv/three-vrm'
import type { EyeController } from './EyeController'

/**
 * Head follow — turns the Neck + Head bones toward the SAME gaze the eyes track,
 * so the avatar's whole head follows the cursor, not just the pupils. Reads the
 * already-smoothed angle from EyeController (single source of truth) and applies
 * it to bones — it never reads the mouse or knows about idle/wander itself.
 *
 * Bones vs blendshapes: this touches ONLY bone rotation; facial expression /
 * blink / lip-sync live on blendshape channels via the ExpressionMixer, so the
 * two systems are fully independent and cannot conflict.
 *
 * Rotation convention (review gap #2): applied on the three-vrm NORMALIZED
 * humanoid bones, whose rest pose is identity-ish and canonical (Y-up, -Z fwd).
 * yaw = rotation about local Y, pitch = about local X, Euler order 'YXZ' (same
 * order three-vrm's own lookAt uses). We compose `rest * offset` so a non-identity
 * rest is preserved and there is no per-frame drift.
 *
 * Pitch (review gap #3): EyeController.currentPitch is already in three-vrm
 * convention (negative = up). PITCH_SIGN maps that to the bone's local X axis.
 *
 * BVH conflict (review gap #1): the tick runs AFTER mixer.update, and we write
 * from the stored REST rotation — so while a body animation drives neck/head this
 * override would win for those two bones. That is acceptable now: body animation
 * is OFF by default (Kimodo drives the body later). When body motion returns,
 * revisit with additive blending — tracked as Phase 2.
 */

const HEAD_FOLLOW_GAIN = 0.6 // head turns 60% of the eye gaze angle (eyes lead)
const NECK_SHARE = 0.4 // of the head rotation, neck takes this, the head bone the rest
const YAW_SIGN = 1 // verified: head turns toward the cursor left/right
// EyeController.currentPitch is negative for "look up" (three-vrm convention). On
// the normalized bone's local +X axis a positive angle tilts the face UP, so we
// negate to keep the head consistent with the eyes (verified visually on seele).
const PITCH_SIGN = -1
const SMOOTH_PER_SEC = 8 // slightly slower than the eyes → natural lead-then-follow
const DEG2RAD = Math.PI / 180

export class HeadController {
  private readonly eye: EyeController
  private readonly neck: THREE.Object3D | null
  private readonly head: THREE.Object3D | null
  private readonly restNeck = new THREE.Quaternion()
  private readonly restHead = new THREE.Quaternion()

  // Smoothed head angles (degrees) — a touch behind the eyes.
  private curYaw = 0
  private curPitch = 0

  // Reusable scratch to avoid per-frame allocation (§8 rule 6).
  private readonly euler = new THREE.Euler(0, 0, 0, 'YXZ')
  private readonly offset = new THREE.Quaternion()

  constructor(vrm: VRM, eye: EyeController) {
    this.eye = eye
    this.neck = vrm.humanoid?.getNormalizedBoneNode(VRMHumanBoneName.Neck) ?? null
    this.head = vrm.humanoid?.getNormalizedBoneNode(VRMHumanBoneName.Head) ?? null
    if (this.neck) this.restNeck.copy(this.neck.quaternion)
    if (this.head) this.restHead.copy(this.head.quaternion)
    if (!this.head && !this.neck) {
      console.warn('[avatar] model has no Neck/Head bone — head follow disabled')
    }
  }

  tick(delta: number): void {
    if (!this.head && !this.neck) return

    const targetYaw = this.eye.currentYaw * HEAD_FOLLOW_GAIN
    const targetPitch = this.eye.currentPitch * HEAD_FOLLOW_GAIN

    const k = 1 - Math.exp(-SMOOTH_PER_SEC * delta)
    this.curYaw += (targetYaw - this.curYaw) * k
    this.curPitch += (targetPitch - this.curPitch) * k

    // Split the rotation across the two bones. If one is missing, the other
    // takes the whole share so the follow magnitude stays the same.
    let neckShare = 0
    let headShare = 0
    if (this.neck && this.head) {
      neckShare = NECK_SHARE
      headShare = 1 - NECK_SHARE
    } else if (this.neck) {
      neckShare = 1
    } else {
      headShare = 1
    }

    if (this.neck) this.apply(this.neck, this.restNeck, neckShare)
    if (this.head) this.apply(this.head, this.restHead, headShare)
  }

  private apply(bone: THREE.Object3D, rest: THREE.Quaternion, share: number): void {
    this.euler.set(
      this.curPitch * share * PITCH_SIGN * DEG2RAD,
      this.curYaw * share * YAW_SIGN * DEG2RAD,
      0,
      'YXZ',
    )
    this.offset.setFromEuler(this.euler)
    bone.quaternion.copy(rest).multiply(this.offset)
  }

  /** Debug read (verification only). */
  debugHead(): { yaw: number; pitch: number } {
    return { yaw: Number(this.curYaw.toFixed(2)), pitch: Number(this.curPitch.toFixed(2)) }
  }

  /** Restore both bones to their rest rotation so nothing is left applied. */
  detach(): void {
    if (this.neck) this.neck.quaternion.copy(this.restNeck)
    if (this.head) this.head.quaternion.copy(this.restHead)
  }
}
