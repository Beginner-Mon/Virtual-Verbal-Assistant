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
 * Where `rest` comes from matters (see HeadController.test.ts). It is read from
 * the model's declared rest pose, NEVER from the bone's live rotation: this class
 * is constructed after `await loadProfileAsync(...)`, a network round-trip, by
 * which point the greeting/idle clip has already posed Neck/Head — and
 * `VRMHumanoidRig.update()` writes normalized → raw without ever resetting the
 * normalized bone, so the mixer's value is still sitting there. Sampling it made
 * `rest` a random animation frame, freezing the head at a wrong angle for the
 * whole session, with fetch latency deciding whether a given refresh was affected.
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
    if (this.neck) readRest(vrm, VRMHumanBoneName.Neck, this.neck, this.restNeck)
    if (this.head) readRest(vrm, VRMHumanBoneName.Head, this.head, this.restHead)
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

/**
 * Fill `out` with the bone's declared rest rotation, most authoritative source
 * first. The live `bone.quaternion` is the last resort precisely because it is
 * the one that can be contaminated by a clip already in flight.
 *
 * 1. `humanoid.normalizedRestPose` — three-vrm's own record of the rig's rest
 *    state, built when the rig is constructed and never written to afterwards.
 * 2. `vrm.scene.userData.restPoses` — snapshotted in the body of VRMCharacter
 *    (CharacterViewer.tsx:92-106), which runs on first render, ahead of every
 *    effect and every frame, so it also predates the mixer.
 * 3. The live bone, for a model that somehow offers neither.
 */
function readRest(
  vrm: VRM,
  boneName: VRMHumanBoneName,
  bone: THREE.Object3D,
  out: THREE.Quaternion,
): void {
  const declared = vrm.humanoid?.normalizedRestPose?.[boneName]?.rotation
  if (declared) {
    out.set(declared[0], declared[1], declared[2], declared[3])
    return
  }

  const captured = (
    vrm.scene?.userData?.restPoses as Map<string, { quaternion: THREE.Quaternion }> | undefined
  )?.get(boneName)
  if (captured) {
    out.copy(captured.quaternion)
    return
  }

  out.copy(bone.quaternion)
}
