import { describe, expect, it } from 'vitest'
import * as THREE from 'three'
import { VRMHumanBoneName } from '@pixiv/three-vrm'
import type { VRM } from '@pixiv/three-vrm'
import { HeadController } from './HeadController'
import type { EyeController } from './EyeController'

/**
 * Regression cover for the "head stuck crooked after a refresh" bug.
 *
 * `AvatarController` — and with it `HeadController` — is constructed only after
 * `await loadProfileAsync(modelId)`, a network round-trip
 * (CharacterViewer.tsx:169-186). By then the greeting/idle clip has already been
 * writing to the normalized Neck/Head bones for several frames, and
 * `VRMHumanoidRig.update()` does not reset them: it reads the normalized bone and
 * writes through to the raw bone, so the mixer's value stays put.
 *
 * So reading `bone.quaternion` in the constructor captured a random animation
 * frame as "rest", and `rest * offset` then froze the head there forever. Fetch
 * latency varies per load, which is why the bug came and went across refreshes.
 */

const DIRTY_HEAD = new THREE.Quaternion().setFromEuler(
  new THREE.Euler(-0.9, 0.3, 0.15, 'YXZ'), // face pitched hard up, as reported
)
const DIRTY_NECK = new THREE.Quaternion().setFromEuler(new THREE.Euler(-0.4, 0.2, 0, 'YXZ'))

function stubEye(yaw = 0, pitch = 0) {
  return { currentYaw: yaw, currentPitch: pitch } as unknown as EyeController
}

interface FakeOptions {
  /** Provide `humanoid.normalizedRestPose` (the three-vrm canonical source). */
  restPose?: boolean
  /** Provide `vrm.scene.userData.restPoses` (the fallback captured at load). */
  userDataRest?: boolean
}

/**
 * A VRM whose Neck/Head bones are already dirty — exactly the state the mixer
 * leaves them in — while both canonical rest sources say identity.
 */
function makeVrm({ restPose = true, userDataRest = false }: FakeOptions = {}) {
  const neck = new THREE.Object3D()
  const head = new THREE.Object3D()
  neck.quaternion.copy(DIRTY_NECK)
  head.quaternion.copy(DIRTY_HEAD)

  const scene = new THREE.Group()
  if (userDataRest) {
    scene.userData.restPoses = new Map<
      string,
      { position: THREE.Vector3; quaternion: THREE.Quaternion }
    >([
      [VRMHumanBoneName.Neck, { position: new THREE.Vector3(), quaternion: new THREE.Quaternion() }],
      [VRMHumanBoneName.Head, { position: new THREE.Vector3(), quaternion: new THREE.Quaternion() }],
    ])
  }

  const vrm = {
    scene,
    humanoid: {
      getNormalizedBoneNode: (name: VRMHumanBoneName) =>
        name === VRMHumanBoneName.Neck ? neck : name === VRMHumanBoneName.Head ? head : null,
      normalizedRestPose: restPose
        ? {
            [VRMHumanBoneName.Neck]: { rotation: [0, 0, 0, 1] },
            [VRMHumanBoneName.Head]: { rotation: [0, 0, 0, 1] },
          }
        : {},
    },
  } as unknown as VRM

  return { vrm, neck, head }
}

const IDENTITY = new THREE.Quaternion()

/** Angle between two rotations, in radians — order-independent, no Euler traps. */
function angleTo(q: THREE.Quaternion) {
  return q.angleTo(IDENTITY)
}

describe('HeadController rest pose', () => {
  it('ignores a bone already posed by the animation mixer', () => {
    const { vrm, neck, head } = makeVrm()
    const controller = new HeadController(vrm, stubEye())

    controller.tick(1 / 60)

    // Zero gaze + canonical rest ⇒ the bones must land on identity. Before the
    // fix they kept DIRTY_*, which is the head frozen at a wrong angle.
    expect(angleTo(head.quaternion)).toBeLessThan(1e-6)
    expect(angleTo(neck.quaternion)).toBeLessThan(1e-6)
  })

  it('stays put across many frames — no drift, no dependence on when it was built', () => {
    const { vrm, head } = makeVrm()
    const controller = new HeadController(vrm, stubEye())

    for (let i = 0; i < 240; i++) controller.tick(1 / 60)

    expect(angleTo(head.quaternion)).toBeLessThan(1e-6)
  })

  it('falls back to the rest poses captured at load when the humanoid has none', () => {
    const { vrm, head, neck } = makeVrm({ restPose: false, userDataRest: true })
    const controller = new HeadController(vrm, stubEye())

    controller.tick(1 / 60)

    expect(angleTo(head.quaternion)).toBeLessThan(1e-6)
    expect(angleTo(neck.quaternion)).toBeLessThan(1e-6)
  })

  it('detaches to the canonical rest, not to the pose it was built on', () => {
    const { vrm, head, neck } = makeVrm()
    const controller = new HeadController(vrm, stubEye(20, -10))

    controller.tick(1)
    controller.detach()

    expect(angleTo(head.quaternion)).toBeLessThan(1e-6)
    expect(angleTo(neck.quaternion)).toBeLessThan(1e-6)
  })

  it('still follows the gaze once rest is canonical', () => {
    const { vrm, head } = makeVrm()
    const controller = new HeadController(vrm, stubEye(20, -10))

    // A whole second: the smoothing constant makes k ≈ 1, so the head has
    // essentially reached its target.
    controller.tick(1)

    const { yaw, pitch } = controller.debugHead()
    expect(yaw).toBeGreaterThan(0)
    expect(pitch).toBeLessThan(0)
    expect(angleTo(head.quaternion)).toBeGreaterThan(0.01)
  })
})
