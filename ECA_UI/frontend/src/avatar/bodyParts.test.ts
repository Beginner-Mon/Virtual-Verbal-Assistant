import { describe, expect, it } from 'vitest'
import * as THREE from 'three'
import { VRMHumanBoneName } from '@pixiv/three-vrm'
import type { VRM } from '@pixiv/three-vrm'
import {
  BODY_PARTS,
  BODY_PART_BY_BONE,
  PART_BY_ID,
  buildBoneToPartMap,
  partId,
  resolvePart,
} from './bodyParts'

/**
 * A minimal humanoid: one Object3D per requested bone, parented into a chain so
 * `resolvePart` has a real ancestry to climb — that climb is what makes twist /
 * spring bones (hair, skirt) resolve instead of falling through.
 */
function makeVrm(boneNames: VRMHumanBoneName[]) {
  const nodes = new Map<VRMHumanBoneName, THREE.Object3D>()
  for (const name of boneNames) {
    const node = new THREE.Object3D()
    node.name = name
    nodes.set(name, node)
  }
  const vrm = {
    humanoid: {
      getRawBoneNode: (name: VRMHumanBoneName) => nodes.get(name) ?? null,
    },
  } as unknown as VRM
  return { vrm, nodes }
}

describe('BODY_PART_BY_BONE', () => {
  it('covers every VRM humanoid bone', () => {
    const unmapped = Object.values(VRMHumanBoneName).filter(
      (bone) => BODY_PART_BY_BONE[bone] === undefined,
    )
    expect(unmapped).toEqual([])
  })

  it('only ever yields a declared part — never a raw bone name', () => {
    const parts = new Set<string>(BODY_PARTS)
    for (const bone of Object.values(VRMHumanBoneName)) {
      expect(parts.has(BODY_PART_BY_BONE[bone])).toBe(true)
    }
  })

  it('folds fingers into the hand, never surfacing on their own', () => {
    // The reported bug: clicking a thigh answered "leftThumbProximal" — a raw
    // bone name, and the wrong limb. An individual finger is never an answer.
    expect(BODY_PART_BY_BONE[VRMHumanBoneName.LeftThumbProximal]).toBe('leftHand')
    expect(BODY_PART_BY_BONE[VRMHumanBoneName.RightLittleDistal]).toBe('rightHand')
    expect(BODY_PART_BY_BONE[VRMHumanBoneName.LeftHand]).toBe('leftHand')
  })

  it('keeps the wrist and ankle as their own regions', () => {
    // Physical-therapy exercises target wrist and ankle in their own right, so
    // they are not a sub-segment of the arm or the leg.
    expect(BODY_PART_BY_BONE[VRMHumanBoneName.RightShoulder]).toBe('rightArm')
    expect(BODY_PART_BY_BONE[VRMHumanBoneName.LeftUpperArm]).toBe('leftArm')
    expect(BODY_PART_BY_BONE[VRMHumanBoneName.LeftLowerArm]).toBe('leftArm')
    expect(BODY_PART_BY_BONE[VRMHumanBoneName.LeftUpperLeg]).toBe('leftLeg')
    expect(BODY_PART_BY_BONE[VRMHumanBoneName.RightLowerLeg]).toBe('rightLeg')
  })

  it('folds toes into the foot', () => {
    expect(BODY_PART_BY_BONE[VRMHumanBoneName.LeftToes]).toBe('leftFoot')
    expect(BODY_PART_BY_BONE[VRMHumanBoneName.RightFoot]).toBe('rightFoot')
    expect(BODY_PART_BY_BONE[VRMHumanBoneName.LeftFoot]).toBe('leftFoot')
  })

  it('never leaks a bone from one side into the other side"s part', () => {
    // Stated as "no cross-side leak" rather than "left bone ⇒ left part",
    // because leftEye/rightEye legitimately answer "head".
    for (const bone of Object.values(VRMHumanBoneName)) {
      const part = BODY_PART_BY_BONE[bone]
      if (bone.startsWith('left')) expect(part.startsWith('right')).toBe(false)
      if (bone.startsWith('right')) expect(part.startsWith('left')).toBe(false)
    }
  })

  it('maps the face bones a model may or may not ship', () => {
    // Rarely present — most VRMs skin the whole face to `head` — but when they
    // are, the eye bones own the eyeball meshes that no morph target moves.
    expect(BODY_PART_BY_BONE[VRMHumanBoneName.LeftEye]).toBe('leftEye')
    expect(BODY_PART_BY_BONE[VRMHumanBoneName.RightEye]).toBe('rightEye')
    expect(BODY_PART_BY_BONE[VRMHumanBoneName.Jaw]).toBe('mouth')
    expect(BODY_PART_BY_BONE[VRMHumanBoneName.Neck]).toBe('head')
    expect(BODY_PART_BY_BONE[VRMHumanBoneName.Head]).toBe('head')
  })

  it('maps the torso and the root', () => {
    expect(BODY_PART_BY_BONE[VRMHumanBoneName.Spine]).toBe('chest')
    expect(BODY_PART_BY_BONE[VRMHumanBoneName.UpperChest]).toBe('chest')
    expect(BODY_PART_BY_BONE[VRMHumanBoneName.Hips]).toBe('hips')
  })
})

describe('part ids', () => {
  it('reserves 0 for "no hit" so a cleared pick buffer reads as a miss', () => {
    expect(PART_BY_ID[0]).toBeUndefined()
    for (const part of BODY_PARTS) expect(partId(part)).toBeGreaterThan(0)
  })

  it('round-trips every part through its id', () => {
    for (const part of BODY_PARTS) expect(PART_BY_ID[partId(part)]).toBe(part)
  })

  it('stays inside one 8-bit channel', () => {
    for (const part of BODY_PARTS) expect(partId(part)).toBeLessThanOrEqual(255)
  })
})

describe('buildBoneToPartMap', () => {
  it('keys on node identity, not on bone name', () => {
    const { vrm, nodes } = makeVrm([VRMHumanBoneName.Head, VRMHumanBoneName.LeftHand])
    const map = buildBoneToPartMap(vrm)

    expect(map.get(nodes.get(VRMHumanBoneName.Head)!)).toBe('head')
    expect(map.get(nodes.get(VRMHumanBoneName.LeftHand)!)).toBe('leftHand')

    // A different node carrying the same name must NOT resolve.
    const impostor = new THREE.Object3D()
    impostor.name = VRMHumanBoneName.Head
    expect(map.get(impostor)).toBeUndefined()
  })

  it('skips bones the model does not define', () => {
    const { vrm } = makeVrm([VRMHumanBoneName.Head])
    expect(buildBoneToPartMap(vrm).size).toBe(1)
  })
})

describe('resolvePart', () => {
  it('resolves a humanoid bone directly', () => {
    const { vrm, nodes } = makeVrm([VRMHumanBoneName.RightUpperArm])
    const map = buildBoneToPartMap(vrm)
    expect(resolvePart(nodes.get(VRMHumanBoneName.RightUpperArm)!, map)).toBe('rightArm')
  })

  it('climbs to the nearest humanoid ancestor for a non-humanoid bone', () => {
    // Hair spring bones and twist bones are skinning targets but are not part of
    // the humanoid map — the old code fell through to a nearest-joint guess here.
    const { vrm, nodes } = makeVrm([VRMHumanBoneName.Head])
    const map = buildBoneToPartMap(vrm)

    const hairRoot = new THREE.Object3D()
    const hairTip = new THREE.Object3D()
    nodes.get(VRMHumanBoneName.Head)!.add(hairRoot)
    hairRoot.add(hairTip)

    expect(resolvePart(hairTip, map)).toBe('head')
  })

  it('stops at the nearest ancestor, not the root', () => {
    const { vrm, nodes } = makeVrm([VRMHumanBoneName.Hips, VRMHumanBoneName.LeftLowerLeg])
    const map = buildBoneToPartMap(vrm)

    const hips = nodes.get(VRMHumanBoneName.Hips)!
    const shin = nodes.get(VRMHumanBoneName.LeftLowerLeg)!
    hips.add(shin)
    const twist = new THREE.Object3D()
    shin.add(twist)

    expect(resolvePart(twist, map)).toBe('leftLeg')
  })

  it('returns null for a node outside the rig', () => {
    const { vrm } = makeVrm([VRMHumanBoneName.Head])
    const map = buildBoneToPartMap(vrm)
    expect(resolvePart(new THREE.Object3D(), map)).toBeNull()
  })

  it('tolerates a null node', () => {
    const { vrm } = makeVrm([VRMHumanBoneName.Head])
    expect(resolvePart(null, buildBoneToPartMap(vrm))).toBeNull()
  })
})
