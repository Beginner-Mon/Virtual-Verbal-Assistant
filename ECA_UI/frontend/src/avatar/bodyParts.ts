import type * as THREE from 'three'
import { VRMHumanBoneName } from '@pixiv/three-vrm'
import type { VRM } from '@pixiv/three-vrm'

/**
 * Bone → body-part mapping for click identification.
 *
 * Deliberately an explicit table rather than substring matching on bone names.
 * The substring version it replaces silently failed on every bone whose name
 * contained none of its probe words — `jaw`, `leftEye`, `leftToes`,
 * `leftThumbProximal` — and leaked the raw bone name out as if it were a part.
 * A table cannot rot that way: `bodyParts.test.ts` asserts full coverage of
 * `VRMHumanBoneName`, so a spec addition fails the suite instead of the UI.
 *
 * Granularity: the seven regions the old code could produce, with hands and feet
 * split out. Wrist and ankle are their own exercise targets in physical therapy,
 * so folding them into "arm" and "leg" would lose a distinction the domain cares
 * about. Individual fingers and toes stay folded in — no exercise is prescribed
 * per phalanx, and a per-knuckle answer is what the old fallback produced by
 * accident. Going finer still (upper vs lower limb) is a table edit plus ids;
 * nothing outside this file assumes the current set.
 */
export const BODY_PARTS = [
  'head',
  'mouth',
  'leftEye',
  'rightEye',
  'chest',
  'hips',
  'leftArm',
  'rightArm',
  'leftHand',
  'rightHand',
  'leftLeg',
  'rightLeg',
  'leftFoot',
  'rightFoot',
] as const

export type BodyPart = (typeof BODY_PARTS)[number]

const B = VRMHumanBoneName

export const BODY_PART_BY_BONE: Record<VRMHumanBoneName, BodyPart> = {
  // Root + torso
  [B.Hips]: 'hips',
  [B.Spine]: 'chest',
  [B.Chest]: 'chest',
  [B.UpperChest]: 'chest',

  // Head and face. Most models skin the whole face to `head` and ship neither
  // eye nor jaw bones, so these three rarely fire — the face is resolved from
  // blendshape regions instead (see BodyPartPicker.refineFaceRegions). They are
  // still worth mapping: when a model DOES have eye bones they own the eyeball
  // meshes, which no morph target touches.
  [B.Neck]: 'head',
  [B.Head]: 'head',
  [B.LeftEye]: 'leftEye',
  [B.RightEye]: 'rightEye',
  [B.Jaw]: 'mouth',

  // Legs — hip through shin. The ankle starts its own region.
  [B.LeftUpperLeg]: 'leftLeg',
  [B.LeftLowerLeg]: 'leftLeg',
  [B.RightUpperLeg]: 'rightLeg',
  [B.RightLowerLeg]: 'rightLeg',

  // Feet — toes fold in; no exercise is prescribed per toe.
  [B.LeftFoot]: 'leftFoot',
  [B.LeftToes]: 'leftFoot',
  [B.RightFoot]: 'rightFoot',
  [B.RightToes]: 'rightFoot',

  // Arms — shoulder through forearm. The wrist starts its own region.
  [B.LeftShoulder]: 'leftArm',
  [B.LeftUpperArm]: 'leftArm',
  [B.LeftLowerArm]: 'leftArm',
  [B.RightShoulder]: 'rightArm',
  [B.RightUpperArm]: 'rightArm',
  [B.RightLowerArm]: 'rightArm',

  // Hands — every finger bone folds into the hand it belongs to.
  [B.LeftHand]: 'leftHand',
  [B.LeftThumbMetacarpal]: 'leftHand',
  [B.LeftThumbProximal]: 'leftHand',
  [B.LeftThumbDistal]: 'leftHand',
  [B.LeftIndexProximal]: 'leftHand',
  [B.LeftIndexIntermediate]: 'leftHand',
  [B.LeftIndexDistal]: 'leftHand',
  [B.LeftMiddleProximal]: 'leftHand',
  [B.LeftMiddleIntermediate]: 'leftHand',
  [B.LeftMiddleDistal]: 'leftHand',
  [B.LeftRingProximal]: 'leftHand',
  [B.LeftRingIntermediate]: 'leftHand',
  [B.LeftRingDistal]: 'leftHand',
  [B.LeftLittleProximal]: 'leftHand',
  [B.LeftLittleIntermediate]: 'leftHand',
  [B.LeftLittleDistal]: 'leftHand',

  [B.RightHand]: 'rightHand',
  [B.RightThumbMetacarpal]: 'rightHand',
  [B.RightThumbProximal]: 'rightHand',
  [B.RightThumbDistal]: 'rightHand',
  [B.RightIndexProximal]: 'rightHand',
  [B.RightIndexIntermediate]: 'rightHand',
  [B.RightIndexDistal]: 'rightHand',
  [B.RightMiddleProximal]: 'rightHand',
  [B.RightMiddleIntermediate]: 'rightHand',
  [B.RightMiddleDistal]: 'rightHand',
  [B.RightRingProximal]: 'rightHand',
  [B.RightRingIntermediate]: 'rightHand',
  [B.RightRingDistal]: 'rightHand',
  [B.RightLittleProximal]: 'rightHand',
  [B.RightLittleIntermediate]: 'rightHand',
  [B.RightLittleDistal]: 'rightHand',
}

/**
 * Wire format for GPU picking: the part index travels in one 8-bit channel, so
 * 0 must stay free — a cleared pick buffer reads back as 0 and that has to mean
 * "nothing was there", not "hips".
 */
export const PART_BY_ID: readonly (BodyPart | undefined)[] = [undefined, ...BODY_PARTS]

const ID_BY_PART = new Map<BodyPart, number>(BODY_PARTS.map((part, i) => [part, i + 1]))

export function partId(part: BodyPart): number {
  return ID_BY_PART.get(part) ?? 0
}

/**
 * Map every humanoid bone NODE to its part, once per model.
 *
 * Keyed on object identity rather than name: bone names are model-authored and
 * routinely collide with hair/accessory nodes. Uses raw bone nodes because those
 * are what `SkinnedMesh.skeleton.bones` actually contains — the normalized rig is
 * a separate object tree that no skin index ever points at.
 */
export function buildBoneToPartMap(vrm: VRM): Map<THREE.Object3D, BodyPart> {
  const map = new Map<THREE.Object3D, BodyPart>()
  const humanoid = vrm.humanoid
  if (!humanoid) return map

  for (const bone of Object.values(VRMHumanBoneName)) {
    const node = humanoid.getRawBoneNode(bone)
    if (node) map.set(node, BODY_PART_BY_BONE[bone])
  }
  return map
}

/**
 * Resolve any node to a body part by climbing to its nearest humanoid ancestor.
 *
 * This is what makes twist bones, spring bones and accessory bones answer
 * correctly: they are legitimate skinning targets but are absent from the
 * humanoid map, and the code this replaces fell back to a nearest-joint distance
 * guess for exactly those hits — which is how a click on a thigh could answer
 * with a thumb joint that happened to hang beside it.
 */
export function resolvePart(
  node: THREE.Object3D | null | undefined,
  map: Map<THREE.Object3D, BodyPart>,
): BodyPart | null {
  let current: THREE.Object3D | null = node ?? null
  while (current) {
    const part = map.get(current)
    if (part) return part
    current = current.parent
  }
  return null
}
