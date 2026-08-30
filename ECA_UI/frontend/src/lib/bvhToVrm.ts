/**
 * BVH → VRM Animation Retargeting Utility
 *
 * Loads a BVH file, maps its skeleton bones to VRM humanoid bones,
 * and produces a THREE.AnimationClip that can be played on a VRM model.
 *
 * Supports both standard BVH naming and Mixamo-style naming.
 */

import * as THREE from 'three'
import { BVHLoader } from 'three/examples/jsm/loaders/BVHLoader.js'
import { VRMHumanBoneName } from '@pixiv/three-vrm'
import type { VRM } from '@pixiv/three-vrm'
import type { RestPoseMap } from './restPose'

export interface RetargetOptions {
  mirrorZ: boolean
  flip180Y: boolean
  hipCompensation: THREE.Quaternion | null
  swapYandZ: boolean
}

export const SMPLX_RETARGET_OPTIONS: RetargetOptions = {
  mirrorZ: true,
  flip180Y: false,
  hipCompensation: null,
  swapYandZ: true,
}

export const STANDARD_RETARGET_OPTIONS: RetargetOptions = {
  mirrorZ: false,
  flip180Y: true,
  hipCompensation: null,
  swapYandZ: false,
}

/* ────────────────────────── Bone Name Mapping ────────────────────── */

/**
 * Maps common BVH bone names (case-insensitive) to VRM humanoid bone names.
 * Supports standard BVH, Mixamo, CMU, and Bandai-Namco naming conventions.
 */
const BVH_TO_VRM_BONE_MAP: Record<string, string> = {
  // Root / Hips
  hips: 'hips',
  pelvis: 'hips',
  mixamorigHips: 'hips',

  // Spine chain
  spine: 'spine',
  mixamorigSpine: 'spine',
  spine1: 'chest',
  mixamorigSpine1: 'chest',
  spine2: 'upperChest',
  mixamorigSpine2: 'upperChest',

  // Neck & Head
  neck: 'neck',
  mixamorigNeck: 'neck',
  head: 'head',
  mixamorigHead: 'head',

  // Left Arm
  leftshoulder: 'leftShoulder',
  mixamorigLeftShoulder: 'leftShoulder',
  leftarm: 'leftUpperArm',
  leftupperarm: 'leftUpperArm',
  leftforearm: 'leftLowerArm',
  mixamorigLeftArm: 'leftUpperArm',
  leftlowerarm: 'leftLowerArm',
  mixamorigLeftForeArm: 'leftLowerArm',
  lefthand: 'leftHand',
  mixamorigLeftHand: 'leftHand',

  // Right Arm
  rightshoulder: 'rightShoulder',
  mixamorigRightShoulder: 'rightShoulder',
  rightarm: 'rightUpperArm',
  rightupperarm: 'rightUpperArm',
  rightforearm: 'rightLowerArm',
  mixamorigRightArm: 'rightUpperArm',
  rightlowerarm: 'rightLowerArm',
  mixamorigRightForeArm: 'rightLowerArm',
  righthand: 'rightHand',
  mixamorigRightHand: 'rightHand',

  // Left Leg
  leftupperleg: 'leftUpperLeg',
  leftupleg: 'leftUpperLeg',
  mixamorigLeftUpLeg: 'leftUpperLeg',
  leftlowerleg: 'leftLowerLeg',
  leftleg: 'leftLowerLeg',
  mixamorigLeftLeg: 'leftLowerLeg',
  leftfoot: 'leftFoot',
  mixamorigLeftFoot: 'leftFoot',
  lefttoe: 'leftToes',
  lefttoebase: 'leftToes',
  mixamorigLeftToeBase: 'leftToes',

  // Right Leg
  rightupperleg: 'rightUpperLeg',
  rightupleg: 'rightUpperLeg',
  mixamorigRightUpLeg: 'rightUpperLeg',
  rightlowerleg: 'rightLowerLeg',
  rightleg: 'rightLowerLeg',
  mixamorigRightLeg: 'rightLowerLeg',
  rightfoot: 'rightFoot',
  mixamorigRightFoot: 'rightFoot',
  righttoe: 'rightToes',
  righttoebase: 'rightToes',
  mixamorigRightToeBase: 'rightToes',

  // Facial extras sometimes present in BVH exports
  jaw: 'jaw',
  lefteye: 'leftEye',
  righteye: 'rightEye',
}

/**
 * Resolve a BVH bone name to a VRM humanoid bone name.
 * Tries exact match first, then case-insensitive lookup.
 */
function resolveVrmBoneName(bvhBoneName: string): string | null {
  // Try exact match
  if (BVH_TO_VRM_BONE_MAP[bvhBoneName]) {
    return BVH_TO_VRM_BONE_MAP[bvhBoneName]
  }
  // Try case-insensitive
  const lower = bvhBoneName.toLowerCase()
  if (BVH_TO_VRM_BONE_MAP[lower]) {
    return BVH_TO_VRM_BONE_MAP[lower]
  }
  return null
}

/* ────────────────────────── BVH Loader ──────────────────────────── */

/**
 * Load a BVH file from a URL and return the parsed result.
 */
export async function loadBVH(url: string) {
  const loader = new BVHLoader()
  return new Promise<ReturnType<BVHLoader['parse']>>((resolve, reject) => {
    loader.load(url, resolve, undefined, reject)
  })
}

/* ──────────────────── Retargeting Core Logic ────────────────────── */

/**
 * Create a retargeted AnimationClip from a BVH result that can be
 * played on a VRM model's scene using THREE.AnimationMixer.
 *
 * The function:
 * 1. Iterates through BVH animation tracks
 * 2. Extracts the bone name from each track name
 * 3. Maps it to the VRM humanoid bone
 * 4. Re-writes the track name to target the VRM bone node's path
 * 5. Optionally applies coordinate-system corrections
 */
export function retargetBVHToVRM(
  bvhResult: { skeleton: THREE.Skeleton; clip: THREE.AnimationClip },
  vrm: VRM,
  options: RetargetOptions = STANDARD_RETARGET_OPTIONS,
): THREE.AnimationClip | null {
  const tracks: THREE.KeyframeTrack[] = []
  const clip = bvhResult.clip
  const sourceBoneByName = new Map<string, THREE.Bone>()

  for (const bone of bvhResult.skeleton.bones) {
    sourceBoneByName.set(bone.name, bone)
    sourceBoneByName.set(bone.name.toLowerCase(), bone)
  }

  for (const track of clip.tracks) {
    // Track names are like "boneName.quaternion" or "boneName.position"
    const dotIndex = track.name.lastIndexOf('.')
    if (dotIndex === -1) continue

    const bvhBoneName = track.name.substring(0, dotIndex)
    const property = track.name.substring(dotIndex + 1) // "quaternion" or "position"

    // Resolve the VRM bone name
    const vrmBoneName = resolveVrmBoneName(bvhBoneName)
    if (!vrmBoneName) continue

    // Get the VRM bone node
    const vrmBoneNode = vrm.humanoid?.getNormalizedBoneNode(vrmBoneName as VRMHumanBoneName)
    if (!vrmBoneNode) continue

    if (vrmBoneNode.name !== vrmBoneName) {
      vrmBoneNode.name = vrmBoneName
    }

    // Prefer the normalized bone node name directly; fall back to the
    // full hierarchy path when the node has no usable name.
    const nodePath = vrmBoneNode.name || getNodePath(vrm.scene, vrmBoneNode)
    if (!nodePath) continue

    // For rotation tracks (quaternion), apply them directly
    // BVH data is already converted to quaternions by BVHLoader
    if (property === 'quaternion') {
      const sourceBone = sourceBoneByName.get(bvhBoneName) ?? sourceBoneByName.get(bvhBoneName.toLowerCase())
      const sourceRest = sourceBone?.quaternion.clone() ?? new THREE.Quaternion()
      const sourceRestConverted = sourceRest.clone()
      if (options.mirrorZ) sourceRestConverted.copy(mirrorQuaternionZ(sourceRestConverted))
      if (options.flip180Y) sourceRestConverted.set(-sourceRestConverted.x, sourceRestConverted.y, -sourceRestConverted.z, sourceRestConverted.w).normalize()
      
      const sourceRestInv = sourceRestConverted.clone().invert()
      
      const vrmRestPose = (vrm.scene.userData.restPoses as RestPoseMap | undefined)?.get(vrmBoneName)
      const targetRest = vrmRestPose ? vrmRestPose.quaternion.clone() : vrmBoneNode.quaternion.clone()

      const retargetedValues = new Float32Array(track.values.length)
      const sourceQuat = new THREE.Quaternion()
      const deltaQuat = new THREE.Quaternion()
      const targetQuat = new THREE.Quaternion()

      for (let i = 0; i < track.values.length; i += 4) {
        sourceQuat.fromArray(track.values as ArrayLike<number>, i)

        if (options.mirrorZ) {
          sourceQuat.copy(mirrorQuaternionZ(sourceQuat))
        }
        if (options.flip180Y) {
          sourceQuat.set(-sourceQuat.x, sourceQuat.y, -sourceQuat.z, sourceQuat.w).normalize()
        }

        // Convert source local rotation to delta from source rest,
        // then re-apply on top of target rest orientation.
        deltaQuat.copy(sourceRestInv).multiply(sourceQuat)
        targetQuat.copy(targetRest).multiply(deltaQuat)
        
        // Compensate for group-rotation change if applicable
        if (vrmBoneName === 'hips' && options.hipCompensation) {
          targetQuat.premultiply(options.hipCompensation)
        }
        // NOTE: the Y-up→Z-up tilt is handled by the parent <group rotation={[π/2,0,0]}>
        // in CharacterViewer. Do NOT tilt the bones here too — it would double-rotate.
        targetQuat.normalize()
        targetQuat.toArray(retargetedValues, i)
      }

      const newTrack = new THREE.QuaternionKeyframeTrack(
        `${nodePath}.quaternion`,
        Array.from(track.times),
        Array.from(retargetedValues),
      )
      tracks.push(newTrack)
    }

    // For position tracks, only apply to hips (root motion)
    if (property === 'position' && vrmBoneName === 'hips') {
      // Scale BVH positions down (BVH uses cm, VRM uses meters).
      const scaledValues = new Float32Array(track.values.length)
      const scale = 0.01 // BVH centimeters → meters
      
      const vrmRestPose = (vrm.scene.userData.restPoses as RestPoseMap | undefined)?.get(vrmBoneName)
      const targetRestPos = vrmRestPose ? vrmRestPose.position.clone() : vrmBoneNode.position.clone()
      
      const bvhStartX = track.values[0]
      const bvhStartY = track.values[1]
      const bvhStartZ = track.values[2]

      for (let i = 0; i < track.values.length; i += 3) {
        // Calculate delta from first frame (in cm)
        const dx = track.values[i] - bvhStartX
        const dy = track.values[i + 1] - bvhStartY
        const dz = track.values[i + 2] - bvhStartZ
        
        // Convert delta to meters
        const deltaX = dx * scale
        const deltaY = dy * scale
        const deltaZ = dz * scale
        
        if (options.swapYandZ) {
          // The parent <group rotation={[π/2,0,0]}> handles Y-up→Z-up visually.
          // Position is in the group's LOCAL space (still Y-up).
          // Negate deltaZ: SMPL forward is +Z, but VRM faces -Z.
          scaledValues[i] = targetRestPos.x + deltaX
          scaledValues[i + 1] = targetRestPos.y + deltaY  // Y = vertical (up)
          scaledValues[i + 2] = targetRestPos.z - deltaZ  // -Z = VRM forward direction
        } else if (options.flip180Y) {
          // Rotated 180 degrees around Y axis
          scaledValues[i] = targetRestPos.x - deltaX
          scaledValues[i + 1] = targetRestPos.y + deltaY
          scaledValues[i + 2] = targetRestPos.z - deltaZ
        } else {
          // Standard Y-up convention
          scaledValues[i] = targetRestPos.x + deltaX
          scaledValues[i + 1] = targetRestPos.y + deltaY
          scaledValues[i + 2] = targetRestPos.z + deltaZ
        }
      }

      const newTrack = new THREE.VectorKeyframeTrack(
        `${nodePath}.position`,
        Array.from(track.times),
        Array.from(scaledValues),
      )
      tracks.push(newTrack)
    }
  }

  if (tracks.length === 0) {
    console.warn('[bvhToVrm] No matching bones found between BVH and VRM model.')
    return null
  }

  return new THREE.AnimationClip('bvh-retargeted', clip.duration, tracks)
}

function mirrorQuaternionZ(q: THREE.Quaternion): THREE.Quaternion {
  // Reflection across Z basis change (x, y, z, w) -> (-x, -y, z, w).
  return new THREE.Quaternion(-q.x, -q.y, q.z, q.w).normalize()
}

/* ───────────────────────── Helper Functions ──────────────────────── */

/**
 * Get the property path from a root object to a target child,
 * suitable for use in AnimationClip track names.
 *
 * Returns a path like "Scene.Children[2].Children[0]..." using object names.
 */
function getNodePath(root: THREE.Object3D, target: THREE.Object3D): string | null {
  if (root === target) return root.name || ''

  const path: string[] = []
  let current: THREE.Object3D | null = target

  while (current && current !== root) {
    path.unshift(current.name)
    current = current.parent
  }

  if (current !== root) return null

  return path.join('.')
}

/* ──────────────────── Convenience "Load & Retarget" ─────────────── */

/**
 * Load a BVH file and retarget it to a VRM model in one call.
 *
 * @param bvhUrl URL/path to the BVH file
 * @param vrm The loaded VRM instance
 * @returns An AnimationClip ready to be used with THREE.AnimationMixer
 */
export async function loadAndRetargetBVH(
  bvhUrl: string,
  vrm: VRM,
  options: RetargetOptions = STANDARD_RETARGET_OPTIONS,
): Promise<THREE.AnimationClip | null> {
  const bvhResult = await loadBVH(bvhUrl)
  return retargetBVHToVRM(bvhResult, vrm, options)
}
