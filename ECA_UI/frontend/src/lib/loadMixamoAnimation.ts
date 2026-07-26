/**
 * Mixamo FBX → VRM Animation Retargeting
 *
 * Loads a Mixamo FBX file (exported as "Without Skin", FBX Binary),
 * retargets the animation to a VRM model, and returns a ready-to-play
 * THREE.AnimationClip.
 *
 * References:
 *   - @pixiv/three-vrm official Mixamo example
 *   - https://github.com/pixiv/three-vrm
 */

import * as THREE from 'three'
import { FBXLoader } from 'three/examples/jsm/loaders/FBXLoader.js'
import type { VRM, VRMHumanBoneName } from '@pixiv/three-vrm'

/* ────────────────────────── Mixamo → VRM Bone Map ────────────────── */

/**
 * Maps Mixamo bone names to VRM humanoid bone names.
 * Mixamo uses the "mixamorig" prefix convention.
 */
const MIXAMO_TO_VRM_BONE: Record<string, string> = {
  // Root
  mixamorigHips: 'hips',

  // Spine
  mixamorigSpine: 'spine',
  mixamorigSpine1: 'chest',
  mixamorigSpine2: 'upperChest',

  // Head
  mixamorigNeck: 'neck',
  mixamorigHead: 'head',

  // Left Arm
  mixamorigLeftShoulder: 'leftShoulder',
  mixamorigLeftArm: 'leftUpperArm',
  mixamorigLeftForeArm: 'leftLowerArm',
  mixamorigLeftHand: 'leftHand',

  // Right Arm
  mixamorigRightShoulder: 'rightShoulder',
  mixamorigRightArm: 'rightUpperArm',
  mixamorigRightForeArm: 'rightLowerArm',
  mixamorigRightHand: 'rightHand',

  // Left Leg
  mixamorigLeftUpLeg: 'leftUpperLeg',
  mixamorigLeftLeg: 'leftLowerLeg',
  mixamorigLeftFoot: 'leftFoot',
  mixamorigLeftToeBase: 'leftToes',

  // Right Leg
  mixamorigRightUpLeg: 'rightUpperLeg',
  mixamorigRightLeg: 'rightLowerLeg',
  mixamorigRightFoot: 'rightFoot',
  mixamorigRightToeBase: 'rightToes',

  // Fingers — Left
  mixamorigLeftHandThumb1: 'leftThumbMetacarpal',
  mixamorigLeftHandThumb2: 'leftThumbProximal',
  mixamorigLeftHandThumb3: 'leftThumbDistal',
  mixamorigLeftHandIndex1: 'leftIndexProximal',
  mixamorigLeftHandIndex2: 'leftIndexIntermediate',
  mixamorigLeftHandIndex3: 'leftIndexDistal',
  mixamorigLeftHandMiddle1: 'leftMiddleProximal',
  mixamorigLeftHandMiddle2: 'leftMiddleIntermediate',
  mixamorigLeftHandMiddle3: 'leftMiddleDistal',
  mixamorigLeftHandRing1: 'leftRingProximal',
  mixamorigLeftHandRing2: 'leftRingIntermediate',
  mixamorigLeftHandRing3: 'leftRingDistal',
  mixamorigLeftHandPinky1: 'leftLittleProximal',
  mixamorigLeftHandPinky2: 'leftLittleIntermediate',
  mixamorigLeftHandPinky3: 'leftLittleDistal',

  // Fingers — Right
  mixamorigRightHandThumb1: 'rightThumbMetacarpal',
  mixamorigRightHandThumb2: 'rightThumbProximal',
  mixamorigRightHandThumb3: 'rightThumbDistal',
  mixamorigRightHandIndex1: 'rightIndexProximal',
  mixamorigRightHandIndex2: 'rightIndexIntermediate',
  mixamorigRightHandIndex3: 'rightIndexDistal',
  mixamorigRightHandMiddle1: 'rightMiddleProximal',
  mixamorigRightHandMiddle2: 'rightMiddleIntermediate',
  mixamorigRightHandMiddle3: 'rightMiddleDistal',
  mixamorigRightHandRing1: 'rightRingProximal',
  mixamorigRightHandRing2: 'rightRingIntermediate',
  mixamorigRightHandRing3: 'rightRingDistal',
  mixamorigRightHandPinky1: 'rightLittleProximal',
  mixamorigRightHandPinky2: 'rightLittleIntermediate',
  mixamorigRightHandPinky3: 'rightLittleDistal',
}

/* ────────────────────────── Main Entry ───────────────────────────── */

/**
 * Load a Mixamo FBX animation and retarget it onto a VRM model.
 *
 * Based on the official @pixiv/three-vrm Mixamo retargeting approach:
 *   1. Load FBX, extract the first AnimationClip
 *   2. For each track, map the Mixamo bone name → VRM bone name
 *   3. Re-write each track to target the VRM normalized bone node
 *   4. Apply rest-pose correction (Mixamo rest → VRM rest)
 *   5. Apply position-delta for root motion (hips)
 */
export async function loadMixamoAnimation(
  fbxUrl: string,
  vrm: VRM,
): Promise<THREE.AnimationClip | null> {
  const loader = new FBXLoader()
  const fbx = await loader.loadAsync(fbxUrl)

  const clip = fbx.animations[0]
  if (!clip) {
    console.warn('[loadMixamoAnimation] No animation clip found in FBX file.')
    return null
  }

  // Build a map of Mixamo bone name → THREE.Bone (from the FBX skeleton)
  const mixamoBoneByName = new Map<string, THREE.Object3D>()
  const mixamoWorldRests = new Map<string, { worldRest: THREE.Quaternion; worldRestInv: THREE.Quaternion }>()
  
  // Ensure matrices are updated before getting world quaternions
  fbx.updateMatrixWorld(true)
  
  fbx.traverse((obj) => {
    if ((obj as THREE.Bone).isBone) {
      mixamoBoneByName.set(obj.name, obj)
      
      const worldRest = new THREE.Quaternion()
      obj.getWorldQuaternion(worldRest)
      mixamoWorldRests.set(obj.name, {
        worldRest,
        worldRestInv: worldRest.clone().invert()
      })
    }
  })

  const tracks: THREE.KeyframeTrack[] = []

  for (const track of clip.tracks) {
    // Track names look like "mixamorigHips.quaternion" or "mixamorigHips.position"
    const dotIndex = track.name.lastIndexOf('.')
    if (dotIndex === -1) continue

    const mixamoBoneName = track.name.substring(0, dotIndex)
    const property = track.name.substring(dotIndex + 1)

    // Resolve to VRM bone name
    const vrmBoneName = MIXAMO_TO_VRM_BONE[mixamoBoneName]
    if (!vrmBoneName) continue

    // Get VRM normalized bone node
    const vrmBoneNode = vrm.humanoid?.getNormalizedBoneNode(vrmBoneName as VRMHumanBoneName)
    if (!vrmBoneNode) continue

    // Ensure the node has its VRM bone name set
    if (vrmBoneNode.name !== vrmBoneName) {
      vrmBoneNode.name = vrmBoneName
    }

    const nodePath = vrmBoneNode.name || getNodePath(vrm.scene, vrmBoneNode)
    if (!nodePath) continue

    // ─── Quaternion tracks ───
    if (property === 'quaternion') {
      const mixamoBone = mixamoBoneByName.get(mixamoBoneName)
      const mixamoRestInv = mixamoBone?.quaternion.clone().invert() ?? new THREE.Quaternion()
      
      const worldRestData = mixamoWorldRests.get(mixamoBoneName)
      if (!worldRestData) continue
      const { worldRest, worldRestInv } = worldRestData

      const vrmRestPose = (vrm.scene.userData.restPoses as Map<string, any>)?.get(vrmBoneName)
      const vrmRest = vrmRestPose ? vrmRestPose.quaternion.clone() : vrmBoneNode.quaternion.clone()

      const retargetedValues = new Float32Array(track.values.length)
      const srcQuat = new THREE.Quaternion()
      const deltaLocal = new THREE.Quaternion()
      const deltaWorld = new THREE.Quaternion()
      const outQuat = new THREE.Quaternion()

      for (let i = 0; i < track.values.length; i += 4) {
        srcQuat.fromArray(track.values as ArrayLike<number>, i)

        // 1. Calculate local rotation delta in Mixamo space
        deltaLocal.copy(mixamoRestInv).multiply(srcQuat)
        
        // 2. Convert to Mixamo World Space
        deltaWorld.copy(worldRest).multiply(deltaLocal).multiply(worldRestInv)
        
        // 3. Convert to VRM Space (Rotate 180 degrees around Y because VRM faces opposite)
        // flip180Y(q) = (-q.x, q.y, -q.z, q.w)
        deltaWorld.set(-deltaWorld.x, deltaWorld.y, -deltaWorld.z, deltaWorld.w).normalize()

        // 4. Apply to VRM Rest Pose
        outQuat.copy(vrmRest).multiply(deltaWorld)
        outQuat.normalize()

        outQuat.toArray(retargetedValues, i)
      }

      tracks.push(
        new THREE.QuaternionKeyframeTrack(
          `${nodePath}.quaternion`,
          Array.from(track.times),
          Array.from(retargetedValues),
        ),
      )
    }

    // ─── Position tracks (root motion — hips only) ───
    if (property === 'position' && vrmBoneName === 'hips') {
      const retargetedValues = new Float32Array(track.values.length)
      
      const vrmRestPose = (vrm.scene.userData.restPoses as Map<string, any>)?.get(vrmBoneName)
      const vrmRestPos = vrmRestPose ? vrmRestPose.position.clone() : vrmBoneNode.position.clone()

      // Mixamo FBX positions are in centimeters
      const scale = 0.01

      // First-frame reference for delta calculation
      const startX = track.values[0]
      const startY = track.values[1]
      const startZ = track.values[2]

      for (let i = 0; i < track.values.length; i += 3) {
        const dx = (track.values[i] - startX) * scale
        const dy = (track.values[i + 1] - startY) * scale
        const dz = (track.values[i + 2] - startZ) * scale

        // Mixamo faces opposite to VRM, so we flip X and Z (180deg around Y)
        retargetedValues[i] = vrmRestPos.x - dx
        retargetedValues[i + 1] = vrmRestPos.y + dy
        retargetedValues[i + 2] = vrmRestPos.z - dz
      }

      tracks.push(
        new THREE.VectorKeyframeTrack(
          `${nodePath}.position`,
          Array.from(track.times),
          Array.from(retargetedValues),
        ),
      )
    }
  }

  if (tracks.length === 0) {
    console.warn('[loadMixamoAnimation] No matching bones found between FBX and VRM model.')
    return null
  }

  console.log(
    `[loadMixamoAnimation] Retargeted ${tracks.length} tracks, duration ${clip.duration.toFixed(2)}s`,
  )

  return new THREE.AnimationClip('mixamo-retargeted', clip.duration, tracks)
}

/* ────────────────────────── Helpers ──────────────────────────────── */

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
