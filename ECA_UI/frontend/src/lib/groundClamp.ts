/**
 * Keeps the character from sinking through the floor.
 *
 * WHY THIS LIVES IN THE FRONTEND, not in the BVH or the converter:
 *
 * `retargetBVHToVRM` anchors frame 0 of a clip to the VRM's rest hip height and
 * then applies the clip's per-frame DELTAS (see bvhToVrm.ts — `dy = values[i] -
 * bvhStartY`). Absolute height in the source file is discarded, so "grounding"
 * the BVH has no effect here at all. Whatever the clip descends, the character
 * descends — from a standing start.
 *
 * Measured on the shipped clips (vertical axis is the BVH Z channel, because
 * these files are read with `swapYandZ`):
 *
 *   motion_7b4b8d9e   descends   1.5 cm from frame 0   → fine
 *   motion_a658       descends   1.7 cm                → fine
 *   motion_b28e8284   descends 120.7 cm                → ~1.04 m BELOW the floor
 *
 * That last one is what made the shadow vanish: the ground plane at z=0 is the
 * shadow RECEIVER, so any part of the body under it stops casting onto its top
 * face. The silhouette gets sliced along the floor line, and once most of the
 * body is under, there is no shadow left at all.
 *
 * A clamp (never push down, only lift) is the right shape for this:
 *   - a jump still leaves the ground, because that moves the character UP;
 *   - a fall settles on the floor instead of passing through it;
 *   - it needs no re-conversion, so clips streamed from Kimodo at runtime are
 *     covered too.
 *
 * The alternative — computing one offset per clip at load time — preserves the
 * vertical curve exactly, but needs full FK over the clip and still cannot help
 * a clip whose descent exceeds the rest hip height. Not worth it while the
 * clamp is this cheap.
 */

import type * as THREE from 'three'
import type { VRM } from '@pixiv/three-vrm'

export interface GroundClampOptions {
  /** World height of the floor. The scene is Z-up: the ground plane sits at 0. */
  groundZ: number
  /** Ignore lift changes smaller than this, so the transform isn't rewritten
   *  every frame over floating-point noise. */
  epsilon: number
}

export const DEFAULT_GROUND_CLAMP: GroundClampOptions = {
  groundZ: 0,
  epsilon: 0.001,
}

export class GroundClamp {
  private readonly target: THREE.Object3D
  private readonly options: GroundClampOptions
  /** The target's authored height, captured once; the lift is added to it. */
  private readonly baseZ: number
  private lift = 0

  private bones: THREE.Object3D[] = []
  private boneSource: VRM | null = null
  private readonly scratch: THREE.Vector3

  constructor(target: THREE.Object3D, scratch: THREE.Vector3, options: GroundClampOptions = DEFAULT_GROUND_CLAMP) {
    this.target = target
    this.options = options
    this.scratch = scratch
    this.baseZ = target.position.z
  }

  /** How far the character is currently being held up, in metres. */
  get currentLift(): number {
    return this.lift
  }

  /** Call once per frame, AFTER the animation mixer has advanced the pose. */
  update(vrm: VRM | null): void {
    if (!vrm) return

    if (this.boneSource !== vrm) {
      this.bones = collectBones(vrm)
      this.boneSource = vrm
      this.lift = 0
      this.target.position.z = this.baseZ
    }
    if (this.bones.length === 0) return

    let lowest = Infinity
    for (const bone of this.bones) {
      bone.getWorldPosition(this.scratch)
      if (this.scratch.z < lowest) lowest = this.scratch.z
    }
    if (!Number.isFinite(lowest)) return

    // The lift already in effect is baked into what we just measured, so remove
    // it to recover the pose's own height before deciding the new lift.
    const rawLowest = lowest - this.lift
    // Lift exactly onto the floor — subtracting a tolerance here would park the
    // character permanently that far under it.
    const wanted = Math.max(0, this.options.groundZ - rawLowest)
    if (Math.abs(wanted - this.lift) < this.options.epsilon) return

    this.lift = wanted
    this.target.position.z = this.baseZ + wanted
  }
}

function collectBones(vrm: VRM): THREE.Object3D[] {
  const humanoid = vrm.humanoid
  if (!humanoid) return []
  const bones: THREE.Object3D[] = []
  for (const name of Object.keys(humanoid.humanBones)) {
    const node =
      humanoid.getNormalizedBoneNode(name as never) ?? humanoid.getRawBoneNode(name as never)
    if (node) bones.push(node)
  }
  return bones
}
