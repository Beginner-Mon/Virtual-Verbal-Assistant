import { describe, expect, it } from 'vitest'
import * as THREE from 'three'
import type { VRM } from '@pixiv/three-vrm'
import { DEFAULT_GROUND_CLAMP, GroundClamp } from './groundClamp'

/**
 * The bones are real children of the target, which is the property the whole
 * class depends on: lifting the target moves the bones, so the next frame
 * measures a pose that already contains the previous lift. A fake whose bones
 * sit outside the target would make the `rawLowest = lowest - lift` correction
 * untestable — and that line is the only non-obvious one in the file.
 */
function makeRig(boneZ: number[], baseZ = 0) {
  const target = new THREE.Object3D()
  target.position.z = baseZ

  const nodes = boneZ.map((z, i) => {
    const bone = new THREE.Object3D()
    bone.position.set(i * 0.1, 0, z)
    target.add(bone)
    return bone
  })

  const humanBones = Object.fromEntries(nodes.map((_, i) => [`bone${i}`, {}]))
  const vrm = {
    humanoid: {
      humanBones,
      getNormalizedBoneNode: (name: string) => nodes[Number(name.replace('bone', ''))] ?? null,
      getRawBoneNode: () => null,
    },
  } as unknown as VRM

  return { target, nodes, vrm }
}

const clampFor = (target: THREE.Object3D) =>
  new GroundClamp(target, new THREE.Vector3(), DEFAULT_GROUND_CLAMP)

describe('GroundClamp', () => {
  it('lifts a pose that sinks below the floor back onto it', () => {
    // The motion_b28e8284 case from the module comment: ~1.04 m under the floor.
    const { target, vrm } = makeRig([-1.04, 0.5, 1.6])
    const clamp = clampFor(target)

    clamp.update(vrm)

    expect(clamp.currentLift).toBeCloseTo(1.04, 6)
    expect(target.position.z).toBeCloseTo(1.04, 6)
  })

  it('leaves a pose that is already above the floor alone', () => {
    const { target, vrm } = makeRig([0.2, 0.9, 1.7])
    const clamp = clampFor(target)

    clamp.update(vrm)

    // Pushing DOWN onto the floor would kill every jump in the animation set.
    expect(clamp.currentLift).toBe(0)
    expect(target.position.z).toBe(0)
  })

  it('holds a constant lift instead of drifting upward frame after frame', () => {
    // Regression guard for the lift-compensation line. Without it, each frame
    // re-measures its own lift as part of the pose and adds it again: 1.04,
    // 2.08, 3.12 — the character climbs out of the scene in under a second.
    const { target, vrm } = makeRig([-1.04, 0.5])
    const clamp = clampFor(target)

    for (let frame = 0; frame < 30; frame++) clamp.update(vrm)

    expect(clamp.currentLift).toBeCloseTo(1.04, 6)
    expect(target.position.z).toBeCloseTo(1.04, 6)
  })

  it('follows the pose down and back up across frames', () => {
    const { target, nodes, vrm } = makeRig([-0.5, 1.0])
    const clamp = clampFor(target)

    clamp.update(vrm)
    expect(clamp.currentLift).toBeCloseTo(0.5, 6)

    // The clip descends further.
    nodes[0].position.z = -1.2
    clamp.update(vrm)
    expect(clamp.currentLift).toBeCloseTo(1.2, 6)

    // ...then rises clear of the floor: the lift must be released, not kept.
    nodes[0].position.z = 0.3
    clamp.update(vrm)
    expect(clamp.currentLift).toBe(0)
    expect(target.position.z).toBe(0)
  })

  it('ignores sub-epsilon changes so the transform is not rewritten on noise', () => {
    const { target, nodes, vrm } = makeRig([-1.0, 0.5])
    const clamp = clampFor(target)
    clamp.update(vrm)

    const settled = target.position.z
    nodes[0].position.z = -1.0 + DEFAULT_GROUND_CLAMP.epsilon / 2
    clamp.update(vrm)

    expect(target.position.z).toBe(settled)
  })

  it('adds the lift to the authored height instead of overwriting it', () => {
    // Authored 0.25 up, lowest bone 0.4 below that ⇒ world -0.15, needs 0.15.
    const { target, nodes, vrm } = makeRig([-0.4, 1.0], 0.25)
    const clamp = clampFor(target)

    clamp.update(vrm)

    expect(clamp.currentLift).toBeCloseTo(0.15, 6)
    expect(target.position.z).toBeCloseTo(0.4, 6)

    // The property that actually matters, and the one that does not depend on
    // how the rig was authored: the lowest joint ends up ON the floor, not
    // under it and not hovering above it.
    const lowest = new THREE.Vector3()
    nodes[0].getWorldPosition(lowest)
    expect(lowest.z).toBeCloseTo(DEFAULT_GROUND_CLAMP.groundZ, 6)
  })

  it('resets the lift when the model is swapped', () => {
    const { target, vrm } = makeRig([-1.04, 0.5])
    const clamp = clampFor(target)
    clamp.update(vrm)
    expect(clamp.currentLift).toBeCloseTo(1.04, 6)

    // A different VRM whose pose needs no lift. Carrying the old lift over
    // would leave the new model floating a metre in the air.
    const second = makeRig([0.4, 1.2])
    second.nodes.forEach((bone) => target.add(bone))
    clamp.update(second.vrm)

    expect(clamp.currentLift).toBe(0)
    expect(target.position.z).toBe(0)
  })

  it('does nothing when there is no model', () => {
    const { target } = makeRig([-1.0])
    const clamp = clampFor(target)

    clamp.update(null)

    expect(clamp.currentLift).toBe(0)
    expect(target.position.z).toBe(0)
  })

  it('does nothing for a VRM with no humanoid bones', () => {
    const { target } = makeRig([-1.0])
    const clamp = clampFor(target)

    clamp.update({ humanoid: null } as unknown as VRM)

    expect(clamp.currentLift).toBe(0)
  })
})
