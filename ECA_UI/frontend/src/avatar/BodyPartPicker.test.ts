import { describe, expect, it } from 'vitest'
import * as THREE from 'three'
import { splitBySide } from './BodyPartPicker'

/**
 * Covers the one piece of BodyPartPicker that is pure geometry: deciding which
 * of two eyes a blink-region vertex belongs to.
 *
 * Everything else in that file needs a live WebGL context — a render target, a
 * shader link, a pixel readback — so it is verified in the browser instead.
 *
 * This split only runs for models that ship a combined `blink` and no
 * `blinkLeft`/`blinkRight`. Authored sides always win; this is the fallback.
 */

/** Two clusters of eye vertices, offset from the origin so a naive "split at
 *  zero" would get it wrong and the centroid-based split gets it right. */
function eyePair(axis: THREE.Vector3, centre: THREE.Vector3, separation: number) {
  const left: THREE.Vector3[] = []
  const right: THREE.Vector3[] = []
  for (let i = 0; i < 8; i++) {
    const jitter = new THREE.Vector3((i % 3) * 0.001, (i % 2) * 0.001, 0)
    left.push(centre.clone().addScaledVector(axis, separation).add(jitter))
    right.push(centre.clone().addScaledVector(axis, -separation).add(jitter))
  }
  return { left, right }
}

describe('splitBySide', () => {
  it('separates two clusters along the axis', () => {
    const axis = new THREE.Vector3(1, 0, 0)
    const { left, right } = eyePair(axis, new THREE.Vector3(0, 1.4, 0.1), 0.03)

    const sides = splitBySide([...left, ...right], axis)

    expect(sides.slice(0, left.length).every(Boolean)).toBe(true)
    expect(sides.slice(left.length).some(Boolean)).toBe(false)
  })

  it('cuts at the set centroid, not at the origin', () => {
    // A face authored well off-centre: every eye vertex has x > 0, so a split at
    // x = 0 would call the whole region "left".
    const axis = new THREE.Vector3(1, 0, 0)
    const { left, right } = eyePair(axis, new THREE.Vector3(5, 1.4, 0), 0.03)

    const sides = splitBySide([...left, ...right], axis)

    expect(sides.filter(Boolean)).toHaveLength(left.length)
  })

  it('follows the axis direction — flipping it swaps the sides', () => {
    const axis = new THREE.Vector3(1, 0, 0)
    const { left, right } = eyePair(axis, new THREE.Vector3(0, 1.4, 0), 0.03)
    const points = [...left, ...right]

    const sides = splitBySide(points, axis)
    const flipped = splitBySide(points, axis.clone().negate())

    expect(flipped).toEqual(sides.map((side) => !side))
  })

  it('works on an axis that is not world-aligned', () => {
    // The lateral axis comes from two bind-pose bone positions, so it is only
    // approximately a world axis on a model authored at an angle.
    const axis = new THREE.Vector3(0.9, 0.1, -0.42).normalize()
    const { left, right } = eyePair(axis, new THREE.Vector3(0.2, 1.4, 0.05), 0.03)

    const sides = splitBySide([...left, ...right], axis)

    expect(sides.slice(0, left.length).every(Boolean)).toBe(true)
    expect(sides.slice(left.length).some(Boolean)).toBe(false)
  })

  it('returns nothing for an empty region', () => {
    expect(splitBySide([], new THREE.Vector3(1, 0, 0))).toEqual([])
  })
})
