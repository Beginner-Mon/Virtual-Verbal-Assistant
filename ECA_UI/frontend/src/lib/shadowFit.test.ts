import { describe, expect, it } from 'vitest'
import * as THREE from 'three'
import type { VRM } from '@pixiv/three-vrm'
import { DEFAULT_SHADOW_FIT, ShadowCameraFitter } from './shadowFit'

/** Matches the authored placement in config/environmentConfig.ts (Z-up scene). */
const LIGHT_POSITION = new THREE.Vector3(0, 3, 8)

function makeLight() {
  const light = new THREE.DirectionalLight(0xffffff, 1)
  light.position.copy(LIGHT_POSITION)
  light.target.position.set(0, 0, 0)
  light.castShadow = true
  light.shadow.mapSize.set(1024, 1024)
  return light
}

/**
 * A skeleton whose joints sit at given world positions, under a group so the
 * whole rig can be moved and rotated the way a character actually is.
 */
function makeRig(points: [number, number, number][]) {
  const group = new THREE.Group()
  const nodes = points.map(([x, y, z]) => {
    const bone = new THREE.Object3D()
    bone.position.set(x, y, z)
    group.add(bone)
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

  return { group, nodes, vrm }
}

/** A standing figure: feet on the floor, head at 1.6 m, arms out to the sides. */
const STANDING: [number, number, number][] = [
  [-0.12, 0, 0], [0.12, 0, 0], // feet
  [0, 0, 0.9],                 // hips
  [-0.35, 0, 1.3], [0.35, 0, 1.3], // hands
  [0, 0, 1.6],                 // head
]

/**
 * Reproduces how three.js aims the shadow camera at render time: it places the
 * camera at the light and looks at the light's target. Testing against a
 * hand-rolled frustum instead would let the fitter's own basis error cancel out.
 */
function viewMatrixOf(light: THREE.DirectionalLight): THREE.Matrix4 {
  const camera = light.shadow.camera as THREE.OrthographicCamera
  camera.position.copy(light.position)
  camera.lookAt(light.target.position)
  camera.updateMatrixWorld()
  return new THREE.Matrix4().copy(camera.matrixWorld).invert()
}

function containment(light: THREE.DirectionalLight, world: THREE.Vector3) {
  const camera = light.shadow.camera as THREE.OrthographicCamera
  const p = world.clone().applyMatrix4(viewMatrixOf(light))
  const depth = -p.z // the camera looks down its own -Z
  return (
    p.x >= camera.left && p.x <= camera.right &&
    p.y >= camera.bottom && p.y <= camera.top &&
    depth >= camera.near && depth <= camera.far
  )
}

/** Every joint, plus where each one's shadow lands on the floor. */
function subjectPoints(nodes: THREE.Object3D[], direction: THREE.Vector3, groundZ = 0) {
  const points: THREE.Vector3[] = []
  for (const node of nodes) {
    const world = new THREE.Vector3()
    node.getWorldPosition(world)
    points.push(world)

    const t = (groundZ - world.z) / direction.z
    if (t > 0) points.push(direction.clone().multiplyScalar(t).add(world))
  }
  return points
}

const lightDirection = () =>
  new THREE.Vector3().copy(new THREE.Vector3(0, 0, 0)).sub(LIGHT_POSITION).normalize()

describe('ShadowCameraFitter', () => {
  it('covers the ground projection, not just the body', () => {
    // The failure this module exists to prevent: with this light angle the
    // head's shadow lands ~2.3 units from the character, so a frustum fitted to
    // the skeleton alone clips the shadow while the body is fully inside it.
    const light = makeLight()
    const { nodes, vrm } = makeRig(STANDING)
    new ShadowCameraFitter(light).update(vrm, 16)

    const direction = lightDirection()
    const projections = subjectPoints(nodes, direction).filter((p) => Math.abs(p.z) < 1e-6)

    expect(projections.length).toBeGreaterThan(0)
    for (const point of projections) {
      expect(containment(light, point)).toBe(true)
    }
  })

  it('covers every joint', () => {
    const light = makeLight()
    const { nodes, vrm } = makeRig(STANDING)
    new ShadowCameraFitter(light).update(vrm, 16)

    for (const node of nodes) {
      const world = new THREE.Vector3()
      node.getWorldPosition(world)
      expect(containment(light, world)).toBe(true)
    }
  })

  it('barely changes size as the character turns, and never clips', () => {
    // The sphere is what keeps this stable, but not perfectly: it is derived
    // from a WORLD-axis-aligned box, and that box does breathe a little as the
    // rig turns (an arm span swaps from the X extent to the Y extent, and the
    // ground projections move with it). Measured spread over a full turn is
    // ~5% of the radius. A light-space box would swing far wider AND change the
    // texel size with it, which is the crawl this avoids.
    //
    // The hard requirement is the second assertion: whatever the size does, the
    // subject stays inside at every angle.
    const { group, nodes, vrm } = makeRig(STANDING)
    const radii: number[] = []

    for (const angle of [0, Math.PI / 4, Math.PI / 2, Math.PI, (3 * Math.PI) / 2]) {
      const light = makeLight()
      group.rotation.z = angle
      group.updateMatrixWorld(true)
      new ShadowCameraFitter(light).update(vrm, 16)
      radii.push((light.shadow.camera as THREE.OrthographicCamera).right)

      for (const point of subjectPoints(nodes, lightDirection())) {
        expect(containment(light, point)).toBe(true)
      }
    }

    const mean = radii.reduce((a, b) => a + b, 0) / radii.length
    const spread = Math.max(...radii) - Math.min(...radii)
    expect(spread / mean).toBeLessThan(0.1)
  })

  it('is sharper than the fixed 2.5-unit frustum it replaced', () => {
    // Same mapSize, so this is texel density bought for free. The module's
    // measured range was 1.4-2.0x; anything at or below the old size means the
    // fit regressed into padding the frustum again.
    const light = makeLight()
    const { vrm } = makeRig(STANDING)
    new ShadowCameraFitter(light).update(vrm, 16)

    const radius = (light.shadow.camera as THREE.OrthographicCamera).right
    expect(radius).toBeLessThan(2.5)
    expect(radius).toBeGreaterThan(0)
  })

  it('does not move the light DIRECTION', () => {
    // MToon shades from NdotL. If fitting rotated the light, the character's
    // shading would drift as it walked — a visible bug with no obvious cause.
    const light = makeLight()
    const { group, vrm } = makeRig(STANDING)
    const fitter = new ShadowCameraFitter(light)

    const before = new THREE.Vector3()
      .copy(light.target.position).sub(light.position).normalize()

    fitter.update(vrm, 16)
    group.position.set(1.5, -0.8, 0)
    group.updateMatrixWorld(true)
    fitter.update(vrm, 16)

    const after = new THREE.Vector3()
      .copy(light.target.position).sub(light.position).normalize()
    expect(after.angleTo(before)).toBeLessThan(1e-6)
  })

  it('follows the subject when it walks away from the origin', () => {
    const light = makeLight()
    const { group, nodes, vrm } = makeRig(STANDING)
    const fitter = new ShadowCameraFitter(light)

    group.position.set(3, 2, 0)
    group.updateMatrixWorld(true)
    fitter.update(vrm, 16)

    for (const point of subjectPoints(nodes, lightDirection())) {
      expect(containment(light, point)).toBe(true)
    }
  })

  it('snaps the frustum centre to whole shadow texels', () => {
    // Residual crawl comes from the frustum SLIDING sub-texel with the
    // character. Snapping removes it; without the snap this offset lands
    // anywhere.
    const light = makeLight()
    const { group, vrm } = makeRig(STANDING)
    const fitter = new ShadowCameraFitter(light)

    group.position.set(0.0137, 0.0091, 0) // deliberately not a texel multiple
    group.updateMatrixWorld(true)
    fitter.update(vrm, 16)

    const camera = light.shadow.camera as THREE.OrthographicCamera
    const texel = (camera.right * 2) / light.shadow.mapSize.x

    // Express the centre in the same light-space basis the fitter snapped in.
    const centre = light.target.position.clone().applyMatrix4(viewMatrixOf(light))
    for (const axis of [centre.x, centre.y]) {
      const offset = Math.abs(axis / texel - Math.round(axis / texel))
      expect(offset).toBeLessThan(1e-4)
    }
  })

  it('keeps the subject between near and far', () => {
    const light = makeLight()
    const { nodes, vrm } = makeRig(STANDING)
    new ShadowCameraFitter(light).update(vrm, 16)

    const camera = light.shadow.camera as THREE.OrthographicCamera
    const view = viewMatrixOf(light)
    for (const point of subjectPoints(nodes, lightDirection())) {
      const depth = -point.clone().applyMatrix4(view).z
      expect(depth).toBeGreaterThanOrEqual(camera.near)
      expect(depth).toBeLessThanOrEqual(camera.far)
    }
  })

  it('grows the frustum by the padding, which covers hair and skirt', () => {
    const { vrm } = makeRig(STANDING)

    const tight = makeLight()
    new ShadowCameraFitter(tight, { ...DEFAULT_SHADOW_FIT, padding: 0 }).update(vrm, 16)

    const padded = makeLight()
    new ShadowCameraFitter(padded, { ...DEFAULT_SHADOW_FIT, padding: 0.5 }).update(vrm, 16)

    const tightR = (tight.shadow.camera as THREE.OrthographicCamera).right
    const paddedR = (padded.shadow.camera as THREE.OrthographicCamera).right
    expect(paddedR - tightR).toBeCloseTo(0.5, 6)
  })

  it('leaves the camera untouched when there is no model', () => {
    const light = makeLight()
    const camera = light.shadow.camera as THREE.OrthographicCamera
    const before = camera.right

    new ShadowCameraFitter(light).update(null, 16)

    expect(camera.right).toBe(before)
  })

  it('honours intervalMs throttling', () => {
    // The default is 0 on purpose (see the options doc), but the throttle must
    // still work for anyone who turns it on.
    const light = makeLight()
    const { group, vrm } = makeRig(STANDING)
    const fitter = new ShadowCameraFitter(light, { ...DEFAULT_SHADOW_FIT, intervalMs: 100 })

    fitter.update(vrm, 1000) // first call always runs
    const settled = light.target.position.clone()

    group.position.set(4, 0, 0)
    group.updateMatrixWorld(true)
    fitter.update(vrm, 10) // under the interval — must be skipped

    expect(light.target.position.distanceTo(settled)).toBe(0)
  })
})
