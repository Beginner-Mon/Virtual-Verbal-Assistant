/**
 * Fits the directional light's shadow camera to the subject, every frame.
 *
 * WHY, measured on the pre-fix scene (1024² shadow map):
 *
 *   frustum          5.00 × 5.00, centred on the world origin
 *   subject (bones)  occupies 7-8% of that width
 *   bottom margin    0.43 units — and the subject translates >1.2 units
 *                    during a motion clip
 *   texel            0.49 cm
 *
 * So ~92% of the shadow map was spent on empty space while the character sat
 * against the edge of the frustum — which is exactly how a shadow ends up sliced
 * off in a dead straight line. Enlarging `cameraSize` "fixes" the clipping by
 * wasting even more of the map and making the shadow blurrier.
 *
 * Fitting instead gives BOTH: the frustum can never clip (it is derived from the
 * subject) and texel density goes UP at the same `mapSize`, so shadows get
 * sharper for free.
 *
 * Two details that are easy to get wrong:
 *
 *  1. **The ground projection must be inside the frustum, not just the body.**
 *     The shadow cast onto the floor lands far from the character — with this
 *     light angle the head's shadow is ~2.3 units away. Fitting to the body
 *     alone clips the shadow even though the body is fully covered.
 *
 *  2. **Fit a SPHERE, not a light-space box.** A sphere is rotation-invariant,
 *     so the frustum size does not change as the character turns. A box would
 *     resize every frame and make the shadow edge crawl.
 *
 * Residual crawl from the frustum *moving* is removed by snapping its centre to
 * whole shadow-map texels.
 *
 * MEASURED (2 models × 3 generated clips × 22 samples each, counting points —
 * joints and their floor projections — that fall outside the frustum):
 *
 *                     clipped frames   texel size
 *   fixed 2.5 box        0 / 22         0.488 cm
 *   fitted               0 / 22         0.245-0.359 cm   ← 1.4-2.0× sharper
 *
 * Same `mapSize`, so the GPU cost is unchanged. Note neither config clipped on
 * the CURRENT assets — the fixed box was simply close to it (0.43 units of
 * margin while the subject drifts >1.2 units per clip), so any taller model or
 * clip with more root translation would cross it.
 *
 * KNOWN LIMIT: this runs in SceneLighting's `useFrame`, which is registered
 * before the one that advances the animation mixer, so the fit is one frame
 * behind the pose. `padding` absorbs normal motion; only a clip *switch* (the
 * character teleports to a new first frame) can exceed it, for a single frame.
 */

import * as THREE from 'three'
import type { VRM } from '@pixiv/three-vrm'

export interface ShadowFitOptions {
  /** Metres added around the skeleton to cover mesh the bones don't reach:
   *  hair, skirt, fingertips, shoes. */
  padding: number
  /** World height of the shadow-receiving floor (the ground plane sits at z=0
   *  in this scene's Z-up world). The subject's silhouette is projected onto
   *  this plane and included in the fit. */
  groundZ: number
  /** Distance kept between the light and the subject centre. Only affects depth
   *  range — a directional light's shading depends on direction alone. */
  standoff: number
  /** Recompute at most this often, ms. Keep at 0: throttling lets the subject
   *  drift past the frustum between updates — measured as a 0.06-unit overshoot
   *  on one frame at 100ms. Reading ~50 bone positions costs microseconds, far
   *  less than the clipping it prevents. */
  intervalMs: number
}

export const DEFAULT_SHADOW_FIT: ShadowFitOptions = {
  padding: 0.18,
  groundZ: 0,
  standoff: 2,
  intervalMs: 0,
}

export class ShadowCameraFitter {
  private readonly light: THREE.DirectionalLight
  private readonly options: ShadowFitOptions
  /** Constant light DIRECTION. Recomputing it from a moving light would change
   *  MToon's NdotL shading as the character walks — the look must not drift. */
  private readonly direction = new THREE.Vector3()
  /** Rotation-only basis of the light's view, used for texel snapping. */
  private readonly lightBasis = new THREE.Matrix4()
  private readonly worldFromLight = new THREE.Matrix4()

  private bones: THREE.Object3D[] = []
  private boneSource: VRM | null = null
  private sinceUpdate = Infinity

  private readonly min = new THREE.Vector3()
  private readonly max = new THREE.Vector3()
  private readonly centre = new THREE.Vector3()
  private readonly scratch = new THREE.Vector3()
  private readonly projected = new THREE.Vector3()

  constructor(light: THREE.DirectionalLight, options: ShadowFitOptions = DEFAULT_SHADOW_FIT) {
    this.light = light
    this.options = options

    // Direction the light travels, captured once from its authored placement.
    this.direction.copy(light.target.position).sub(light.position).normalize()

    const eye = this.direction.clone().multiplyScalar(-1)
    this.lightBasis.lookAt(eye, new THREE.Vector3(), light.shadow.camera.up)
    this.worldFromLight.copy(this.lightBasis)
    this.lightBasis.invert() // world → light (rotation only, so invert == transpose)
  }

  /** Call once per frame; internally throttled to `intervalMs`. */
  update(vrm: VRM | null, deltaMs: number): void {
    this.sinceUpdate += deltaMs
    if (this.sinceUpdate < this.options.intervalMs) return
    this.sinceUpdate = 0
    if (!vrm) return

    if (this.boneSource !== vrm) {
      this.bones = collectBones(vrm)
      this.boneSource = vrm
    }
    if (this.bones.length === 0) return

    this.computeBounds()
    this.apply()
  }

  /** Bounds over the skeleton PLUS its silhouette projected onto the floor. */
  private computeBounds(): void {
    const { groundZ } = this.options
    this.min.set(Infinity, Infinity, Infinity)
    this.max.set(-Infinity, -Infinity, -Infinity)

    const dirZ = this.direction.z
    for (const bone of this.bones) {
      bone.getWorldPosition(this.scratch)
      this.min.min(this.scratch)
      this.max.max(this.scratch)

      // Where this joint's shadow lands. Skip when the light is horizontal
      // (never hits the floor) or the joint is already below it.
      if (dirZ >= -1e-4) continue
      const t = (groundZ - this.scratch.z) / dirZ
      if (t <= 0) continue
      this.projected.copy(this.direction).multiplyScalar(t).add(this.scratch)
      this.min.min(this.projected)
      this.max.max(this.projected)
    }
  }

  private apply(): void {
    const { padding, standoff } = this.options
    const light = this.light
    const camera = light.shadow.camera as THREE.OrthographicCamera

    this.centre.addVectors(this.min, this.max).multiplyScalar(0.5)
    // Bounding sphere of the box: rotation-invariant, so turning the character
    // never resizes the frustum.
    const radius = this.scratch.subVectors(this.max, this.min).length() * 0.5 + padding
    if (!Number.isFinite(radius) || radius <= 0) return

    // Snap the centre to whole shadow texels, otherwise the shadow edge crawls
    // as the frustum slides with the character.
    const texel = (radius * 2) / light.shadow.mapSize.x
    this.centre.applyMatrix4(this.lightBasis)
    this.centre.x = Math.round(this.centre.x / texel) * texel
    this.centre.y = Math.round(this.centre.y / texel) * texel
    this.centre.applyMatrix4(this.worldFromLight)

    const distance = radius + standoff
    light.target.position.copy(this.centre)
    light.position.copy(this.centre).addScaledVector(this.direction, -distance)
    // `light.target` is not part of the scene graph, so three.js never refreshes
    // it — without this the shadow camera keeps aiming at the old spot.
    light.target.updateMatrixWorld()
    light.updateMatrixWorld()

    camera.left = -radius
    camera.right = radius
    camera.top = radius
    camera.bottom = -radius
    camera.near = Math.max(0.01, distance - radius)
    camera.far = distance + radius
    camera.updateProjectionMatrix()
  }
}

/** Humanoid bones drive the skinned mesh, so they bound it closely enough —
 *  far cheaper than a per-frame Box3 over skinned geometry, which is also wrong
 *  unless the skinning is re-evaluated on the CPU. */
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
