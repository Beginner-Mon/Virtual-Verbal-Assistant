import * as THREE from 'three'
import { VRMExpressionMorphTargetBind, VRMHumanBoneName } from '@pixiv/three-vrm'
import type { VRM } from '@pixiv/three-vrm'
import {
  BODY_PARTS,
  PART_BY_ID,
  buildBoneToPartMap,
  partId,
  resolvePart,
  type BodyPart,
} from './bodyParts'

/**
 * Identifies which body part a click landed on, by GPU picking.
 *
 * Why not a raycast. `Raycaster.intersectObject(vrm.scene, true)` walks every
 * triangle of every mesh, and on a SkinnedMesh three has to skin each vertex on
 * the CPU first — `SkinnedMesh.getVertexPosition` calls `applyBoneTransform` per
 * vertex (three/src/objects/SkinnedMesh.js:213-217), four Matrix4 multiplies
 * each. A ~100k-triangle model therefore costs ~300k bone transforms per click,
 * synchronously, on the main thread; the first click additionally pays
 * `computeBoundingSphere()` over every vertex (:138-154). That is the freeze,
 * and it scales with the model — which is why one avatar felt fine and another
 * stalled. Deferring it to a rAF does not help: the same blocking work simply
 * lands one frame later.
 *
 * Why not nearest-bone either. Distance from the hit point to the closest joint
 * has no notion of what is in front: with the arms hanging at the sides, a point
 * on the thigh is genuinely nearer the thumb joints than to the upper-leg joint,
 * so clicking a leg answered with a finger. No radius tuning fixes that — the
 * information needed (which surface is in front) is simply not in the distance.
 *
 * What this does instead: render the model into a 1×1 render target aimed at the
 * clicked pixel, with a material that writes a part id instead of a colour, and
 * read that one pixel back. The depth test resolves occlusion for free, skinning
 * and morphs are applied by the GPU as part of the normal pipeline, and the cost
 * is one draw call plus one readback regardless of polygon count.
 */

/** Encoded id travels in the red channel, so 0 is reserved for "no hit". */
const PICK_ATTRIBUTE = 'partIndex'

/**
 * Alpha threshold for source materials that are blended rather than masked.
 * Without a discard, a transparent hair card would occlude everything behind it
 * in the pick pass even where it is invisible on screen.
 */
const BLEND_ALPHA_CUTOFF = 0.5

/** VRM viseme presets — between them they cover the whole mouth. */
const MOUTH_EXPRESSIONS = ['aa', 'ih', 'ou', 'ee', 'oh']

/**
 * A vertex belongs to a face region when the morph target moves it by at least
 * this fraction of that target's own largest displacement. Relative, so it holds
 * for a model of any scale; loose enough to keep the soft edge of a mouth shape,
 * tight enough to drop the cheek vertices a viseme barely brushes.
 */
const REGION_THRESHOLD = 0.15

interface RegionHit {
  mesh: THREE.Mesh
  ids: Float32Array
  /** A Set, not an array: the five visemes overlap heavily, and a vertex counted
   *  twice would drag the centroid that decides left eye from right. */
  vertices: Set<number>
}

// Chunk order mirrors three's own meshbasic vertex shader exactly (ShaderLib/
// meshbasic.glsl.js): skinbase before begin_vertex, then morph → skin → project.
// The batching/instancing chunks are carried along because <project_vertex>
// references batchingMatrix/instanceMatrix under their defines — cheap insurance
// against a shader that fails to compile, which would silently turn every pick
// into a miss.
const pickVertexShader = /* glsl */ `
#include <common>
#include <batching_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>

attribute float ${PICK_ATTRIBUTE};

// flat: a triangle spanning two parts must NOT interpolate its id. Averaging
// e.g. 3 and 7 would yield 5 — a third part, wrong everywhere along the seam.
flat out float vPart;

#ifdef PICK_ALPHA
out vec2 vPickUv;
#endif

void main() {
  vPart = ${PICK_ATTRIBUTE};

  #ifdef PICK_ALPHA
  vPickUv = uv;
  #endif

  #include <morphinstance_vertex>
  #include <batching_vertex>
  #include <skinbase_vertex>
  #include <begin_vertex>
  #include <morphtarget_vertex>
  #include <skinning_vertex>
  #include <project_vertex>
}
`

const pickFragmentShader = /* glsl */ `
flat in float vPart;

#ifdef PICK_ALPHA
uniform sampler2D pickMap;
uniform float pickAlphaCutoff;
in vec2 vPickUv;
#endif

void main() {
  #ifdef PICK_ALPHA
  if (texture2D(pickMap, vPickUv).a < pickAlphaCutoff) discard;
  #endif

  gl_FragColor = vec4(vPart / 255.0, 0.0, 0.0, 1.0);
}
`

interface Swap {
  mesh: THREE.Mesh
  original: THREE.Material | THREE.Material[]
  pick: THREE.Material | THREE.Material[]
}

export class BodyPartPicker {
  private readonly vrm: VRM
  private readonly renderer: THREE.WebGLRenderer
  private readonly target: THREE.WebGLRenderTarget
  private readonly size = new THREE.Vector2()
  private readonly clearColor = new THREE.Color()
  private readonly swaps: Swap[] = []
  private readonly ownedMaterials: THREE.ShaderMaterial[] = []
  private readonly taggedGeometries = new Set<THREE.BufferGeometry>()
  private readonly idsByGeometry = new Map<THREE.BufferGeometry, Float32Array>()
  private built = false
  private disposed = false

  /**
   * Breakdown of the last pick, in milliseconds. Mutated in place — read it
   * right after the call, do not keep the object as a record.
   *
   * - `render`: CPU only. `renderer.render()` returns once the draw commands are
   *   queued, so this is scene-graph traversal, frustum culling and render-list
   *   building — it scales with the NODE count of the VRM, which on a model with
   *   spring-boned hair and cloth runs to thousands. Measured here at 0.4–0.8 ms.
   * - `blocking`: everything the click actually pays on the main thread, i.e. up
   *   to and including issuing the readback. This is the number that decides
   *   whether a click feels smooth.
   * - `latency`: until the answer is available. Larger than `blocking` by the GPU
   *   fence wait, but that wait is off-thread — it delays the answer, not the UI.
   * - `warm`: the one-off `warm()` pass. Shader link and uniform-location fetch
   *   land here (measured at ~140 ms on a heavy model) instead of on a click.
   */
  readonly timings = { render: 0, blocking: 0, latency: 0, warm: 0 }

  constructor(vrm: VRM, renderer: THREE.WebGLRenderer) {
    this.vrm = vrm
    this.renderer = renderer
    this.target = new THREE.WebGLRenderTarget(1, 1, {
      depthBuffer: true,
      stencilBuffer: false,
      format: THREE.RGBAFormat,
      type: THREE.UnsignedByteType,
      minFilter: THREE.NearestFilter,
      magFilter: THREE.NearestFilter,
    })
    // No colour management on the way out — the red channel is an integer id,
    // and an sRGB transfer curve would quietly renumber it.
    this.target.texture.colorSpace = THREE.NoColorSpace
    this.target.texture.generateMipmaps = false
  }

  get isReady(): boolean {
    return this.built && !this.disposed
  }

  /**
   * Tag every vertex with its part and prepare the pick materials.
   *
   * Idempotent, and heavy enough (one pass over every vertex) that the caller
   * should run it off the interaction path — the same reasoning as the clip
   * warm-up in AnimationRegistry.
   */
  build(): void {
    if (this.built || this.disposed) return

    const boneToPart = buildBoneToPartMap(this.vrm)
    if (boneToPart.size === 0) {
      console.warn('[bodyPart] model has no humanoid bones — picking disabled')
      this.built = true
      return
    }

    this.vrm.scene.traverse((object) => {
      const mesh = object as THREE.Mesh
      if (!mesh.isMesh) return

      this.tagGeometry(mesh, boneToPart)
      this.swaps.push({
        mesh,
        original: mesh.material,
        pick: this.pickMaterialFor(mesh),
      })
    })

    this.refineFaceRegions()
    this.reportCoverage()

    this.built = true
  }

  /**
   * One line per model saying which parts actually got vertices.
   *
   * Coverage is genuinely per-model — a model with no blendshapes has no mouth
   * region and never will — so this makes the difference visible at load instead
   * of leaving it to be discovered by clicking a face and getting `head`.
   */
  private reportCoverage(): void {
    const counts = new Map<BodyPart, number>()
    for (const ids of this.idsByGeometry.values()) {
      for (let v = 0; v < ids.length; v++) {
        const part = PART_BY_ID[ids[v]]
        if (part) counts.set(part, (counts.get(part) ?? 0) + 1)
      }
    }

    const found = [...counts.entries()]
      .sort((a, b) => b[1] - a[1])
      .map(([part, n]) => `${part}:${n}`)
      .join(' ')
    const missing = BODY_PARTS.filter((part) => !counts.has(part))

    console.log('[bodyPart] coverage', found || '(none)')
    if (missing.length > 0) console.log('[bodyPart] no vertices for', missing.join(', '))
  }

  /**
   * Compile the pick programs ahead of the first click.
   *
   * Swapping a material makes three link a new program on first draw, so without
   * this the first pick pays a shader link — measured at 140 ms on a heavy model,
   * a smaller repeat of the very stall this class exists to remove.
   *
   * `compile()` alone is NOT enough: it issues linkProgram, but three defers
   * `getProgramParameter(LINK_STATUS)` and the uniform-location fetch until the
   * program is actually drawn with. So this renders once as well, and pays that
   * cost here instead of under the user's cursor.
   *
   * That throwaway render deliberately has NO view offset. A pick renders through
   * a one-pixel sliver of a frustum, which culls nearly every mesh — warming
   * through the same path would link only the handful of programs visible in that
   * sliver and leave the rest to hitch on some later click somewhere else. Full
   * frustum squeezed into the 1×1 target draws everything, so every program is
   * linked exactly once, here.
   *
   * Synchronous on purpose, and deliberately not `compileAsync`: the window in
   * which the pick materials are attached must not span an await. R3F keeps
   * rendering, and a model wearing its pick materials for even one frame is a
   * visible flash of flat red. Visibility is forced for the same reason
   * `traverseVisible` inside `compile` needs it — a model still behind the
   * loading overlay would otherwise contribute no materials at all.
   */
  warm(camera: THREE.Camera): void {
    if (!this.isReady || this.swaps.length === 0) return

    const renderer = this.renderer
    const startedAt = performance.now()

    const wasVisible = this.vrm.scene.visible
    const prevTarget = renderer.getRenderTarget()
    const prevActiveCubeFace = renderer.getActiveCubeFace()
    const prevActiveMipmapLevel = renderer.getActiveMipmapLevel()
    const prevAutoClear = renderer.autoClear
    const prevClearAlpha = renderer.getClearAlpha()
    renderer.getClearColor(this.clearColor)

    this.vrm.scene.visible = true
    for (const swap of this.swaps) swap.mesh.material = swap.pick

    try {
      renderer.compile(this.vrm.scene, camera)
      renderer.setRenderTarget(this.target)
      renderer.autoClear = false
      renderer.setClearColor(0x000000, 1)
      renderer.clear(true, true, false)
      renderer.render(this.vrm.scene, camera)
    } catch {
      // A warm-up failure costs one slow click, nothing more.
    } finally {
      for (const swap of this.swaps) swap.mesh.material = swap.original
      this.vrm.scene.visible = wasVisible
      renderer.setClearColor(this.clearColor, prevClearAlpha)
      renderer.autoClear = prevAutoClear
      renderer.setRenderTarget(prevTarget, prevActiveCubeFace, prevActiveMipmapLevel)
      this.timings.warm = performance.now() - startedAt
    }
  }

  /**
   * Which part is drawn at this pixel? Coordinates are CSS pixels relative to
   * the canvas, top-left origin — the same space `getBoundingClientRect` yields.
   * Resolves to null for a miss (background) or before `build()` has run.
   *
   * Async because the readback, not the render, was the cost. Measured on this
   * project: render 0.4–0.8 ms, blocking readback 5.4–22.1 ms — `readPixels` is
   * synchronous and stalls until the draw it depends on has actually finished on
   * the GPU. `readRenderTargetPixelsAsync` issues the read into a pixel-pack
   * buffer and waits on a fence instead, so the stall leaves the main thread.
   *
   * The ordering below matters and is not incidental. Everything up to three's
   * first `await` — including the `readPixels` into the PBO — runs synchronously
   * when the promise is created (r184 WebGLRenderer.js:3190-3200). So the promise
   * is captured, ALL renderer state is restored synchronously, and only then is
   * the promise awaited. Awaiting first would leave the model wearing its pick
   * materials, and the render target bound, across frames that R3F is drawing.
   *
   * Leaves a breakdown in {@link timings}.
   */
  async pickAsync(px: number, py: number, camera: THREE.Camera): Promise<BodyPart | null> {
    if (!this.isReady || this.swaps.length === 0) return null
    if (!isOffsetCamera(camera)) return null

    const startedAt = performance.now()
    const renderer = this.renderer
    renderer.getSize(this.size)
    const width = Math.max(1, Math.round(this.size.x))
    const height = Math.max(1, Math.round(this.size.y))

    const x = Math.floor(px)
    const y = Math.floor(py)
    if (x < 0 || y < 0 || x >= width || y >= height) return null

    const prevTarget = renderer.getRenderTarget()
    const prevActiveCubeFace = renderer.getActiveCubeFace()
    const prevActiveMipmapLevel = renderer.getActiveMipmapLevel()
    const prevAutoClear = renderer.autoClear
    const prevClearAlpha = renderer.getClearAlpha()
    renderer.getClearColor(this.clearColor)

    // Per-call buffer, not a shared one: two clicks in quick succession overlap,
    // and a shared buffer would let the second readback land in the first's result.
    const pixel = new Uint8Array(4)
    // Left unassigned rather than seeded with null: if the render throws, the
    // finally restores state and the error propagates — the seed is never read.
    let readback: Promise<unknown> | undefined

    for (const swap of this.swaps) swap.mesh.material = swap.pick

    try {
      // Squeeze the whole viewport down to the single clicked pixel, so the one
      // texel we read back is exactly what the user aimed at.
      camera.setViewOffset(width, height, x, y, 1, 1)

      renderer.setRenderTarget(this.target)
      renderer.autoClear = false
      renderer.setClearColor(0x000000, 1) // 0 = "nothing here"
      renderer.clear(true, true, false)

      const beforeRender = performance.now()
      renderer.render(this.vrm.scene, camera)
      this.timings.render = performance.now() - beforeRender

      readback = renderer.readRenderTargetPixelsAsync(this.target, 0, 0, 1, 1, pixel)
    } finally {
      camera.clearViewOffset()
      for (const swap of this.swaps) swap.mesh.material = swap.original
      renderer.setClearColor(this.clearColor, prevClearAlpha)
      renderer.autoClear = prevAutoClear
      renderer.setRenderTarget(prevTarget, prevActiveCubeFace, prevActiveMipmapLevel)
      this.timings.blocking = performance.now() - startedAt
    }

    if (!readback) return null
    try {
      await readback
    } catch {
      return null // a lost context or a resized target — one click, not a crash
    }
    this.timings.latency = performance.now() - startedAt

    return PART_BY_ID[pixel[0]] ?? null
  }

  dispose(): void {
    if (this.disposed) return
    this.disposed = true

    for (const swap of this.swaps) swap.mesh.material = swap.original
    this.swaps.length = 0

    for (const geometry of this.taggedGeometries) geometry.deleteAttribute(PICK_ATTRIBUTE)
    this.taggedGeometries.clear()
    this.idsByGeometry.clear()

    for (const material of this.ownedMaterials) material.dispose()
    this.ownedMaterials.length = 0

    this.target.dispose()
  }

  /* ─────────────────────────── build helpers ──────────────────────────── */

  /**
   * Write a part id per vertex.
   *
   * The bone→part resolution runs once per skeleton bone, not once per vertex:
   * a skeleton has tens of bones and a mesh has tens of thousands of vertices,
   * so the per-vertex loop is reduced to a table lookup. That table is also what
   * handles twist and spring bones, which are real skinning targets but are not
   * humanoid bones — `resolvePart` climbs to their nearest humanoid ancestor.
   */
  private tagGeometry(mesh: THREE.Mesh, boneToPart: Map<THREE.Object3D, BodyPart>): void {
    const geometry = mesh.geometry
    // Shared geometry is tagged once; the first mesh to claim it wins. VRM
    // exports do not share geometry across body regions, so this is a guard
    // against double work rather than a policy.
    if (this.taggedGeometries.has(geometry)) return

    const position = geometry.attributes.position
    if (!position) return

    const count = position.count
    const ids = new Float32Array(count)
    const skinned = mesh as THREE.SkinnedMesh
    const skinIndex = geometry.attributes.skinIndex
    const skinWeight = geometry.attributes.skinWeight

    if (skinned.isSkinnedMesh && skinned.skeleton && skinIndex && skinWeight) {
      const bones = skinned.skeleton.bones
      const idByBone = new Uint8Array(bones.length)
      for (let i = 0; i < bones.length; i++) {
        const part = resolvePart(bones[i], boneToPart)
        idByBone[i] = part ? partId(part) : 0
      }

      for (let v = 0; v < count; v++) {
        let bestWeight = 0
        let bestId = 0
        for (let i = 0; i < 4; i++) {
          const weight = skinWeight.getComponent(v, i)
          if (weight > bestWeight) {
            bestWeight = weight
            bestId = idByBone[skinIndex.getComponent(v, i)] ?? 0
          }
        }
        ids[v] = bestId
      }
    } else {
      // A static mesh — hair or an accessory parented straight onto a bone.
      const part = resolvePart(mesh, boneToPart)
      if (part) ids.fill(partId(part))
    }

    geometry.setAttribute(PICK_ATTRIBUTE, new THREE.BufferAttribute(ids, 1))
    this.taggedGeometries.add(geometry)
    this.idsByGeometry.set(geometry, ids)
  }

  /* ────────────────────── face regions (mouth / eyes) ──────────────────── */

  /**
   * Refine the face from blendshape data: mouth, left eye, right eye.
   *
   * Bones cannot answer this. Almost every VRM skins the entire face to the
   * single `head` bone and ships no jaw bone at all, so the bone pass paints the
   * whole face one colour. The authored signal lives in the expressions: the
   * viseme morph targets (`aa`/`ih`/`ou`/`ee`/`oh`) displace exactly the mouth
   * vertices, and `blink` exactly the eye vertices. That is the model author's
   * own definition of those regions, not a guess about where a mouth ought to be.
   *
   * Degrades honestly. A model with no `expressionManager` — or one with zero
   * blendshape groups, as bronya_long.vrm has (VRMExpressionAdapter.ts:12) —
   * keeps `head` for the whole face. That is the right answer to give when the
   * model carries no information about where its mouth is; the alternative would
   * be a geometric guess, and guessing is what put a thumb joint on a thigh.
   */
  private refineFaceRegions(): void {
    const manager = this.vrm.expressionManager
    if (!manager) return

    this.assignRegion(this.collectRegion(MOUTH_EXPRESSIONS), partId('mouth'))

    // Authored sides beat anything inferred, so try them first.
    const left = this.collectRegion(['blinkLeft'])
    const right = this.collectRegion(['blinkRight'])
    if (hasVertices(left) && hasVertices(right)) {
      this.assignRegion(left, partId('leftEye'))
      this.assignRegion(right, partId('rightEye'))
      return
    }

    // Only a combined `blink`: split it down the model's own lateral axis.
    const both = this.collectRegion(['blink'])
    if (!hasVertices(both)) return

    const axis = this.lateralAxis()
    if (!axis) {
      // No way to tell left from right. Leaving the region as `head` beats a
      // coin flip that would mislabel one eye on every single click.
      console.warn('[bodyPart] cannot orient the eye region — leaving the face as head')
      return
    }
    this.assignEyeSides(both, axis)
  }

  /**
   * Vertices displaced by any of the named expressions, grouped per geometry.
   *
   * The threshold is a fraction of each morph target's OWN largest displacement,
   * so it carries across models of any scale without a magic world-space number.
   */
  private collectRegion(names: string[]): RegionHit[] {
    const manager = this.vrm.expressionManager
    if (!manager) return []

    const byGeometry = new Map<THREE.BufferGeometry, RegionHit>()

    for (const name of names) {
      const expression = manager.getExpression(name)
      if (!expression) continue

      for (const bind of expression.binds) {
        if (!(bind instanceof VRMExpressionMorphTargetBind)) continue

        for (const mesh of bind.primitives) {
          const geometry = mesh.geometry
          const ids = this.idsByGeometry.get(geometry)
          const morph = geometry.morphAttributes.position?.[bind.index]
          if (!ids || !morph) continue

          const threshold = maxDisplacement(morph) * REGION_THRESHOLD
          if (threshold <= 0) continue

          let hit = byGeometry.get(geometry)
          if (!hit) {
            hit = { mesh, ids, vertices: new Set<number>() }
            byGeometry.set(geometry, hit)
          }

          const thresholdSq = threshold * threshold
          for (let v = 0; v < morph.count; v++) {
            const dx = morph.getX(v)
            const dy = morph.getY(v)
            const dz = morph.getZ(v)
            if (dx * dx + dy * dy + dz * dz < thresholdSq) continue
            hit.vertices.add(v)
          }
        }
      }
    }

    return [...byGeometry.values()]
  }

  /**
   * Overwrite ids for a region — but only where the bone pass said `head`.
   *
   * Emotion visemes brush cheeks and eyelids, and a face mesh can carry a few
   * neck vertices. Refusing to overwrite anything that is not already `head`
   * means a sloppy morph target can never repaint part of a shoulder.
   */
  private assignRegion(hits: RegionHit[], id: number): void {
    const headId = partId('head')
    for (const hit of hits) {
      for (const v of hit.vertices) {
        if (hit.ids[v] === headId) hit.ids[v] = id
      }
    }
  }

  /** Split one combined blink region into left and right along `axis`. */
  private assignEyeSides(hits: RegionHit[], axis: THREE.Vector3): void {
    const points: THREE.Vector3[] = []
    const owners: RegionHit[] = []
    const indices: number[] = []
    const scratch = new THREE.Vector3()

    for (const hit of hits) {
      const position = hit.mesh.geometry.attributes.position
      const toBind = (hit.mesh as THREE.SkinnedMesh).bindMatrix
      for (const v of hit.vertices) {
        scratch.fromBufferAttribute(position, v)
        // Bind space, so the reference bones below live in the same frame and
        // no animation pose can shift the split.
        if (toBind) scratch.applyMatrix4(toBind)
        points.push(scratch.clone())
        owners.push(hit)
        indices.push(v)
      }
    }

    const sides = splitBySide(points, axis)
    const headId = partId('head')
    const leftId = partId('leftEye')
    const rightId = partId('rightEye')

    for (let i = 0; i < sides.length; i++) {
      const ids = owners[i].ids
      const v = indices[i]
      if (ids[v] !== headId) continue
      ids[v] = sides[i] ? leftId : rightId
    }
  }

  /**
   * A unit vector pointing toward the character's LEFT, in bind space.
   *
   * Taken from the upper-leg bones' bind positions rather than their live ones:
   * bind positions come from `skeleton.boneInverses`, so they are fixed by the
   * asset and no greeting wave or idle pose can flip the sign. Legs are required
   * humanoid bones and never cross, which makes them a safer pair than the arms.
   */
  private lateralAxis(): THREE.Vector3 | null {
    const humanoid = this.vrm.humanoid
    if (!humanoid) return null

    let skeleton: THREE.Skeleton | null = null
    this.vrm.scene.traverse((object) => {
      const mesh = object as THREE.SkinnedMesh
      if (!skeleton && mesh.isSkinnedMesh && mesh.skeleton) skeleton = mesh.skeleton
    })
    if (!skeleton) return null

    const left = bindPosition(skeleton, humanoid.getRawBoneNode(VRMHumanBoneName.LeftUpperLeg))
    const right = bindPosition(skeleton, humanoid.getRawBoneNode(VRMHumanBoneName.RightUpperLeg))
    if (!left || !right) return null

    const axis = left.sub(right)
    if (axis.lengthSq() === 0) return null
    return axis.normalize()
  }

  private pickMaterialFor(mesh: THREE.Mesh): THREE.Material | THREE.Material[] {
    const hasUv = mesh.geometry.attributes.uv !== undefined
    const build = (source: THREE.Material) => this.buildPickMaterial(source, hasUv)
    return Array.isArray(mesh.material)
      ? mesh.material.map(build)
      : build(mesh.material)
  }

  /**
   * One pick material per source material, so `side` and alpha cut-out match
   * what is actually on screen. Without the cut-out, clicking between two strands
   * of hair would answer "head" instead of whatever is visible behind them.
   */
  private buildPickMaterial(source: THREE.Material, hasUv: boolean): THREE.ShaderMaterial {
    const map = (source as THREE.MeshBasicMaterial).map
    const masked = source.alphaTest > 0
    const useAlpha = hasUv && map != null && (masked || source.transparent)
    const cutoff = masked ? source.alphaTest : BLEND_ALPHA_CUTOFF

    const material = new THREE.ShaderMaterial({
      vertexShader: pickVertexShader,
      fragmentShader: pickFragmentShader,
      uniforms: useAlpha
        ? { pickMap: { value: map }, pickAlphaCutoff: { value: cutoff } }
        : {},
      defines: useAlpha ? { PICK_ALPHA: '' } : {},
      side: source.side,
      // Opaque on purpose: the pick pass needs depth writes to resolve which
      // surface is in front, which is the whole point of doing this on the GPU.
      transparent: false,
      depthTest: true,
      depthWrite: true,
      fog: false,
      lights: false,
    })

    this.ownedMaterials.push(material)
    return material
  }
}

/** An expression can exist and still bind nothing — see the bind-repair note in
 *  VRMExpressionAdapter. Presence of the expression is not presence of a region. */
function hasVertices(hits: readonly RegionHit[]): boolean {
  return hits.some((hit) => hit.vertices.size > 0)
}

/** Largest vertex displacement in a morph target, used to scale the threshold. */
function maxDisplacement(morph: THREE.BufferAttribute | THREE.InterleavedBufferAttribute): number {
  let maxSq = 0
  for (let v = 0; v < morph.count; v++) {
    const x = morph.getX(v)
    const y = morph.getY(v)
    const z = morph.getZ(v)
    const lengthSq = x * x + y * y + z * z
    if (lengthSq > maxSq) maxSq = lengthSq
  }
  return Math.sqrt(maxSq)
}

/**
 * Bind-space position of a bone, read from the skeleton's inverse bind matrix.
 *
 * Pose-independent by construction: `boneInverses` is baked into the asset, so
 * unlike `bone.getWorldPosition()` this cannot move when an animation plays.
 */
function bindPosition(
  skeleton: THREE.Skeleton,
  node: THREE.Object3D | null,
): THREE.Vector3 | null {
  if (!node) return null
  const index = skeleton.bones.indexOf(node as THREE.Bone)
  if (index < 0) return null
  const bindMatrix = new THREE.Matrix4().copy(skeleton.boneInverses[index]).invert()
  return new THREE.Vector3().setFromMatrixPosition(bindMatrix)
}

/**
 * Split points into two sides along `axis`, cutting at the set's own centroid.
 *
 * Returns true for the side `axis` points toward. Self-calibrating: the cut
 * lands between the two eyes wherever the model actually puts them, so it does
 * not assume the face is centred on the origin.
 *
 * Exported for tests — it is the one piece of this file that is pure geometry.
 */
export function splitBySide(points: readonly THREE.Vector3[], axis: THREE.Vector3): boolean[] {
  if (points.length === 0) return []

  let centre = 0
  for (const point of points) centre += point.dot(axis)
  centre /= points.length

  return points.map((point) => point.dot(axis) > centre)
}

type OffsetCamera = THREE.Camera & {
  setViewOffset: (
    fullWidth: number,
    fullHeight: number,
    x: number,
    y: number,
    width: number,
    height: number,
  ) => void
  clearViewOffset: () => void
}

function isOffsetCamera(camera: THREE.Camera): camera is OffsetCamera {
  return typeof (camera as OffsetCamera).setViewOffset === 'function'
}
