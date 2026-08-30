import { beforeEach, describe, expect, it, vi } from 'vitest'
import * as THREE from 'three'
import type { VRM } from '@pixiv/three-vrm'

const loadAndRetargetBVH = vi.fn()
const loadMixamoAnimation = vi.fn()

vi.mock('./bvhToVrm', async () => {
  const actual = await vi.importActual<typeof import('./bvhToVrm')>('./bvhToVrm')
  return { ...actual, loadAndRetargetBVH: (...a: unknown[]) => loadAndRetargetBVH(...a) }
})
vi.mock('./loadMixamoAnimation', () => ({
  loadMixamoAnimation: (...a: unknown[]) => loadMixamoAnimation(...a),
}))

const { AnimationRegistry } = await import('./AnimationRegistry')

/** Minimal stand-in: the registry only reads `vrm` to hand to the loaders. */
function makeVrm(): VRM {
  return { scene: new THREE.Object3D() } as unknown as VRM
}

function makeClip(name: string) {
  return new THREE.AnimationClip(name, 1, [])
}

/**
 * A CloudFront signed URL for the same object, signed twice. Only the query
 * differs — that is the whole problem: the signature is regenerated per request
 * and the URL is never byte-identical twice.
 */
const SIGNED_A = 'https://cdn/motions/abc123.bvh?Policy=eyJ&Signature=AAA&Key-Pair-Id=K1'
const SIGNED_B = 'https://cdn/motions/abc123.bvh?Policy=eyJ&Signature=BBB&Key-Pair-Id=K1'

describe('AnimationRegistry dynamic clip caching', () => {
  beforeEach(() => {
    loadAndRetargetBVH.mockReset()
    loadMixamoAnimation.mockReset()
    loadAndRetargetBVH.mockResolvedValue(makeClip('motion'))
  })

  it('retargets once for two signed URLs that share a cacheKey', async () => {
    // The clip cache is keyed `vrmUrl|url`. A signed URL puts a fresh
    // Signature in that key every time, so the same rendered motion would be
    // fetched and retargeted again on every replay — and THREE.Cache, keyed by
    // full URL, would keep every copy. job_id is the content hash of the
    // request, which is exactly the stable identity the key wants.
    const registry = new AnimationRegistry(makeVrm(), 'model.vrm')

    registry.update('exercise', { url: SIGNED_A, loader: 'bvh', retarget: 'smplx', cacheKey: 'abc123' })
    await registry.get('exercise')

    registry.update('exercise', { url: SIGNED_B, loader: 'bvh', retarget: 'smplx', cacheKey: 'abc123' })
    await registry.get('exercise')

    expect(loadAndRetargetBVH).toHaveBeenCalledTimes(1)
  })

  it('still keys on the URL when no cacheKey is given', async () => {
    // The debug motion picker passes bundled URLs and no key; those are stable
    // and distinct, so URL-keying stays correct for them.
    const registry = new AnimationRegistry(makeVrm(), 'model.vrm')

    registry.update('exercise', { url: '/motions/one.bvh', loader: 'bvh' })
    await registry.get('exercise')

    registry.update('exercise', { url: '/motions/two.bvh', loader: 'bvh' })
    await registry.get('exercise')

    expect(loadAndRetargetBVH).toHaveBeenCalledTimes(2)
  })

  it('keeps the cache per model — same motion, two avatars, two retargets', async () => {
    // Retargeting is per-skeleton; sharing a clip across models would put one
    // avatar's bone lengths on another.
    const a = new AnimationRegistry(makeVrm(), 'alice.vrm')
    const b = new AnimationRegistry(makeVrm(), 'bob.vrm')
    const clip = { url: SIGNED_A, loader: 'bvh' as const, retarget: 'smplx' as const, cacheKey: 'abc123' }

    a.update('exercise', clip)
    await a.get('exercise')
    b.update('exercise', clip)
    await b.get('exercise')

    expect(loadAndRetargetBVH).toHaveBeenCalledTimes(2)
  })
})
