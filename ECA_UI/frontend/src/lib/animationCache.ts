/**
 * Animation clip cache.
 *
 * Retargeting (Mixamo FBX / BVH → VRM) is per-model and costs a network fetch
 * plus a CPU pass, so clips are cached by "vrmUrl|animUrl". The cached value is
 * the in-flight PROMISE itself, which dedupes concurrent requests for the same
 * pair (e.g. user spam-clicking a motion) and makes re-selecting a previously
 * used (model, motion) pair instant — no async gap, no bind-pose flash.
 *
 * Clips are immutable after creation, so sharing one clip across actions is
 * safe. The cache is unbounded in theory but in practice is limited to
 * (#VRM models × #motion assets), which is tiny.
 */

import * as THREE from 'three'

// Fetch-level cache for VRM/FBX/BVH bytes: repeat loads skip the network.
// Generated motion files have unique names (motion_<uuid>.bvh), so stale
// content is not a concern.
THREE.Cache.enabled = true

const clipCache = new Map<string, Promise<THREE.AnimationClip | null>>()

/**
 * Get the clip for `key`, running `factory` (fetch + retarget) only on the
 * first call. Failed/empty loads are evicted so a later attempt can retry.
 */
export function getAnimationClip(
  key: string,
  factory: () => Promise<THREE.AnimationClip | null>,
): Promise<THREE.AnimationClip | null> {
  let entry = clipCache.get(key)
  if (!entry) {
    entry = factory().then(
      (clip) => {
        if (!clip) clipCache.delete(key)
        return clip
      },
      (err) => {
        clipCache.delete(key)
        throw err
      },
    )
    clipCache.set(key, entry)
  }
  return entry
}
