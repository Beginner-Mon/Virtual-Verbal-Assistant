/**
 * Asset layer for the animation FSM: state → ready-to-play `AnimationClip`.
 *
 * Load, retarget, subclip and cache all live here, so `AnimationController`
 * never learns what a URL, a file format or a retarget matrix is (plan §3.1).
 *
 * One registry per VRM instance. Clips are retargeted against a specific
 * skeleton, so reusing them across models distorts the pose — the instance
 * boundary is what prevents that (plus `invalidate()` for the in-place case).
 */

import * as THREE from 'three'
import type { VRM } from '@pixiv/three-vrm'
import { loadAndRetargetBVH, SMPLX_RETARGET_OPTIONS, STANDARD_RETARGET_OPTIONS } from './bvhToVrm'
import { loadMixamoAnimation } from './loadMixamoAnimation'
import { getAnimationClip } from './animationCache'
import { resolveMotionUrl } from './motionAssets'
import { CHAR_STATES, staticSourceOf, type CharState, type StaticSource, type SubclipRange } from './AnimationStates'

export class AnimationRegistry {
  /**
   * state → subclipped, ready-to-use clip. Caching the FINAL clip (not just the
   * base file) matters: `mixer.clipAction()` is keyed by clip identity, so
   * re-deriving a subclip per transition would mint a fresh action every time —
   * leaking actions and defeating the `reset()` fix for replaying one-shots.
   */
  private clips = new Map<CharState, Promise<THREE.AnimationClip | null>>()

  /** URLs injected at runtime (generated motion). Only `exercise` uses this. */
  private dynamicUrls = new Map<CharState, string>()

  private readonly vrm: VRM
  /** Cache-key namespace: the same motion retargets differently per model. */
  private readonly vrmUrl: string

  constructor(vrm: VRM, vrmUrl: string) {
    this.vrm = vrm
    this.vrmUrl = vrmUrl
  }

  /** Ready clip for `state`, or null if its asset is missing or failed to load. */
  get(state: CharState): Promise<THREE.AnimationClip | null> {
    let entry = this.clips.get(state)
    if (!entry) {
      entry = this.build(state).catch((err) => {
        // Evict so a later attempt can retry (e.g. transient network failure).
        this.clips.delete(state)
        console.error(`[AnimationRegistry] Failed to build clip for "${state}":`, err)
        return null
      })
      this.clips.set(state, entry)
    }
    return entry
  }

  /**
   * Point `state` at a runtime URL (SSE motion, or a file picked in the debug
   * selector) and drop any clip cached for it. Generated motion is always
   * SMPL-X BVH — the output of `scripts/kimodo_npz_to_bvh.py`.
   */
  update(state: CharState, url: string): void {
    this.dynamicUrls.set(state, url)
    this.clips.delete(state)
  }

  /** Drop every cached clip. Required when the VRM changes in place. */
  invalidate(): void {
    this.clips.clear()
  }

  /**
   * Warm every static clip during idle time.
   *
   * Retargeting is synchronous CPU work on the main thread: loading a clip for
   * the first time blocks the render loop long enough to be seen as a freeze /
   * flash (measured 111ms for Thinking.fbx, 214ms for random_Bored.fbx). Doing
   * it lazily meant that cost landed on the user's FIRST question, and again on
   * the first idle filler — i.e. exactly mid-interaction.
   *
   * One clip per idle callback, so the warm-up never blocks a frame itself.
   * `get()` is cached and dedupes, so a real transition arriving mid-prefetch
   * simply awaits the same promise instead of loading twice.
   */
  prefetchStatic(): void {
    const pending = CHAR_STATES.filter((state) => staticSourceOf(state) && !this.clips.has(state))

    const step = () => {
      const next = pending.shift()
      if (next === undefined) return
      void this.get(next).finally(() => schedule())
    }
    const schedule = () => {
      if (typeof requestIdleCallback === 'function') requestIdleCallback(step, { timeout: 2000 })
      else setTimeout(step, 200)
    }
    schedule()
  }

  private async build(state: CharState): Promise<THREE.AnimationClip | null> {
    const dynamicUrl = this.dynamicUrls.get(state)
    if (dynamicUrl) {
      return this.load({ loader: 'bvh', match: /^$/, retarget: 'smplx' }, dynamicUrl)
    }

    const source = staticSourceOf(state)
    if (!source) {
      // 'dynamic' state with nothing registered yet — a legitimate miss, not an
      // error: the caller (transitionTo) simply reports the transition failed.
      return null
    }

    const url = resolveMotionUrl(source.match)
    if (!url) {
      console.warn(`[AnimationRegistry] No asset matches ${source.match} for state "${state}"`)
      return null
    }
    return this.load(source, url)
  }

  private async load(source: StaticSource, url: string): Promise<THREE.AnimationClip | null> {
    // The shared cache dedupes concurrent loads and is keyed per model, so the
    // fetch + retarget pass happens once per (model, file) pair.
    const base = await getAnimationClip(`${this.vrmUrl}|${url}`, () =>
      source.loader === 'fbx'
        ? loadMixamoAnimation(url, this.vrm)
        : loadAndRetargetBVH(
            url,
            this.vrm,
            source.retarget === 'smplx' ? SMPLX_RETARGET_OPTIONS : STANDARD_RETARGET_OPTIONS,
          ),
    )
    if (!base) return null
    return source.subclip ? subclip(base, source.subclip) : base
  }
}

function subclip(clip: THREE.AnimationClip, range: SubclipRange): THREE.AnimationClip {
  return THREE.AnimationUtils.subclip(clip, range.name, range.start, range.end, range.fps)
}
