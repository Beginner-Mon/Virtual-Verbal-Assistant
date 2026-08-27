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
import { CHAR_STATES, staticSourceOf, type CharState, type SubclipRange } from './AnimationStates'

/**
 * A clip handed to a `'dynamic'` state at runtime: a resolved URL plus how to
 * read it. Generated Kimodo motion is SMPL-X BVH; a character's own gesture is
 * usually a bundled Mixamo FBX — the loader has to travel with the URL.
 */
export interface DynamicClip {
  url: string
  loader: 'fbx' | 'bvh'
  /** BVH only. Defaults to `smplx`, which is what Kimodo emits. */
  retarget?: 'smplx' | 'standard'
}

/**
 * Infer the loader from a URL. Vite keeps the original extension on hashed
 * asset names, and generated motion is always `motion_<uuid>.bvh`, so the
 * extension is a reliable signal for both origins.
 */
export function loaderForUrl(url: string): 'fbx' | 'bvh' {
  return /\.fbx(\?|$)/i.test(url) ? 'fbx' : 'bvh'
}

export class AnimationRegistry {
  /**
   * state → subclipped, ready-to-use clip. Caching the FINAL clip (not just the
   * base file) matters: `mixer.clipAction()` is keyed by clip identity, so
   * re-deriving a subclip per transition would mint a fresh action every time —
   * leaking actions and defeating the `reset()` fix for replaying one-shots.
   */
  private clips = new Map<CharState, Promise<THREE.AnimationClip | null>>()

  /**
   * Clips injected at runtime, for states whose source is `'dynamic'`.
   *
   * Carries the loader, not just the URL. It used to be a bare URL because the
   * only consumer was `exercise`, whose clips are always SMPL-X BVH from Kimodo,
   * and `build()` hard-coded that. Per-character gestures broke the assumption —
   * a bundled `.fbx` loaded as BVH does not fail loudly, it produces nothing.
   */
  private dynamicClips = new Map<CharState, DynamicClip>()

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
   * Point `state` at a runtime clip (SSE motion, a file picked in the debug
   * selector, or a character's gesture) and drop any clip cached for it.
   *
   * Dropping the cache entry is cheap even when the same gesture is replayed:
   * `build()` goes back through `getAnimationClip`, which is keyed by
   * `vrmUrl|animUrl` and hands back the SAME clip object — so the retarget does
   * not repeat and `mixer.clipAction()` still resolves to the same action.
   */
  update(state: CharState, clip: DynamicClip): void {
    this.dynamicClips.set(state, clip)
    this.clips.delete(state)
  }

  /**
   * Warm a character's gestures during idle time.
   *
   * Same reasoning as `prefetchStatic`, and the same measurements apply: the
   * first load of a clip is a fetch plus a synchronous retarget, 111 ms for
   * Thinking.fbx and 214 ms for random_Bored.fbx. Without this, the first click
   * on a body part pays that stall.
   *
   * This is only possible because gestures are DECLARED per character rather
   * than passed ad-hoc at the moment of the click — knowing the set up front is
   * what makes it warmable.
   */
  prefetchGestures(clips: readonly DynamicClip[]): void {
    const pending = [...clips]

    const step = () => {
      const next = pending.shift()
      if (next === undefined) return
      void this.load({ loader: next.loader, retarget: next.retarget }, next.url)
        .catch(() => null)
        .finally(() => schedule())
    }
    const schedule = () => {
      if (typeof requestIdleCallback === 'function') requestIdleCallback(step, { timeout: 2000 })
      else setTimeout(step, 200)
    }
    schedule()
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
    const dynamic = this.dynamicClips.get(state)
    if (dynamic) {
      return this.load({ loader: dynamic.loader, retarget: dynamic.retarget }, dynamic.url)
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

  /**
   * Structural parameter rather than `StaticSource`: the dynamic path has a URL
   * already and no `match` to give, and inventing a throwaway RegExp to satisfy
   * the type is how the hard-coded `{ loader: 'bvh', retarget: 'smplx' }` slipped
   * in and made every dynamic clip a Kimodo clip.
   */
  private async load(
    source: { loader: 'fbx' | 'bvh'; retarget?: 'smplx' | 'standard'; subclip?: SubclipRange },
    url: string,
  ): Promise<THREE.AnimationClip | null> {
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
