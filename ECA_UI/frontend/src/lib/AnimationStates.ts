/**
 * Animation FSM — SINGLE SOURCE OF TRUTH.
 *
 * See `.claude/plans/animation-fsm-refactor.md` §3.0. Everything about a
 * character state lives in ONE entry of `STATES`: which clip, loop mode, where
 * it goes when it ends, who may enter it, camera framing, facial policy, debug
 * label, cross-fade. Adding an animation means adding one entry — the previous
 * design spread the same information across six parallel maps, three of which
 * had no compile-time safety net, and that is exactly how the two 🔴 bugs in
 * plan v1.0 happened.
 *
 * Two type-level guarantees make omissions impossible:
 *   1. `Record<CharState, StateDef>` (not `Partial`) — a new state without an
 *      entry fails the build.
 *   2. `loop: 'once'` requires `onFinished`; `loop: 'repeat'` forbids it — a
 *      one-shot can never be left without a successor (that would freeze the
 *      character on the clip's last frame).
 */

import type { CanonicalEmotion } from '../avatar/AvatarProfile'

export type CharState =
  | 'idle' //            Default: Standard Idle, looping, close camera
  | 'greeting' //        One-shot on boot: action_greeting → idle
  | 'bored' //           One-shot idle filler: random_* → idle
  | 'thinking_intro' //  Sequence: → thinking_loop
  | 'thinking_loop' //   Sequence: → thinking_outro (when the answer arrives)
  | 'thinking_outro' //  Sequence: → idle
  | 'exercise' //        One-shot generated motion → idle (wide camera + cooldown)

export type CameraMode = 'head' | 'hips'

export interface SubclipRange {
  name: string
  /** Frame indices, inclusive start / exclusive end, at `fps`. */
  start: number
  end: number
  fps: number
}

/**
 * Where a state's clip comes from. Discriminated on `loader` so an impossible
 * combination (FBX + BVH retarget options) cannot be expressed.
 */
export type StaticSource =
  | { loader: 'fbx'; match: RegExp; subclip?: SubclipRange }
  | { loader: 'bvh'; match: RegExp; retarget: 'smplx' | 'standard'; subclip?: SubclipRange }

/**
 * Who is allowed to enter this state. Declared ON THE DESTINATION — that is the
 * whole point: adding a state can never break another state's reachability, and
 * you never edit anyone else's row. Plan v1.0 hand-listed per-source arrays and
 * silently dropped `thinking_loop → exercise` (bug 🔴 #2).
 */
export type Reach =
  /** User/backend-driven — reachable from every state. Chat and motion-ready
   *  events arrive at arbitrary times; blocking them drops user interaction. */
  | 'anytime'
  /** Only while the character is otherwise unoccupied. */
  | 'from-idle'
  /** Link in a fixed sequence: only from these predecessors. */
  | { after: CharState[] }

interface StateBase {
  /** `'dynamic'` = supplied at runtime via `registry.update()` (generated motion). */
  source: StaticSource | 'dynamic'
  reach: Reach
  camera: CameraMode
  /** Facial policy for this body state (plan §9.4): may the face idle-wander? */
  facial: { wander: boolean; hold?: CanonicalEmotion }
  /** Present ⇒ selectable in the debug state dropdown. Absent ⇒ not manual. */
  debugLabel?: string
  /** Cross-fade seconds when LEAVING this state. Default 0.3. */
  crossfade?: number
  /** Per-destination override of `crossfade`, for the rare asymmetric case. */
  crossfadeTo?: Partial<Record<CharState, number>>
  /** Declarative timer trigger: after this long IN this state, go to `to`. */
  autoAfter?: { to: CharState; minSec: number; maxSec: number }
}

export type StateDef = StateBase &
  ({ loop: 'once'; onFinished: CharState } | { loop: 'repeat'; onFinished: null })

export const STATES: Record<CharState, StateDef> = {
  idle: {
    source: { loader: 'fbx', match: /standard idle/i },
    loop: 'repeat',
    onFinished: null,
    reach: 'anytime',
    camera: 'head',
    facial: { wander: true },
    debugLabel: 'Standard Idle',
    // Replaces the old "random idle action" timer in MotionContext.
    autoAfter: { to: 'bored', minSec: 60, maxSec: 120 },
  },
  greeting: {
    source: { loader: 'bvh', match: /action_greeting/i, retarget: 'standard' },
    loop: 'once',
    onFinished: 'idle',
    reach: 'from-idle',
    camera: 'head',
    facial: { wander: true },
    debugLabel: 'Action Greeting',
    crossfade: 0.5,
  },
  bored: {
    source: { loader: 'fbx', match: /random_bored/i },
    loop: 'once',
    onFinished: 'idle',
    reach: 'from-idle',
    camera: 'head',
    facial: { wander: true },
    debugLabel: 'Random Bored',
    crossfade: 0.5,
  },
  thinking_intro: {
    source: {
      loader: 'fbx',
      match: /thinking/i,
      subclip: { name: 'intro', start: 0, end: 38, fps: 30 },
    },
    loop: 'once',
    onFinished: 'thinking_loop',
    reach: 'anytime',
    camera: 'head',
    facial: { wander: false, hold: 'neutral' },
    debugLabel: 'Thinking (intro→loop)',
  },
  thinking_loop: {
    source: {
      loader: 'fbx',
      match: /thinking/i,
      subclip: { name: 'loop', start: 38, end: 75, fps: 30 },
    },
    loop: 'repeat',
    onFinished: null,
    reach: { after: ['thinking_intro'] },
    camera: 'head',
    facial: { wander: false, hold: 'neutral' },
  },
  thinking_outro: {
    source: {
      loader: 'fbx',
      match: /thinking/i,
      subclip: { name: 'outro', start: 75, end: 127, fps: 30 },
    },
    loop: 'once',
    onFinished: 'idle',
    // The answer can arrive before the intro finishes, so both are predecessors.
    reach: { after: ['thinking_intro', 'thinking_loop'] },
    camera: 'head',
    facial: { wander: false, hold: 'neutral' },
  },
  exercise: {
    source: 'dynamic',
    loop: 'once',
    onFinished: 'idle',
    reach: 'anytime',
    camera: 'hips',
    facial: { wander: false, hold: 'neutral' },
    crossfade: 0.8,
  },
}

/**
 * Transition rule as a FUNCTION over `STATES[to].reach`, not a hand-written
 * matrix. Verified equivalent to the v1.1 `ALLOWED_TRANSITIONS` table, with the
 * single harmless addition that `idle → idle` is legal (a no-op).
 */
export function canTransition(from: CharState, to: CharState): boolean {
  if (to === 'idle') return true // every state has a safe way home
  const reach = STATES[to].reach
  if (reach === 'anytime') return true // incl. exercise → exercise (new motion wins)
  if (reach === 'from-idle') return from === 'idle'
  return reach.after.includes(from)
}

/** One-shot successor, or null for looping states. */
export const onFinishedOf = (state: CharState): CharState | null => STATES[state].onFinished

export const loopModeOf = (state: CharState): 'once' | 'repeat' => STATES[state].loop

export const cameraModeOf = (state: CharState): CameraMode => STATES[state].camera

export const facialOf = (state: CharState) => STATES[state].facial

/** Static clip descriptor, or null when the clip is supplied at runtime. */
export function staticSourceOf(state: CharState): StaticSource | null {
  const source = STATES[state].source
  return source === 'dynamic' ? null : source
}

const DEFAULT_CROSSFADE = 0.3

/** Cross-fade seconds for FROM→TO. Replaces the old TransitionConfig module. */
export function crossfadeFor(from: CharState, to: CharState): number {
  const def = STATES[from]
  return def.crossfadeTo?.[to] ?? def.crossfade ?? DEFAULT_CROSSFADE
}

export const CHAR_STATES = Object.keys(STATES) as CharState[]

/**
 * Debug dropdown contents — derived, never hand-maintained. Give a state a
 * `debugLabel` and it appears here with zero UI edits.
 */
export const STATE_OPTIONS: { id: CharState; label: string }[] = CHAR_STATES.filter(
  (id) => STATES[id].debugLabel,
).map((id) => ({ id, label: STATES[id].debugLabel! }))
