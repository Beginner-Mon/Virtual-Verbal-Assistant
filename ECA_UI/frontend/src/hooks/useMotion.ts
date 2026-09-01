import { createContext, useContext } from 'react'
import type { AvatarController } from '../avatar/AvatarController'
import type { AnimationController } from '../lib/AnimationController'
import type { AnimationRegistry } from '../lib/AnimationRegistry'
import type { CameraMode, CharState } from '../lib/AnimationStates'
import { STATE_OPTIONS } from '../lib/AnimationStates'
import type { MotionFile } from '../lib/motionAssets'
import type { Character } from '../lib/characters'
import type { UserActivity } from '../avatar/userActivity'
import type { CameraConfig } from '../lib/CameraConfig'

/** One motion the backend rendered and this page can replay. */
export interface SessionMotion {
  /** Content hash of the request. Doubles as the clip cache key. */
  jobId: string
  /**
   * Signed CloudFront URL, when one is already in hand.
   *
   * Absent for a motion restored from conversation history: that page load has
   * no cached clip and no URL, and a signed URL only lives five minutes, so
   * fetching one at restore time would hand the picker a link that is dead
   * before anybody clicks it. The picker resolves a fresh one when picked.
   */
  url?: string
  /** What the user asked for, e.g. "động tác squat". */
  label: string
}

export type { CameraMode }

type AssetOption = {
  /** characters.slug — also what gets sent back as ChatRequest.persona_id. */
  id: string
  label: string
  url: string
  character?: Character
}

export interface MotionContextType {
  selectedVrmId: string
  setSelectedVrmId: (id: string) => void
  vrmOptions: AssetOption[]
  /** False while the initial character request is in flight — nothing to render yet. */
  vrmOptionsLoading: boolean
  /** Set when the character could not be fetched; the picker shows it verbatim. */
  vrmOptionsError: string | null
  /** Ensure lite catalog (4-col) for AvatarsPanel — lazy, called when panel opens. */
  ensureCatalogLoaded: () => Promise<void>

  /**
   * FSM — the single entry point for every state change. Returns false when the
   * transition is disallowed or the clip is unavailable (e.g. before the model
   * has finished loading), so callers can stay silent instead of guessing.
   */
  transitionTo: (state: CharState) => Promise<boolean>
  /** Current body state, pushed from the controller's `stateChanged` event. */
  currentState: CharState
  /** Debug dropdown contents — derived from STATES, never hand-maintained. */
  stateOptions: typeof STATE_OPTIONS

  /**
   * Play an arbitrary motion file through the `exercise` state. Used by the SSE
   * motion handler and by the debug file selector, which is the only way to
   * verify the Kimodo NPZ→BVH pipeline without a backend (plan §4.3).
   */
  playMotionFile: (url: string, cacheKey?: string, label?: string) => Promise<boolean>
  /** List a motion for replay without playing it — used when restoring a
   *  conversation, where the avatar has nothing loaded yet. */
  registerSessionMotion: (m: SessionMotion) => void
  /**
   * Motions the backend rendered during this page session, newest first.
   *
   * `exercise` is a one-shot: it plays through and returns to idle, so a user
   * who looked away has missed it. This list is how they get it back — the
   * motion picker replays from here. Nothing is re-downloaded: the clip is
   * cached under its job_id, so a replay never touches the network, and the
   * stored `url` is only a fallback for the first play (its CloudFront
   * signature expires after five minutes and is not needed once the clip is
   * in memory).
   *
   * Page-session scoped, deliberately. Surviving a reload needs the job ids in
   * the conversation history, which is a separate piece of work.
   */
  sessionMotions: SessionMotion[]
  motionFileOptions: MotionFile[]

  /** Camera framing. FSM-driven; the setter is a manual debug override. */
  cameraMode: CameraMode
  setCameraMode: (mode: CameraMode) => void
  cameraConfig: CameraConfig
  setCameraConfig: (config: CameraConfig) => void

  /**
   * Report something the user did and let the CHARACTER decide what it means.
   *
   * The caller names an interaction, never an animation or an emotion: the
   * binding lives in the character's profile, so a model can bring its own
   * reactions from the database without any call site changing. Resolves true
   * when the character actually reacted.
   */
  dispatchActivity: (activity: UserActivity) => Promise<boolean>
  /**
   * Warm this character's gestures during idle time. Call once the avatar has
   * attached — that is when its gesture set becomes known.
   */
  prefetchGestures: () => void

  /**
   * Wiring hook for the component that owns the VRM: it creates the controller
   * pair and hands them over here. Returns the detach function.
   */
  attachControllers: (controller: AnimationController, registry: AnimationRegistry) => () => void

  isPlaying: boolean
  setIsPlaying: (playing: boolean) => void
  speed: number
  setSpeed: (speed: number) => void
  handleReset: () => void
  clipInfo: { tracks: number; duration: number } | null
  setClipInfo: (info: { tracks: number; duration: number } | null) => void

  /**
   * Imperative handle to the active facial-animation controller. Set by
   * CharacterViewer when a VRM attaches; consumed by the dev panel (and later the
   * chat SSE handler) to drive emotions WITHOUT routing frame data through React
   * state (facial-animation-plan.md §8 rule 3).
   */
  avatarRef: React.MutableRefObject<AvatarController | null>
  isMusicPlaying: boolean
  toggleMusic: () => void
}

export const MotionContext = createContext<MotionContextType | null>(null)

export function useMotion(): MotionContextType {
  const ctx = useContext(MotionContext)
  if (!ctx) {
    throw new Error('useMotion must be used within MotionProvider')
  }
  return ctx
}

// Re-export for consumers that need the type without importing from context file
export type { AssetOption }
export { STATE_OPTIONS }
