import {
  createContext,
  useContext,
  useRef,
  useState,
  useCallback,
  useEffect,
  useMemo,
  type ReactNode,
} from 'react'
import type { AvatarController } from '../avatar/AvatarController'
import type { AnimationController } from '../lib/AnimationController'
import { loaderForUrl } from '../lib/AnimationRegistry'
import type { AnimationRegistry } from '../lib/AnimationRegistry'
import { ActivityDispatcher } from '../avatar/ActivityDispatcher'
import type { UserActivity } from '../avatar/userActivity'
import { CameraController } from '../lib/CameraController'
import { STATE_OPTIONS, type CameraMode, type CharState } from '../lib/AnimationStates'
import { MOTION_FILES, resolveMotionByName, type MotionFile } from '../lib/motionAssets'
import { fetchCharacters, isCompatible, type Character } from '../lib/characters'
import { useAutoAfterTrigger } from '../hooks/useFsmTriggers'
import instrumentalUrl from '../asset/audio/instrumental-ver.mp3'
import { DEFAULT_CAMERA_CONFIG, type CameraConfig } from '../lib/CameraConfig'

/** One motion the backend rendered this session. */
export interface SessionMotion {
  /** Content hash of the request. Doubles as the clip cache key. */
  jobId: string
  /** Signed CloudFront URL from the first play. Expires; the cache does not. */
  url: string
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

function toAssetOption(character: Character): AssetOption {
  return {
    id: character.slug,
    label: character.display_name,
    url: character.vrm_url,
    character,
  }
}

interface MotionContextType {
  selectedVrmId: string
  setSelectedVrmId: (id: string) => void
  vrmOptions: AssetOption[]
  /** False while the catalog request is in flight — nothing to render yet. */
  vrmOptionsLoading: boolean
  /** Set when the catalog could not be fetched; the picker shows it verbatim. */
  vrmOptionsError: string | null

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

const MotionContext = createContext<MotionContextType | null>(null)

export function MotionProvider({ children }: { children: ReactNode }) {
  const [vrmOptions, setVrmOptions] = useState<AssetOption[]>([])
  const [vrmOptionsLoading, setVrmOptionsLoading] = useState(true)
  const [vrmOptionsError, setVrmOptionsError] = useState<string | null>(null)
  const [selectedVrmId, setSelectedVrmId] = useState('')

  // The catalog comes from the CDN now, so the option list starts empty and the
  // default selection can only be made once it arrives — hence the effect
  // rather than a lazily-initialised useState.
  useEffect(() => {
    const abort = new AbortController()

    async function load() {
      try {
        // A model with incompatible_reasons still appears in the picker, greyed
        // out with the reason — it is data the user should see, not a row to
        // hide. Only the default selection skips them.
        const options = (await fetchCharacters(abort.signal)).map(toAssetOption)

        if (abort.signal.aborted) return

        setVrmOptions(options)
        setSelectedVrmId((current) => {
          if (current && options.some((o) => o.id === current)) return current
          const usable = options.filter((o) => !o.character || isCompatible(o.character))
          const preferred = usable.find((o) => /anne/i.test(o.id) || /anne/i.test(o.label))
          return preferred?.id ?? usable[0]?.id ?? options[0]?.id ?? ''
        })
        setVrmOptionsError(null)
      } catch (err) {
        if (abort.signal.aborted || (err as Error)?.name === 'AbortError') return
        console.error('[MotionContext] character catalog failed', err)
        setVrmOptionsError(
          err instanceof Error ? err.message : 'Could not load the character catalog'
        )
      } finally {
        if (!abort.signal.aborted) setVrmOptionsLoading(false)
      }
    }

    void load()
    return () => abort.abort()
  }, [])

  const [isPlaying, setIsPlaying] = useState(true)
  const [speed, setSpeed] = useState(1.0)
  const [sessionMotions, setSessionMotions] = useState<SessionMotion[]>([])
  const [clipInfo, setClipInfo] = useState<{ tracks: number; duration: number } | null>(null)
  const [cameraMode, setCameraModeState] = useState<CameraMode>('head')
  const [cameraConfig, setCameraConfig] = useState<CameraConfig>(DEFAULT_CAMERA_CONFIG)

  // Held as state, not a ref: the play/pause effect and the auto-after timer
  // must re-run when the controller is swapped (model change / StrictMode).
  const [animController, setAnimController] = useState<AnimationController | null>(null)
  const [currentState, setCurrentState] = useState<CharState>('idle')
  const registryRef = useRef<AnimationRegistry | null>(null)
  const avatarRef = useRef<AvatarController | null>(null)
  // Mirrors `animController` for consumers that must read it outside a render —
  // the dispatcher below is built once and would otherwise close over the first
  // (null) value forever.
  const animControllerRef = useRef<AnimationController | null>(null)

  /**
   * User activity -> what this character does about it (ActivityDispatcher).
   *
   * Built here because this is where all three handles converge: the avatar
   * controller carries the profile with the bindings, the animation controller
   * runs the clip, and the registry resolves it. Reading each through a getter
   * keeps a click from driving a model that has since been swapped out.
   */
  const dispatcherRef = useRef<ActivityDispatcher | null>(null)
  // Built on first use rather than in a useMemo or an effect. A useMemo factory
  // that closes over refs reads as a render-time ref access; an effect would not
  // have run yet when a child's effect calls prefetchGestures, since effects
  // fire child-first. Lazy init depends on neither.
  const getDispatcher = useCallback(() => {
    dispatcherRef.current ??= new ActivityDispatcher({
      getAvatar: () => avatarRef.current,
      getAnim: () => animControllerRef.current,
      getRegistry: () => registryRef.current,
      resolveBuiltIn: resolveMotionByName,
    })
    return dispatcherRef.current
  }, [])

  // Outlives model swaps: camera framing is a property of the FSM state, not of
  // the loaded VRM.
  const cameraController = useMemo(() => new CameraController(setCameraModeState), [])
  useEffect(() => () => cameraController.dispose(), [cameraController])

  const attachControllers = useCallback(
    (controller: AnimationController, registry: AnimationRegistry) => {
      setAnimController(controller)
      animControllerRef.current = controller
      registryRef.current = registry
      setCurrentState(controller.currentState)

      const off = controller.on('stateChanged', (state) => {
        setCurrentState(state)
        cameraController.onStateChanged(state)
      })

      return () => {
        off()
        setAnimController((prev) => (prev === controller ? null : prev))
        if (animControllerRef.current === controller) animControllerRef.current = null
        if (registryRef.current === registry) registryRef.current = null
      }
    },
    [cameraController],
  )

  // Debug playback controls are pushed into the controller; the mixer owns the
  // actual clock, so nothing here runs per frame.
  useEffect(() => {
    animController?.setPlaying(isPlaying)
  }, [animController, isPlaying])

  useEffect(() => {
    animController?.setSpeed(speed)
  }, [animController, speed])

  // Declarative timer trigger (idle → bored), driven by STATES[state].autoAfter.
  useAutoAfterTrigger(animController, currentState)

  const transitionTo = useCallback(
    async (state: CharState) => (await animController?.transitionTo(state)) ?? false,
    [animController],
  )

  const playMotionFile = useCallback(
    async (url: string, cacheKey?: string, label?: string) => {
      const registry = registryRef.current
      if (!registry || !animController || !url) return false

      // Only backend renders are listed: a cacheKey means this came from the
      // motion queue, where `label` is what the user asked for. The bundled
      // debug files carry neither and must not accumulate in the list.
      // Newest first, deduped by job id — asking twice for the same movement
      // is one entry, the same way it is one render.
      if (cacheKey) {
        setSessionMotions((prev) => [
          { jobId: cacheKey, url, label: label?.trim() || cacheKey.slice(0, 8) },
          ...prev.filter((m) => m.jobId !== cacheKey),
        ])
      }
      // Registry first: `transitionTo('exercise')` resolves the clip through it,
      // so updating afterwards would play the previous motion.
      // The loader is inferred rather than assumed: the debug selector lists the
      // .fbx files alongside the .bvh ones, and every dynamic clip used to be
      // loaded as SMPL-X BVH regardless.
      //
      // cacheKey is the motion's job_id when this came from the chat stream.
      // The URL is a CloudFront signature that changes per request, so without
      // it a replayed motion is re-fetched and re-retargeted every time — see
      // DynamicClip.cacheKey. The debug selector passes bundled URLs and no
      // key, which stays correct because those URLs are stable.
      registry.update('exercise', { url, loader: loaderForUrl(url), retarget: 'smplx', cacheKey })
      return animController.transitionTo('exercise')
    },
    [animController],
  )

  const dispatchActivity = useCallback(
    (activity: UserActivity) => getDispatcher().dispatch(activity),
    [getDispatcher],
  )

  const prefetchGestures = useCallback(() => getDispatcher().prefetch(), [getDispatcher])

  const setCameraMode = useCallback(
    (mode: CameraMode) => cameraController.setMode(mode),
    [cameraController],
  )

  const handleReset = useCallback(() => animController?.restart(), [animController])

  // Dev-only test handle. Lives here (always mounted) rather than in the debug
  // panel, so automated checks don't depend on a panel being open.
  const stateHistoryRef = useRef<CharState[]>([])
  useEffect(() => {
    if (!import.meta.env.DEV) return
    stateHistoryRef.current.push(currentState)
  }, [currentState])

  useEffect(() => {
    if (!import.meta.env.DEV) return
    ;(window as unknown as { __fsm?: unknown }).__fsm = {
      transitionTo,
      playMotionFile,
      setVrm: setSelectedVrmId,
      vrmIds: vrmOptions.map((o) => o.id),
      get state() {
        return animController?.currentState ?? null
      },
      get hasPose() {
        return animController?.hasPose ?? false
      },
      get cameraMode() {
        return cameraController.cameraMode
      },
      get history() {
        return [...stateHistoryRef.current]
      },
    }
  }, [transitionTo, playMotionFile, animController, cameraController, currentState, vrmOptions])

  // Background music state
  const audioRef = useRef<HTMLAudioElement | null>(null)
  const [isMusicPlaying, setIsMusicPlaying] = useState(false)

  const toggleMusic = useCallback(() => {
    if (!audioRef.current) {
      const audio = new Audio(instrumentalUrl)
      audio.loop = true
      audio.volume = 0.35
      audioRef.current = audio
    }
    const audio = audioRef.current
    if (isMusicPlaying) {
      audio.pause()
      setIsMusicPlaying(false)
    } else {
      audio.play().then(() => setIsMusicPlaying(true)).catch(() => {})
    }
  }, [isMusicPlaying])

  // Memoized: an inline object literal here would be a new value on EVERY
  // provider render, re-rendering every consumer — including CharacterViewer,
  // which owns the <Canvas>. FSM transitions push `currentState` through this
  // provider, so that cascade would fire several times per chat turn.
  const value = useMemo<MotionContextType>(
    () => ({
      selectedVrmId,
      setSelectedVrmId,
      vrmOptions,
      vrmOptionsLoading,
      vrmOptionsError,
      transitionTo,
      currentState,
      stateOptions: STATE_OPTIONS,
      playMotionFile,
      sessionMotions,
      motionFileOptions: MOTION_FILES,
      cameraMode,
      setCameraMode,
      cameraConfig,
      setCameraConfig,
      dispatchActivity,
      prefetchGestures,
      attachControllers,
      isPlaying,
      setIsPlaying,
      speed,
      setSpeed,
      handleReset,
      clipInfo,
      setClipInfo,
      avatarRef,
      isMusicPlaying,
      toggleMusic,
    }),
    [
      selectedVrmId,
      vrmOptions,
      vrmOptionsLoading,
      vrmOptionsError,
      transitionTo,
      currentState,
      playMotionFile,
      cameraMode,
      setCameraMode,
      cameraConfig,
      setCameraConfig,
      dispatchActivity,
      prefetchGestures,
      attachControllers,
      isPlaying,
      speed,
      sessionMotions,
      handleReset,
      clipInfo,
      isMusicPlaying,
      toggleMusic,
    ],
  )

  return <MotionContext.Provider value={value}>{children}</MotionContext.Provider>
}

export function useMotion(): MotionContextType {
  const ctx = useContext(MotionContext)
  if (!ctx) {
    throw new Error('useMotion must be used within MotionProvider')
  }
  return ctx
}
