import {
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
import { MOTION_FILES, resolveMotionByName } from '../lib/motionAssets'
import { fetchCharacters, isCompatible, type Character } from '../lib/characters'
import { useAutoAfterTrigger } from '../hooks/useFsmTriggers'
import instrumentalUrl from '../asset/audio/instrumental-ver.mp3'
import { fetchAuthSession } from 'aws-amplify/auth'
import { fetchPreferences } from '../lib/preferences'
import { DEFAULT_CAMERA_CONFIG, type CameraConfig } from '../lib/CameraConfig'
import { MotionContext, type MotionContextType, type SessionMotion, type AssetOption } from '../hooks/useMotion'

export type { SessionMotion, AssetOption, MotionContextType } from '../hooks/useMotion'

function toAssetOption(character: Character): AssetOption {
  return {
    id: character.slug,
    label: character.display_name,
    url: character.vrm_url,
    character,
  }
}

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

        // Synced default character (Neon user_preferences.selected_character_slug)
        // overrides the hardcoded /anne/ fallback — cross-device.
        let syncedSlug: string | null = null
        try {
          const s = await fetchAuthSession()
          if (s.tokens?.idToken) {
            const prefs = await fetchPreferences(abort.signal)
            syncedSlug = prefs.selected_character_slug ?? null
          }
        } catch {
          // guest or 401 → keep local default
        }
        if (abort.signal.aborted) return

        setSelectedVrmId((current) => {
          if (current && options.some((o) => o.id === current)) return current
          if (syncedSlug && options.some((o) => o.id === syncedSlug)) return syncedSlug
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

  /** Add a motion to the replay list, newest first, one entry per job.
   *
   * Separate from playMotionFile because restoring a conversation has to list
   * motions the avatar is not going to play right now — and at mount there is
   * no registry to play them with anyway. */
  const registerSessionMotion = useCallback((m: SessionMotion) => {
    setSessionMotions((prev) => [
      { ...m, label: m.label.trim() || m.jobId.slice(0, 8) },
      ...prev.filter((p) => p.jobId !== m.jobId),
    ])
  }, [])

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
        registerSessionMotion({ jobId: cacheKey, url, label: label ?? '' })
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
    [animController, registerSessionMotion],
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
      registerSessionMotion,
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
      registerSessionMotion,
      handleReset,
      clipInfo,
      isMusicPlaying,
      toggleMusic,
    ],
  )

  return <MotionContext.Provider value={value}>{children}</MotionContext.Provider>
}
