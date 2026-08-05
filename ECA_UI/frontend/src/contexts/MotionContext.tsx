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
import type { AnimationRegistry } from '../lib/AnimationRegistry'
import { CameraController } from '../lib/CameraController'
import { STATE_OPTIONS, type CameraMode, type CharState } from '../lib/AnimationStates'
import { MOTION_FILES, type MotionFile } from '../lib/motionAssets'
import { useAutoAfterTrigger } from '../hooks/useFsmTriggers'
import instrumentalUrl from '../asset/audio/instrumental-ver.mp3'
import { DEFAULT_CAMERA_CONFIG, type CameraConfig } from '../lib/CameraConfig'

export type { CameraMode }

type AssetOption = {
  id: string
  label: string
  url: string
}

const VRM_ASSET_MODULES = import.meta.glob('../asset/**/*.vrm', {
  eager: true,
  import: 'default',
}) as Record<string, string>

function buildVrmOptions(modules: Record<string, string>): AssetOption[] {
  return Object.entries(modules)
    .map(([assetPath, url]) => ({
      id: assetPath,
      label: assetPath.replace(/^\.\.\/asset\//, '').replace(/\\/g, '/'),
      url,
    }))
    .filter((o) => !/bronya_long/i.test(o.label)) // 0 blendshape groups — unusable
    .sort((left, right) => left.label.localeCompare(right.label))
}

const BUILTIN_VRM_OPTIONS = buildVrmOptions(VRM_ASSET_MODULES)

interface MotionContextType {
  selectedVrmId: string
  setSelectedVrmId: (id: string) => void
  vrmOptions: AssetOption[]

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
  playMotionFile: (url: string) => Promise<boolean>
  motionFileOptions: MotionFile[]

  /** Camera framing. FSM-driven; the setter is a manual debug override. */
  cameraMode: CameraMode
  setCameraMode: (mode: CameraMode) => void
  cameraConfig: CameraConfig
  setCameraConfig: (config: CameraConfig) => void

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
  const defaultVrmId =
    BUILTIN_VRM_OPTIONS.find((o) => /anne/i.test(o.label))?.id ?? BUILTIN_VRM_OPTIONS[0]?.id ?? ''

  const [selectedVrmId, setSelectedVrmId] = useState(defaultVrmId)
  const [isPlaying, setIsPlaying] = useState(true)
  const [speed, setSpeed] = useState(1.0)
  const [clipInfo, setClipInfo] = useState<{ tracks: number; duration: number } | null>(null)
  const [cameraMode, setCameraModeState] = useState<CameraMode>('head')
  const [cameraConfig, setCameraConfig] = useState<CameraConfig>(DEFAULT_CAMERA_CONFIG)

  // Held as state, not a ref: the play/pause effect and the auto-after timer
  // must re-run when the controller is swapped (model change / StrictMode).
  const [animController, setAnimController] = useState<AnimationController | null>(null)
  const [currentState, setCurrentState] = useState<CharState>('idle')
  const registryRef = useRef<AnimationRegistry | null>(null)
  const avatarRef = useRef<AvatarController | null>(null)

  // Outlives model swaps: camera framing is a property of the FSM state, not of
  // the loaded VRM.
  const cameraController = useMemo(() => new CameraController(setCameraModeState), [])
  useEffect(() => () => cameraController.dispose(), [cameraController])

  const attachControllers = useCallback(
    (controller: AnimationController, registry: AnimationRegistry) => {
      setAnimController(controller)
      registryRef.current = registry
      setCurrentState(controller.currentState)

      const off = controller.on('stateChanged', (state) => {
        setCurrentState(state)
        cameraController.onStateChanged(state)
      })

      return () => {
        off()
        setAnimController((prev) => (prev === controller ? null : prev))
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
    async (url: string) => {
      const registry = registryRef.current
      if (!registry || !animController || !url) return false
      // Registry first: `transitionTo('exercise')` resolves the clip through it,
      // so updating afterwards would play the previous motion.
      registry.update('exercise', url)
      return animController.transitionTo('exercise')
    },
    [animController],
  )

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
      vrmIds: BUILTIN_VRM_OPTIONS.map((o) => o.id),
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
  }, [transitionTo, playMotionFile, animController, cameraController, currentState])

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
      vrmOptions: BUILTIN_VRM_OPTIONS,
      transitionTo,
      currentState,
      stateOptions: STATE_OPTIONS,
      playMotionFile,
      motionFileOptions: MOTION_FILES,
      cameraMode,
      setCameraMode,
      cameraConfig,
      setCameraConfig,
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
      transitionTo,
      currentState,
      playMotionFile,
      cameraMode,
      setCameraMode,
      cameraConfig,
      setCameraConfig,
      attachControllers,
      isPlaying,
      speed,
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
