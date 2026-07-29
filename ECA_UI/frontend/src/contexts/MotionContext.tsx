import { createContext, useContext, useRef, useState, useCallback, useEffect, type ReactNode } from 'react'
import type { AvatarController } from '../avatar/AvatarController'
import instrumentalUrl from '../asset/audio/instrumental-ver.mp3'

type AssetOption = {
  id: string
  label: string
  url: string
}

export type CameraMode = 'head' | 'hips'

const VRM_ASSET_MODULES = import.meta.glob('../asset/**/*.vrm', {
  eager: true,
  import: 'default',
}) as Record<string, string>

const BVH_ASSET_MODULES = import.meta.glob('../asset/**/*.bvh', {
  eager: true,
  query: '?url',
  import: 'default',
}) as Record<string, string>

const FBX_ASSET_MODULES = import.meta.glob('../asset/**/*.fbx', {
  eager: true,
  query: '?url',
  import: 'default',
}) as Record<string, string>

function toLabel(assetPath: string) {
  return assetPath.replace(/^\.\.\/asset\//, '').replace(/\\/g, '/')
}

function buildAssetOptions(modules: Record<string, string>): AssetOption[] {
  return Object.entries(modules)
    .map(([assetPath, url]) => ({
      id: assetPath,
      label: toLabel(assetPath),
      url,
    }))
    .filter((o) => !/bronya_long/i.test(o.label)) // 0 blendshape groups — unusable
    .sort((left, right) => left.label.localeCompare(right.label))
}

const BUILTIN_VRM_OPTIONS = buildAssetOptions(VRM_ASSET_MODULES)
const BUILTIN_MOTION_OPTIONS = buildAssetOptions({
  ...BVH_ASSET_MODULES,
  ...FBX_ASSET_MODULES,
})

interface MotionContextType {
  selectedVrmId: string
  setSelectedVrmId: (id: string) => void
  selectedMotionId: string
  setSelectedMotionId: (id: string) => void
  cameraMode: CameraMode
  setCameraMode: (mode: CameraMode) => void
  isPlaying: boolean
  setIsPlaying: (playing: boolean) => void
  speed: number
  setSpeed: (speed: number) => void
  clipInfo: { tracks: number; duration: number } | null
  setClipInfo: (info: { tracks: number; duration: number } | null) => void
  vrmOptions: AssetOption[]
  motionOptions: AssetOption[]
  onResetRef: React.MutableRefObject<(() => void) | null>
  handleReset: () => void
  /**
   * Imperative handle to the active facial-animation controller. Set by
   * CharacterViewer when a VRM attaches; consumed by DevPanel (and later the
   * chat SSE handler) to drive emotions WITHOUT routing frame data through React
   * state (facial-animation-plan.md §8 rule 3).
   */
  avatarRef: React.MutableRefObject<AvatarController | null>
  isMusicPlaying: boolean
  toggleMusic: () => void
  handleAnimationFinished: () => void
}

const MotionContext = createContext<MotionContextType | null>(null)

export function MotionProvider({ children }: { children: ReactNode }) {
  // Default: Standard Idle animation so the avatar shows a natural idle pose
  // on load instead of the stiff T-pose.
  const [speed, setSpeed] = useState(1.0)
  const [clipInfo, setClipInfo] = useState<{ tracks: number; duration: number } | null>(null)
  const defaultVrmId =
    BUILTIN_VRM_OPTIONS.find((o) => /seele/i.test(o.label))?.id ??
    BUILTIN_VRM_OPTIONS[0]?.id ??
    ''
  const defaultMotionId =
    BUILTIN_MOTION_OPTIONS.find((o) => /action_greeting/i.test(o.label))?.id ??
    BUILTIN_MOTION_OPTIONS.find((o) => /standard idle/i.test(o.label))?.id ??
    BUILTIN_MOTION_OPTIONS[0]?.id ??
    ''
  const [selectedVrmId, setSelectedVrmId] = useState(defaultVrmId)
  const [selectedMotionId, setSelectedMotionId] = useState(defaultMotionId)
  const [isPlaying, setIsPlaying] = useState(true)
  const [cameraMode, setCameraMode] = useState<CameraMode>('head')

  const onResetRef = useRef<(() => void) | null>(null)
  const avatarRef = useRef<AvatarController | null>(null)

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

  const handleReset = () => {
    if (onResetRef.current) {
      onResetRef.current()
    }
  }

  const handleAnimationFinished = useCallback(() => {
    const idleId = BUILTIN_MOTION_OPTIONS.find((o) => /standard idle/i.test(o.label))?.id ?? BUILTIN_MOTION_OPTIONS[0]?.id ?? ''
    setSelectedMotionId(idleId)
  }, [])

  // Random Idle Action Logic
  useEffect(() => {
    const currentOption = BUILTIN_MOTION_OPTIONS.find(o => o.id === selectedMotionId)
    if (!isPlaying || !currentOption) return

    // Only trigger random action if we are currently in an "idle" state
    if (!/idle/i.test(currentOption.label)) return

    const randomActions = BUILTIN_MOTION_OPTIONS.filter(o => /random_/i.test(o.label))
    if (randomActions.length === 0) return

    // Random timeout between 10s and 30s
    const timeoutSeconds = 10 + Math.random() * 20
    const timerId = setTimeout(() => {
      const randomAction = randomActions[Math.floor(Math.random() * randomActions.length)]
      setSelectedMotionId(randomAction.id)
    }, timeoutSeconds * 1000)

    return () => clearTimeout(timerId)
  }, [selectedMotionId, isPlaying])

  return (
    <MotionContext.Provider
      value={{
        selectedVrmId,
        setSelectedVrmId,
        selectedMotionId,
        setSelectedMotionId,
        cameraMode,
        setCameraMode,
        isPlaying,
        setIsPlaying,
        speed,
        setSpeed,
        clipInfo,
        setClipInfo,
        vrmOptions: BUILTIN_VRM_OPTIONS,
        motionOptions: BUILTIN_MOTION_OPTIONS,
        onResetRef,
        handleReset,
        avatarRef,
        isMusicPlaying,
        toggleMusic,
        handleAnimationFinished,
      }}
    >
      {children}
    </MotionContext.Provider>
  )
}

export function useMotion(): MotionContextType {
  const ctx = useContext(MotionContext)
  if (!ctx) {
    throw new Error('useMotion must be used within MotionProvider')
  }
  return ctx
}
