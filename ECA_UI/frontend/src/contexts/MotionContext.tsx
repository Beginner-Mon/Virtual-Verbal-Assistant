import { createContext, useContext, useRef, useState, type ReactNode } from 'react'
import type { AvatarController } from '../avatar/AvatarController'

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
const BUILTIN_MOTION_OPTIONS = buildAssetOptions(BVH_ASSET_MODULES)

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
}

const MotionContext = createContext<MotionContextType | null>(null)

export function MotionProvider({ children }: { children: ReactNode }) {
  // Default: no BVH motion — avatar renders in its rest pose. Motion is only
  // applied when the user explicitly picks a motion source AND presses play.
  const [isPlaying, setIsPlaying] = useState(false)
  const [speed, setSpeed] = useState(1.0)
  const [clipInfo, setClipInfo] = useState<{ tracks: number; duration: number } | null>(null)
  const defaultVrmId =
    BUILTIN_VRM_OPTIONS.find((o) => /seele/i.test(o.label))?.id ??
    BUILTIN_VRM_OPTIONS[0]?.id ??
    ''
  const [selectedVrmId, setSelectedVrmId] = useState(defaultVrmId)
  const [selectedMotionId, setSelectedMotionId] = useState('')
  const [cameraMode, setCameraMode] = useState<CameraMode>('head')

  const onResetRef = useRef<(() => void) | null>(null)
  const avatarRef = useRef<AvatarController | null>(null)

  const handleReset = () => {
    if (onResetRef.current) {
      onResetRef.current()
    }
  }

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
