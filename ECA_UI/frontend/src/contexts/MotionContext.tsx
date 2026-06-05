import { createContext, useContext, useRef, useState, type ReactNode } from 'react'

type AssetOption = {
  id: string
  label: string
  url: string
}

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
    .sort((left, right) => left.label.localeCompare(right.label))
}

const BUILTIN_VRM_OPTIONS = buildAssetOptions(VRM_ASSET_MODULES)
const BUILTIN_MOTION_OPTIONS = buildAssetOptions(BVH_ASSET_MODULES)

interface MotionContextType {
  selectedVrmId: string
  setSelectedVrmId: (id: string) => void
  selectedMotionId: string
  setSelectedMotionId: (id: string) => void
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
}

const MotionContext = createContext<MotionContextType | null>(null)

export function MotionProvider({ children }: { children: ReactNode }) {
  const [isPlaying, setIsPlaying] = useState(true)
  const [speed, setSpeed] = useState(1.0)
  const [clipInfo, setClipInfo] = useState<{ tracks: number; duration: number } | null>(null)
  const [selectedVrmId, setSelectedVrmId] = useState(BUILTIN_VRM_OPTIONS[0]?.id ?? '')
  const [selectedMotionId, setSelectedMotionId] = useState(BUILTIN_MOTION_OPTIONS[0]?.id ?? '')

  const onResetRef = useRef<(() => void) | null>(null)

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
