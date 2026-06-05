import { Canvas, useFrame, useLoader, useThree } from '@react-three/fiber'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'
import { VRMLoaderPlugin, VRMHumanBoneName } from '@pixiv/three-vrm'
import type { VRM } from '@pixiv/three-vrm'
import seeleUrl from '../asset/seele.vrm'
import { useTheme } from '../contexts/ThemeContext'
import {
  ContactShadows,
  Environment,
  Stars,
  OrbitControls,
} from '@react-three/drei'
import { useRef, useEffect, useState, Suspense, useMemo } from 'react'
import * as THREE from 'three'
import { loadAndRetargetBVH } from '../lib/bvhToVrm'
import { Play, Pause, RotateCcw, Activity, Sparkles, Sliders } from 'lucide-react'

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

/* ───────────────────── VRM Character with BVH Animation ──────────── */

interface VRMCharacterProps {
  vrmUrl: string
  animationUrl: string
  isPlaying: boolean
  speed: number
  onResetRef: React.MutableRefObject<(() => void) | null>
  onLoaded: (info: { tracks: number; duration: number }) => void
  vrmRef: React.MutableRefObject<VRM | null>
}

function VRMCharacter({
  vrmUrl,
  animationUrl,
  isPlaying,
  speed,
  onResetRef,
  onLoaded,
  vrmRef,
}: VRMCharacterProps) {
  const gltf = useLoader(GLTFLoader, vrmUrl, (loader) => {
    loader.register((parser) => new VRMLoaderPlugin(parser))
  })

  const vrm: VRM = gltf.userData.vrm

  // Expose the VRM instance so the parent can read bone positions
  useEffect(() => {
    vrmRef.current = vrm
    return () => {
      vrmRef.current = null
    }
  }, [vrm, vrmRef])

  const mixerRef = useRef<THREE.AnimationMixer | null>(null)
  const [animLoaded, setAnimLoaded] = useState(false)

  // Load and apply BVH animation after VRM is ready
  useEffect(() => {
    if (!vrm) return

    let cancelled = false

    async function applyBVH() {
      try {
        const clip = await loadAndRetargetBVH(animationUrl, vrm)

        if (cancelled || !clip) return

        // Create mixer on the VRM scene
        const mixer = new THREE.AnimationMixer(vrm.scene)
        const action = mixer.clipAction(clip)

        // Set loop mode for idle animation
        action.setLoop(THREE.LoopRepeat, Infinity)
        action.clampWhenFinished = false
        action.play()

        mixerRef.current = mixer
        setAnimLoaded(true)
        onLoaded({ tracks: clip.tracks.length, duration: clip.duration })

        console.log(
          `[CharacterViewer] BVH animation loaded: ${clip.tracks.length} tracks, ${clip.duration.toFixed(2)}s`,
        )
      } catch (err) {
        console.error('[CharacterViewer] Failed to load BVH animation:', err)
      }
    }

    applyBVH()

    return () => {
      cancelled = true
      if (mixerRef.current) {
        mixerRef.current.stopAllAction()
        mixerRef.current = null
      }
    }
  }, [vrm, animationUrl])

  // Handle play/pause and speed changes
  useEffect(() => {
    if (!mixerRef.current) return
    mixerRef.current.timeScale = isPlaying ? speed : 0
  }, [isPlaying, speed, animLoaded])

  // Expose reset action
  useEffect(() => {
    onResetRef.current = () => {
      if (mixerRef.current) {
        mixerRef.current.setTime(0)
      }
    }
    return () => {
      onResetRef.current = null
    }
  }, [animLoaded])

  // Update both the animation mixer and VRM (for SpringBone physics) every frame
  useFrame((_state, delta) => {
    if (mixerRef.current) {
      mixerRef.current.update(delta)
    }
    if (vrm) {
      vrm.update(delta)
    }
  })

  return <primitive object={vrm.scene} position={[0, -1.5, 0]} />
}

/* ───────────────────────── Floating Particles ────────────────────── */

function FloatingParticles() {
  const count = 200
  const pointsRef = useRef<THREE.Points>(null!)

  const positions = useMemo(() => {
    const arr = new Float32Array(count * 3)
    for (let i = 0; i < count; i++) {
      arr[i * 3 + 0] = (Math.random() - 0.5) * 10
      arr[i * 3 + 1] = (Math.random() - 0.5) * 10
      arr[i * 3 + 2] = (Math.random() - 0.5) * 10
    }
    return arr
  }, [])

  useFrame(({ clock }) => {
    const t = clock.getElapsedTime()
    pointsRef.current.rotation.y = t * 0.02
    pointsRef.current.rotation.x = t * 0.01
  })

  return (
    <points ref={pointsRef}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          args={[positions, 3]}
        />
      </bufferGeometry>
      <pointsMaterial
        size={0.02}
        color="#a78bfa"
        transparent
        opacity={0.6}
        sizeAttenuation
      />
    </points>
  )
}

/* ────────────────────────────── Scene ─────────────────────────────── */

interface SceneProps {
  theme: 'light' | 'dark'
  vrmUrl: string
  animationUrl: string
  isPlaying: boolean
  speed: number
  onResetRef: React.MutableRefObject<(() => void) | null>
  onLoaded: (info: { tracks: number; duration: number }) => void
}

function Scene({
  theme,
  vrmUrl,
  animationUrl,
  isPlaying,
  speed,
  onResetRef,
  onLoaded,
}: SceneProps) {
  const controlsRef = useRef<any>(null)
  const vrmRef = useRef<VRM | null>(null)
  const { camera } = useThree()

  // Reusable vectors to avoid GC pressure
  const hipsPos = useMemo(() => new THREE.Vector3(), [])
  const deltaVec = useMemo(() => new THREE.Vector3(), [])

  // Every frame: make the camera orbit target follow the VRM hips.
  // We also shift the camera position by the same delta so the orbital
  // offset (angle + distance) is preserved while the rig moves.
  useFrame(() => {
    if (!vrmRef.current || !controlsRef.current) return

    const hips = vrmRef.current.humanoid.getNormalizedBoneNode(
      VRMHumanBoneName.Hips,
    )
    if (!hips) return

    hips.getWorldPosition(hipsPos)

    // How far did the hips move since the last frame?
    deltaVec.subVectors(hipsPos, controlsRef.current.target)

    // Shift the camera by the exact same amount so the orbit "diameter"
    // around the skeleton stays constant.
    camera.position.add(deltaVec)

    // Update the controls target to the new hips position
    controlsRef.current.target.copy(hipsPos)
  })

return (
    <>
      <ambientLight intensity={theme === 'dark' ? 0.15 : 0.6} />
      <directionalLight position={[5, 5, 5]} intensity={0.5} castShadow color={theme === 'dark' ? "#e9d5ff" : "#ffffff"} />
      <directionalLight position={[-5, 3, -5]} intensity={0.3} color="#7c3aed" />
      <spotLight position={[0, 5, 0]} angle={0.4} penumbra={1} intensity={0.5} color="#a78bfa" />

      <VRMCharacter
        vrmRef={vrmRef}
        vrmUrl={vrmUrl}
        animationUrl={animationUrl}
        isPlaying={isPlaying}
        speed={speed}
        onResetRef={onResetRef}
        onLoaded={onLoaded}
      />
      <FloatingParticles />

      {/* Ground disc */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -1.5, 0]} receiveShadow>
        <circleGeometry args={[4, 64]} />
        <meshStandardMaterial
          color={theme === 'dark' ? '#2d1b4e' : '#e9d5ff'}
          roughness={0.8}
          metalness={0.1}
          transparent
          opacity={0.85}
        />
      </mesh>

      {/* Grid overlay */}
      <gridHelper
        args={[8, 16, theme === 'dark' ? '#6d28d9' : '#c084fc', theme === 'dark' ? '#3b1265' : '#ddd6fe']}
        position={[0, -1.5, 0]}
      />

      <ContactShadows
        position={[0, -1.5, 0]}
        opacity={theme === 'dark' ? 0.4 : 0.15}
        scale={8}
        blur={2.5}
        far={4}
        color={theme === 'dark' ? "#4c1d95" : "#000000"}
      />

      {theme === 'dark' && <Stars radius={50} depth={50} count={1000} factor={2} saturation={0.5} fade speed={0.5} />}
      <Environment preset={theme === 'dark' ? 'night' : 'city'} />

      {/* Orbital camera: follows hips, enforces minimum distance (radius) */}
      <OrbitControls
        ref={controlsRef}
        enablePan={false}
        enableZoom={true}
        minDistance={2}
        maxDistance={8}
        target={[0, 0, 0]}
      />
    </>
  )
}

/* ───────────────────────── Exported Component ────────────────────── */

export default function CharacterViewer() {
  const { theme } = useTheme()
  const [isPlaying, setIsPlaying] = useState(true)
  const [speed, setSpeed] = useState(1.0)
  const [clipInfo, setClipInfo] = useState<{ tracks: number; duration: number } | null>(null)
  const [selectedVrmId, setSelectedVrmId] = useState(BUILTIN_VRM_OPTIONS[0]?.id ?? '')
  const [selectedMotionId, setSelectedMotionId] = useState(BUILTIN_MOTION_OPTIONS[0]?.id ?? '')

  const onResetRef = useRef<(() => void) | null>(null)

  const vrmUrl = BUILTIN_VRM_OPTIONS.find((option) => option.id === selectedVrmId)?.url ?? seeleUrl
  const animationUrl = BUILTIN_MOTION_OPTIONS.find((option) => option.id === selectedMotionId)?.url ?? ''

  useEffect(() => {
    setClipInfo(null)
    if (onResetRef.current) {
      onResetRef.current()
    }
  }, [vrmUrl, animationUrl])

  const handleReset = () => {
    if (onResetRef.current) {
      onResetRef.current()
    }
  }

  return (
    <div
      className="relative w-full h-full overflow-hidden"
      style={{
        background: theme === 'dark'
          ? 'radial-gradient(ellipse at center, #1a0533 0%, #0a0a12 70%)'
          : 'radial-gradient(ellipse at center, #f3e8ff 0%, #ffffff 70%)',
      }}
    >
      <Canvas camera={{ position: [0, 0, 4], fov: 45 }} gl={{ antialias: true, alpha: true }}>
        <Suspense fallback={null}>
          <Scene
            theme={theme}
            vrmUrl={vrmUrl}
            animationUrl={animationUrl}
            isPlaying={isPlaying}
            speed={speed}
            onResetRef={onResetRef}
            onLoaded={setClipInfo}
          />
        </Suspense>
      </Canvas>

      {/* Bottom gradient */}
      <div className="absolute bottom-0 left-0 right-0 h-24 pointer-events-none bg-gradient-to-t from-background/80 to-transparent" />

      {/* Floating Control panel */}
      <div
        className="absolute bottom-6 right-6 w-80 rounded-2xl p-4 flex flex-col gap-4 backdrop-blur-md bg-card/60 border border-border/40 shadow-2xl transition-all duration-300 hover:bg-card/70"
        style={{ zIndex: 100 }}
      >
        <div className="flex items-center justify-between border-b border-border/20 pb-2">
          <div className="flex items-center gap-2">
            <Activity className="w-4 h-4 text-purple-400 animate-pulse" />
            <span className="text-xs font-semibold tracking-wider text-muted-foreground uppercase">
              Motion Player
            </span>
          </div>
          <div className="flex items-center gap-1.5 bg-purple-500/10 text-purple-300 text-[10px] font-bold px-2 py-0.5 rounded-full border border-purple-500/20">
            <Sparkles className="w-3 h-3 text-purple-400" />
            3D BVH Retargeting
          </div>
        </div>

        <div className="flex items-center justify-between gap-2 p-2 bg-secondary/30 rounded-xl border border-border/10">
          <span className="text-xs font-medium text-muted-foreground">Motion source</span>
          <select
            value={selectedMotionId}
            onChange={(e) => {
              setSelectedMotionId(e.target.value)
            }}
            className="max-w-[180px] bg-transparent text-[10px] text-foreground font-bold uppercase tracking-wider border-none outline-none cursor-pointer"
          >
            {BUILTIN_MOTION_OPTIONS.map((option) => (
              <option key={option.id} value={option.id} className="bg-card text-foreground">
                {option.label}
              </option>
            ))}
          </select>
        </div>

        <div className="grid grid-cols-1 gap-2">
          <label className="flex flex-col gap-1.5 p-2 rounded-xl bg-secondary/20 border border-border/10">
            <span className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">VRM avatar</span>
            <select
              value={selectedVrmId}
              onChange={(e) => {
                setSelectedVrmId(e.target.value)
              }}
              className="bg-transparent text-xs text-foreground font-medium border-none outline-none cursor-pointer"
            >
              {BUILTIN_VRM_OPTIONS.map((option) => (
                <option key={option.id} value={option.id} className="bg-card text-foreground">
                  {option.label}
                </option>
              ))}
            </select>
          </label>
        </div>

        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <button
              onClick={() => setIsPlaying(!isPlaying)}
              className="w-10 h-10 rounded-xl flex items-center justify-center bg-primary text-primary-foreground shadow-lg hover:bg-primary/95 transition-all active:scale-95"
            >
              {isPlaying ? (
                <Pause className="w-4 h-4 fill-current" />
              ) : (
                <Play className="w-4 h-4 fill-current ml-0.5" />
              )}
            </button>
            <button
              onClick={handleReset}
              title="Reset animation"
              className="w-10 h-10 rounded-xl flex items-center justify-center border border-border/40 hover:bg-secondary/40 text-foreground transition-all active:scale-95"
            >
              <RotateCcw className="w-4 h-4" />
            </button>
          </div>

          <div className="flex items-center gap-1.5 bg-secondary/20 border border-border/20 px-2 py-1 rounded-xl">
            <Sliders className="w-3.5 h-3.5 text-muted-foreground" />
            <select
              value={speed}
              onChange={(e) => setSpeed(parseFloat(e.target.value))}
              className="bg-transparent text-xs text-foreground font-medium border-none outline-none cursor-pointer"
            >
              <option value="0.5" className="bg-card text-foreground">0.5x</option>
              <option value="1.0" className="bg-card text-foreground">1.0x</option>
              <option value="1.5" className="bg-card text-foreground">1.5x</option>
              <option value="2.0" className="bg-card text-foreground">2.0x</option>
            </select>
          </div>
        </div>

        {clipInfo && (
          <div className="flex justify-between items-center text-[10px] text-muted-foreground border-t border-border/10 pt-2">
            <span>Bones: {clipInfo.tracks} tracks</span>
            <span>Duration: {clipInfo.duration.toFixed(2)}s</span>
          </div>
        )}
      </div>
    </div>
  )
}