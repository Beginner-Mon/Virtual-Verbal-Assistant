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
import { useMotion } from '../contexts/MotionContext'

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
  const {
    selectedVrmId,
    selectedMotionId,
    isPlaying,
    speed,
    onResetRef,
    setClipInfo,
    vrmOptions,
    motionOptions,
  } = useMotion()

  const vrmUrl = vrmOptions.find((o) => o.id === selectedVrmId)?.url ?? seeleUrl
  const animationUrl = motionOptions.find((o) => o.id === selectedMotionId)?.url ?? ''

  useEffect(() => {
    setClipInfo(null)
    if (onResetRef.current) {
      onResetRef.current()
    }
  }, [vrmUrl, animationUrl])

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
    </div>
  )
}