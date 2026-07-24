import { Canvas, useFrame, useLoader, useThree } from '@react-three/fiber'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'
import { VRMLoaderPlugin, VRMHumanBoneName } from '@pixiv/three-vrm'
import type { VRM } from '@pixiv/three-vrm'
import seeleUrl from '../asset/seele.vrm'
import { Html } from '@react-three/drei'
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
import { type CameraMode, useMotion } from '../contexts/MotionContext'
import { AvatarController } from '../avatar/AvatarController'
import { loadProfile } from '../avatar/AvatarProfile'
import AvatarDevPanel from '../avatar/AvatarDevPanel'

const CAMERA_MODES: Record<CameraMode, { boneName: VRMHumanBoneName; cameraOffset: THREE.Vector3 }> = {
  head: {
    boneName: VRMHumanBoneName.Head,
    cameraOffset: new THREE.Vector3(0, 0.5, 0),
  },
  hips: {
    boneName: VRMHumanBoneName.Hips,
    cameraOffset: new THREE.Vector3(0, 2.1, 0),
  },
}

/* ───────────────────── VRM Character with BVH Animation ──────────── */

interface VRMCharacterProps {
  vrmUrl: string
  modelId: string
  animationUrl: string
  isPlaying: boolean
  speed: number
  onResetRef: React.MutableRefObject<(() => void) | null>
  onLoaded: (info: { tracks: number; duration: number }) => void
  vrmRef: React.MutableRefObject<VRM | null>
  avatarRef: React.MutableRefObject<AvatarController | null>
}

function VRMCharacter({
  vrmUrl,
  modelId,
  animationUrl,
  isPlaying,
  speed,
  onResetRef,
  onLoaded,
  vrmRef,
  avatarRef,
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
  const avatarControllerRef = useRef<AvatarController | null>(null)
  const [animLoaded, setAnimLoaded] = useState(false)

  // Facial-animation controller lifecycle: attach on VRM load, detach on
  // model change / unmount. Kept out of React state — this ref IS the handle
  // (facial-animation-plan.md §8 rules 2-4).
  useEffect(() => {
    if (!vrm) return
    const controller = new AvatarController(vrm, loadProfile(modelId))
    avatarControllerRef.current = controller
    avatarRef.current = controller
    return () => {
      controller.detach()
      if (avatarRef.current === controller) avatarRef.current = null
      avatarControllerRef.current = null
    }
  }, [vrm, modelId, avatarRef])

  // Load and apply BVH animation after VRM is ready. Skip when no motion is
  // selected (animationUrl === '') so the avatar rests in its natural pose.
  useEffect(() => {
    if (!vrm || !animationUrl) return

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

  // Update body animation, then facial expressions, then the VRM itself.
  // Order is mandatory (§8 rule 1): the avatar controller calls setValue, and
  // vrm.update applies those weights via expressionManager.update() — so the
  // tick must land BETWEEN mixer.update and vrm.update.
  useFrame((_state, delta) => {
    if (mixerRef.current) {
      mixerRef.current.update(delta)
    }
    if (avatarControllerRef.current) {
      avatarControllerRef.current.tick(delta)
    }
    if (vrm) {
      vrm.update(delta)
    }
  })

  return (
    <group position={[0, -1.5, 0]} rotation={[0, Math.PI, 0]}>
      <primitive object={vrm.scene} />
    </group>
  )
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
  modelId: string
  animationUrl: string
  isPlaying: boolean
  speed: number
  onResetRef: React.MutableRefObject<(() => void) | null>
  onLoaded: (info: { tracks: number; duration: number }) => void
  avatarRef: React.MutableRefObject<AvatarController | null>
}

function Scene({
  theme,
  vrmUrl,
  modelId,
  animationUrl,
  isPlaying,
  speed,
  onResetRef,
  onLoaded,
  avatarRef,
}: SceneProps) {
  const controlsRef = useRef<any>(null)
  const vrmRef = useRef<VRM | null>(null)
  const { camera } = useThree()
  const cameraInitializedRef = useRef(false)
  const { cameraMode } = useMotion()

  useEffect(() => {
    camera.up.set(0, 0, 1)
  }, [camera])

  useEffect(() => {
    cameraInitializedRef.current = false
  }, [cameraMode, vrmUrl])

  // Reusable vectors to avoid GC pressure
  const followPos = useMemo(() => new THREE.Vector3(), [])
  const deltaVec = useMemo(() => new THREE.Vector3(), [])

  // Every frame: make the camera orbit target follow the selected bone.
  // We also shift the camera position by the same delta so the orbital
  // offset (angle + distance) is preserved while the rig moves.
  useFrame(() => {
    if (!vrmRef.current || !controlsRef.current) return

    const mode = CAMERA_MODES[cameraMode]
    const bone =
      vrmRef.current.humanoid.getNormalizedBoneNode(mode.boneName) ??
      vrmRef.current.humanoid.getNormalizedBoneNode(VRMHumanBoneName.Hips)
    if (!bone) return

    bone.getWorldPosition(followPos)

    if (!cameraInitializedRef.current) {
      controlsRef.current.target.copy(followPos)
      camera.position.copy(followPos).add(mode.cameraOffset)
      camera.lookAt(followPos)
      controlsRef.current.update()
      cameraInitializedRef.current = true
      return
    }

    // How far did the follow point move since the last frame?
    deltaVec.subVectors(followPos, controlsRef.current.target)

    // Move the camera by the same 3D delta so the distance to the model
    // stays identical even if the model shifts on any axis.
    camera.position.add(deltaVec)

    // Update the controls target to the new follow point
    controlsRef.current.target.copy(followPos)
    controlsRef.current.update()
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
        modelId={modelId}
        animationUrl={animationUrl}
        isPlaying={isPlaying}
        speed={speed}
        onResetRef={onResetRef}
        onLoaded={onLoaded}
        avatarRef={avatarRef}
      />
      <FloatingParticles />

     {/* Ground disc — XZ plane in a Y-up world */}
      <mesh position={[0, -1.5, 0]} rotation={[-Math.PI / 2, 0, 0]} receiveShadow>
        <circleGeometry args={[4, 64]} />
        <meshStandardMaterial
          color={theme === 'dark' ? '#7c3aed' : '#7c3aed'}
          roughness={0.8}
          metalness={0.1}
          transparent
          opacity={0.65}
        />
      </mesh>

      {/* Grid on XZ plane */}
      <mesh position={[0, -1.5, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <planeGeometry args={[8, 8, 16, 16]} />
        <meshStandardMaterial
          color={theme === 'dark' ? '#7c3aed' : '#7c3aed'}
          roughness={1}
          metalness={0}
          transparent
          opacity={0.4}
          wireframe
        />
      </mesh>

      {/* Grid overlay */}
      <gridHelper
        args={[8, 16, theme === 'dark' ? '#6d28d9' : '#c084fc', theme === 'dark' ? '#3b1265' : '#ddd6fe']}
        position={[0, -1.5, 0]}
      />
      <primitive object={new THREE.AxesHelper(3)} />

      {/* Axis labels */}
      <Html position={[3.2, 0, 0]}>
        <span style={{ color: 'red', fontWeight: 'bold', fontSize: 14 }}>X</span>
      </Html>
      <Html position={[0, 3.2, 0]}>
        <span style={{ color: 'green', fontWeight: 'bold', fontSize: 14 }}>Y</span>
      </Html>
      <Html position={[0, 0, 3.2]}>
        <span style={{ color: 'blue', fontWeight: 'bold', fontSize: 14 }}>Z</span>
      </Html>

      <ContactShadows
        position={[0, -1.5, 0]}
        opacity={theme === 'dark' ? 0.4 : 0.15}
        scale={8}
        blur={2.5}
        far={4}
        color={theme === 'dark' ? "#4c1d95" : "#000000"}
      />

      {theme === 'dark' && <Stars radius={50} depth={50} count={1000} factor={2} saturation={0.5} fade speed={0.5} />}
      {/*
        resolution={64}: the default PMREM environment texture (256px cubemap) is large
        enough that PMREM prefiltering (PMREMGGXConvolution) triggers a D3D11 device
        removal (DXGI 0x887A0020) → WebGL context loss → blank canvas on GPUs with a
        tight texture budget (repro'd via ANGLE/D3D11). A 64px IBL map keeps the
        image-based lighting while staying well under the allocation limit.
      */}
      <Environment preset={theme === 'dark' ? 'night' : 'city'} resolution={64} />

      {/* Orbital camera: follows hips, enforces minimum distance (radius) */}
      <OrbitControls
        ref={controlsRef}
        enablePan={false}
        enableZoom={true}
        minDistance={1}
        maxDistance={20}
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
    avatarRef,
  } = useMotion()

  const selectedVrm = vrmOptions.find((o) => o.id === selectedVrmId)
  const vrmUrl = selectedVrm?.url ?? seeleUrl
  // Derive a stable model id ("seele", "bronya", "bronya_long") from the asset
  // label so loadProfile can pick a per-model override.
  const modelId = (selectedVrm?.label ?? 'seele.vrm')
    .replace(/\.vrm$/i, '')
    .replace(/^.*\//, '')
    .toLowerCase()
  const animationUrl = motionOptions.find((o) => o.id === selectedMotionId)?.url ?? ''

  useEffect(() => {
    setClipInfo(null)
    if (onResetRef.current) {
      onResetRef.current()
    }
  }, [vrmUrl, animationUrl])

  // Feed normalized mouse position to the avatar's eye gaze (§4.2 / EyeController).
  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    const controller = avatarRef.current
    if (!controller) return
    const rect = e.currentTarget.getBoundingClientRect()
    const nx = ((e.clientX - rect.left) / rect.width) * 2 - 1
    const ny = -(((e.clientY - rect.top) / rect.height) * 2 - 1)
    controller.setMouse(nx, ny)
  }

  return (
    <div
      className="relative w-full h-full overflow-hidden"
      onMouseMove={handleMouseMove}
      style={{
        background: theme === 'dark'
          ? 'radial-gradient(ellipse at center, #1a0533 0%, #0a0a12 70%)'
          : 'radial-gradient(ellipse at center, #f3e8ff 0%, #ffffff 70%)',
      }}
    >
      <Canvas camera={{ position: [0, 2.05, 0], fov: 45 }} gl={{ antialias: true, alpha: true }}>
        <Suspense fallback={null}>
          <Scene
            theme={theme}
            vrmUrl={vrmUrl}
            modelId={modelId}
            animationUrl={animationUrl}
            isPlaying={isPlaying}
            speed={speed}
            onResetRef={onResetRef}
            onLoaded={setClipInfo}
            avatarRef={avatarRef}
          />
        </Suspense>
      </Canvas>

      {/* Bottom gradient */}
      <div className="absolute bottom-0 left-0 right-0 h-24 pointer-events-none bg-gradient-to-t from-background/80 to-transparent" />

      {/* Dev-only facial-animation test panel (§10 Phase A acceptance). */}
      {import.meta.env.DEV && <AvatarDevPanel />}
    </div>
  )
}