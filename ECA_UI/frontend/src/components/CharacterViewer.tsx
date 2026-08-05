import { Canvas, useFrame, useLoader, useThree } from '@react-three/fiber'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'
import { VRMLoaderPlugin, VRMHumanBoneName } from '@pixiv/three-vrm'
import type { VRM } from '@pixiv/three-vrm'
import anneUrl from '../asset/models/anne.vrm'
import { useTheme } from '../contexts/ThemeContext'
import { OrbitControls, Html } from '@react-three/drei'
import { useRef, useEffect, useState, Suspense, useMemo } from 'react'
import * as THREE from 'three'
import { AnimationController } from '../lib/AnimationController'
import { AnimationRegistry } from '../lib/AnimationRegistry'
import { DEFAULT_GROUND_CLAMP, GroundClamp } from '../lib/groundClamp'
import { ASPECT_RANGE, type CameraResponsivePreset } from '../lib/CameraConfig'
import type { CameraMode } from '../lib/AnimationStates'
import { useFsmBoot } from '../hooks/useFsmTriggers'
import { useMotion } from '../contexts/MotionContext'
import { AvatarController } from '../avatar/AvatarController'
import { loadProfile } from '../avatar/AvatarProfile'
import LoadingOverlay from './ui/LoadingOverlay'
import { disposeVRM } from '../lib/vrmDispose'
import { ENV_CONFIG } from '../config/environmentConfig'
import RendererSetup from './scene/RendererSetup'
import SceneLighting from './scene/SceneLighting'
import SceneEnvironment from './scene/SceneEnvironment'
import ScenePostProcessing from './scene/ScenePostProcessing'
import { GraphicsProvider, useGraphics } from '../contexts/GraphicsContext'

const CAMERA_MODES: Record<CameraMode, { boneName: VRMHumanBoneName }> = {
  head: { boneName: VRMHumanBoneName.Head },
  hips: { boneName: VRMHumanBoneName.Head },
}

/** Responsive presets per camera mode. wideFraming = current desktop-tuned offsets;
 *  narrowFraming = mobile-portrait offsets (tune by eye: lower Y, push Z further). */
const CAMERA_RESPONSIVE_PRESETS: Record<CameraMode, CameraResponsivePreset> = {
  head: {
    wideFraming: [0, 0.5, 0],
    narrowFraming: [0, -0.3, 2.0],
  },
  hips: {
    wideFraming: [0.5, 2.2, 1.0],
    narrowFraming: [0.5, 0.5, 4.0],
  },
}

/**
 * Static <Canvas> configuration, hoisted OUT of the render.
 *
 * R3F calls `root.configure(props)` on every render of <Canvas> and writes
 * these onto the live renderer. Passing fresh object literals meant that work
 * repeated on every React re-render — and re-renders got more frequent once FSM
 * state started flowing through context. Module constants make the config what
 * it actually is: fixed for the lifetime of the app.
 */
const CANVAS_SHADOWS = { type: ENV_CONFIG.shadows.type }
const CANVAS_CAMERA = { position: [0, 2.05, 0] as [number, number, number], fov: 45 }
const CANVAS_GL = {
  antialias: true,
  alpha: true,
  toneMapping: ENV_CONFIG.renderer.toneMapping,
  toneMappingExposure: ENV_CONFIG.renderer.toneMappingExposure,
}

/* ───────────────────── VRM Character with FSM-driven animation ──────────── */

interface VRMCharacterProps {
  vrmUrl: string
  modelId: string
  /** Readiness gate: false while the model has no pose yet (bind pose hidden). */
  onReady: (ready: boolean) => void
  vrmRef: React.MutableRefObject<VRM | null>
  avatarRef: React.MutableRefObject<AvatarController | null>
}

function VRMCharacter({ vrmUrl, modelId, onReady, vrmRef, avatarRef }: VRMCharacterProps) {
  const { attachControllers, setClipInfo } = useMotion()

  const gltf = useLoader(GLTFLoader, vrmUrl, (loader) => {
    loader.register((parser) => new VRMLoaderPlugin(parser))
  })

  const vrm: VRM = gltf.userData.vrm

  // Cache rest poses immediately upon load, before any animations mutate the bones.
  // The retargeter (BVH/Mixamo) needs these pure bind poses to calculate offsets.
  if (!vrm.scene.userData.restPoses) {
    const restPoses = new Map<string, { position: THREE.Vector3; quaternion: THREE.Quaternion }>()
    if (vrm.humanoid) {
      Object.values(VRMHumanBoneName).forEach((boneName) => {
        const bone = vrm.humanoid?.getNormalizedBoneNode(boneName as VRMHumanBoneName)
        if (bone) {
          restPoses.set(boneName, {
            position: bone.position.clone(),
            quaternion: bone.quaternion.clone(),
          })
        }
      })
    }
    vrm.scene.userData.restPoses = restPoses
  }

  // Expose the VRM instance so the parent can read bone positions.
  useEffect(() => {
    vrmRef.current = vrm
    return () => {
      vrmRef.current = null
    }
  }, [vrm, vrmRef])

  // Dispose GPU resources (geometry/material/texture) when the VRM truly
  // unmounts (model switch). React 19 StrictMode double-fires effects in dev
  // (mount → cleanup → remount): a naive cleanup would destroy the
  // useLoader-cached VRM on the first cycle, making the model invisible on
  // remount. The `mountedRef` flag lets us distinguish: on cleanup we set it
  // false, then on the synchronous remount we set it true again — the
  // microtask only fires dispose if the ref is still false (real unmount).
  const disposeGuardRef = useRef(true)
  useEffect(() => {
    disposeGuardRef.current = true
    return () => {
      disposeGuardRef.current = false
      const vrmToDispose = vrm
      queueMicrotask(() => {
        if (!disposeGuardRef.current) {
          disposeVRM(vrmToDispose)
        }
      })
    }
  }, [vrm])

  const avatarControllerRef = useRef<AvatarController | null>(null)
  const animControllerRef = useRef<AnimationController | null>(null)
  const modelGroupRef = useRef<THREE.Group>(null)
  const groundClampRef = useRef<GroundClamp | null>(null)
  const revealedRef = useRef(false)
  const emotionTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  // Drives <primitive visible={...}>: false until the first pose is applied.
  // Per-instance state — key={vrmUrl} remounts reset it on model switch.
  const [revealed, setRevealed] = useState(false)
  // Also kept in state so the boot effect re-runs when the controller is swapped.
  const [animController, setAnimController] = useState<AnimationController | null>(null)
  const [registry, setRegistry] = useState<AnimationRegistry | null>(null)
  // Reused by the ground clamp so it allocates nothing per frame.
  const groundScratch = useMemo(() => new THREE.Vector3(), [])

  // Latest-value refs: the animation controller lives outside React, so its
  // callbacks must not capture a stale render's props.
  const onReadyRef = useRef(onReady)
  const setClipInfoRef = useRef(setClipInfo)
  useEffect(() => {
    onReadyRef.current = onReady
    setClipInfoRef.current = setClipInfo
  }, [onReady, setClipInfo])

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

  // Animation FSM lifecycle. The registry is per-VRM because clips are
  // retargeted against a specific skeleton (plan lỗi #6); a new instance per
  // model IS the invalidation.
  useEffect(() => {
    if (!vrm) return

    const registry = new AnimationRegistry(vrm, vrmUrl)
    const controller = new AnimationController(vrm, registry, {
      onClipApplied: (info) => {
        setClipInfoRef.current({ tracks: info.tracks, duration: info.duration })
        // Reveal only once an actual pose has reached the bones. Event-driven
        // readiness — never a timed wait.
        if (!revealedRef.current) {
          revealedRef.current = true
          setRevealed(true)
          onReadyRef.current(true)
        }
        // Schedule emotion at midpoint of greeting clip (plan greeting-midpoint-emotion).
        if (info.state === 'greeting') {
          if (emotionTimerRef.current) clearTimeout(emotionTimerRef.current)
          const delayMs = info.duration * 0.25 * 1000
          emotionTimerRef.current = setTimeout(() => {
            const emotion = avatarControllerRef.current?.profile.greetingEmotion ?? 'happy'
            avatarControllerRef.current?.setEmotion(emotion, 1, 600)
          }, delayMs)
        }
      },
    })

    animControllerRef.current = controller
    setAnimController(controller)
    setRegistry(registry)
    const detach = attachControllers(controller, registry)
    onReadyRef.current(false)

    return () => {
      if (emotionTimerRef.current) clearTimeout(emotionTimerRef.current)
      detach()
      controller.dispose()
      animControllerRef.current = null
      setAnimController(null)
      setRegistry(null)
      revealedRef.current = false
      setRevealed(false)
      onReadyRef.current(false)
    }
  }, [vrm, vrmUrl, attachControllers])

  // Boot: greet once, then idle (plan §2.5).
  useFsmBoot(animController, registry)

  // Update body animation, then facial expressions, then the VRM itself.
  // Order is mandatory (§8 rule 1): the avatar controller calls setValue, and
  // vrm.update applies those weights via expressionManager.update() — so the
  // tick must land BETWEEN the mixer update and vrm.update.
  useFrame((_state, delta) => {
    animControllerRef.current?.update(delta)
    avatarControllerRef.current?.tick(delta)
    vrm?.update(delta)
    // Last: the clamp reads the pose this frame actually produced. A generated
    // clip can descend further than the character is tall (measured: 1.21 m on
    // motion_b28e8284), which would otherwise sink it through the floor and
    // kill its shadow — see lib/groundClamp.ts.
    if (modelGroupRef.current) {
      if (!groundClampRef.current) {
        groundClampRef.current = new GroundClamp(modelGroupRef.current, groundScratch, {
          ...DEFAULT_GROUND_CLAMP,
          groundZ: ENV_CONFIG.shadows.fitGroundZ,
        })
      }
      groundClampRef.current.update(vrm)
    }
  })

  return (
    <group ref={modelGroupRef} position={[0, 1.5, 0]} rotation={[Math.PI / 2, 0, 0]}>
      {/* visible=false until the first animation pose is applied — the model
          never renders in bind pose (T-pose). */}
      <primitive object={vrm.scene} visible={revealed} />
    </group>
  )
}

/* ───────────────────────── Floating Particles ────────────────────── */

function FloatingParticles() {
  const { particles } = ENV_CONFIG
  const count = particles.count
  const pointsRef = useRef<THREE.Points>(null!)
  const { settings } = useGraphics()

  const positions = useMemo(() => {
    const arr = new Float32Array(count * 3)
    for (let i = 0; i < count; i++) {
      arr[i * 3 + 0] = (Math.random() - 0.5) * 10
      arr[i * 3 + 1] = (Math.random() - 0.5) * 10
      arr[i * 3 + 2] = (Math.random() - 0.5) * 10
    }
    return arr
  }, [count])

  useFrame(({ clock }) => {
    if (!settings.particles || !pointsRef.current) return
    const t = clock.getElapsedTime()
    pointsRef.current.rotation.y = t * 0.02
    pointsRef.current.rotation.x = t * 0.01
  })

  if (!settings.particles) return null

  return (
    <points ref={pointsRef}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          args={[positions, 3]}
        />
      </bufferGeometry>
      <pointsMaterial
        size={particles.size}
        color={particles.color}
        transparent
        opacity={particles.opacity}
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
  onReady: (ready: boolean) => void
  avatarRef: React.MutableRefObject<AvatarController | null>
}

function Scene({ theme, vrmUrl, modelId, onReady, avatarRef }: SceneProps) {
  // Camera mode is owned by CameraController and driven by FSM state
  // (exercise → wide + 3s cooldown). The old URL-substring heuristic is gone.
  const { cameraMode, cameraConfig } = useMotion()
  const { settings: gfx } = useGraphics()
  const controlsRef = useRef<any>(null)
  const vrmRef = useRef<VRM | null>(null)
  const { camera, size } = useThree()
  const cameraInitializedRef = useRef(false)
  const cameraTransitionRef = useRef<{
    startPos: THREE.Vector3
    startTarget: THREE.Vector3
    elapsed: number
  } | null>(null)
  const prevCameraModeRef = useRef(cameraMode)

  const responsiveTargetRef = useRef(new THREE.Vector3(0, 0.5, 0))
  const responsiveDisplayRef = useRef(new THREE.Vector3(0, 0.5, 0))

  const TRANSITION_DURATION = 0.6

  useEffect(() => {
    camera.up.set(0, 0, 1)
  }, [camera])

  useEffect(() => {
    cameraInitializedRef.current = false
  }, [vrmUrl])

  useEffect(() => {
    const prev = prevCameraModeRef.current
    prevCameraModeRef.current = cameraMode

    if (prev === cameraMode) return
    if (!cameraInitializedRef.current || !controlsRef.current || !vrmRef.current) return

    const mode = CAMERA_MODES[cameraMode]
    const bone = vrmRef.current.humanoid.getNormalizedBoneNode(mode.boneName)
    if (!bone) return

    const tempVec = new THREE.Vector3()
    bone.getWorldPosition(tempVec)

    cameraTransitionRef.current = {
      startPos: camera.position.clone(),
      startTarget: controlsRef.current.target.clone(),
      elapsed: 0,
    }
  }, [cameraMode])

  // Reusable vectors to avoid GC pressure
  const followPos = useMemo(() => new THREE.Vector3(), [])
  const deltaVec = useMemo(() => new THREE.Vector3(), [])
  // Built once: a fresh helper per render would make R3F detach/dispose and
  // re-attach the object on every state change.
  const axesHelper = useMemo(() => new THREE.AxesHelper(3), [])

  // Every frame: make the camera orbit target follow the selected bone.
  // We also shift the camera position by the same delta so the orbital
  // offset (angle + distance) is preserved while the rig moves.
  useFrame((_state, delta) => {
    if (!vrmRef.current || !controlsRef.current) return

    // Compute target responsive offset from canvas aspect ratio
    const aspect = size.width / size.height
    const tClamped = Math.max(0, Math.min(1,
      (ASPECT_RANGE.wide - aspect) / (ASPECT_RANGE.wide - ASPECT_RANGE.narrow)
    ))
    const preset = CAMERA_RESPONSIVE_PRESETS[cameraMode]
    responsiveTargetRef.current.set(
      THREE.MathUtils.lerp(preset.wideFraming[0], preset.narrowFraming[0], tClamped),
      THREE.MathUtils.lerp(preset.wideFraming[1], preset.narrowFraming[1], tClamped),
      THREE.MathUtils.lerp(preset.wideFraming[2], preset.narrowFraming[2], tClamped),
    )

    // Smooth towards target on resize; snap during camera-mode transition
    if (cameraTransitionRef.current) {
      responsiveDisplayRef.current.copy(responsiveTargetRef.current)
    } else {
      responsiveDisplayRef.current.lerp(responsiveTargetRef.current, 0.2)
    }

    const mode = CAMERA_MODES[cameraMode]
    const bone =
      vrmRef.current.humanoid.getNormalizedBoneNode(mode.boneName) ??
      vrmRef.current.humanoid.getNormalizedBoneNode(VRMHumanBoneName.Hips)
    if (!bone) return

    bone.getWorldPosition(followPos)

    if (cameraTransitionRef.current) {
      const t = cameraTransitionRef.current
      t.elapsed += delta
      const progress = Math.min(t.elapsed / TRANSITION_DURATION, 1)
      const eased = 1 - Math.pow(1 - progress, 3)

      const currentCustomOffset = new THREE.Vector3(cameraConfig.offsetX, cameraConfig.offsetY, cameraConfig.offsetZ)
      const endTarget = followPos.clone().add(currentCustomOffset)
      const endPos = followPos.clone().add(responsiveDisplayRef.current).add(currentCustomOffset)

      camera.position.lerpVectors(t.startPos, endPos, eased)
      controlsRef.current.target.lerpVectors(t.startTarget, endTarget, eased)
      controlsRef.current.update()

      if (progress >= 1) {
        cameraTransitionRef.current = null
      }
      return
    }

    const currentCustomOffset = new THREE.Vector3(cameraConfig.offsetX, cameraConfig.offsetY, cameraConfig.offsetZ)
    const targetPos = followPos.clone().add(currentCustomOffset)

    if (!cameraInitializedRef.current) {
      controlsRef.current.target.copy(targetPos)
      camera.position.copy(followPos).add(responsiveDisplayRef.current).add(currentCustomOffset)
      camera.lookAt(targetPos)
      controlsRef.current.update()
      cameraInitializedRef.current = true
      return
    }

    if (!cameraConfig.followTarget) return

    // How far did the follow point move since the last frame?
    deltaVec.subVectors(targetPos, controlsRef.current.target)

    // Move the camera by the same 3D delta so the distance to the model
    // stays identical even if the model shifts on any axis.
    camera.position.add(deltaVec)

    // Update the controls target to the new follow point
    controlsRef.current.target.copy(targetPos)
    controlsRef.current.update()
  })

return (
    <>
      {/* ── Phase 1: Renderer color pipeline + material audit ────── */}
      <RendererSetup vrm={vrmRef.current} />

      {/* ── Phase 2+3+4: Lighting, shadows & ground ─────────────── */}
      <SceneLighting vrm={vrmRef.current} />

      {/* ── Phase 5: Background (gradient / HDRI + stars) ────────── */}
      <SceneEnvironment theme={theme} />

      {/* ── Phase 6: Post processing (Bloom / SSAO / Vignette) ──── */}
      <ScenePostProcessing />

      {/* key={vrmUrl}: a model switch is a clean remount — the old model is
          dropped immediately and the new one stays hidden (plus a loading
          overlay) until its first pose is applied. */}
      <VRMCharacter
        key={vrmUrl}
        vrmRef={vrmRef}
        vrmUrl={vrmUrl}
        modelId={modelId}
        onReady={onReady}
        avatarRef={avatarRef}
      />
      <FloatingParticles />

      {/* ── Debug Overlays ─────────────────────────────────────── */}
      {gfx.showGrid && (
        <group rotation={[Math.PI / 2, 0, 0]}>
          <gridHelper
            args={[8, 16, theme === 'dark' ? '#666688' : '#808080', theme === 'dark' ? '#2a2a3e' : '#c0c0c0']}
            position={[0, 0, -1.5]}
          />
        </group>
      )}

      {gfx.showAxes && (
        <>
          <primitive object={axesHelper} />
          <Html position={[3.2, 0, 0]}>
            <span style={{ color: 'red', fontWeight: 'bold', fontSize: 14 }}>X</span>
          </Html>
          <Html position={[0, 3.2, 0]}>
            <span style={{ color: 'green', fontWeight: 'bold', fontSize: 14 }}>Y</span>
          </Html>
          <Html position={[0, 0, 3.2]}>
            <span style={{ color: 'blue', fontWeight: 'bold', fontSize: 14 }}>Z</span>
          </Html>
        </>
      )}

      {/* Orbital camera: follows hips, enforces minimum distance (radius) */}
      <OrbitControls
        ref={controlsRef}
        enablePan={cameraConfig.enablePan}
        enableZoom={cameraConfig.enableZoom}
        minDistance={cameraConfig.minDistance}
        maxDistance={cameraConfig.maxDistance}
        target={[0, 0, 0]}
      />
    </>
  )
}

/* ───────────────────────── Exported Component ────────────────────── */

export default function CharacterViewer() {
  const { theme } = useTheme()
  const { selectedVrmId, vrmOptions, avatarRef, setClipInfo } = useMotion()

  const selectedVrm = vrmOptions.find((o) => o.id === selectedVrmId)
  const vrmUrl = selectedVrm?.url ?? anneUrl
  // Readiness gate driven by VRMCharacter: the model (and this overlay) swap
  // only when the first pose is actually applied — never a timed wait.
  const [viewerReady, setViewerReady] = useState(false)
  // Derive a stable model id ("seele", "bronya", "bronya_long") from the asset
  // label so loadProfile can pick a per-model override.
  const modelId = (selectedVrm?.label ?? 'anne.vrm')
    .replace(/\.vrm$/i, '')
    .replace(/^.*\//, '')
    .toLowerCase()

  useEffect(() => {
    setClipInfo(null)
  }, [vrmUrl, setClipInfo])

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
        background: 'transparent',
      }}
    >
      <Canvas
        shadows={CANVAS_SHADOWS}
        camera={CANVAS_CAMERA}
        gl={CANVAS_GL}
        onCreated={({ gl }) => {
          gl.outputColorSpace = ENV_CONFIG.renderer.outputColorSpace
        }}
      >
        <Suspense fallback={null}>
          <GraphicsProvider>
          <Scene
            theme={theme}
            vrmUrl={vrmUrl}
            modelId={modelId}
            onReady={setViewerReady}
            avatarRef={avatarRef}
          />
          </GraphicsProvider>
        </Suspense>
      </Canvas>

      {/* Loading overlay: shown while the model has no pose yet (initial load
          / model switch). Replaces the old T-pose flash with a spinner. */}
      {!viewerReady && <LoadingOverlay text="Loading 3D Avatar..." />}

      {/* Bottom gradient */}
      <div className="absolute bottom-0 left-0 right-0 h-24 pointer-events-none bg-gradient-to-t from-background/80 to-transparent" />
    </div>
  )
}
