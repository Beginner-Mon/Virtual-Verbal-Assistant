import { Canvas, useFrame, useLoader, useThree } from '@react-three/fiber'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'
import { VRMLoaderPlugin, VRMHumanBoneName } from '@pixiv/three-vrm'
import type { VRM } from '@pixiv/three-vrm'
import seeleUrl from '../asset/models/seele.vrm'
import { useTheme } from '../contexts/ThemeContext'
import { OrbitControls, Html } from '@react-three/drei'
import { useRef, useEffect, useState, Suspense, useMemo } from 'react'
import * as THREE from 'three'
import { loadAndRetargetBVH, SMPLX_RETARGET_OPTIONS, STANDARD_RETARGET_OPTIONS } from '../lib/bvhToVrm'
import { loadMixamoAnimation } from '../lib/loadMixamoAnimation'
import { getAnimationClip } from '../lib/animationCache'
import { type CameraMode, useMotion } from '../contexts/MotionContext'
import { AvatarController } from '../avatar/AvatarController'
import { loadProfile } from '../avatar/AvatarProfile'
import LoadingOverlay from './ui/LoadingOverlay'
import { disposeVRM } from '../lib/vrmDispose'
import { ENV_CONFIG } from '../config/environmentConfig'
import RendererSetup from './scene/RendererSetup'
import SceneLighting from './scene/SceneLighting'
import SceneEnvironment from './scene/SceneEnvironment'
import ScenePostProcessing from './scene/ScenePostProcessing'

const CAMERA_MODES: Record<CameraMode, { boneName: VRMHumanBoneName; cameraOffset: THREE.Vector3 }> = {
  head: {
    boneName: VRMHumanBoneName.Head,
    cameraOffset: new THREE.Vector3(0, 0.5, 0),
  },
  hips: {
    boneName: VRMHumanBoneName.Head,
    cameraOffset: new THREE.Vector3(0.5, 2.2, 1.0),
  },
}



/* ───────────────────── VRM Character with BVH Animation ──────────── */

/** Crossfade duration (s) when swapping motion clips. */
const FADE_SEC = 0.3

interface VRMCharacterProps {
  vrmUrl: string
  modelId: string
  animationUrl: string
  isPlaying: boolean
  speed: number
  onResetRef: React.MutableRefObject<(() => void) | null>
  onLoaded: (info: { tracks: number; duration: number }) => void
  /** Readiness gate: false while the model has no pose yet (bind pose hidden). */
  onReady: (ready: boolean) => void
  vrmRef: React.MutableRefObject<VRM | null>
  avatarRef: React.MutableRefObject<AvatarController | null>
  onAnimationFinished: () => void
}

function VRMCharacter({
  vrmUrl,
  modelId,
  animationUrl,
  isPlaying,
  speed,
  onResetRef,
  onLoaded,
  onReady,
  vrmRef,
  avatarRef,
  onAnimationFinished,
}: VRMCharacterProps) {
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

  const mixerRef = useRef<THREE.AnimationMixer | null>(null)
  const actionRef = useRef<THREE.AnimationAction | null>(null)
  const fadingOutRef = useRef<THREE.AnimationAction[]>([])
  const loadGenRef = useRef(0)
  const isPlayingRef = useRef(isPlaying)
  const revealedRef = useRef(false)
  const avatarControllerRef = useRef<AvatarController | null>(null)
  const [animLoaded, setAnimLoaded] = useState(false)
  // Drives <primitive visible={...}>: false until the first pose is applied.
  // Per-instance state — key={vrmUrl} remounts reset it on model switch.
  const [revealed, setRevealed] = useState(false)
  const onAnimationFinishedRef = useRef(onAnimationFinished)

  useEffect(() => {
    onAnimationFinishedRef.current = onAnimationFinished
  }, [onAnimationFinished])

  useEffect(() => {
    isPlayingRef.current = isPlaying
  }, [isPlaying])

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

  // One AnimationMixer per VRM, living for the model's whole lifetime. Actions
  // are added/retired on this mixer — stopAllAction() only happens here, on
  // teardown. Invariant: from the moment the model is visible, at least one
  // action drives the skeleton, so the bind pose (T-pose) is never rendered.
  useEffect(() => {
    if (!vrm) return
    const mixer = new THREE.AnimationMixer(vrm.scene)
    mixerRef.current = mixer
    
    const handleFinished = (e: any) => {
      if (e.action === actionRef.current) {
        onAnimationFinishedRef.current?.()
      }
    }
    mixer.addEventListener('finished', handleFinished)

    onReady(false)
    return () => {
      mixer.removeEventListener('finished', handleFinished)
      mixer.stopAllAction()
      mixer.uncacheRoot(vrm.scene)
      mixerRef.current = null
      actionRef.current = null
      fadingOutRef.current = []
      onReady(false)
    }
  }, [vrm, onReady])

  // Load (or reuse a cached) clip when a motion source is selected. The
  // current action keeps driving the skeleton until its replacement is ready,
  // then a crossfade swaps them — there is never a frame with zero active
  // actions, so switching motions never flashes the bind pose.
  useEffect(() => {
    if (!vrm || !animationUrl) return
    const gen = ++loadGenRef.current

    async function applyAnimation() {
      try {
        const isFbx = animationUrl.includes('.fbx')
        const clip = await getAnimationClip(`${vrmUrl}|${animationUrl}`, () =>
          isFbx
            ? loadMixamoAnimation(animationUrl, vrm)
            : loadAndRetargetBVH(
                animationUrl,
                vrm,
                animationUrl.includes('motion_') || animationUrl.includes('smplx')
                  ? SMPLX_RETARGET_OPTIONS
                  : STANDARD_RETARGET_OPTIONS,
              ),
        )

        // A newer request (or a teardown) superseded this one — the clip
        // stays cached for future use, but we don't touch the active action.
        if (gen !== loadGenRef.current || !clip) return
        const mixer = mixerRef.current
        if (!mixer) return

        const action = mixer.clipAction(clip)
        const isAction = !animationUrl.toLowerCase().includes('idle_') && (animationUrl.toLowerCase().includes('action_') || animationUrl.toLowerCase().includes('random_'))
        action.setLoop(isAction ? THREE.LoopOnce : THREE.LoopRepeat, isAction ? 1 : Infinity)
        action.clampWhenFinished = isAction

        const prev = actionRef.current
        if (isPlayingRef.current) {
          action.play()
          if (prev && prev !== action && prev.isRunning()) {
            // Crossfade: new clip fades in over the old one; the old action is
            // retired in useFrame once its weight reaches zero.
            prev.fadeOut(FADE_SEC)
            action.fadeIn(FADE_SEC)
            fadingOutRef.current = fadingOutRef.current.filter((a) => a !== action)
            fadingOutRef.current.push(prev)
          } else {
            // First action on this model (or previous one was stopped): apply
            // at full weight immediately — fading in from nothing would show
            // the bind pose for the fade duration.
            action.setEffectiveWeight(1)
          }
        } else if (prev && prev !== action) {
          prev.stop()
          mixer.uncacheAction(prev.getClip())
        }
        actionRef.current = action
        setAnimLoaded(true)
        onLoaded({ tracks: clip.tracks.length, duration: clip.duration })

        // Reveal only after a pose has actually been applied to the bones.
        // This is an event-driven readiness gate, NOT a wait time.
        if (!revealedRef.current) {
          revealedRef.current = true
          mixer.update(0)
          setRevealed(true)
          onReady(true)
        }

        console.log(
          `[CharacterViewer] Animation ready (${isFbx ? 'FBX' : 'BVH'}): ${clip.tracks.length} tracks, ${clip.duration.toFixed(2)}s`,
        )
      } catch (err) {
        console.error('[CharacterViewer] Failed to load animation:', err)
      }
    }

    applyAnimation()
  }, [vrm, vrmUrl, animationUrl, onLoaded, onReady])

  // Handle play/pause and speed changes. Pause is a debug-only control: the
  // action is fully stopped (not just timeScale=0), which restores the bind
  // pose — acceptable for debugging, not part of the production UX.
  useEffect(() => {
    const action = actionRef.current
    const mixer = mixerRef.current
    if (!action || !mixer || !animLoaded) return
    if (isPlaying) {
      if (!action.isRunning()) {
        action.reset()
        action.play()
      } else {
        action.paused = false
      }
      mixer.timeScale = speed
    } else {
      action.stop()
    }
  }, [isPlaying, speed, animLoaded])

  // Expose reset action — restarts if currently playing, no-op otherwise.
  useEffect(() => {
    onResetRef.current = () => {
      const action = actionRef.current
      if (!action) return
      action.reset()
      if (isPlaying) action.play()
    }
    return () => {
      onResetRef.current = null
    }
  }, [animLoaded, isPlaying])

  // Update body animation, then facial expressions, then the VRM itself.
  // Order is mandatory (§8 rule 1): the avatar controller calls setValue, and
  // vrm.update applies those weights via expressionManager.update() — so the
  // tick must land BETWEEN mixer.update and vrm.update.
  useFrame((_state, delta) => {
    const mixer = mixerRef.current
    if (mixer) {
      mixer.update(delta)
      // Retire actions whose crossfade-out has completed.
      const fading = fadingOutRef.current
      for (let i = fading.length - 1; i >= 0; i--) {
        const a = fading[i]
        if (a === actionRef.current || a.getEffectiveWeight() > 0.001) continue
        a.stop()
        mixer.uncacheAction(a.getClip())
        fading.splice(i, 1)
      }
    }
    if (avatarControllerRef.current) {
      avatarControllerRef.current.tick(delta)
    }
    if (vrm) {
      vrm.update(delta)
    }
  })

  return (
    <group position={[0, -1.5, 0]} rotation={[Math.PI / 2, 0, 0]}>
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
    if (!particles.enabled || !pointsRef.current) return
    const t = clock.getElapsedTime()
    pointsRef.current.rotation.y = t * 0.02
    pointsRef.current.rotation.x = t * 0.01
  })

  if (!particles.enabled) return null

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
  animationUrl: string
  isPlaying: boolean
  speed: number
  onResetRef: React.MutableRefObject<(() => void) | null>
  onLoaded: (info: { tracks: number; duration: number }) => void
  onReady: (ready: boolean) => void
  avatarRef: React.MutableRefObject<AvatarController | null>
  onAnimationFinished: () => void
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
  onReady,
  avatarRef,
  onAnimationFinished,
}: SceneProps) {
  const { cameraMode, setCameraMode } = useMotion()
  const controlsRef = useRef<any>(null)
  const vrmRef = useRef<VRM | null>(null)
  const { camera } = useThree()
  const cameraInitializedRef = useRef(false)
  const cameraTransitionRef = useRef<{
    startPos: THREE.Vector3
    startTarget: THREE.Vector3
    elapsed: number
  } | null>(null)
  const prevCameraModeRef = useRef(cameraMode)

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

  useEffect(() => {
    if (animationUrl.includes('generated')) {
      setCameraMode('hips')
    } else if (animationUrl.includes('built-in')) {
      setCameraMode('head')
    }
  }, [animationUrl])

  // Reusable vectors to avoid GC pressure
  const followPos = useMemo(() => new THREE.Vector3(), [])
  const deltaVec = useMemo(() => new THREE.Vector3(), [])

  // Every frame: make the camera orbit target follow the selected bone.
  // We also shift the camera position by the same delta so the orbital
  // offset (angle + distance) is preserved while the rig moves.
  useFrame((_state, delta) => {
    if (!vrmRef.current || !controlsRef.current) return

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

      const endTarget = followPos.clone()
      const endPos = followPos.clone().add(mode.cameraOffset)

      camera.position.lerpVectors(t.startPos, endPos, eased)
      controlsRef.current.target.lerpVectors(t.startTarget, endTarget, eased)
      controlsRef.current.update()

      if (progress >= 1) {
        cameraTransitionRef.current = null
      }
      return
    }

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
        animationUrl={animationUrl}
        isPlaying={isPlaying}
        speed={speed}
        onResetRef={onResetRef}
        onLoaded={onLoaded}
        onReady={onReady}
        avatarRef={avatarRef}
        onAnimationFinished={onAnimationFinished}
      />
      <FloatingParticles />

      {/* ── Debug Overlays ─────────────────────────────────────── */}
      {ENV_CONFIG.debug.showGrid && (
        <group rotation={[Math.PI / 2, 0, 0]}>
          <gridHelper
            args={[8, 16, theme === 'dark' ? '#444466' : '#c0c0c0', theme === 'dark' ? '#1a1a2e' : '#e8e8e8']}
            position={[0, 0, -1.5]}
          />
        </group>
      )}
      
      {ENV_CONFIG.debug.showAxes && (
        <>
          <primitive object={new THREE.AxesHelper(3)} />
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
    handleAnimationFinished,
  } = useMotion()

  const selectedVrm = vrmOptions.find((o) => o.id === selectedVrmId)
  const vrmUrl = selectedVrm?.url ?? seeleUrl
  // Readiness gate driven by VRMCharacter: the model (and this overlay) swap
  // only when the first pose is actually applied — never a timed wait.
  const [viewerReady, setViewerReady] = useState(false)
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
        background: 'transparent',
      }}
    >
      <Canvas
        shadows={{ type: ENV_CONFIG.shadows.type }}
        camera={{ position: [0, 2.05, 0], fov: 45 }}
        gl={{
          antialias: true,
          alpha: true,
          toneMapping: ENV_CONFIG.renderer.toneMapping,
          toneMappingExposure: ENV_CONFIG.renderer.toneMappingExposure,
        }}
        onCreated={({ gl }) => {
          gl.outputColorSpace = ENV_CONFIG.renderer.outputColorSpace
        }}
      >
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
            onReady={setViewerReady}
            avatarRef={avatarRef}
            onAnimationFinished={handleAnimationFinished}
          />
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