import { Canvas, useFrame, useLoader, useThree } from '@react-three/fiber'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'
import { VRMLoaderPlugin, VRMHumanBoneName } from '@pixiv/three-vrm'
import type { VRM } from '@pixiv/three-vrm'
import { useTheme } from '../hooks/useTheme'
import { OrbitControls, Html } from '@react-three/drei'
import { useRef, useEffect, useState, Suspense, useMemo } from 'react'
import * as THREE from 'three'
import { AnimationController } from '../lib/AnimationController'
import { AnimationRegistry } from '../lib/AnimationRegistry'
import { DEFAULT_GROUND_CLAMP, GroundClamp } from '../lib/groundClamp'
import { RootMotionAccumulator } from '../lib/rootMotionAccumulator'
import type { CameraResponsivePreset } from '../lib/CameraConfig'
import type { CameraMode } from '../lib/AnimationStates'
import { cameraModeOf, loopModeOf } from '../lib/AnimationStates'
import { useFsmBoot } from '../hooks/useFsmTriggers'
import { useMotion } from '../hooks/useMotion'
import { AvatarController } from '../avatar/AvatarController'
import { BodyPartPicker } from '../avatar/BodyPartPicker'
import { bodyPartClick } from '../avatar/userActivity'
import { loadProfileAsync } from '../avatar/AvatarProfile'
import LoadingOverlay from './ui/LoadingOverlay'
import { disposeVRM } from '../lib/vrmDispose'
import { ENV_CONFIG } from '../config/environmentConfig'
import RendererSetup from './scene/RendererSetup'
import SceneLighting from './scene/SceneLighting'
import SceneEnvironment from './scene/SceneEnvironment'
import ScenePostProcessing from './scene/ScenePostProcessing'
import ClickRipple from './scene/ClickRipple'
import { GraphicsProvider } from '../contexts/GraphicsContext'
import { useGraphics } from '../hooks/useGraphics'

const CAMERA_MODES: Record<CameraMode, { boneName: VRMHumanBoneName }> = {
  head: { boneName: VRMHumanBoneName.Head },
  hips: { boneName: VRMHumanBoneName.Head },
}

/** Responsive presets per camera mode. wideFraming = current desktop-tuned offsets;
 *  narrowFraming = mobile-portrait offsets (increase Y to push back, Z stays at eye level).
 *  narrowTargetZ shifts the look-at target DOWN so the model appears higher, compensating
 *  for the chat panel that occupies the bottom ~40% on mobile. */
const CAMERA_RESPONSIVE_PRESETS: Record<CameraMode, CameraResponsivePreset> = {
  head: {
    wideFraming: [0, 0.5, 0],
    narrowFraming: [0, 2.0, 0],
    narrowTargetZ: -0.6,
  },
  hips: {
    wideFraming: [1.5, 3.8, 1.5],
    narrowFraming: [1.5, 4.5, 1.5],
    narrowTargetZ: -0.4,
  },
}

/**
 * Static <Canvas> configuration, hoisted OUT of the render.
 *
 * R3F calls `root.configure(props)` on every render of <Canvas> and writes
 * these onto the live renderer. Passing fresh object literals meant that work
 * repeated on every React re-render â€” and re-renders got more frequent once FSM
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

/* â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ VRM Character with FSM-driven animation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */

interface VRMCharacterProps {
  vrmUrl: string
  modelId: string
  /** Readiness gate: false while the model has no pose yet (bind pose hidden). */
  onReady: (ready: boolean) => void
  vrmRef: React.MutableRefObject<VRM | null>
  avatarRef: React.MutableRefObject<AvatarController | null>
}

function VRMCharacter({ vrmUrl, modelId, onReady, vrmRef, avatarRef }: VRMCharacterProps) {
  const { attachControllers, setClipInfo, prefetchGestures } = useMotion()

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
  // (mount â†’ cleanup â†’ remount): a naive cleanup would destroy the
  // useLoader-cached VRM on the first cycle, making the model invisible on
  // remount. The `mountedRef` flag lets us distinguish: on cleanup we set it
  // false, then on the synchronous remount we set it true again â€” the
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
  const rootMotionRef = useRef<RootMotionAccumulator | null>(null)
  const posedRef = useRef(false)
  const emotionTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  // Per-instance state â€” key={vrmUrl} remounts resets these on model switch.
  // `posed`: the first animation pose has reached the bones.
  const [posed, setPosed] = useState(false)
  // `avatarAttached`: AvatarController exists, so eye/head follow is driving.
  const [avatarAttached, setAvatarAttached] = useState(false)

  /**
   * Reveal needs BOTH conditions, not just a pose.
   *
   * AvatarController is built behind `await loadProfileAsync(...)`, a network
   * request, while the greeting clip starts as soon as it is retargeted. Showing
   * the model on the pose alone exposed that gap: for its duration nothing holds
   * Neck/Head, so the greeting clip's own head track shows through â€” the head
   * leans down â€” and then snaps straight the moment HeadController attaches.
   * A warm HTTP cache shrinks the gap to nothing, which is why a second refresh
   * "fixed" it and why HMR (VRM served instantly from the useLoader cache, profile
   * still refetched) made it easiest to reproduce. Waiting costs one small request
   * that already falls back rather than failing, behind a spinner that is up
   * anyway for the 10-18 MB model download.
   */
  const revealed = posed && avatarAttached
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

  // Single source of truth for the loading overlay: it hides exactly when the
  // model becomes visible. Previously readiness was pushed from three different
  // places (clip callback, effect body, effect cleanup), which is how it could
  // drift from what <primitive visible> was actually doing.
  useEffect(() => {
    onReadyRef.current(revealed)
  }, [revealed])

  // Facial-animation controller lifecycle: attach on VRM load, detach on
  // model change / unmount. Kept out of React state â€” this ref IS the handle
  // (facial-animation-plan.md Â§8 rules 2-4).
  // The profile now comes from the character record, so this awaits before
  // attaching. Cheap in context: the VRM this runs after is a 10-18 MB
  // download, and loadProfileAsync falls back to the bundled registry rather
  // than failing, so the wait is bounded by one small request either way.
  useEffect(() => {
    if (!vrm) return
    const abort = new AbortController()
    let controller: AvatarController | null = null
    let prefetchIdle: { cancel: () => void } | null = null

    void (async () => {
      let profile
      try {
        profile = await loadProfileAsync(modelId, abort.signal)
      } catch (err) {
        // loadProfileAsync rethrows ONLY AbortError â€” every other failure falls
        // back to the bundled registry (AvatarProfile.ts:131-136). That used to
        // be a detail; now that reveal waits on this effect it is load-bearing,
        // so an unexpected throw must release the gate rather than strand the
        // model behind the loading overlay forever.
        if (!abort.signal.aborted) {
          console.error('[avatar] profile load failed â€” revealing without facial controller', err)
          setAvatarAttached(true)
        }
        return
      }
      if (abort.signal.aborted) return

      controller = new AvatarController(vrm, profile)
      avatarControllerRef.current = controller
      avatarRef.current = controller
      setAvatarAttached(true)
      // The profile is what declares this character's gestures, so this is the
      // first moment the set is known. Warming them on idle keeps the first
      // click off the fetch-and-retarget path (111-214 ms, measured in
      // AnimationRegistry) â€” the whole reason gestures are declared rather than
      // named ad-hoc at the moment of the click.
      prefetchIdle = scheduleIdle(prefetchGestures)
    })()

    return () => {
      abort.abort()
      prefetchIdle?.cancel()
      setAvatarAttached(false)
      if (controller) {
        controller.detach()
        if (avatarRef.current === controller) avatarRef.current = null
        avatarControllerRef.current = null
      }
    }
  }, [vrm, modelId, avatarRef, prefetchGestures])

  // Animation FSM lifecycle. The registry is per-VRM because clips are
  // retargeted against a specific skeleton (plan lá»—i #6); a new instance per
  // model IS the invalidation.
  useEffect(() => {
    if (!vrm) return

    const registry = new AnimationRegistry(vrm, vrmUrl)
    const controller = new AnimationController(vrm, registry, {
      onClipApplied: (info) => {
        setClipInfoRef.current({ tracks: info.tracks, duration: info.duration })
        // Record that a real pose has reached the bones. Event-driven â€” never a
        // timed wait. Reveal itself is derived from this plus the avatar attach.
        if (!posedRef.current) {
          posedRef.current = true
          setPosed(true)
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
        // Begin root-motion tracking for one-shots that use a wide camera
        // (currently only `exercise`). The wide camera signals that the clip
        // has meaningful root translation.
        if (loopModeOf(info.state) === 'once' && cameraModeOf(info.state) === 'hips') {
          rootMotionRef.current?.beginOneShot(vrm)
        }
      },
      onBeforeAutoTransition: (_completed, _next, crossfadeSec) => {
        // Commit the hips displacement BEFORE the successor clip (idle) resets
        // hips to rest position. This offsets the model group so the character
        // visually stays at its final position. The offset is ramped in over
        // the crossfade duration to avoid double-displacement.
        rootMotionRef.current?.commitOneShot(vrm, crossfadeSec)
      },
    })

    animControllerRef.current = controller
    setAnimController(controller)
    setRegistry(registry)
    const detach = attachControllers(controller, registry)

    return () => {
      if (emotionTimerRef.current) clearTimeout(emotionTimerRef.current)
      detach()
      controller.dispose()
      animControllerRef.current = null
      setAnimController(null)
      setRegistry(null)
      posedRef.current = false
      setPosed(false)
    }
  }, [vrm, vrmUrl, attachControllers])

  // Boot: greet once, then idle (plan Â§2.5).
  useFsmBoot(animController, registry)

  // Update body animation, then facial expressions, then the VRM itself.
  // Order is mandatory (Â§8 rule 1): the avatar controller calls setValue, and
  // vrm.update applies those weights via expressionManager.update() â€” so the
  // tick must land BETWEEN the mixer update and vrm.update.
  useFrame((_state, delta) => {
    animControllerRef.current?.update(delta)
    avatarControllerRef.current?.tick(delta)
    vrm?.update(delta)
    // Last: the clamp reads the pose this frame actually produced. A generated
    // clip can descend further than the character is tall (measured: 1.21 m on
    // motion_b28e8284), which would otherwise sink it through the floor and
    // kill its shadow â€” see lib/groundClamp.ts.
    if (modelGroupRef.current) {
      if (!groundClampRef.current) {
        groundClampRef.current = new GroundClamp(modelGroupRef.current, groundScratch, {
          ...DEFAULT_GROUND_CLAMP,
          groundZ: ENV_CONFIG.shadows.fitGroundZ,
        })
      }
      if (!rootMotionRef.current) {
        rootMotionRef.current = new RootMotionAccumulator(modelGroupRef.current)
      }
      rootMotionRef.current.update(delta)
      groundClampRef.current.update(vrm)
    }
  })

  return (
    <group ref={modelGroupRef} position={[0, 1.5, 0]} rotation={[Math.PI / 2, 0, 0]}>
      {/* visible=false until the first animation pose is applied â€” the model
          never renders in bind pose (T-pose). */}
      <primitive object={vrm.scene} visible={revealed} />
    </group>
  )
}

/* â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ Floating Particles â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */

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

/* â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ Scene â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */

interface SceneProps {
  theme: 'light' | 'dark'
  vrmUrl: string
  modelId: string
  onReady: (ready: boolean) => void
  avatarRef: React.MutableRefObject<AvatarController | null>
}

function Scene({ theme, vrmUrl, modelId, onReady, avatarRef }: SceneProps) {
  // Camera mode is owned by CameraController and driven by FSM state
  // (exercise â†’ wide + 3s cooldown). The old URL-substring heuristic is gone.
  const { cameraMode, cameraConfig } = useMotion()
  const { settings: gfx } = useGraphics()
  // eslint-disable-next-line @typescript-eslint/no-explicit-any -- drei OrbitControls ref is an untyped Three.js controls instance; precise typing adds no safety here
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
  }, [cameraMode]) // eslint-disable-line react-hooks/exhaustive-deps -- camera.position is mutable and should not trigger a transition; only cameraMode matters

  // Reusable vectors to avoid GC pressure
  const followPos = useMemo(() => new THREE.Vector3(), [])
  const deltaVec = useMemo(() => new THREE.Vector3(), [])
  const offsetDeltaVec = useMemo(() => new THREE.Vector3(), [])
  const lastAppliedOffsetRef = useRef(new THREE.Vector3(0, 0.5, 0))
  const targetZRef = useRef(0)
  // Built once: a fresh helper per render would make R3F detach/dispose and
  // re-attach the object on every state change.
  const axesHelper = useMemo(() => new THREE.AxesHelper(3), [])

  // Lock X/Y/Z â€” freeze per-axis when toggled (Plan C). Stored frozen value is
  // the last unrestricted position; right-drag delta on a locked axis is reverted.
  const lockPrevTargetRef = useRef(new THREE.Vector3())
  const lockPrevPosRef = useRef(new THREE.Vector3())
  const lockInitializedRef = useRef(false)

  // Every frame: make the camera orbit target follow the selected bone.
  // We also shift the camera position by the same delta so the orbital
  // offset (angle + distance) is preserved while the rig moves.
  useFrame((_state, delta) => {
    if (!vrmRef.current || !controlsRef.current) return

    // Compute t from canvas width: desktop (>768px) â†’ t=0 (wideFraming only),
    // mobile (<360px) â†’ t=1 (full narrowFraming + targetZ shift for chat panel).
    // Matches Tailwind md breakpoint where MobileNavBar/ChatPanel activate.
    const tClamped = Math.max(0, Math.min(1,
      (768 - size.width) / (768 - 360)
    ))
    const preset = CAMERA_RESPONSIVE_PRESETS[cameraMode]
    responsiveTargetRef.current.set(
      THREE.MathUtils.lerp(preset.wideFraming[0], preset.narrowFraming[0], tClamped),
      THREE.MathUtils.lerp(preset.wideFraming[1], preset.narrowFraming[1], tClamped),
      THREE.MathUtils.lerp(preset.wideFraming[2], preset.narrowFraming[2], tClamped),
    )

    const targetZTarget = THREE.MathUtils.lerp(0, preset.narrowTargetZ, tClamped)

    // Smooth towards target on resize; snap during camera-mode transition
    if (cameraTransitionRef.current) {
      responsiveDisplayRef.current.copy(responsiveTargetRef.current)
      targetZRef.current = targetZTarget
    } else {
      responsiveDisplayRef.current.lerp(responsiveTargetRef.current, 0.2)
      targetZRef.current = THREE.MathUtils.lerp(targetZRef.current, targetZTarget, 0.2)
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
      endTarget.z += targetZRef.current
      const endPos = followPos.clone().add(responsiveDisplayRef.current).add(currentCustomOffset)

      camera.position.lerpVectors(t.startPos, endPos, eased)
      controlsRef.current.target.lerpVectors(t.startTarget, endTarget, eased)
      controlsRef.current.update()

      if (progress >= 1) {
        cameraTransitionRef.current = null
        lastAppliedOffsetRef.current.copy(responsiveDisplayRef.current)
      }
      return
    }

    const currentCustomOffset = new THREE.Vector3(cameraConfig.offsetX, cameraConfig.offsetY, cameraConfig.offsetZ)
    const targetPos = followPos.clone().add(currentCustomOffset)
    targetPos.z += targetZRef.current

    if (!cameraInitializedRef.current) {
      controlsRef.current.target.copy(targetPos)
      camera.position.copy(followPos).add(responsiveDisplayRef.current).add(currentCustomOffset)
      lastAppliedOffsetRef.current.copy(responsiveDisplayRef.current)
      camera.lookAt(targetPos)
      controlsRef.current.update()
      cameraInitializedRef.current = true
      return
    }

    if (!cameraConfig.followTarget) return

    // How far did the follow point move since the last frame?
    deltaVec.subVectors(targetPos, controlsRef.current.target)

    // How much did the responsive offset change since last frame?
    offsetDeltaVec.subVectors(responsiveDisplayRef.current, lastAppliedOffsetRef.current)

    // Move the camera by both the bone delta and the offset delta
    camera.position.add(deltaVec).add(offsetDeltaVec)

    lastAppliedOffsetRef.current.copy(responsiveDisplayRef.current)

    // Update the controls target to the new follow point
    controlsRef.current.target.copy(targetPos)
    controlsRef.current.update()
  })

  // Second plane: enforce Lock X/Y/Z after the main follow/transition logic and
  // after OrbitControls has applied any user drag. Registration order guarantees
  // this runs after the preceding useFrame.
  useFrame(() => {
    if (!controlsRef.current) return
    if (!cameraInitializedRef.current) return
    if (!lockInitializedRef.current) {
      lockPrevTargetRef.current.copy(controlsRef.current.target)
      lockPrevPosRef.current.copy(camera.position)
      lockInitializedRef.current = true
      return
    }
    const target = controlsRef.current.target as THREE.Vector3
    const pos = camera.position as THREE.Vector3
    let needsUpdate = false
    if (cameraConfig.lockX) {
      if (target.x !== lockPrevTargetRef.current.x) { target.x = lockPrevTargetRef.current.x; needsUpdate = true }
      if (pos.x !== lockPrevPosRef.current.x) { pos.x = lockPrevPosRef.current.x; needsUpdate = true }
    } else {
      lockPrevTargetRef.current.x = target.x
      lockPrevPosRef.current.x = pos.x
    }
    if (cameraConfig.lockY) {
      if (target.y !== lockPrevTargetRef.current.y) { target.y = lockPrevTargetRef.current.y; needsUpdate = true }
      if (pos.y !== lockPrevPosRef.current.y) { pos.y = lockPrevPosRef.current.y; needsUpdate = true }
    } else {
      lockPrevTargetRef.current.y = target.y
      lockPrevPosRef.current.y = pos.y
    }
    if (cameraConfig.lockZ) {
      if (target.z !== lockPrevTargetRef.current.z) { target.z = lockPrevTargetRef.current.z; needsUpdate = true }
      if (pos.z !== lockPrevPosRef.current.z) { pos.z = lockPrevPosRef.current.z; needsUpdate = true }
    } else {
      lockPrevTargetRef.current.z = target.z
      lockPrevPosRef.current.z = pos.z
    }
    if (needsUpdate) controlsRef.current.update()
  })

return (
    <>
      {/* â”€â”€ Phase 1: Renderer color pipeline + material audit â”€â”€â”€â”€â”€â”€ */}
      <RendererSetup vrm={vrmRef.current} />

      {/* â”€â”€ Phase 2+3+4: Lighting, shadows & ground â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */}
      <SceneLighting vrm={vrmRef.current} />

      {/* â”€â”€ Phase 5: Background (gradient / HDRI + stars) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */}
      <SceneEnvironment theme={theme} />

      {/* â”€â”€ Phase 6: Post processing (Bloom / SSAO / Vignette) â”€â”€â”€â”€ */}
      <ScenePostProcessing />

      {/* key={vrmUrl}: a model switch is a clean remount â€” the old model is
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

      {/* â”€â”€ Debug Overlays â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */}
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
      <BodyPartClickLogger vrmRef={vrmRef} />
    </>
  )
}

/* â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ Body-part picking â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */

/** Pointer travel, in CSS px, still counted as a click rather than a drag. */
const DRAG_SLOP_PX = 5

/**
 * Turns a click on the avatar into a `UserActivity`.
 *
 * This component knows which body part was hit and nothing else â€” not which
 * animation plays, not which expression follows. It reports the interaction and
 * `dispatchActivity` asks the character's own profile what that means, so a
 * model can bring different reactions from the database without this file
 * changing.
 *
 * The picking itself lives in BodyPartPicker (a GPU pick, not a raycast â€” see
 * that file). What stays here is plumbing: rebuild the picker when the VRM is
 * swapped, keep its one-off vertex pass off the frame that swapped the model in,
 * and tell a click apart from a camera drag.
 */
function BodyPartClickLogger({ vrmRef }: { vrmRef: React.MutableRefObject<VRM | null> }) {
  const { camera, gl } = useThree()
  const { dispatchActivity } = useMotion()
  const pickerRef = useRef<BodyPartPicker | null>(null)
  const builtForRef = useRef<VRM | null>(null)
  const idleRef = useRef<{ cancel: () => void } | null>(null)

  // The VRM arrives â€” and is replaced on a model switch â€” through a ref, so
  // there is no render to hang an effect on. One identity compare per frame is
  // cheaper than routing the model through context just for this.
  useFrame(() => {
    const vrm = vrmRef.current
    if (vrm === builtForRef.current) return
    builtForRef.current = vrm

    idleRef.current?.cancel()
    idleRef.current = null
    pickerRef.current?.dispose()
    pickerRef.current = null
    if (!vrm) return

    const picker = new BodyPartPicker(vrm, gl)
    pickerRef.current = picker
    // Tagging every vertex is a single pass over the whole model. Same reasoning
    // as the clip warm-up in AnimationRegistry: do it while nothing is waiting.
    idleRef.current = scheduleIdle(() => {
      idleRef.current = null
      if (pickerRef.current !== picker) return
      picker.build()
      picker.warm(camera)
    })
  })

  useEffect(() => {
    const el = gl.domElement
    // Where the gesture that is currently down began. A pointerup further than
    // DRAG_SLOP_PX away was a camera orbit, not a click on the character.
    let down: { id: number; x: number; y: number } | null = null

    const onDown = (e: PointerEvent) => {
      down = { id: e.pointerId, x: e.clientX, y: e.clientY }
    }

    const onUp = (e: PointerEvent) => {
      const start = down
      down = null
      if (!start || start.id !== e.pointerId) return
      if (Math.abs(e.clientX - start.x) > DRAG_SLOP_PX) return
      if (Math.abs(e.clientY - start.y) > DRAG_SLOP_PX) return

      const picker = pickerRef.current
      if (!picker?.isReady) return
      const rect = el.getBoundingClientRect()

      void picker.pickAsync(e.clientX - rect.left, e.clientY - rect.top, camera).then((part) => {
        // `blocking` is what the click costs the main thread; `latency` includes
        // the GPU fence wait, which happens off-thread. See BodyPartPicker.timings.
        const { render, blocking, latency } = picker.timings
        console.log(
          '[bodyPart]',
          part ?? 'miss',
          `blocking ${blocking.toFixed(2)}ms (render ${render.toFixed(2)}) Â· answer in ${latency.toFixed(2)}ms`,
        )
        if (part) void dispatchActivity(bodyPartClick(part))
      })
    }

    const onCancel = () => {
      down = null
    }

    // Capture phase, like the click ripple: OrbitControls calls
    // setPointerCapture on pointerdown, and this sidesteps any question of what
    // it does with the event on the way back up.
    el.addEventListener('pointerdown', onDown, { capture: true })
    el.addEventListener('pointerup', onUp, { capture: true })
    el.addEventListener('pointercancel', onCancel, { capture: true })
    return () => {
      el.removeEventListener('pointerdown', onDown, { capture: true })
      el.removeEventListener('pointerup', onUp, { capture: true })
      el.removeEventListener('pointercancel', onCancel, { capture: true })
    }
  }, [camera, gl, dispatchActivity])

  useEffect(
    () => () => {
      idleRef.current?.cancel()
      idleRef.current = null
      pickerRef.current?.dispose()
      pickerRef.current = null
    },
    [],
  )

  return null
}

/** requestIdleCallback with a setTimeout fallback, cancellable either way. */
function scheduleIdle(task: () => void): { cancel: () => void } {
  if (typeof requestIdleCallback === 'function') {
    const handle = requestIdleCallback(task, { timeout: 2000 })
    return { cancel: () => cancelIdleCallback(handle) }
  }
  const handle = setTimeout(task, 200)
  return { cancel: () => clearTimeout(handle) }
}

/* â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ Exported Component â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */

export default function CharacterViewer() {
  const { theme } = useTheme()
  const { selectedVrmId, vrmOptions, vrmOptionsError, avatarRef, setClipInfo, currentState } = useMotion()

  const selectedVrm = vrmOptions.find((o) => o.id === selectedVrmId)
  // No local fallback any more: the models live on the CDN, so until the
  // catalog answers there is genuinely no URL to render. Bundling anne.vrm as a
  // safety net would put the 15.8 MB this change removes straight back in.
  const vrmUrl = selectedVrm?.url
  // Readiness gate driven by VRMCharacter: the model (and this overlay) swap
  // only when the first pose is actually applied â€” never a timed wait.
  const [viewerReady, setViewerReady] = useState(false)
  // Stable model id for loadProfile. With the catalog it is already the slug;
  // the bundled-fallback path still yields a label like "models/anne.vrm".
  const modelId = (selectedVrm?.id ?? selectedVrm?.label ?? 'anne')
    .replace(/\.vrm$/i, '')
    .replace(/^.*\//, '')
    .toLowerCase()

  useEffect(() => {
    setClipInfo(null)
  }, [vrmUrl, setClipInfo])

  // Click ripple state â€” uses native pointerdown with capture to fire before
  // R3F's internal event system calls stopPropagation on the canvas element.
  const [clicks, setClicks] = useState<{ id: number; x: number; y: number }[]>([])
  const clickIdRef = useRef(0)
  const lastClickTimeRef = useRef(0)
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const handler = (e: PointerEvent) => {
      if (e.pointerType !== 'mouse' && e.pointerType !== 'touch' && e.pointerType !== 'pen') return
      const now = Date.now()
      if (now - lastClickTimeRef.current < 200) return
      lastClickTimeRef.current = now
      const rect = el.getBoundingClientRect()
      const id = ++clickIdRef.current
      setClicks((prev) => [...prev, { id, x: e.clientX - rect.left, y: e.clientY - rect.top }])
    }
    el.addEventListener('pointerdown', handler, { capture: true })
    return () => el.removeEventListener('pointerdown', handler, { capture: true })
  }, [])

  const removeClick = (id: number) => {
    setClicks((prev) => prev.filter((c) => c.id !== id))
  }

  // Feed normalized mouse position to the avatar's eye gaze (Â§4.2 / EyeController).
  // Suppressed during the greeting animation so the model performs its scripted
  // wave without being pulled toward the cursor.
  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (currentState === 'greeting') return
    const controller = avatarRef.current
    if (!controller) return
    const rect = e.currentTarget.getBoundingClientRect()
    const nx = ((e.clientX - rect.left) / rect.width) * 2 - 1
    const ny = -(((e.clientY - rect.top) / rect.height) * 2 - 1)
    controller.setMouse(nx, ny)
  }

  return (
    <div
      ref={containerRef}
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
          {vrmUrl && (
            <Scene
              theme={theme}
              vrmUrl={vrmUrl}
              modelId={modelId}
              onReady={setViewerReady}
              avatarRef={avatarRef}
            />
          )}
          </GraphicsProvider>
        </Suspense>
      </Canvas>

      {/* Click ripple effects */}
      {clicks.map((c) => (
        <ClickRipple key={c.id} x={c.x} y={c.y} theme={theme} onDone={() => removeClick(c.id)} />
      ))}

      {/* Loading overlay: shown while the model has no pose yet (initial load
          / model switch). Replaces the old T-pose flash with a spinner.
          A catalog failure is called out by name â€” without this the screen is
          an indistinguishable spinner whether the CDN is unreachable or the
          model is merely still downloading. */}
      {!viewerReady && (
        <LoadingOverlay
          text={
            vrmOptionsError
              ? `Could not load characters: ${vrmOptionsError}`
              : vrmUrl
                ? 'Loading 3D Avatar...'
                : 'Loading characters...'
          }
        />
      )}

      {/* Bottom gradient */}
      <div className="absolute bottom-0 left-0 right-0 h-24 pointer-events-none bg-gradient-to-t from-background/80 to-transparent" />
    </div>
  )
}

