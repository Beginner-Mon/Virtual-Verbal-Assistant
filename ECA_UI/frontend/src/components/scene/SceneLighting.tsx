/**
 * SceneLighting — Phase 2 (Lighting) + Phase 3 (Shadows) + Phase 4 (Ground)
 *
 * MToon-safe lighting: exactly ONE directional light for toon-shading (NdotL),
 * plus a hemisphere light for ambient fill. No multi-light PBR setup.
 *
 * The directional light also carries the shadow configuration: PCFSoft shadow
 * map, tight frustum, tuned bias values.
 *
 * Includes the ground plane (invisible shadow receiver) and contact shadows.
 */

import { useRef, useEffect } from 'react'
import * as THREE from 'three'
import { ContactShadows } from '@react-three/drei'
import type { VRM } from '@pixiv/three-vrm'
import { ENV_CONFIG } from '../../config/environmentConfig'

interface SceneLightingProps {
  vrm: VRM | null
}

export default function SceneLighting({ vrm }: SceneLightingProps) {
  const lightRef = useRef<THREE.DirectionalLight>(null!)

  const {
    lighting: { main, ambient },
    shadows,
    ground,
  } = ENV_CONFIG

  // ── Configure shadow camera on the main directional light ─────────────
  useEffect(() => {
    const light = lightRef.current
    if (!light) return

    light.shadow.mapSize.set(shadows.mapSize, shadows.mapSize)
    light.shadow.bias = shadows.bias
    light.shadow.normalBias = shadows.normalBias

    const cam = light.shadow.camera as THREE.OrthographicCamera
    cam.left = -shadows.cameraSize
    cam.right = shadows.cameraSize
    cam.top = shadows.cameraSize
    cam.bottom = -shadows.cameraSize
    cam.near = shadows.cameraNear
    cam.far = shadows.cameraFar
    cam.updateProjectionMatrix()
  }, [shadows])

  // ── Configure VRM meshes: castShadow / receiveShadow ──────────────────
  // Outline meshes (BackSide) must NOT cast shadows — they create doubled
  // shadow artifacts.
  useEffect(() => {
    if (!vrm?.scene) return

    vrm.scene.traverse((object) => {
      if (object instanceof THREE.Mesh || object instanceof THREE.SkinnedMesh) {
        const mat = object.material as THREE.Material
        const isOutline =
          mat.side === THREE.BackSide ||
          (mat.name && mat.name.toLowerCase().includes('outline'))

        object.castShadow = !isOutline
        object.receiveShadow = true
      }
    })
  }, [vrm])

  return (
    <>
      {/* ── Main Light: the ONLY NdotL contributor for MToon ────────── */}
      <directionalLight
        ref={lightRef}
        color={main.color}
        intensity={main.intensity}
        position={main.position}
        castShadow={main.castShadow}
      />

      {/* ── Hemisphere: ambient fill, no directional influence ───────── */}
      <hemisphereLight
        color={ambient.skyColor}
        groundColor={ambient.groundColor}
        intensity={ambient.intensity}
      />

      {/* ── Ground plane: catches real directional shadow ────────────── */}
      <mesh
        rotation={[-Math.PI / 2, 0, 0]}
        position={[0, ground.y, 0]}
        receiveShadow
      >
        <planeGeometry args={[ground.planeSize, ground.planeSize]} />
        <shadowMaterial
          transparent
          opacity={ground.shadowMaterialOpacity}
        />
      </mesh>

      {/* ── Contact Shadow: soft puddle under feet ──────────────────── */}
      <ContactShadows
        position={[0, ground.y + 0.001, 0]}
        opacity={ground.contactShadow.opacity}
        scale={ground.contactShadow.scale}
        blur={ground.contactShadow.blur}
        far={ground.contactShadow.far}
        color={ground.contactShadow.color}
      />
    </>
  )
}
