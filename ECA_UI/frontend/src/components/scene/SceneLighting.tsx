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
import { useFrame } from '@react-three/fiber'
import { ContactShadows } from '@react-three/drei'
import type { VRM } from '@pixiv/three-vrm'
import { ENV_CONFIG } from '../../config/environmentConfig'
import { DEFAULT_SHADOW_FIT, ShadowCameraFitter } from '../../lib/shadowFit'

interface SceneLightingProps {
  vrm: VRM | null
}

export default function SceneLighting({ vrm }: SceneLightingProps) {
  const lightRef = useRef<THREE.DirectionalLight>(null!)
  const fitterRef = useRef<ShadowCameraFitter | null>(null)

  const {
    lighting: { main, ambient },
    shadows,
    ground,
  } = ENV_CONFIG

  // ── Configure shadow map + build the auto-fitter ──────────────────────
  // The frustum extents are NOT set here any more: they are derived from the
  // subject every frame (see lib/shadowFit.ts). A fixed frustum centred on the
  // world origin is what made shadows clip in a straight line while wasting
  // ~92% of the shadow map.
  useEffect(() => {
    const light = lightRef.current
    if (!light) return

    light.shadow.mapSize.set(shadows.mapSize, shadows.mapSize)
    light.shadow.bias = shadows.bias
    light.shadow.normalBias = shadows.normalBias

    fitterRef.current = new ShadowCameraFitter(light, {
      ...DEFAULT_SHADOW_FIT,
      padding: shadows.fitPadding,
      groundZ: shadows.fitGroundZ,
    })
    return () => {
      fitterRef.current = null
    }
  }, [shadows])

  // Track the subject. Throttled internally — this is not per-frame work.
  useFrame((_state, delta) => {
    fitterRef.current?.update(vrm, delta * 1000)
  })

  // DEV handle for the shadow-frustum probe: compare the fitted frustum against
  // the skeleton AND its floor projection. Cheaper than guessing why a shadow
  // looks cut off.
  useEffect(() => {
    if (!import.meta.env.DEV) return
    ;(window as unknown as { __shadow?: unknown }).__shadow = () => ({
      light: lightRef.current,
      vrm,
    })
  }, [vrm])

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

            // Xóa outline pass
          if (isOutline) {
            object.visible = false
          } else {
            // TODO: Tạm tắt ghi đè material — giữ nguyên MToon để giữ màu gốc
            // object.material = new THREE.MeshStandardMaterial({
            //   color: 0xdddddd,
            //   roughness: 0.8,
            // })
            object.castShadow = true
            object.receiveShadow = true
          }
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

      {/* ── Ground plane: catches real directional shadow (XY plane) ─── */}
      <mesh
        position={[0, 0, 0]}
        receiveShadow
      >
        <planeGeometry args={[ground.planeSize, ground.planeSize]} />
        <shadowMaterial
          transparent
          opacity={ground.shadowMaterialOpacity}
        />
      </mesh>

      {/* ── Contact Shadow: soft puddle under feet (XY plane) ──────── */}
      <group rotation={[-Math.PI / 2, 0, 0]}>
        <ContactShadows
          position={[0, 0, 0]}
          opacity={ground.contactShadow.opacity}
          scale={ground.contactShadow.scale}
          blur={ground.contactShadow.blur}
          far={ground.contactShadow.far}
          color={ground.contactShadow.color}
        />
      </group>
    </>
  )
}
