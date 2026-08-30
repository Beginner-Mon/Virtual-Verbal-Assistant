/**
 * RendererSetup â€” Phase 1: Color Pipeline & Material Audit
 *
 * Configures the WebGL renderer for proper color management and performs
 * a one-time material/texture audit on every VRM load to correct color spaces.
 *
 * This component renders nothing â€” it only applies side effects via hooks.
 */

import { useThree } from '@react-three/fiber'
import { useEffect } from 'react'
import * as THREE from 'three'
import type { VRM, MToonMaterial } from '@pixiv/three-vrm'
import { ENV_CONFIG } from '../../config/environmentConfig'
import { useGraphics } from '../../hooks/useGraphics'

/** Texture property names that should be in linear space (non-color data). */
const LINEAR_TEXTURE_PROPS = new Set([
  'normalMap',
  'roughnessMap',
  'metalnessMap',
  'aoMap',
  'bumpMap',
  'displacementMap',
  // MToon-specific data textures
  'shadeMultiplyTexture',
  'shadingShiftTexture',
  'matcapTexture',
  'uvAnimationMaskTexture',
])

/** Texture property names that should be in sRGB space (color data). */
const SRGB_TEXTURE_PROPS = new Set([
  'map',
  'emissiveMap',
  'rimMultiplyTexture',
])

interface RendererSetupProps {
  vrm: VRM | null
}

export default function RendererSetup({ vrm }: RendererSetupProps) {
  const { gl } = useThree()
  const { settings: gfx } = useGraphics()

  // â”€â”€ Renderer config (once) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  useEffect(() => {
    gl.toneMapping = ENV_CONFIG.renderer.toneMapping
    gl.toneMappingExposure = ENV_CONFIG.renderer.toneMappingExposure
    gl.outputColorSpace = ENV_CONFIG.renderer.outputColorSpace
  }, [gl])

  // â”€â”€ Material & Texture Audit (per VRM load) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  useEffect(() => {
    if (!vrm?.scene) return

    let corrections = 0

    vrm.scene.traverse((object) => {
      if (!(object instanceof THREE.Mesh || object instanceof THREE.SkinnedMesh)) return

      const materials = Array.isArray(object.material)
        ? object.material
        : [object.material]

      for (const mat of materials) {
        if (!mat) continue

        // MToon shade override (driven by Graphics Settings toggle)
        if (gfx.mtoon && (mat as MToonMaterial).isMToonMaterial) {
          const mtoon = mat as MToonMaterial
          mtoon.shadingShiftFactor = ENV_CONFIG.mtoon.shadingShiftFactor
          mtoon.shadeColorFactor = new THREE.Color(ENV_CONFIG.mtoon.shadeColorHex)
          corrections++
        }

        // Check each texture property for correct color space
        for (const [key, value] of Object.entries(mat)) {
          if (!(value instanceof THREE.Texture)) continue

          if (LINEAR_TEXTURE_PROPS.has(key)) {
            if (value.colorSpace !== THREE.LinearSRGBColorSpace) {
              value.colorSpace = THREE.LinearSRGBColorSpace
              value.needsUpdate = true
              corrections++
            }
          } else if (SRGB_TEXTURE_PROPS.has(key)) {
            if (value.colorSpace !== THREE.SRGBColorSpace) {
              value.colorSpace = THREE.SRGBColorSpace
              value.needsUpdate = true
              corrections++
            }
          }
        }
      }
    })

    if (corrections > 0 && import.meta.env.DEV) {
      console.log(`[RendererSetup] Corrected ${corrections} material setting(s)`)
    }
  }, [vrm, gfx.mtoon])

  return null
}

