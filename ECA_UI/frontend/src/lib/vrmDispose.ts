/**
 * Dispose all GPU resources owned by a VRM scene.
 *
 * When a VRM model is swapped out (`key={vrmUrl}` remounts VRMCharacter),
 * React unmounts the component but Three.js does NOT automatically free the
 * underlying WebGL buffers. This utility walks the scene graph and explicitly
 * disposes every geometry, material, and texture.
 *
 * Call this in the VRMCharacter cleanup effect — before the component unmounts
 * and the old `vrm` reference becomes unreachable.
 */

import * as THREE from 'three'
import type { VRM } from '@pixiv/three-vrm'

/**
 * Recursively dispose all GPU resources in a VRM's scene graph.
 */
export function disposeVRM(vrm: VRM): void {
  if (!vrm?.scene) return

  vrm.scene.traverse((object) => {
    // Dispose geometry
    if (object instanceof THREE.Mesh || object instanceof THREE.SkinnedMesh) {
      if (object.geometry) {
        object.geometry.dispose()
      }

      // Dispose material(s) and their textures
      const materials = Array.isArray(object.material)
        ? object.material
        : [object.material]

      for (const material of materials) {
        if (!material) continue
        disposeTextures(material)
        material.dispose()
      }
    }
  })

  // Remove from parent so Three.js internal references are dropped
  if (vrm.scene.parent) {
    vrm.scene.parent.remove(vrm.scene)
  }
}

/**
 * Dispose every texture property on a material.
 *
 * Three.js materials store textures as `.map`, `.normalMap`, `.emissiveMap`,
 * etc. MToon (from @pixiv/three-vrm) adds its own properties like `.shadeMultiplyTexture`,
 * `.rimMultiplyTexture`, etc. We iterate all own properties and dispose anything
 * that is a `THREE.Texture`.
 */
function disposeTextures(material: THREE.Material): void {
  const record = material as unknown as Record<string, unknown>
  for (const key of Object.keys(record)) {
    const value = record[key]
    if (value instanceof THREE.Texture) {
      value.dispose()
    }
  }
}
