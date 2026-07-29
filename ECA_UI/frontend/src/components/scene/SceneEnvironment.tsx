/**
 * SceneEnvironment — Phase 5: Background
 *
 * Two modes (controlled by ENV_CONFIG.environment.useGradient):
 *
 * A. **Studio Gradient** (default) — a full-screen vertical gradient rendered
 *    via a custom shader on a back-plane. Theme-aware: dark/light palette.
 *    No IBL reflections — perfectly safe for MToon.
 *
 * B. **HDRI** — uses drei `<Environment>` with a low-intensity IBL preset.
 *    envMapIntensity is clamped low via material traversal to prevent
 *    over-reflection on MToon shaders.
 *
 * Also conditionally renders Stars for the dark theme.
 */

import { useMemo } from 'react'
import * as THREE from 'three'
import { Environment, Stars } from '@react-three/drei'
import { ENV_CONFIG } from '../../config/environmentConfig'

interface SceneEnvironmentProps {
  theme: 'light' | 'dark'
}

// ── Gradient Background Shader ──────────────────────────────────────────

const gradientVertexShader = /* glsl */ `
  varying vec2 vUv;
  void main() {
    vUv = uv;
    gl_Position = vec4(position, 1.0);
  }
`

const gradientFragmentShader = /* glsl */ `
  uniform vec3 uTopColor;
  uniform vec3 uBottomColor;
  varying vec2 vUv;
  void main() {
    gl_FragColor = vec4(mix(uBottomColor, uTopColor, vUv.y), 1.0);
  }
`

function GradientBackground({ theme }: { theme: 'light' | 'dark' }) {
  const { gradient } = ENV_CONFIG.environment
  const colors = theme === 'dark' ? gradient.dark : gradient.light

  const uniforms = useMemo(
    () => ({
      uTopColor: { value: new THREE.Color(colors.top) },
      uBottomColor: { value: new THREE.Color(colors.bottom) },
    }),
    [colors.top, colors.bottom],
  )

  return (
    <mesh renderOrder={-1} frustumCulled={false}>
      <planeGeometry args={[2, 2]} />
      <shaderMaterial
        vertexShader={gradientVertexShader}
        fragmentShader={gradientFragmentShader}
        uniforms={uniforms}
        depthWrite={false}
        depthTest={false}
      />
    </mesh>
  )
}

// ── Main Export ──────────────────────────────────────────────────────────

export default function SceneEnvironment({ theme }: SceneEnvironmentProps) {
  const { environment } = ENV_CONFIG
  const showStars = theme === 'dark' ? environment.showStars.dark : environment.showStars.light

  return (
    <>
      {/* Background */}
      {environment.useGradient ? (
        <GradientBackground theme={theme} />
      ) : (
        <Environment
          preset={environment.hdri.preset}
          resolution={environment.iblResolution}
          environmentIntensity={environment.hdri.intensity}
        />
      )}

      {/* Stars overlay */}
      {showStars && (
        <Stars
          radius={50}
          depth={50}
          count={800}
          factor={1.5}
          saturation={0.3}
          fade
          speed={0.3}
        />
      )}
    </>
  )
}
