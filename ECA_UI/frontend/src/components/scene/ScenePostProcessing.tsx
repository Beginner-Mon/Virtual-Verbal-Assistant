/**
 * ScenePostProcessing â€” Phase 6: Post Processing
 *
 * Very light post-processing stack: Bloom + SSAO + Vignette.
 * All values pulled from ENV_CONFIG â€” no hardcoding.
 *
 * Design principle: enhance depth, not drama. Keep the anime/VRM look.
 * SSAO can be individually disabled via config if it causes artifacts
 * with MToon outlines.
 */

import {
  EffectComposer,
  Bloom,
  Vignette,
  SSAO,
} from '@react-three/postprocessing'
import { BlendFunction } from 'postprocessing'
import { memo, type ReactElement } from 'react'
import { ENV_CONFIG } from '../../config/environmentConfig'
import { useGraphics } from '../../hooks/useGraphics'

function ScenePostProcessing() {
  const { settings: gfx } = useGraphics()
  const { postProcessing: pp } = ENV_CONFIG

  if (!pp.enabled) return null

  // Built as an array rather than with inline `&&`: EffectComposer types its
  // children as elements, so a disabled effect must be absent, not `false`.
  const effects: ReactElement[] = []

  // OFF by default: Bloom intermittently emits an all-black frame.
  // Measurements + reasoning in ENV_CONFIG.postProcessing.bloom.enabled.
  if (pp.bloom.enabled) {
    effects.push(
      <Bloom
        key="bloom"
        intensity={pp.bloom.intensity}
        luminanceThreshold={pp.bloom.luminanceThreshold}
        luminanceSmoothing={pp.bloom.luminanceSmoothing}
        blendFunction={BlendFunction.ADD}
      />,
    )
  }

  if (gfx.ssao) {
    effects.push(
      <SSAO
        key="ssao"
        intensity={pp.ssao.intensity}
        radius={pp.ssao.radius}
        samples={pp.ssao.samples}
        blendFunction={BlendFunction.MULTIPLY}
      />,
    )
  }

  if (gfx.vignette) {
    effects.push(
      <Vignette
        key="vignette"
        offset={pp.vignette.offset}
        darkness={pp.vignette.darkness}
        blendFunction={BlendFunction.NORMAL}
      />,
    )
  }

  // SSAO reads the scene normals, and the r3f wrapper bails out with
  // `console.error("Please enable the NormalPassâ€¦")` and an empty object when
  // the composer has none â€” so until now the SSAO toggle produced a console
  // error and nothing else. Tied to the flag rather than switched on: the normal
  // pass is a second full pass over the scene, not worth paying for when SSAO
  // is off.
  return <EffectComposer enableNormalPass={gfx.ssao}>{effects}</EffectComposer>
}

/**
 * Takes no props and reads only static config, so it never needs to re-render.
 * Without this, every FSM state change re-rendered the whole scene subtree and
 * re-created the effect elements inside EffectComposer.
 */
export default memo(ScenePostProcessing)

