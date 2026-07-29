/**
 * ScenePostProcessing — Phase 6: Post Processing
 *
 * Very light post-processing stack: Bloom + SSAO + Vignette.
 * All values pulled from ENV_CONFIG — no hardcoding.
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
import { ENV_CONFIG } from '../../config/environmentConfig'

export default function ScenePostProcessing() {
  const { postProcessing: pp } = ENV_CONFIG

  if (!pp.enabled) return null

  return (
    <EffectComposer>
      <Bloom
        intensity={pp.bloom.intensity}
        luminanceThreshold={pp.bloom.luminanceThreshold}
        luminanceSmoothing={pp.bloom.luminanceSmoothing}
        blendFunction={BlendFunction.ADD}
      />

      {pp.ssao.enabled && (
        <SSAO
          intensity={pp.ssao.intensity}
          radius={pp.ssao.radius}
          samples={pp.ssao.samples}
          blendFunction={BlendFunction.MULTIPLY}
        />
      )}

      <Vignette
        offset={pp.vignette.offset}
        darkness={pp.vignette.darkness}
        blendFunction={BlendFunction.NORMAL}
      />
    </EffectComposer>
  )
}
