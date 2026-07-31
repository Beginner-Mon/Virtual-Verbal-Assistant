import type { AvatarProfile } from '../AvatarProfile'
import { defaultProfile } from './default'

/**
 * Override for bronya.vrm. Inspecting the GLB shows an 18th blendshape group
 * with presetName "unknown" and name "Surprised" (4 binds) — a custom VRM 0.x
 * expression. three-vrm keeps custom expressions keyed by their raw name, so
 * the `surprised` canonical emotion must target the "Surprised" channel here
 * instead of the (absent) standard `surprised` preset.
 */
export const bronyaProfile: AvatarProfile = {
  ...defaultProfile,
  modelId: 'bronya',
  recipes: {
    ...defaultProfile.recipes,
    surprised: { Surprised: 1.0 },
  },
  morphRepairMap: {
    blink: ['43', '44'],
    aa: '30',
    ih: '31',
    ou: '32',
    ee: '34',
    oh: '35',
  },
  binaryEmotions: ['sad'],
}
