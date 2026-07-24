import type { AvatarProfile } from '../AvatarProfile'
import { defaultProfile } from './default'

/**
 * Override for seele.vrm. Inspecting the GLB shows all 17 blendshape groups
 * ship with EMPTY `binds` — the expressions register but drive nothing
 * (facial-animation-plan.md §6.1 note). The mesh does carry morph targets with
 * MMD-style Japanese names (extras.targetNames), so we repair the binds by
 * name at attach time via `morphRepairMap`.
 *
 * Mapping (targetNames index -> channel):
 *   にこり smile -> happy, 哀 sadness -> sad, 怒り anger -> angry,
 *   なごみ calm -> relaxed, びっくり surprised -> surprised,
 *   まばたき blink -> blink, あ/い/う/え/お -> aa/ih/ou/ee/oh
 */
export const seeleProfile: AvatarProfile = {
  ...defaultProfile,
  modelId: 'seele',
  morphRepairMap: {
    happy: 'にこり',
    sad: '哀',
    angry: '怒り',
    relaxed: 'なごみ',
    surprised: 'びっくり',
    blink: 'まばたき',
    aa: 'あ',
    ih: 'い',
    ou: 'う',
    ee: 'え',
    oh: 'お',
  },
  binaryEmotions: ['relaxed'],
}
