import type { AvatarProfile } from '../AvatarProfile'

/**
 * Default profile for VRM 0.x models whose presets migrate cleanly to the
 * three-vrm v3 standard 1.0 names. Verified against seele.vrm (17 blendshape
 * groups: neutral, a/i/u/e/o, blink, joy, angry, sorrow, fun, look*, blink_l/r).
 *
 * `surprised` is intentionally mapped to the `surprised` preset even though a
 * plain VRM 0.x model has no such preset — VRMExpressionAdapter's capability
 * detection makes it a safe no-op where the channel is absent. Models that DO
 * have it (as a custom expression) provide a per-model override — see bronya.ts.
 */
export const defaultProfile: AvatarProfile = {
  version: 1,
  modelId: 'default',
  recipes: {
    neutral: {},
    happy: { happy: 1.0 },
    sad: { sad: 1.0 },
    angry: { angry: 1.0 },
    relaxed: { relaxed: 1.0 },
    surprised: { surprised: 1.0 },
  },
  visemes: { A: 'aa', I: 'ih', U: 'ou', E: 'ee', O: 'oh' },
  blinkChannel: 'blink',
}
