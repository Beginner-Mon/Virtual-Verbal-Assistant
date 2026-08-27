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
  greetingEmotion: 'happy',
  // Bundled gestures every model gets. A character that brings its own set from
  // the database overrides this; one that brings none keeps it. These are ADDED
  // to the built-in FSM states (idle / greeting / bored / thinking / exercise),
  // which are untouched.
  gestures: {
    kiss: { source: { builtIn: 'kiss' }, crossfade: 0.5 },
  },
  reactions: {
    // Animation and emotion together in one binding — the click plays the clip
    // and lifts the expression, from a single per-model record.
    'bodyPartClick:mouth': { gesture: 'kiss', emotion: { name: 'happy', durationMs: 600 } },
  },
}
