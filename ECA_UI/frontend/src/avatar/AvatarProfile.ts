/**
 * Avatar profile types + loader.
 *
 * A profile maps CANONICAL emotion names (what business logic / backend speak)
 * to a recipe of VRM expression CHANNELS (runtime preset names as exposed by
 * @pixiv/three-vrm v3 after it migrates a VRM 0.x model on load).
 *
 * Migration reference (three-vrm v3, VRM 0.x -> 1.0 preset):
 *   joy -> happy, sorrow -> sad, angry -> angry, fun -> relaxed,
 *   a -> aa, i -> ih, u -> ou, e -> ee, o -> oh, blink -> blink
 *
 * Profiles are `.ts` typed consts (not `.json`) because tsconfig does not enable
 * `resolveJsonModule` and `verbatimModuleSyntax` is on — see facial-animation-plan.md §3.
 */

// Canonical emotion vocabulary. Backend / DevPanel may only speak these names.
export const CANONICAL_EMOTIONS = [
  'neutral',
  'happy',
  'sad',
  'angry',
  'relaxed',
  'surprised',
] as const

export type CanonicalEmotion = (typeof CANONICAL_EMOTIONS)[number]

/** A recipe: how much weight each VRM channel gets for one canonical emotion. */
export type EmotionRecipe = Record<string, number>

/** Canonical vowel keys used by the (future) lip-sync viseme mapping. */
export type Viseme = 'A' | 'I' | 'U' | 'E' | 'O'

export interface AvatarProfile {
  version: 1
  modelId: string
  /** Every canonical emotion maps to a channel recipe (may be empty = neutral). */
  recipes: Record<CanonicalEmotion, EmotionRecipe>
  /** Canonical vowel -> VRM viseme channel (Phase C lip-sync). */
  visemes: Record<Viseme, string>
  /** Channel that carries the blink expression on this model. */
  blinkChannel: string
}

export function isCanonicalEmotion(name: string): name is CanonicalEmotion {
  return (CANONICAL_EMOTIONS as readonly string[]).includes(name)
}

// Per-model profile registry. `default` is the fallback for any VRM 0.x model
// whose migrated preset names match the standard set.
import { defaultProfile } from './profiles/default'
import { bronyaProfile } from './profiles/bronya'

const PROFILE_REGISTRY: Record<string, AvatarProfile> = {
  default: defaultProfile,
  bronya: bronyaProfile,
}

/**
 * Resolve a profile for a model id (e.g. "seele", "bronya", "bronya_long").
 * Falls back to `default` when no per-model override exists — most VRM 0.x
 * models fit the default because three-vrm migrates their presets to the
 * standard 1.0 names.
 */
export function loadProfile(modelId: string): AvatarProfile {
  const key = modelId.toLowerCase()
  const override = PROFILE_REGISTRY[key]
  if (override) return override
  return { ...PROFILE_REGISTRY.default, modelId }
}
