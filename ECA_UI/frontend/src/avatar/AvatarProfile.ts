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
  /**
   * Optional repair map for VRM 0.x files that declare expression groups with
   * EMPTY binds (e.g. seele.vrm — the groups exist but reference no morph
   * targets, so the expressions register but drive nothing). Maps an expression
   * channel (runtime preset name) to one or MORE morph target names on the
   * model's meshes; VRMExpressionAdapter patches the missing binds in at attach
   * time. An array is used when one channel must drive several morphs together.
   */
  morphRepairMap?: Record<string, string | string[]>
  /**
   * Canonical emotions whose morph targets are purely on/off — they have no
   * intensity gradation. The intensity slider is ignored and the expression
   * always renders at weight 1 (fully visible) whenever triggered.
   * (e.g. seele.vrm "なごみ" relaxed, bronya.vrm sorrow morphs.)
   */
  binaryEmotions?: CanonicalEmotion[]
  /** Which emotion to trigger at the midpoint of the greeting animation. Defaults to 'happy'. */
  greetingEmotion?: CanonicalEmotion
}

export function isCanonicalEmotion(name: string): name is CanonicalEmotion {
  return (CANONICAL_EMOTIONS as readonly string[]).includes(name)
}

// Per-model profile registry. `default` is the fallback for any VRM 0.x model
// whose migrated preset names match the standard set.
import { defaultProfile } from './profiles/default'
import { bronyaProfile } from './profiles/bronya'
import { seeleProfile } from './profiles/seele'

const PROFILE_REGISTRY: Record<string, AvatarProfile> = {
  default: defaultProfile,
  bronya: bronyaProfile,
  seele: seeleProfile,
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

// ── Remote profiles (characters.avatar_profile) ──────────────────────────────

const remoteProfileCache = new Map<string, AvatarProfile>()

function isUsableProfile(value: unknown): value is AvatarProfile {
  // The column defaults to '{}', so a character seeded before its profile was
  // written would otherwise hand back an object with no recipes and no viseme
  // map — the avatar attaches, drives nothing, and looks frozen rather than
  // broken. Falling back to the bundled profile is strictly better than that.
  if (!value || typeof value !== 'object') return false
  const p = value as Partial<AvatarProfile>
  return Boolean(p.recipes && p.visemes && p.blinkChannel)
}

/**
 * Resolve a profile for a model, preferring the one stored alongside the
 * character in the database so a new character needs no frontend deploy.
 *
 * Falls back to the bundled `PROFILE_REGISTRY` on any failure — a CDN blip
 * should cost the model its per-model overrides, not its face.
 */
export async function loadProfileAsync(
  modelId: string,
  signal?: AbortSignal
): Promise<AvatarProfile> {
  const key = modelId.toLowerCase()

  const cached = remoteProfileCache.get(key)
  if (cached) return cached

  try {
    const { fetchAvatarProfile } = await import('../lib/characters')
    const raw = await fetchAvatarProfile(key, signal)
    if (isUsableProfile(raw)) {
      const profile = { ...raw, modelId: key }
      remoteProfileCache.set(key, profile)
      return profile
    }
    console.warn(`[AvatarProfile] ${key}: remote profile incomplete, using bundled`)
  } catch (err) {
    if ((err as Error)?.name === 'AbortError') throw err
    console.warn(`[AvatarProfile] ${key}: remote profile unavailable, using bundled`, err)
  }

  return loadProfile(key)
}
