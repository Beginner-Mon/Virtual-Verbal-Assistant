/**
 * Character catalog client.
 *
 * Two different origins, and the split is not arbitrary:
 *
 *   the CATALOG (this file's fetches)  → API Gateway. It is an API call.
 *   the MODEL FILES (`vrm_url` below)  → CloudFront. They are 9-17 MB binaries,
 *                                        over API Gateway's 10 MB limit.
 *
 * Until 20-08 both went through CloudFront, on a /characters* behavior pointing
 * at a Lambda Function URL. The catalog moved to the gateway with the rest of the
 * API; the .vrm files stayed where a CDN is the right answer.
 *
 * `vrm_url` is an absolute URL stored in the database by
 * scripts/upload_characters_to_s3.py, so it already points at CloudFront — this
 * file does not build it.
 */

import { API_GATEWAY, ASSET_BASE } from './apiBase'

export { ASSET_BASE }

/** Auto-extracted from the .vrm GLB by scripts/upload_characters_to_s3.py. */
export interface VrmMetadata {
  joint_count: number
  spine_count: number
  has_humanoid_rig: boolean
  blendshape_groups: {
    total: number
    emotions: number
    visemes: number
    blinks: number
    look_ats: number
    customs: number
  }
  has_blink: boolean
  has_look_at: boolean
  /** Empty when the model is fully usable; each entry is a reason it is not. */
  incompatible_reasons: string[]
  vrm_version: string
  file_size_bytes: number
  extracted_at: string
}

export interface Character {
  slug: string
  display_name: string
  description: string | null
  vrm_url: string
  thumbnail_url: string | null
  vrm_metadata: VrmMetadata
  voice_language: string
  sort_order: number
  /** Chat-surface copy — greeting, stage labels, error line, input placeholder.
   *  Optional because the catalog Lambda only returns it once redeployed with
   *  the column; see characterCopy.uiStringsFor for the fallback. */
  ui_strings?: Record<string, string>
}

interface CharacterListResponse {
  characters: Character[]
  total: number
}

/** True when nothing blocks this model from being used. */
export function isCompatible(character: Character): boolean {
  return (character.vrm_metadata?.incompatible_reasons?.length ?? 0) === 0
}

/** Human-readable reason a model is greyed out, or null when it is fine. */
export function incompatibilityReason(character: Character): string | null {
  const reasons = character.vrm_metadata?.incompatible_reasons ?? []
  if (reasons.length === 0) return null
  const readable: Record<string, string> = {
    'spine_count < 3': 'Rig has too few spine joints for motion retargeting',
    'blendshape_groups.total == 0': 'Model has no expression blendshapes',
    'no humanoid rig': 'Model has no humanoid rig',
  }
  return reasons.map((r) => readable[r] ?? r).join('; ')
}

export async function fetchCharacters(signal?: AbortSignal): Promise<Character[]> {
  const res = await fetch(`${API_GATEWAY}/characters`, { signal })
  if (!res.ok) throw new Error(`GET /characters failed: ${res.status}`)
  const data = (await res.json()) as CharacterListResponse
  return data.characters ?? []
}

export async function fetchAvatarProfile(
  slug: string,
  signal?: AbortSignal
): Promise<unknown> {
  const res = await fetch(`${API_GATEWAY}/characters/${slug}/avatar-profile`, { signal })
  if (!res.ok) throw new Error(`GET /characters/${slug}/avatar-profile failed: ${res.status}`)
  return res.json()
}
