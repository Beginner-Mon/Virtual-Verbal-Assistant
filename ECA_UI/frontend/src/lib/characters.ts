/**
 * Character catalog client.
 *
 * The catalog lives behind CloudFront, not behind the chat API: `/characters*`
 * is routed to a Lambda and everything else on that distribution is the VRM
 * files themselves. So it reads VITE_ASSET_BASE_URL, while streaming chat keeps
 * using VITE_API_BASE_URL in lib/api.ts. Two origins, split by what they serve
 * — SSE deliberately does not go through the CDN.
 */

const DEFAULT_ASSET_BASE = 'http://localhost:8000'

const _rawAssetBase = import.meta.env.VITE_ASSET_BASE_URL as string | undefined
if (!_rawAssetBase) {
  console.warn(
    `[characters] VITE_ASSET_BASE_URL is not set — falling back to ${DEFAULT_ASSET_BASE}. ` +
      'Set it to the AssetStack CloudFront domain (CfnOutput AssetBaseUrl). Vite reads ' +
      'this at BUILD time: .env.local for local dev, Amplify env vars for a deployed build.'
  )
}
export const ASSET_BASE: string = _rawAssetBase || DEFAULT_ASSET_BASE

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
  const res = await fetch(`${ASSET_BASE}/characters`, { signal })
  if (!res.ok) throw new Error(`GET /characters failed: ${res.status}`)
  const data = (await res.json()) as CharacterListResponse
  return data.characters ?? []
}

export async function fetchAvatarProfile(
  slug: string,
  signal?: AbortSignal
): Promise<unknown> {
  const res = await fetch(`${ASSET_BASE}/characters/${slug}/avatar-profile`, { signal })
  if (!res.ok) throw new Error(`GET /characters/${slug}/avatar-profile failed: ${res.status}`)
  return res.json()
}
