/**
 * User preferences API — cross-device synced UI prefs (Neon + Bearer token).
 *
 * No user_id param anywhere — /me means "whoever the token says you are"
 * (api/auth.py current_user_id). No cookie, no DynamoDB.
 * prefs is UI-only (notifications/locale), never PHI.
 */

import { fetchAuthSession } from 'aws-amplify/auth'
import { API_GATEWAY } from './apiBase'

export type AvatarBgId = 'slate' | 'violet' | 'blue' | 'emerald' | 'amber' | 'rose' | 'cyan' | 'indigo'

export interface UserPreferences {
  avatar_bg: AvatarBgId
  selected_character_slug: string | null
  display_name: string | null
  prefs: Record<string, unknown>
  version: number
  updated_at: string
}

export interface PreferencesPatch {
  avatar_bg?: AvatarBgId
  selected_character_slug?: string | null
  display_name?: string | null
  prefs?: Record<string, unknown>
  version: number
}

async function authHeader(): Promise<Record<string, string>> {
  try {
    const session = await fetchAuthSession()
    const token = session.tokens?.idToken?.toString()
    return token ? { Authorization: `Bearer ${token}` } : {}
  } catch {
    return {}
  }
}

function ensureOk(res: Response, body: string): void {
  if (!res.ok) {
    // 409 carries JSON {detail:{error:"version_conflict"...}}
    throw Object.assign(new Error(`HTTP ${res.status}: ${body}`), {
      status: res.status,
      body,
    })
  }
}

export async function fetchPreferences(signal?: AbortSignal): Promise<UserPreferences> {
  const headers = await authHeader()
  const res = await fetch(`${API_GATEWAY}/me/preferences`, {
    method: 'GET',
    headers: { Accept: 'application/json', ...headers },
    signal,
  })
  const text = await res.text()
  ensureOk(res, text)
  return JSON.parse(text) as UserPreferences
}

export async function patchPreferences(
  patch: PreferencesPatch,
  signal?: AbortSignal,
): Promise<UserPreferences> {
  const headers = await authHeader()
  const res = await fetch(`${API_GATEWAY}/me/preferences`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json', Accept: 'application/json', ...headers },
    body: JSON.stringify(patch),
    signal,
  })
  const text = await res.text()
  ensureOk(res, text)
  return JSON.parse(text) as UserPreferences
}
