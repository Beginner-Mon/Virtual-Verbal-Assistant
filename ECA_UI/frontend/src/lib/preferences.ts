/**
 * User preferences API — cross-device synced UI prefs (Neon + Bearer token).
 *
 * No user_id param anywhere — /me means "whoever the token says you are"
 * (api/auth.py current_user_id). No cookie, no DynamoDB.
 *
 * Preferences are UI-only. Clinical facts live in user_memory; the backend
 * refuses any key its SyncedPrefs model does not declare, so sending one here
 * is a 422 rather than something that quietly lands in the database.
 *
 * There is no request cache and no shared in-flight promise. There used to be
 * both, because three components fetched this independently — and because the
 * shared promise was created with one caller's AbortSignal, whichever of them
 * unmounted first cancelled the fetch the other two were awaiting (React's
 * StrictMode double-mount made that reliable in development). The deduplication
 * belongs one level up: PreferencesProvider fetches once and passes the result
 * down. See contexts/PreferencesContext.tsx.
 */

import { fetchAuthSession } from 'aws-amplify/auth'
import { API_GATEWAY } from './apiBase'

/**
 * Everything that follows a user between devices.
 *
 * Mirrors SyncedPrefs in agenticRAG/langgraph_agents/api/schemas.py. Adding a
 * synced preference means adding a field in both places and nothing else — the
 * column is JSONB so there is no migration.
 *
 * `avatar_bg` is a plain string rather than AvatarBgId on purpose: the backend
 * does not know the palette (it lives in avatarPalette.ts alone, so adding a
 * colour needs no backend deploy), which means a value from a newer build can
 * reach an older one. Look it up and fall back; never assume it is known.
 */
export interface SyncedPrefs {
  avatar_bg?: string | null
  selected_character_slug?: string | null
}

export interface UserPreferences {
  preferences: SyncedPrefs
  updated_at: string | null
}

/** A partial write. Only the keys present are changed; `null` clears one. */
export type PreferencesPatch = SyncedPrefs

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
    throw Object.assign(new Error(`HTTP ${res.status}: ${body}`), {
      status: res.status,
      body,
    })
  }
}

export async function fetchPreferences(): Promise<UserPreferences> {
  const headers = await authHeader()
  const res = await fetch(`${API_GATEWAY}/me/preferences`, {
    method: 'GET',
    headers: { Accept: 'application/json', ...headers },
  })
  const text = await res.text()
  ensureOk(res, text)
  return JSON.parse(text) as UserPreferences
}

export async function patchPreferences(
  patch: PreferencesPatch,
): Promise<UserPreferences> {
  const headers = await authHeader()
  const res = await fetch(`${API_GATEWAY}/me/preferences`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json', Accept: 'application/json', ...headers },
    body: JSON.stringify({ preferences: patch }),
  })
  const text = await res.text()
  ensureOk(res, text)
  return JSON.parse(text) as UserPreferences
}
