/**
 * useUserPreferences — SWR for cross-device synced UI prefs (Neon).
 *
 * Fetch GET /me/preferences once after auth, cache 5m, refetch on
 * visibilitychange. PATCH with optimistic lock (version), debounce 300ms,
 * rollback on 409.
 *
 * Guest (no token) → null, callers fall back to localStorage.
 */

import { useCallback, useEffect, useRef, useState } from 'react'
import { fetchAuthSession } from 'aws-amplify/auth'
import {
  fetchPreferences,
  patchPreferences,
  type UserPreferences,
  type PreferencesPatch,
} from '@/lib/preferences'

const REFRESH_MS = 5 * 60 * 1000

export function useUserPreferences() {
  const [data, setData] = useState<UserPreferences | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const abortRef = useRef<AbortController | null>(null)
  const patchTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const isAuthed = useCallback(async () => {
    try {
      const s = await fetchAuthSession()
      return !!s.tokens?.idToken
    } catch {
      return false
    }
  }, [])

  const refresh = useCallback(async () => {
    if (!(await isAuthed())) {
      setData(null)
      setLoading(false)
      return
    }
    abortRef.current?.abort()
    const ac = new AbortController()
    abortRef.current = ac
    try {
      const prefs = await fetchPreferences(ac.signal)
      if (!ac.signal.aborted) {
        setData(prefs)
        setError(null)
      }
    } catch (e) {
      if ((e as Error).name === 'AbortError') return
      // 401 for guest or expired token → treat as no prefs, not error toast
      const status = (e as { status?: number }).status
      if (status === 401) {
        setData(null)
      } else {
        setError(e instanceof Error ? e.message : String(e))
      }
    } finally {
      if (!ac.signal.aborted) setLoading(false)
    }
  }, [isAuthed])

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- intentional initial sync: hydrate prefs once after mount
    void refresh()
    const onVisible = () => {
      if (document.visibilityState === 'visible') void refresh()
    }
    document.addEventListener('visibilitychange', onVisible)
    const id = window.setInterval(() => void refresh(), REFRESH_MS)
    return () => {
      abortRef.current?.abort()
      document.removeEventListener('visibilitychange', onVisible)
      window.clearInterval(id)
      if (patchTimerRef.current) window.clearTimeout(patchTimerRef.current)
    }
  }, [refresh])

  // Optimistic PATCH with debounce 300ms (Q7 rare). Caller passes partial patch
  // without version — we fill current version. On 409, refetch.
  const patch = useCallback(
    async (partial: Omit<PreferencesPatch, 'version'>) => {
      if (!data) return
      const nextVersion = data.version
      const optimistic: UserPreferences = {
        ...data,
        ...partial,
        // prefs is merged shallow — keep existing keys not in patch
        prefs: partial.prefs ? { ...data.prefs, ...partial.prefs } : data.prefs,
        version: data.version + 1,
      }
      // Apply optimistic immediately
      setData(optimistic)

      // Debounce the network call — multiple rapid patches (theme toggles) coalesce
      if (patchTimerRef.current) window.clearTimeout(patchTimerRef.current)
      return new Promise<UserPreferences | null>((resolve) => {
        patchTimerRef.current = window.setTimeout(async () => {
          try {
            const saved = await patchPreferences({ ...partial, version: nextVersion })
            setData(saved)
            setError(null)
            resolve(saved)
          } catch (e) {
            const status = (e as { status?: number }).status
            if (status === 409) {
              // Version conflict — refetch canonical
              await refresh()
            } else {
              // Rollback on other error
              setData(data)
              setError(e instanceof Error ? e.message : String(e))
            }
            resolve(null)
          }
        }, 300)
      })
    },
    [data, refresh],
  )

  return { data, loading, error, refresh, patch }
}
