/**
 * useUserPreferences — the one place that reads and writes synced prefs.
 *
 * Fetch once after auth, refresh on `visibilitychange` and every five minutes,
 * write through an optimistic PATCH debounced by 300ms.
 *
 * Call this exactly once, in PreferencesProvider. Every consumer reads the
 * result out of that context. Three components calling it independently is the
 * arrangement this replaced.
 *
 * Guest (no token) → data stays null, and callers fall back to localStorage.
 *
 * Writes are last-write-wins, so there is no 409 to handle: the server merges
 * per key, and two devices changing two different preferences never disagree.
 */

import { useCallback, useEffect, useRef, useState } from 'react'
import { fetchAuthSession } from 'aws-amplify/auth'
import {
  fetchPreferences,
  patchPreferences,
  type PreferencesPatch,
  type UserPreferences,
} from '@/lib/preferences'

const REFRESH_MS = 5 * 60 * 1000
const PATCH_DEBOUNCE_MS = 300

export function useUserPreferences() {
  const [data, setData] = useState<UserPreferences | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // A generation counter rather than an AbortController: fetchPreferences takes
  // no signal any more (one shared promise cancelled by one caller's unmount was
  // the bug that removed it), so what is needed here is only to ignore the
  // result of a refresh that a newer one has overtaken.
  const generationRef = useRef(0)
  const patchTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const pendingRef = useRef<PreferencesPatch>({})

  const refresh = useCallback(async () => {
    const generation = ++generationRef.current
    try {
      const session = await fetchAuthSession()
      if (!session.tokens?.idToken) {
        if (generation === generationRef.current) {
          setData(null)
          setLoading(false)
        }
        return
      }
      const prefs = await fetchPreferences()
      if (generation !== generationRef.current) return
      setData(prefs)
      setError(null)
    } catch (e) {
      if (generation !== generationRef.current) return
      // 401 is a guest or an expired token, not something to show anyone.
      if ((e as { status?: number }).status === 401) setData(null)
      else setError(e instanceof Error ? e.message : String(e))
    } finally {
      if (generation === generationRef.current) setLoading(false)
    }
  }, [])

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- intentional initial sync: hydrate prefs once after mount
    void refresh()
    const onVisible = () => {
      if (document.visibilityState === 'visible') void refresh()
    }
    document.addEventListener('visibilitychange', onVisible)
    const id = window.setInterval(() => void refresh(), REFRESH_MS)
    return () => {
      // Not a DOM ref: bumping the counter is exactly the point, so that any
      // refresh still in flight discards its result instead of setting state on
      // an unmounted component.
      // eslint-disable-next-line react-hooks/exhaustive-deps
      generationRef.current++
      document.removeEventListener('visibilitychange', onVisible)
      window.clearInterval(id)
      if (patchTimerRef.current) window.clearTimeout(patchTimerRef.current)
    }
  }, [refresh])

  /**
   * Apply a partial change locally, then send it.
   *
   * Debounced so a run of quick changes coalesces into one request, and
   * accumulated rather than replaced while the timer runs — otherwise picking a
   * colour and then a character within 300ms would send only the character.
   */
  const patch = useCallback(
    (partial: PreferencesPatch) => {
      setData((current) =>
        current
          ? { ...current, preferences: { ...current.preferences, ...partial } }
          : current,
      )
      pendingRef.current = { ...pendingRef.current, ...partial }

      if (patchTimerRef.current) window.clearTimeout(patchTimerRef.current)
      patchTimerRef.current = window.setTimeout(() => {
        const body = pendingRef.current
        pendingRef.current = {}
        void (async () => {
          try {
            const saved = await patchPreferences(body)
            setData(saved)
            setError(null)
          } catch (e) {
            // The optimistic value is now wrong and there is nothing to reconcile
            // against, so take the server's word for it.
            setError(e instanceof Error ? e.message : String(e))
            void refresh()
          }
        })()
      }, PATCH_DEBOUNCE_MS)
    },
    [refresh],
  )

  return { data, loading, error, refresh, patch }
}
