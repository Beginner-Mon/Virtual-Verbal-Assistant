import { useMemo, type ReactNode } from 'react'
import { useUserPreferences } from '@/hooks/useUserPreferences'
import { PreferencesContext } from '@/hooks/usePreferences'

/**
 * One fetch of GET /me/preferences per page load, shared by everyone who needs it.
 *
 * AvatarBgContext, MotionContext and AvatarsPanel each used to fetch this for
 * themselves. Three requests were fired in parallel on every load, and the
 * deduplication bolted on afterwards — a module-level promise shared between
 * callers that still carried one caller's AbortSignal — meant whichever consumer
 * unmounted first cancelled the fetch the other two were waiting on. React's
 * StrictMode double-mount made that happen on every development reload.
 *
 * Must sit outside MotionProvider: the character to render is read from here.
 */
export function PreferencesProvider({ children }: { children: ReactNode }) {
  const { data, loading, error, patch } = useUserPreferences()
  const value = useMemo(
    () => ({ data, loading, error, patch }),
    [data, loading, error, patch],
  )
  return <PreferencesContext.Provider value={value}>{children}</PreferencesContext.Provider>
}
