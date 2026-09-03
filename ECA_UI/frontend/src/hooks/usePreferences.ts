/**
 * The preferences context object and its accessor.
 *
 * Split from contexts/PreferencesContext.tsx for the same reason useMotion and
 * useTheme are: a file that exports both a component and a hook breaks Fast
 * Refresh, so the provider lives alone in contexts/ and everything else here.
 *
 * Not to be confused with useUserPreferences, which is the hook that actually
 * fetches and writes. That one is called once, by the provider. This is what
 * consumers use.
 */

import { createContext, useContext } from 'react'
import type { PreferencesPatch, UserPreferences } from '@/lib/preferences'

export interface PreferencesContextType {
  /** Null for a guest, or before the first fetch lands. */
  data: UserPreferences | null
  /** True until the first fetch settles — including for guests. */
  loading: boolean
  error: string | null
  /** Optimistic, debounced, partial write. Absent keys are left alone. */
  patch: (partial: PreferencesPatch) => void
}

export const PreferencesContext = createContext<PreferencesContextType | null>(null)

export function usePreferences(): PreferencesContextType {
  const ctx = useContext(PreferencesContext)
  if (!ctx) throw new Error('usePreferences must be used within PreferencesProvider')
  return ctx
}
