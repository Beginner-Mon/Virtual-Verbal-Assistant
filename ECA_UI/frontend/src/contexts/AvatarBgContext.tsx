import { createContext, useContext, useState, useEffect, type ReactNode } from 'react'
import { AVATAR_BG_OPTIONS, type AvatarBgId } from '@/lib/avatarPalette'
import { fetchAuthSession } from 'aws-amplify/auth'
import { fetchPreferences, patchPreferences } from '@/lib/preferences'

interface AvatarBgContextType {
  colorId: AvatarBgId
  bg: (typeof AVATAR_BG_OPTIONS)[number]
  setColorId: (id: AvatarBgId) => void
}

const AvatarBgContext = createContext<AvatarBgContextType | null>(null)

const STORAGE_KEY = 'vva_avatar_bg'

/**
 * Synced preference (Neon user_preferences.avatar_bg) with localStorage fallback
 * for guests (no token / vva_demo_user). Authed users read/write Neon;
 * guests keep the old localStorage behaviour.
 */
export function AvatarBgProvider({ children }: { children: ReactNode }) {
  const [colorId, setColorId] = useState<AvatarBgId>(() => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY) as AvatarBgId | null
      if (stored && AVATAR_BG_OPTIONS.some((o) => o.id === stored)) return stored
    } catch {
      // localStorage may be unavailable (SSR/private mode)
    }
    return 'slate'
  })

  const [version, setVersion] = useState<number | null>(null)
  const [isAuthed, setIsAuthed] = useState(false)

  // Hydrate from Neon on mount for authed users (cross-device sync)
  useEffect(() => {
    let cancelled = false
    void (async () => {
      try {
        const s = await fetchAuthSession()
        const authed = !!s.tokens?.idToken
        if (cancelled) return
        setIsAuthed(authed)
        if (!authed) return
        const prefs = await fetchPreferences()
        if (cancelled) return
        if (prefs.avatar_bg && AVATAR_BG_OPTIONS.some((o) => o.id === prefs.avatar_bg)) {
          setColorId(prefs.avatar_bg as AvatarBgId)
        }
        setVersion(prefs.version)
      } catch {
        // guest or 401 → keep localStorage value
      }
    })()
    return () => {
      cancelled = true
    }
  }, [])

  // Persist: Neon when authed, localStorage always (fallback + guest)
  const persist = (next: AvatarBgId) => {
    try {
      localStorage.setItem(STORAGE_KEY, next)
    } catch {
      // ignore write errors
    }
    if (isAuthed && version !== null) {
      // Fire-and-forget with optimistic version; 409 will be resolved on next fetch
      void patchPreferences({ avatar_bg: next, version }).catch(() => {})
      setVersion((v) => (v === null ? v : v + 1))
    } else if (isAuthed && version === null) {
      // First write before version known — fetch then patch
      void (async () => {
        try {
          const prefs = await fetchPreferences()
          await patchPreferences({ avatar_bg: next, version: prefs.version })
          setVersion(prefs.version + 1)
        } catch {
          // ignore
        }
      })()
    }
  }

  const setColorIdWrapped = (id: AvatarBgId) => {
    setColorId(id)
    persist(id)
  }

  const bg = AVATAR_BG_OPTIONS.find((o) => o.id === colorId) ?? AVATAR_BG_OPTIONS[0]

  return (
    <AvatarBgContext.Provider value={{ colorId, bg, setColorId: setColorIdWrapped }}>
      {children}
    </AvatarBgContext.Provider>
  )
}

export function useAvatarBg(): AvatarBgContextType {
  const ctx = useContext(AvatarBgContext)
  if (!ctx) throw new Error('useAvatarBg must be used within AvatarBgProvider')
  return ctx
}
