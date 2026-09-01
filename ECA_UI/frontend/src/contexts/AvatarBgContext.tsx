import { createContext, useContext, useState, useEffect, type ReactNode } from 'react'
import { AVATAR_BG_OPTIONS, type AvatarBgId } from '@/lib/avatarPalette'

interface AvatarBgContextType {
  colorId: AvatarBgId
  bg: (typeof AVATAR_BG_OPTIONS)[number]
  setColorId: (id: AvatarBgId) => void
}

const AvatarBgContext = createContext<AvatarBgContextType | null>(null)

const STORAGE_KEY = 'vva_avatar_bg'

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

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, colorId)
    } catch {
      // ignore write errors
    }
  }, [colorId])

  const bg = AVATAR_BG_OPTIONS.find((o) => o.id === colorId) ?? AVATAR_BG_OPTIONS[0]

  return (
    <AvatarBgContext.Provider value={{ colorId, bg, setColorId }}>
      {children}
    </AvatarBgContext.Provider>
  )
}

export function useAvatarBg(): AvatarBgContextType {
  const ctx = useContext(AvatarBgContext)
  if (!ctx) throw new Error('useAvatarBg must be used within AvatarBgProvider')
  return ctx
}
