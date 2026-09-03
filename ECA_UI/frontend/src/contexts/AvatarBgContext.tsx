import { useState, type ReactNode } from 'react'
import { AVATAR_BG_OPTIONS, type AvatarBgId } from '@/lib/avatarPalette'
import { AvatarBgContext } from '@/hooks/useAvatarBg'
import { usePreferences } from '@/hooks/usePreferences'

const STORAGE_KEY = 'vva_avatar_bg'

function readLocal(): AvatarBgId | null {
  try {
    const stored = localStorage.getItem(STORAGE_KEY) as AvatarBgId | null
    return stored && AVATAR_BG_OPTIONS.some((o) => o.id === stored) ? stored : null
  } catch {
    // Unavailable in private mode; not worth distinguishing from unset.
    return null
  }
}

function known(id: string | null | undefined): id is AvatarBgId {
  return !!id && AVATAR_BG_OPTIONS.some((o) => o.id === id)
}

/**
 * The avatar backdrop: synced through users.preferences for signed-in users,
 * localStorage for guests.
 *
 * The server value wins when it is one this build knows about. It may not be:
 * the palette lives in avatarPalette.ts and only there, so a colour added in a
 * newer frontend can reach an older one, and an unknown id has to fall through
 * to the local value rather than render as nothing.
 *
 * localStorage is written either way. It is all a guest has, and for a signed-in
 * user it is what paints the first frame on the next visit, before
 * GET /me/preferences has answered.
 */
export function AvatarBgProvider({ children }: { children: ReactNode }) {
  const { data, patch } = usePreferences()
  const [localId, setLocalId] = useState<AvatarBgId>(() => readLocal() ?? 'slate')

  const synced = data?.preferences.avatar_bg
  const colorId: AvatarBgId = known(synced) ? synced : localId

  const setColorId = (id: AvatarBgId) => {
    setLocalId(id)
    try {
      localStorage.setItem(STORAGE_KEY, id)
    } catch {
      // Quota or private mode — the synced copy is the one that matters.
    }
    // `data` is non-null exactly when there is a session to write to. The write
    // is optimistic and debounced inside the hook, so this returns immediately.
    if (data) patch({ avatar_bg: id })
  }

  const bg = AVATAR_BG_OPTIONS.find((o) => o.id === colorId) ?? AVATAR_BG_OPTIONS[0]

  return (
    <AvatarBgContext.Provider value={{ colorId, bg, setColorId }}>
      {children}
    </AvatarBgContext.Provider>
  )
}
