/**
 * The avatar-backdrop context object and its accessor.
 *
 * Split from contexts/AvatarBgContext.tsx for the same reason useMotion and
 * useTheme are: exporting a component and a hook from one file breaks Fast
 * Refresh.
 */

import { createContext, useContext } from 'react'
import type { AVATAR_BG_OPTIONS, AvatarBgId } from '@/lib/avatarPalette'

export interface AvatarBgContextType {
  colorId: AvatarBgId
  bg: (typeof AVATAR_BG_OPTIONS)[number]
  setColorId: (id: AvatarBgId) => void
}

export const AvatarBgContext = createContext<AvatarBgContextType | null>(null)

export function useAvatarBg(): AvatarBgContextType {
  const ctx = useContext(AvatarBgContext)
  if (!ctx) throw new Error('useAvatarBg must be used within AvatarBgProvider')
  return ctx
}
