import { createContext, useContext } from 'react'

export type Theme = 'dark' | 'light'

export type ThemeProviderState = {
  theme: Theme
  setTheme: (theme: Theme) => void
}

export type ThemeProviderProps = {
  children: React.ReactNode
  defaultTheme?: Theme
  storageKey?: string
}

/** Just enough of an Element to ask which theme class it carries. */
export interface ThemeRoot {
  classList: { contains(token: string): boolean }
}

/**
 * Which theme the app starts in.
 *
 * The blocking script in index.html has already decided this and painted with
 * it, before React exists — reading `localStorage` here and deciding again is
 * what caused the flash it was added to prevent. The script falls back to
 * `prefers-color-scheme` when nothing is stored, so on a first visit from a
 * dark-mode machine it paints dark; a provider defaulting to 'light' then
 * repainted. So: whatever is on `<html>` wins, and the rest is only for the
 * case where the script did not run at all.
 */
export function resolveInitialTheme(
  root: ThemeRoot,
  stored: string | null,
  defaultTheme: Theme,
): Theme {
  if (root.classList.contains('dark')) return 'dark'
  if (root.classList.contains('light')) return 'light'
  if (stored === 'dark' || stored === 'light') return stored
  return defaultTheme
}

export const ThemeContext = createContext<ThemeProviderState>({
  theme: 'light',
  setTheme: () => null,
})

export const useTheme = () => {
  const context = useContext(ThemeContext)
  if (context === undefined)
    throw new Error('useTheme must be used within a ThemeProvider')
  return context
}
