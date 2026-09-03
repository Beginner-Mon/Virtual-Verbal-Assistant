import { useEffect, useState } from 'react'
import {
  ThemeContext,
  resolveInitialTheme,
  type Theme,
  type ThemeProviderProps,
} from '../hooks/useTheme'

export function ThemeProvider({
  children,
  defaultTheme = 'light',
  storageKey = 'vite-ui-theme',
  ...props
}: ThemeProviderProps) {
  // Decided by resolveInitialTheme, from the class the blocking script in
  // index.html has already put on <html> — see the reasoning there. Kept out of
  // this file so it can be tested without a DOM.
  const [theme, setTheme] = useState<Theme>(() =>
    resolveInitialTheme(
      window.document.documentElement,
      localStorage.getItem(storageKey),
      defaultTheme,
    ),
  )

  useEffect(() => {
    const root = window.document.documentElement

    root.classList.remove('light', 'dark')

    root.classList.add(theme)
  }, [theme])

  const value = {
    theme,
    setTheme: (theme: Theme) => {
      localStorage.setItem(storageKey, theme)
      setTheme(theme)
    },
  }

  return (
    <ThemeContext.Provider {...props} value={value}>
      {children}
    </ThemeContext.Provider>
  )
}

export type { Theme, ThemeProviderState, ThemeProviderProps } from '../hooks/useTheme'
