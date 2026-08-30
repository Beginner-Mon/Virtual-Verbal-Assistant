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
