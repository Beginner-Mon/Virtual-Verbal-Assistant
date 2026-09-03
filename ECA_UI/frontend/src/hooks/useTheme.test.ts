/**
 * The theme must not be decided twice.
 *
 * index.html runs a blocking script before first paint: it reads `eca-theme`
 * from localStorage, falls back to `prefers-color-scheme`, and puts the result
 * on <html> as a class. Whatever React does afterwards has to agree with that,
 * because disagreeing means repainting the page in front of the user.
 *
 * It disagreed. ThemeProvider read localStorage itself and fell back to
 * `defaultTheme`, which main.tsx passes as "light" — so a first visit from a
 * dark-mode machine painted dark, then flipped to light on mount. The script
 * had been added to stop exactly that flash, in the other direction.
 */

import { describe, expect, it } from 'vitest'
import { resolveInitialTheme, type ThemeRoot } from './useTheme'

const root = (...classes: string[]): ThemeRoot => ({
  classList: { contains: (token: string) => classes.includes(token) },
})

describe('resolveInitialTheme', () => {
  it('takes dark from the class the blocking script painted with', () => {
    // The regression: nothing stored, defaultTheme 'light', script chose dark
    // from prefers-color-scheme. Returning 'light' here is the flash.
    expect(resolveInitialTheme(root('dark'), null, 'light')).toBe('dark')
  })

  it('takes light from the class as well', () => {
    expect(resolveInitialTheme(root('light'), null, 'dark')).toBe('light')
  })

  it('lets the painted class outrank a stale stored value', () => {
    expect(resolveInitialTheme(root('dark'), 'light', 'light')).toBe('dark')
  })

  it('falls back to storage when no class is present', () => {
    expect(resolveInitialTheme(root(), 'dark', 'light')).toBe('dark')
  })

  it('ignores a stored value that is not a theme', () => {
    expect(resolveInitialTheme(root(), 'purple', 'light')).toBe('light')
  })

  it('falls back to the default when there is neither', () => {
    expect(resolveInitialTheme(root(), null, 'light')).toBe('light')
    expect(resolveInitialTheme(root(), null, 'dark')).toBe('dark')
  })

  it('is not confused by unrelated classes on <html>', () => {
    expect(resolveInitialTheme(root('h-full', 'dark', 'antialiased'), null, 'light')).toBe('dark')
  })
})
