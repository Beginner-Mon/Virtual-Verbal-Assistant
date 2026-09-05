import { useTranslation } from 'react-i18next'
import { DEFAULT_LOCALE, isLocale, type Locale } from '@/i18n/locale'

/**
 * The active website locale, and the way to change it.
 *
 * Thin on purpose: `react-i18next` already re-renders every `useTranslation`
 * consumer on `languageChanged`, so a context of our own would only be a second
 * copy of state that can disagree with i18next's. This exists so call sites that
 * need the locale as a VALUE — `uiStringsFor(character, locale)` — can read it
 * without importing i18next directly.
 *
 * Persistence is i18next's `caches: ['localStorage']`, not ours.
 */
export function useLocale(): {
  locale: Locale
  setLocale: (next: Locale) => void
} {
  const { i18n } = useTranslation()

  const resolved = i18n.resolvedLanguage
  const locale: Locale = isLocale(resolved) ? resolved : DEFAULT_LOCALE

  return {
    locale,
    setLocale: (next: Locale) => {
      void i18n.changeLanguage(next)
    },
  }
}
