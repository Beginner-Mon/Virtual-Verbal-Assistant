/**
 * What a locale IS, kept apart from the i18next runtime that consumes it.
 *
 * `characterCopy.ts` needs the type and the two constants but has no business
 * importing i18next — it resolves backend-served copy, not translation
 * catalogues. Splitting them also keeps this module importable from a test
 * without booting i18next.
 */

/**
 * Locale is the language of the WEBSITE — buttons, labels, placeholders, error
 * lines. It is NOT the language the assistant replies in: that mirrors the
 * user's question, per turn, and is decided by the backend. The two are allowed
 * to differ, and an English UI showing a Vietnamese reply is correct behaviour
 * rather than a bug.
 */
export const LOCALES = ['en', 'vi'] as const

export type Locale = (typeof LOCALES)[number]

/**
 * English, per the product rule that everything defaults to English. This is
 * the last resort in the detection chain, not a guess about who is visiting.
 */
export const DEFAULT_LOCALE: Locale = 'en'

export function isLocale(value: unknown): value is Locale {
  return typeof value === 'string' && (LOCALES as readonly string[]).includes(value)
}

/**
 * Narrow a BCP-47 tag to a supported locale: 'vi-VN' → 'vi', 'en-GB' → 'en'.
 *
 * Prefix matching rather than exact, because `navigator.language` reports a
 * region almost always ('en-US', 'vi-VN') and an exact table would miss every
 * one of them.
 */
export function normalizeLocale(tag: string | null | undefined): Locale | null {
  if (!tag) return null
  const base = tag.toLowerCase().split('-')[0]
  return isLocale(base) ? base : null
}

/** Native-language label, so someone stranded in the wrong language can still find the way out. */
export const LOCALE_LABELS: Record<Locale, string> = {
  en: 'English',
  vi: 'Tiếng Việt',
}
