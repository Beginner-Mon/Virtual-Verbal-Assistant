/**
 * Locale runtime. Import once from main.tsx, before the first render.
 *
 * Locale is the language of the WEBSITE only. It does NOT decide what language
 * the assistant answers in — that mirrors the user's question, per turn, and is
 * decided by the backend. A viewer reading the site in English and typing
 * Vietnamese should get a Vietnamese reply on an English page; if changing this
 * setting ever changes the reply language, the two axes have been wired
 * together and that is the bug.
 */

import i18next from 'i18next'
import { initReactI18next } from 'react-i18next'
import LanguageDetector from 'i18next-browser-languagedetector'

import en from './locales/en.json'
import vi from './locales/vi.json'
import { DEFAULT_LOCALE, LOCALES, type Locale } from './locale'

export const LOCALE_STORAGE_KEY = 'eca-locale'

/**
 * Ordered — first detector to return a supported locale wins.
 *
 * Declared as an array even though it holds two entries, because the next
 * change to it is an insertion: Phase 2 adds a `geo` detector that routes by
 * the viewer's location. It goes AFTER `localStorage` deliberately — a person
 * who has chosen a language must outrank a guess made from where they happen to
 * be. A Vietnamese speaker abroad has already answered the question.
 */
const DETECTION_ORDER = ['localStorage', 'navigator'] as const

void i18next
  .use(LanguageDetector)
  .use(initReactI18next)
  .init({
    resources: {
      en: { translation: en },
      vi: { translation: vi },
    },
    supportedLngs: [...LOCALES],
    fallbackLng: DEFAULT_LOCALE,
    // 'vi-VN' → 'vi'. Without this every region-tagged browser misses the
    // catalogue and silently falls back to English.
    load: 'languageOnly',
    detection: {
      order: [...DETECTION_ORDER],
      lookupLocalStorage: LOCALE_STORAGE_KEY,
      caches: ['localStorage'],
    },
    interpolation: {
      // React escapes for us; letting i18next escape as well double-encodes
      // any apostrophe that reaches an interpolated value.
      escapeValue: false,
    },
    // A missing key should render as the key, loudly, rather than as an empty
    // element that looks like a layout bug.
    returnNull: false,
  })

/**
 * Keep `<html lang>` truthful.
 *
 * index.html hard-codes `lang="en"`, which is a lie the moment anyone switches.
 * Screen readers choose a voice from it, and CSS `:lang()` and hyphenation
 * rules key off it too.
 *
 * Not done with a blocking script the way the theme is: a wrong `lang` for one
 * frame is inaudible and invisible, whereas a wrong theme for one frame is a
 * white flash.
 */
function syncDocumentLang(lng: string) {
  // Guarded because this module is imported for its side effect — the `init`
  // call above — and that must work anywhere. Without the guard, importing it
  // outside a browser throws at module scope and takes the whole import graph
  // with it. The catalogue and the DOM attribute are separate concerns and only
  // one of them needs a document.
  if (typeof document === 'undefined') return
  document.documentElement.setAttribute('lang', lng)
}

syncDocumentLang(i18next.resolvedLanguage ?? DEFAULT_LOCALE)
i18next.on('languageChanged', syncDocumentLang)

export function currentLocale(): Locale {
  return (i18next.resolvedLanguage as Locale) ?? DEFAULT_LOCALE
}

export default i18next
