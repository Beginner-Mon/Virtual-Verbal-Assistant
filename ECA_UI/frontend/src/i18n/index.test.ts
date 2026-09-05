import { describe, expect, it } from 'vitest'
import i18n from './index'
import { DEFAULT_LOCALE, LOCALES } from './locale'

/**
 * Proves the runtime is wired, not just that the JSON files parse.
 *
 * `catalog.test.ts` reads the two files directly and would stay green through a
 * broken `init` — a missing `resources` entry, a `supportedLngs` that excludes a
 * locale, a `load` mode that drops the region tag. Every one of those renders
 * raw key strings on screen and none of them fail a build.
 */
describe('i18n runtime', () => {
  it('initialises', () => {
    expect(i18n.isInitialized).toBe(true)
  })

  it('defaults to English', () => {
    expect(i18n.resolvedLanguage).toBe(DEFAULT_LOCALE)
  })

  it('resolves a real key rather than echoing it back', () => {
    const value = i18n.t('chat.new_conversation')
    expect(value).toBe('New conversation')
    expect(value).not.toBe('chat.new_conversation')
  })

  it('serves every supported locale from bundled resources', async () => {
    for (const locale of LOCALES) {
      await i18n.changeLanguage(locale)
      expect(i18n.resolvedLanguage).toBe(locale)
      expect(i18n.t('chat.new_conversation')).not.toBe('chat.new_conversation')
    }
    await i18n.changeLanguage(DEFAULT_LOCALE)
  })

  it('narrows a region tag to its base locale', async () => {
    // `navigator.language` is 'vi-VN' far more often than plain 'vi'. Without
    // `load: 'languageOnly'` this resolves to nothing and every Vietnamese
    // browser silently gets English.
    await i18n.changeLanguage('vi-VN')
    expect(i18n.resolvedLanguage).toBe('vi')
    expect(i18n.t('chat.new_conversation')).toBe('Cuộc trò chuyện mới')
    await i18n.changeLanguage(DEFAULT_LOCALE)
  })

  it('falls back to English for an unsupported language', async () => {
    await i18n.changeLanguage('fr')
    expect(i18n.t('chat.new_conversation')).toBe('New conversation')
    await i18n.changeLanguage(DEFAULT_LOCALE)
  })
})
