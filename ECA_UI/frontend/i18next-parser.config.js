/**
 * Extracts `t('...')` keys from source into src/i18n/locales/*.json.
 *
 * Run it to find keys a component uses that no catalogue defines:
 *
 *     npx i18next
 *
 * `keepRemoved: true` on purpose — the parser only sees keys it can read
 * statically, and a key looked up through a variable would otherwise be deleted
 * from the catalogue on the next run and take the translation with it. Removing
 * a dead key is a deliberate act, not a side effect of running a tool.
 *
 * This complements, and does not replace, the ESLint rule: the parser finds
 * MISSING translations, the rule finds text that never went through `t()` at
 * all. Neither catches the other's failure.
 */
export default {
  locales: ['en', 'vi'],
  defaultNamespace: 'translation',
  input: ['src/**/*.{ts,tsx}', '!src/**/*.test.{ts,tsx}'],
  output: 'src/i18n/locales/$LOCALE.json',

  // Keys are authored as 'section.name', so the separator must be active — but
  // plural/context suffixes are not used anywhere and would turn a literal
  // underscore in a key into an accidental plural form.
  keySeparator: '.',
  namespaceSeparator: false,
  pluralSeparator: false,
  contextSeparator: false,

  keepRemoved: true,
  sort: true,
  // An empty string here would render as blank on screen. Failing loudly with
  // the key text is the behaviour `returnNull: false` in src/i18n/index.ts is
  // already chosen for; keep the two consistent.
  defaultValue: '',
  createOldCatalogs: false,
  indentation: 2,
}
