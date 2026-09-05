import js from '@eslint/js'
import globals from 'globals'
import reactHooks from 'eslint-plugin-react-hooks'
import reactRefresh from 'eslint-plugin-react-refresh'
import i18next from 'eslint-plugin-i18next'
import tseslint from 'typescript-eslint'
import { defineConfig, globalIgnores } from 'eslint/config'

export default defineConfig([
  globalIgnores(['dist', '.amplify/**']),
  {
    files: ['**/*.{ts,tsx}'],
    extends: [
      js.configs.recommended,
      tseslint.configs.recommended,
      reactHooks.configs.flat.recommended,
      reactRefresh.configs.vite,
    ],
    languageOptions: {
      globals: globals.browser,
    },
  },

  /*
   * Hard-coded user-facing strings fail the build.
   *
   * This is the whole reason the project took on i18next rather than a
   * hand-rolled `t()`: translating the existing screens once is a morning's
   * work, but keeping them translated is forever, and nothing was stopping the
   * next PR from adding another literal. ChatPanel had
   * 'Đang tải hội thoại trước...' and 'Online · Ready to chat' on the SAME LINE
   * before this rule existed — nobody chose that, it just accumulated.
   *
   * Scoped deliberately rather than switched on globally:
   *
   *   - Only the directories that render UI. `src/avatar/` and `src/lib/` are
   *     full of string literals that are not copy at all — VRM bone names,
   *     animation state ids, cache keys, Tailwind class fragments — and a rule
   *     that flagged those would be turned off within a week.
   *   - Only JSX text and the four attributes a human actually reads. Without
   *     `callees`/`words` narrowing, every className would be a violation.
   */
  {
    files: ['src/components/**/*.tsx', 'src/contexts/**/*.tsx', 'src/hooks/**/*.ts'],
    ignores: ['**/*.test.ts', '**/*.test.tsx'],
    plugins: { i18next },
    rules: {
      'i18next/no-literal-string': [
        'error',
        {
          mode: 'jsx-text-only',
          'should-validate-template': false,
          message: 'Move user-facing text into src/i18n/locales/*.json and read it with t().',
          callees: { exclude: ['.*'] },
          'jsx-attributes': {
            include: ['aria-label', 'placeholder', 'title', 'alt'],
          },
          words: {
            // Punctuation and symbols carry no language. Anything with a letter
            // in it does, and must go through t().
            exclude: ['^[^\\p{L}]+$'],
          },
        },
      ],
    },
  },
])
