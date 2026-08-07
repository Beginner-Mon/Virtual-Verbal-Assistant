import { defineConfig } from 'vitest/config'
import path from 'path'

/**
 * Deliberately NOT `mergeConfig(viteConfig, …)`.
 *
 * `vite.config.ts` registers `motionAssetSyncPlugin`, whose `buildStart` shells
 * out to `sync-npz-to-bvh.mjs`. Inheriting it would run that script on every
 * test invocation — seconds of file copying to run tests that touch none of it,
 * and a hard failure whenever the motion assets are absent (CI). The tests under
 * test are pure TypeScript plus three.js, so the only thing worth sharing is the
 * `@` alias.
 *
 * Environment stays `node`: three.js constructs Object3D/AnimationMixer without
 * a DOM, so jsdom would only add startup cost. A test that needs the DOM should
 * opt in per-file with `// @vitest-environment jsdom`.
 */
export default defineConfig({
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  test: {
    environment: 'node',
    include: ['src/**/*.test.ts'],
  },
})
