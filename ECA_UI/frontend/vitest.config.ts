import { defineConfig } from 'vitest/config'
import path from 'path'

/**
 * Deliberately NOT `mergeConfig(viteConfig, …)`.
 *
 * The tests are pure TypeScript plus three.js — none of them go through React,
 * Tailwind or asset handling — so inheriting the app's plugin chain would only
 * add startup cost. The one thing worth sharing is the `@` alias.
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
