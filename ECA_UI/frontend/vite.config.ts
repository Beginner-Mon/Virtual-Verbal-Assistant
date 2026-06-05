import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import path from 'path'
import { spawnSync } from 'node:child_process'

function syncNpzToBvh() {
  const result = spawnSync('node', ['./scripts/sync-npz-to-bvh.mjs'], {
    cwd: __dirname,
    stdio: 'inherit',
  })

  if (result.status !== 0) {
    throw new Error(`NPZ-to-BVH sync failed with exit code ${result.status ?? 'unknown'}`)
  }
}

function motionAssetSyncPlugin() {
  return {
    name: 'motion-asset-sync',
    buildStart() {
      syncNpzToBvh()
    },
    configureServer(server: { watcher: { on: Function }, ws: { send: Function } }) {
      const rerun = () => {
        syncNpzToBvh()
        server.ws.send({ type: 'full-reload' })
      }

      server.watcher.on('add', (filePath: string) => {
        if (filePath.toLowerCase().endsWith('.npz')) rerun()
      })
      server.watcher.on('change', (filePath: string) => {
        if (filePath.toLowerCase().endsWith('.npz')) rerun()
      })
      server.watcher.on('unlink', (filePath: string) => {
        if (filePath.toLowerCase().endsWith('.npz')) rerun()
      })
    },
  }
}

// https://vite.dev/config/
export default defineConfig({
  plugins: [motionAssetSyncPlugin(), react(), tailwindcss()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  assetsInclude: ['**/*.vrm', '**/*.bvh'],
})
