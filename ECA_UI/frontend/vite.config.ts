import { defineConfig, loadEnv, type Plugin } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import fs from 'fs'
import path from 'path'

/**
 * Variables a production bundle cannot be correct without.
 *
 * Vite inlines env vars at BUILD time, and the client code falls back to
 * localhost when one is missing — deliberately, because throwing at module load
 * once shipped a blank page. The cost of that choice is that a missing variable
 * is invisible until a browser somewhere fails a fetch: `VITE_ASSET_BASE_URL`
 * went unset in one deployment and the only symptom was `/characters` 404ing
 * against `localhost:8000`, half a day away from anything that named the cause.
 *
 * So the check belongs at build time instead. `npm run dev` stays permissive —
 * the fallbacks exist for exactly that — and CI catches the real thing.
 */
const REQUIRED_FOR_BUILD = [
  ['VITE_API_GATEWAY_URL', 'REST API base, output RestApiUrl of VvaRestApiStack'],
  ['VITE_ASSET_BASE_URL', 'CloudFront domain for .vrm files, output AssetBaseUrl of VvaAssetStack'],
] as const

/**
 * Auth configuration has TWO sources, and requiring the environment one broke
 * the release build on 20-08.
 *
 * src/config/amplify.ts reads amplify_outputs.json first and only falls back to
 * VITE_USER_POOL_ID / VITE_AUTH_API_URL when that file is absent. On Amplify the
 * file is generated during the build, so those variables are genuinely
 * unnecessary there — and a gate that demanded them stopped a deployment that
 * would have worked.
 *
 * So the rule is not "this variable must be set", it is "auth must be
 * configurable from somewhere". Check for the file; require the variables only
 * when it is missing.
 */
const AUTH_ENV_FALLBACK = [
  ['VITE_AUTH_API_URL', 'API Gateway for the auth Lambdas; the login screen calls lookup-email before any password is typed'],
  ['VITE_USER_POOL_ID', 'Cognito pool id'],
  ['VITE_USER_POOL_WEB_CLIENT_ID', 'Cognito app client id'],
] as const

/** Variables whose presence in a production build is itself the bug. */
const FORBIDDEN_IN_BUILD = [
  [
    'VITE_AUTH_DISABLED',
    'skips the sign-in gate. The backend has no matching flag, so a build with ' +
      'this set renders the shell and 401s every request.',
  ],
] as const

function requireBuildEnv(mode: string): Plugin {
  return {
    name: 'vva-require-build-env',
    apply: 'build',
    config() {
      // loadEnv, not process.env: it applies Vite's own .env resolution order,
      // so this sees exactly what the bundle will be built with.
      const env = loadEnv(mode, process.cwd(), 'VITE_')

      const missing = [...REQUIRED_FOR_BUILD.filter(([name]) => !env[name])]

      // amplify_outputs.json is written by `ampx` during the Amplify build and is
      // gitignored, so it is present there and usually absent locally. When it
      // exists it supplies the pool and the auth API URL, and the VITE_* fallback
      // is dead weight.
      const hasAmplifyOutputs = fs.existsSync(
        path.resolve(__dirname, 'amplify_outputs.json')
      )
      if (!hasAmplifyOutputs) {
        missing.push(...AUTH_ENV_FALLBACK.filter(([name]) => !env[name]))
      }

      const forbidden = FORBIDDEN_IN_BUILD.filter(([name]) => env[name])
      if (!missing.length && !forbidden.length) return

      const lines = [
        '',
        'Build stopped: the bundle would ship misconfigured.',
        '',
        ...missing.map(([name, why]) => `  MISSING    ${name}  — ${why}`),
        ...forbidden.map(([name, why]) => `  MUST UNSET ${name}  — ${why}`),
        '',
        'Set these in ECA_UI/frontend/.env.local for a local build, or in the',
        'Amplify Console branch environment for a deployed one. `npm run dev`',
        'runs without them on purpose.',
      ]
      if (!hasAmplifyOutputs) {
        lines.push(
          '',
          'The auth variables above are only required because amplify_outputs.json',
          'is absent. Running `npx ampx sandbox` writes it and they stop being needed.',
        )
      }
      lines.push('')
      throw new Error(lines.join('\n'))
    },
  }
}

// https://vite.dev/config/
export default defineConfig(({ mode }) => ({
  plugins: [requireBuildEnv(mode), react(), tailwindcss()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  assetsInclude: ['**/*.vrm', '**/*.bvh', '**/*.fbx'],
}))
