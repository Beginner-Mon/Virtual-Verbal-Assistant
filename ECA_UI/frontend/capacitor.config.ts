import type { CapacitorConfig } from '@capacitor/cli'

/**
 * `hostname` is the important line.
 *
 * Left at Capacitor's default the app serves itself from `capacitor://localhost`
 * (iOS) or `http://localhost` (Android), and Amplify — which chooses a redirect
 * URI by substring-matching the browser's hostname — then selects
 * `http://localhost:5173/`, the Vite dev server, on a phone. Serving from the
 * real domain makes that choice correct with no patching, and is also what lets
 * App Links / Universal Links deliver the OAuth callback into the app.
 *
 * The domain must serve /.well-known/assetlinks.json (Android) and
 * /.well-known/apple-app-site-association (iOS) — see docs/mobile-app-links.md.
 */
const MOBILE_HOST = process.env.MOBILE_APP_HOST

if (!MOBILE_HOST) {
  // Failing loudly beats shipping a build that silently authenticates against
  // the wrong origin and only breaks on a device.
  console.warn(
    '[capacitor] MOBILE_APP_HOST is not set — native OAuth will not work.\n' +
      '            Set it to the host you own, e.g. app.example.com',
  )
}

const config: CapacitorConfig = {
  appId: 'com.eca.assistant',
  appName: 'ECA',
  // Vite's default output. `npx cap sync` copies this, so build before syncing.
  webDir: 'dist',
  server: {
    ...(MOBILE_HOST ? { hostname: MOBILE_HOST } : {}),
    androidScheme: 'https',
  },
}

export default config
