import { Capacitor } from '@capacitor/core'

/**
 * Native-only auth plumbing for the Capacitor build. No-ops on the web.
 *
 * Two things the web flow gets for free and a packaged app does not:
 *
 * 1. **Where the sign-in page opens.** Google refuses OAuth inside an embedded
 *    webview (`disallowed_useragent`), so the authorize URL has to go to the
 *    system browser — Chrome Custom Tabs or SFSafariViewController. That is the
 *    real reason mobile sign-in appears "unsupported"; it has nothing to do with
 *    PKCE, which Amplify has always done (S256).
 *
 * 2. **How the callback gets back in.** The browser lands on the App Link
 *    (`https://<host>/auth/callback/?code=…`). The OS hands that URL to the app
 *    rather than the browser, and it arrives as an `appUrlOpen` event — not as a
 *    page load. Amplify's listener only ever reads `window.location.href`, so
 *    the code has to be placed there before it will do anything.
 */

export const isNative = (): boolean => Capacitor.isNativePlatform()

/**
 * Hand the authorize URL to the system browser.
 *
 * Passed to `signInWithRedirect({ options: { authSessionOpener } })`. Amplify
 * types this as `OpenAuthSession`; on the web it simply assigns
 * `window.location`, which is exactly what Google rejects on a device.
 */
export async function openAuthSessionInSystemBrowser(url: string): Promise<void> {
  const { Browser } = await import('@capacitor/browser')
  await Browser.open({ url, presentationStyle: 'popover' })
}

/**
 * Start listening for the OAuth callback App Link.
 *
 * Call once, before Amplify is configured, so no event is missed. Returns a
 * function that removes the listener.
 */
export async function listenForAuthCallback(): Promise<() => void> {
  if (!isNative()) return () => {}

  const [{ App }, { Browser }] = await Promise.all([
    import('@capacitor/app'),
    import('@capacitor/browser'),
  ])

  const handle = await App.addListener('appUrlOpen', async ({ url }) => {
    let incoming: URL
    try {
      incoming = new URL(url)
    } catch {
      return
    }

    const code = incoming.searchParams.get('code')
    const state = incoming.searchParams.get('state')
    const error = incoming.searchParams.get('error')

    // Some other deep link into the app — leave it alone.
    if (!code && !error) return

    // The custom tab sits on top of the app until dismissed. Close it before
    // touching app state so the user sees the result rather than the tab.
    await Browser.close().catch(() => {})

    // Copy the parameters onto the app's own URL. Same origin, because the app
    // is served from the App Link host — this is a rewrite, not a navigation,
    // so the SPA is not reloaded and no in-flight state is lost.
    const target = new URL(window.location.href)
    target.search = ''
    if (code) target.searchParams.set('code', code)
    if (state) target.searchParams.set('state', state)
    if (error) {
      target.searchParams.set('error', error)
      const description = incoming.searchParams.get('error_description')
      if (description) target.searchParams.set('error_description', description)
    }
    window.history.replaceState({}, '', target.toString())

    // Amplify registers its OAuth listener with the Amplify singleton and runs
    // it on configure. Re-applying the existing configuration is what replays
    // it now that the URL carries the code.
    const [{ Amplify }] = await Promise.all([import('aws-amplify')])
    Amplify.configure(Amplify.getConfig())
  })

  return () => {
    void handle.remove()
  }
}

/**
 * Keep Cognito tokens out of the webview's localStorage.
 *
 * localStorage in a packaged app is readable by anything that reaches the
 * webview and survives with no OS-level protection. Capacitor Preferences maps
 * to SharedPreferences / UserDefaults, which is not a secret store either but is
 * at least app-private. A hardware-backed keystore would be better and is the
 * obvious follow-up.
 */
export async function useNativeTokenStorage(): Promise<void> {
  if (!isNative()) return

  const [{ Preferences }, { cognitoUserPoolsTokenProvider }] = await Promise.all([
    import('@capacitor/preferences'),
    import('aws-amplify/auth/cognito'),
  ])

  cognitoUserPoolsTokenProvider.setKeyValueStorage({
    setItem: async (key, value) => {
      await Preferences.set({ key, value })
    },
    getItem: async (key) => (await Preferences.get({ key })).value,
    removeItem: async (key) => {
      await Preferences.remove({ key })
    },
    clear: async () => {
      await Preferences.clear()
    },
  })
}
