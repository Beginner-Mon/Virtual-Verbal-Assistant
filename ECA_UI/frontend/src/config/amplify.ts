export let isAmplifyConfigured = false
export let customOutputs: Record<string, unknown> = {}

/**
 * Amplify configuration, from one of two places.
 *
 * 1. `amplify_outputs.json` — written by `npx ampx sandbox`. Authoritative, and
 *    what CI/deploys use.
 * 2. `VITE_*` environment variables — for pointing a local frontend at a pool
 *    that already exists.
 *
 * The second path is why this file is longer than it looks like it should be.
 * `.env.example` has documented `VITE_USER_POOL_ID` and friends for a while, but
 * nothing read them: the only configuration path was the JSON file, and that
 * file can only be produced by `ampx`, which needs AWS credentials. So "turn
 * auth on locally" was gated behind credentials even when the Cognito pool was
 * already deployed and its ids were known. Now the documented variables do what
 * they say.
 *
 * Without either, the app runs in demo mode: no sign-in, no `Authorization`
 * header, and the backend falls back to the client-supplied user id — which is
 * only safe while the backend has `REQUIRE_AUTH=false`.
 */

interface AmplifyOutputs {
  [key: string]: unknown
  custom?: Record<string, unknown>
}

/** What we hand to `Amplify.configure`, plus the `custom` block for our own use. */
interface ResolvedConfig {
  /** `ResourcesConfig` (camelCase) or a Gen2 outputs object — both accepted. */
  amplify: Record<string, unknown>
  custom: Record<string, unknown>
  source: string
}

/**
 * Build a v6 config from env vars, or null when they are not set.
 *
 * Emits the NATIVE `ResourcesConfig` shape (`Auth.Cognito.userPoolId`), not the
 * snake_case `amplify_outputs.json` shape. `Amplify.configure` accepts both, but
 * it tells them apart by the `version` field that `ampx` writes into the JSON.
 * A hand-built snake_case object without it is read as a ResourcesConfig, where
 * `auth.user_pool_id` means nothing — so configure() succeeds, `Auth` ends up
 * empty, and the first call to `signInWithRedirect` throws
 * `AuthUserPoolException: Auth UserPool not configured`. The camelCase shape has
 * no such ambiguity.
 */
function configFromEnv(): ResolvedConfig | null {
  const userPoolId = import.meta.env.VITE_USER_POOL_ID as string | undefined
  const userPoolClientId = import.meta.env.VITE_USER_POOL_WEB_CLIENT_ID as string | undefined
  const domain = import.meta.env.VITE_COGNITO_DOMAIN as string | undefined

  // The pool id and client id are the minimum. Without the hosted-UI domain,
  // email/password still works but "Continue with Google" cannot: it redirects
  // to that domain.
  if (!userPoolId || !userPoolClientId) return null

  const origin = window.location.origin

  return {
    amplify: {
      Auth: {
        Cognito: {
          userPoolId,
          userPoolClientId,
          ...(domain
            ? {
                loginWith: {
                  oauth: {
                    // Domain only — Amplify adds the scheme. A value with
                    // `https://` produces a malformed authorize URL.
                    domain: domain.replace(/^https?:\/\//, '').replace(/\/$/, ''),
                    scopes: ['email', 'profile', 'openid'],
                    // Must match the URLs registered on the app client EXACTLY,
                    // or Cognito answers with its own error page instead of
                    // coming back. `amplify/shared/origins.ts` registers the dev
                    // origin with a trailing slash, and `/` + `/login` for
                    // sign-out.
                    redirectSignIn: [`${origin}/`],
                    redirectSignOut: [`${origin}/`, `${origin}/login`],
                    responseType: 'code' as const,
                  },
                },
              }
            : {}),
        },
      },
    },
    custom: {
      // The API Gateway base for the auth Lambdas (set-password, lookup-email,
      // link-google). Normally comes from amplify_outputs.custom.authApiUrl.
      ...(import.meta.env.VITE_AUTH_API_URL
        ? { authApiUrl: import.meta.env.VITE_AUTH_API_URL as string }
        : {}),
    },
    source: `VITE_* env vars${domain ? '' : ' (no VITE_COGNITO_DOMAIN — Google sign-in disabled)'}`,
  }
}

async function outputsFromFile(): Promise<AmplifyOutputs | null> {
  // import.meta.glob resolves to {} when the file is missing, so Vite does NOT
  // fail to build without it.
  const loaders = import.meta.glob('../../amplify_outputs.json')
  const loader = loaders['../../amplify_outputs.json']
  if (!loader) return null

  const mod = (await loader()) as { default?: AmplifyOutputs }
  if (!mod.default || Object.keys(mod.default).length === 0) return null
  return mod.default
}

export async function initializeAmplify(): Promise<void> {
  try {
    const { Amplify } = await import('aws-amplify')

    const fromFile = await outputsFromFile()
    const resolved: ResolvedConfig | null = fromFile
      ? { amplify: fromFile, custom: fromFile.custom ?? {}, source: 'amplify_outputs.json' }
      : configFromEnv()

    if (!resolved) {
      throw new Error('no amplify_outputs.json and no VITE_USER_POOL_ID — demo mode')
    }

    // Both must be in place BEFORE configure: the callback listener so an
    // App Link arriving during start-up is not dropped, and the token storage so
    // Amplify never writes a session to the webview's localStorage first.
    const { isNative, listenForAuthCallback, useNativeTokenStorage } = await import(
      '../lib/nativeAuth'
    )
    if (isNative()) {
      // Not a React hook despite the name — it installs Amplify's token storage
      // adapter for Capacitor. Renaming it would be the real fix; that touches
      // the mobile branch, so it is disabled here rather than churned.
      // eslint-disable-next-line react-hooks/rules-of-hooks
      await useNativeTokenStorage()
      await listenForAuthCallback()
    }

    Amplify.configure(resolved.amplify)
    customOutputs = resolved.custom

    // Read the config BACK rather than trusting that configure() understood it.
    // It does not throw on a shape it fails to parse — it just ends up with no
    // user pool, and the failure surfaces much later as
    // `AuthUserPoolException: Auth UserPool not configured` from whichever
    // sign-in call the user happened to press first.
    const applied = Amplify.getConfig().Auth?.Cognito
    if (!applied?.userPoolId) {
      throw new Error(
        `configure() accepted the ${resolved.source} config but no user pool came ` +
          'back out of getConfig() — the shape was not understood'
      )
    }

    isAmplifyConfigured = true
    console.log(
      `[Auth] Amplify configured from ${resolved.source} — pool ${applied.userPoolId}` +
        `${applied.loginWith?.oauth ? ', hosted UI ready' : ', NO hosted UI (Google sign-in will fail)'}`
    )
  } catch (err) {
    console.warn(
      '%c⚠ Amplify not configured — running in demo mode',
      'color: #f59e0b; font-weight: bold'
    )
    console.warn('[Auth]', (err as Error).message)
    console.warn(
      '[Auth] Either run `npx ampx sandbox`, or set VITE_USER_POOL_ID + ' +
        'VITE_USER_POOL_WEB_CLIENT_ID (+ VITE_COGNITO_DOMAIN for Google sign-in) ' +
        'in ECA_UI/frontend/.env.local'
    )
    console.warn(
      '[Auth] Demo mode sends NO Authorization header. The backend must have ' +
        'REQUIRE_AUTH=false or every request will be 401.'
    )
  }
}
