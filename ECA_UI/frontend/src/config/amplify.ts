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

/** Build a v6 Amplify config from env vars, or null when they are not set. */
function configFromEnv(): AmplifyOutputs | null {
  const userPoolId = import.meta.env.VITE_USER_POOL_ID as string | undefined
  const userPoolClientId = import.meta.env.VITE_USER_POOL_WEB_CLIENT_ID as string | undefined
  const domain = import.meta.env.VITE_COGNITO_DOMAIN as string | undefined

  // The pool id and client id are the minimum. Without the hosted-UI domain,
  // email/password still works but "Continue with Google" cannot: it redirects
  // to that domain.
  if (!userPoolId || !userPoolClientId) return null

  const origin = window.location.origin

  return {
    auth: {
      user_pool_id: userPoolId,
      user_pool_client_id: userPoolClientId,
      aws_region: (import.meta.env.VITE_COGNITO_REGION as string | undefined)
        // v6 does not need the region separately — it is the prefix of the pool
        // id — but the field is part of the outputs shape.
        ?? userPoolId.split('_')[0],
      ...(domain
        ? {
            oauth: {
              domain,
              scopes: ['email', 'profile', 'openid'],
              // Must match the URLs registered on the app client EXACTLY, or
              // Cognito answers with its own error page instead of coming back.
              // `amplify/shared/origins.ts` registers the dev origin with a
              // trailing slash, and `/` + `/login` for sign-out.
              redirect_sign_in_uri: [`${origin}/`],
              redirect_sign_out_uri: [`${origin}/`, `${origin}/login`],
              response_type: 'code',
            },
          }
        : {}),
    },
    custom: {
      // The API Gateway base for the auth Lambdas (set-password, lookup-email,
      // link-google). Normally comes from amplify_outputs.custom.authApiUrl.
      ...(import.meta.env.VITE_AUTH_API_URL
        ? { authApiUrl: import.meta.env.VITE_AUTH_API_URL as string }
        : {}),
    },
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
    const outputs = fromFile ?? configFromEnv()
    if (!outputs) {
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

    Amplify.configure(outputs)
    customOutputs = outputs.custom || {}
    isAmplifyConfigured = true
    console.log(
      `[Auth] Amplify configured from ${fromFile ? 'amplify_outputs.json' : 'VITE_* env vars'}`
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
