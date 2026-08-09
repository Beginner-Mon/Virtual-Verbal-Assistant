import { Amplify } from 'aws-amplify'
import { signInWithRedirect } from 'aws-amplify/auth'
import { isNative, openAuthSessionInSystemBrowser } from './nativeAuth'

/**
 * The single entry point for "Continue with Google".
 *
 * Every caller used to inline `signInWithRedirect({ provider: 'Google',
 * options: { prompt: 'SELECT_ACCOUNT' } })`, which is OIDC `prompt=select_account`
 * — the code was *asking* Google to show the account chooser. When the app
 * already knows which account the user is signing into, offering them a list of
 * their other Google accounts is how they end up on the wrong one.
 *
 * Only one call site recorded the intended email, so a mismatch on the login
 * pages was never detected at all: picking a different Google account silently
 * signed you in as that other person.
 *
 * SCOPE — this is SIGN-IN only. Linking Google to an account you are already
 * signed into does not come through here any more; it uses `lib/googleLink.ts`,
 * which talks to Google directly. The reason is not tidiness: through the hosted
 * UI a link is a *sign-up*, so choosing the wrong account made Cognito create a
 * real account for that address before any check could run. An `alreadySignedIn`
 * option used to exist here for that flow and is gone — the flow is gone.
 *
 * The remaining callers are all genuinely signed out, where any Google account
 * the user picks is a legitimate answer and creating one is the correct
 * behaviour.
 */

const EXPECTED_EMAIL_KEY = 'expectedEmail'
/** Superseded by EXPECTED_EMAIL_KEY. Purged so a stale value from an older
 *  build cannot fire a spurious mismatch. */
const LEGACY_EXPECTED_EMAIL_KEY = 'linkingEmail'

function write(key: string, value: string) {
  // Both stores on purpose: sessionStorage survives the redirect in the same
  // tab, localStorage covers providers that hand control back in a new one.
  sessionStorage.setItem(key, value)
  localStorage.setItem(key, value)
}

function clear(key: string) {
  sessionStorage.removeItem(key)
  localStorage.removeItem(key)
}

/**
 * Start the Google redirect.
 *
 * @param expectedEmail The account the UI has committed to — from the email
 *   lookup, or the signed-in user when linking from Profile. Pass `undefined`
 *   only when the app genuinely does not know yet (a cold "sign in with Google"),
 *   in which case any account is a legitimate answer.
 *
 * ⚠️ `loginHint` does NOT reach Google. The Cognito docs are explicit:
 *
 *   "When your authorization request invokes a redirect to OIDC IdPs, Amazon
 *    Cognito adds a login_hint parameter to the request to that third-party
 *    authorizer. You can't forward login hints to SAML, Apple, Login With
 *    Amazon, Google, or Facebook (Meta) IdPs."
 *   — docs.aws.amazon.com/cognito/latest/developerguide/authorization-endpoint.html
 *
 * It is kept because it costs nothing and does work for a generic OIDC provider,
 * should one be added. But for Google there is simply no way to pre-select the
 * account through Cognito, so the account the user lands on cannot be
 * constrained — it can only be *checked afterwards*. That makes
 * `takeExpectedEmail` the primary defence here, not a backstop.
 *
 * `prompt` is different: Cognito forwards every value except `none` to the IdP.
 * That is why passing SELECT_ACCOUNT literally asked Google for the picker.
 */
export function startGoogleSignIn(
  expectedEmail?: string,
  { forceAccountChoice = false }: { forceAccountChoice?: boolean } = {},
): Promise<void> {
  const hint = expectedEmail?.trim()

  if (hint) {
    write(EXPECTED_EMAIL_KEY, hint)
  } else {
    // Must clear: a leftover value from an abandoned attempt would reject a
    // perfectly valid sign-in later.
    clear(EXPECTED_EMAIL_KEY)
  }
  clear(LEGACY_EXPECTED_EMAIL_KEY)

  const options: {
    loginHint?: string
    prompt?: 'LOGIN' | 'SELECT_ACCOUNT'
    authSessionOpener?: (url: string) => Promise<void>
  } = {}
  if (hint) options.loginHint = hint

  if (forceAccountChoice) {
    // The one case where the account picker is the right answer.
    //
    // Ending the Cognito session is not enough on a retry after a rejected
    // sign-in: per the Cognito docs, "the logout endpoint doesn't sign users out
    // of OIDC or social identity providers". Google still holds its own session,
    // so a request carrying no `prompt` lets it silently re-select the very
    // account we just rejected — the bug reappears with nothing on screen to
    // explain it. Cognito forwards every `prompt` value except `none` to the
    // IdP, so this is what makes Google ask again.
    options.prompt = 'SELECT_ACCOUNT'
  }

  if (isNative()) {
    // Google returns `disallowed_useragent` for OAuth inside an embedded
    // webview, so the sign-in page has to open in the system browser.
    options.authSessionOpener = openAuthSessionInSystemBrowser
  }

  return signInWithRedirect({
    provider: 'Google',
    ...(Object.keys(options).length > 0 ? { options } : {}),
  })
}

/** Read the expected email without consuming it (safe for StrictMode double-execution). */
export function peekExpectedEmail(): string | null {
  return sessionStorage.getItem(EXPECTED_EMAIL_KEY) ??
    localStorage.getItem(EXPECTED_EMAIL_KEY) ??
    sessionStorage.getItem(LEGACY_EXPECTED_EMAIL_KEY) ??
    localStorage.getItem(LEGACY_EXPECTED_EMAIL_KEY)
}

/** Clear the expected email once it has been successfully processed or explicitly rejected. */
export function clearExpectedEmail(): void {
  clear(EXPECTED_EMAIL_KEY)
  clear(LEGACY_EXPECTED_EMAIL_KEY)
}

/** Case-insensitive: IdPs are not consistent about the casing they return. */
export function emailsMatch(a: string, b: string): boolean {
  return a.trim().toLowerCase() === b.trim().toLowerCase()
}

/** Where the login page reads a message left behind by a rejected sign-in. */
export const AUTH_ERROR_KEY = 'authError'

/**
 * URL that genuinely ends the Cognito session.
 *
 * Clearing localStorage does NOT sign a user out. Cognito keeps its own
 * managed-login session as a cookie on its own domain, so after a rejected
 * sign-in the next trip to /oauth2/authorize is authorised straight back into
 * the same account — no account picker, no Google round-trip, no way for the
 * user to escape it. That is the "it just logs me in as the wrong account
 * again" bug; the surviving cookie is Cognito's, not Google's.
 *
 * Returns null when OAuth is not configured (demo mode), leaving the caller to
 * fall back to a local-only clear.
 */
export function cognitoLogoutUrl(returnPath = '/'): string | null {
  const cognito = Amplify.getConfig().Auth?.Cognito
  const domain = cognito?.loginWith?.oauth?.domain
  const clientId = cognito?.userPoolClientId
  if (!domain || !clientId) return null

  // Must match a registered logout URL EXACTLY, or Cognito shows its own error
  // page instead of coming back. '/' is the one every pool has had since the
  // beginning, so this works against the currently deployed stack too — the
  // app then falls through AuthGuard to /login on its own, where the reason is
  // read back out of sessionStorage.
  const logoutUri = `${window.location.origin}${returnPath}`
  return (
    `https://${domain}/logout` +
    `?client_id=${encodeURIComponent(clientId)}` +
    `&logout_uri=${encodeURIComponent(logoutUri)}`
  )
}
