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
  { alreadySignedIn = false }: { alreadySignedIn?: boolean } = {},
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
    prompt?: any
    authSessionOpener?: (url: string) => Promise<void>
  } = {}
  
  if (hint) {
    options.loginHint = hint
    // When we know which email the user should sign in with, force Google to
    // re-authenticate instead of silently picking from its session cookie.
    options.prompt = 'LOGIN'
  } else {
    // When there is no hint (cold login), force Google to show the account
    // picker. Without this, if the user was just kicked for a mismatch,
    // Google will silently auto-pick the active session (the wrong account)
    // again, trapping them in a loop.
    options.prompt = 'select_account'
  }

  if (isNative()) {
    // Google returns `disallowed_useragent` for OAuth inside an embedded
    // webview, so the sign-in page has to open in the system browser.
    options.authSessionOpener = openAuthSessionInSystemBrowser
  }

  if (alreadySignedIn) {
    // Not cosmetic. Amplify does:
    //     if (!input?.options?.prompt) await assertUserNotAuthenticated()
    // so linking a provider from inside an authenticated session throws
    // UserAlreadyAuthenticatedException unless *some* prompt is set.
    // 'LOGIN' is already set above when hint is present, but if somehow
    // called without a hint while already signed in, ensure the bypass.
    options.prompt = 'LOGIN'
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
