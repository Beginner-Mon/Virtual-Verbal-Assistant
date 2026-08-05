import { signInWithRedirect } from 'aws-amplify/auth'

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
 * `loginHint` is a *hint*: it pre-fills the account but the user can still
 * switch. It narrows the accident, it does not make the mismatch check
 * redundant — `readExpectedEmail` still has to run after the redirect.
 */
export function startGoogleSignIn(expectedEmail?: string): Promise<void> {
  const hint = expectedEmail?.trim()

  if (hint) {
    write(EXPECTED_EMAIL_KEY, hint)
  } else {
    // Must clear: a leftover value from an abandoned attempt would reject a
    // perfectly valid sign-in later.
    clear(EXPECTED_EMAIL_KEY)
  }
  clear(LEGACY_EXPECTED_EMAIL_KEY)

  return signInWithRedirect({
    provider: 'Google',
    ...(hint ? { options: { loginHint: hint } } : {}),
  })
}

/** Read and consume the expected email. Returns null when none was recorded. */
export function takeExpectedEmail(): string | null {
  const value =
    sessionStorage.getItem(EXPECTED_EMAIL_KEY) ??
    localStorage.getItem(EXPECTED_EMAIL_KEY) ??
    sessionStorage.getItem(LEGACY_EXPECTED_EMAIL_KEY) ??
    localStorage.getItem(LEGACY_EXPECTED_EMAIL_KEY)

  clear(EXPECTED_EMAIL_KEY)
  clear(LEGACY_EXPECTED_EMAIL_KEY)
  return value
}

/** Case-insensitive: IdPs are not consistent about the casing they return. */
export function emailsMatch(a: string, b: string): boolean {
  return a.trim().toLowerCase() === b.trim().toLowerCase()
}
