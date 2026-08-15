import { beforeEach, describe, expect, it, vi } from 'vitest'

/**
 * These tests exist because the account chooser has now been switched off and
 * back on twice.
 *
 * `prompt: 'SELECT_ACCOUNT'` looks like a cosmetic detail — it reads as "show an
 * extra screen the user has to click through" — so it keeps getting removed on
 * the reasonable-sounding grounds that a chooser is how people pick the wrong
 * account. What that misses is that signing out does not end Google's session,
 * so with no `prompt` the next sign-in is silently answered with the account the
 * user just signed out of, and there is no screen on which to pick another one.
 *
 * Nothing in the type system says this. A test does.
 */

interface RedirectInput {
  provider: string
  options?: {
    prompt?: string
    loginHint?: string
    authSessionOpener?: unknown
  }
}

// `vi.mock` factories are hoisted above every const in the file, so the mocks
// they close over have to be hoisted too.
const { signInWithRedirect, isNative } = vi.hoisted(() => ({
  signInWithRedirect: vi.fn<(input: RedirectInput) => Promise<void>>(),
  isNative: vi.fn<() => boolean>(),
}))

vi.mock('aws-amplify/auth', () => ({ signInWithRedirect }))
vi.mock('aws-amplify', () => ({ Amplify: { getConfig: () => ({}) } }))
vi.mock('./nativeAuth', () => ({
  isNative,
  openAuthSessionInSystemBrowser: vi.fn(),
}))

/** Minimal Storage stand-in — the module reads both stores at import time. */
function memoryStorage(): Storage {
  const map = new Map<string, string>()
  return {
    get length() { return map.size },
    key: (i: number) => [...map.keys()][i] ?? null,
    getItem: (k: string) => map.get(k) ?? null,
    setItem: (k: string, v: string) => { map.set(k, String(v)) },
    removeItem: (k: string) => { map.delete(k) },
    clear: () => { map.clear() },
  } as Storage
}

beforeEach(() => {
  vi.stubGlobal('sessionStorage', memoryStorage())
  vi.stubGlobal('localStorage', memoryStorage())
  signInWithRedirect.mockClear()
  signInWithRedirect.mockResolvedValue(undefined)
  isNative.mockReturnValue(false)
})

/** Import after the globals exist, and fresh each time. */
async function load() {
  vi.resetModules()
  return import('./googleSignIn')
}

/** The single request that went out. Fails loudly if none did. */
function sentRequest(): RedirectInput {
  const call = signInWithRedirect.mock.calls[0]
  if (!call) throw new Error('signInWithRedirect was never called')
  return call[0]
}

describe('startGoogleSignIn', () => {
  it('asks Google for the account chooser when the app knows the email', async () => {
    const { startGoogleSignIn } = await load()

    await startGoogleSignIn('a@example.com')

    expect(sentRequest().options?.prompt).toBe('SELECT_ACCOUNT')
  })

  it('asks Google for the account chooser when the app knows nothing', async () => {
    // The cold "Continue with Google" straight off the login page. This is the
    // exact path that regressed: after signing out, Google still holds a
    // session, so without the chooser the user is put back into the account
    // they just left with no screen in between.
    const { startGoogleSignIn } = await load()

    await startGoogleSignIn()

    expect(sentRequest().options?.prompt).toBe('SELECT_ACCOUNT')
  })

  it('never sends a request without a prompt', async () => {
    // Guards the shape, not just the value: the previous version spread
    // `options` only when it was non-empty, so a sign-in could go out with no
    // options object at all.
    const { startGoogleSignIn } = await load()

    for (const email of [undefined, '', '  ', 'a@example.com']) {
      signInWithRedirect.mockClear()
      await startGoogleSignIn(email)
      expect(sentRequest().options?.prompt).toBe('SELECT_ACCOUNT')
    }
  })

  it('records the expected email so a wrong account can be detected', async () => {
    const { startGoogleSignIn, peekExpectedEmail } = await load()

    await startGoogleSignIn('  A@Example.com  ')

    // The chooser cannot constrain what comes back — Cognito does not forward
    // login_hint to Google — so this record is the actual defence.
    expect(peekExpectedEmail()).toBe('A@Example.com')
  })

  it('clears a stale expectation when the app does not know the email', async () => {
    const { startGoogleSignIn, peekExpectedEmail } = await load()

    await startGoogleSignIn('a@example.com')
    await startGoogleSignIn()

    // A leftover value would reject the next perfectly valid sign-in.
    expect(peekExpectedEmail()).toBeNull()
  })

  it('does not consume the expectation on read', async () => {
    // This is why `takeExpectedEmail` was split into peek + clear: React
    // StrictMode runs effects twice in development, and a read-once accessor
    // returned the email on the first pass and null on the second. The check
    // then compared the signed-in account against nothing and let a mismatch
    // through — in dev only, which is the worst place for it to differ.
    const { startGoogleSignIn, peekExpectedEmail, clearExpectedEmail } = await load()

    await startGoogleSignIn('a@example.com')

    expect(peekExpectedEmail()).toBe('a@example.com')
    expect(peekExpectedEmail()).toBe('a@example.com')

    clearExpectedEmail()
    expect(peekExpectedEmail()).toBeNull()
  })

  it('opens the system browser on native, where an embedded webview is refused', async () => {
    isNative.mockReturnValue(true)
    const { startGoogleSignIn } = await load()

    await startGoogleSignIn('a@example.com')

    expect(sentRequest().options?.authSessionOpener).toBeTypeOf('function')
    expect(sentRequest().options?.prompt).toBe('SELECT_ACCOUNT')
  })
})

describe('emailsMatch', () => {
  it('ignores case and surrounding space, which IdPs are inconsistent about', async () => {
    const { emailsMatch } = await load()

    expect(emailsMatch('A@Example.com', ' a@example.com ')).toBe(true)
    expect(emailsMatch('a@example.com', 'b@example.com')).toBe(false)
  })
})
