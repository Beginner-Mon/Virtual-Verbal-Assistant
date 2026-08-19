import { useState, useEffect, useCallback, useRef } from 'react'
import { Outlet, Navigate } from 'react-router-dom'
import { fetchAuthSession, fetchUserAttributes, signOut } from 'aws-amplify/auth'
import {
  AuthContext,
  type AuthUser,
  type FetchUserAttributesOutput,
} from '../contexts/AuthContext'
import { AUTH_ERROR_KEY, clearExpectedEmail, cognitoLogoutUrl, emailsMatch, peekExpectedEmail } from '../lib/googleSignIn'
import LoadingOverlay from './ui/LoadingOverlay'

function clearLocalAuthStorage() {
  const purge = (storage: Storage) => {
    const doomed: string[] = []
    for (let i = 0; i < storage.length; i++) {
      const key = storage.key(i)
      if (
        key &&
        (key.startsWith('CognitoIdentityServiceProvider') ||
          key.startsWith('amplify-') ||
          key.startsWith('com.amplify.'))
      ) {
        doomed.push(key)
      }
    }
    doomed.forEach((key) => storage.removeItem(key))
  }

  purge(localStorage)
  purge(sessionStorage)
}

function CognitoAuthGuard() {
  const [user, setUser] = useState<AuthUser | undefined>(undefined)
  const [attrs, setAttrs] = useState<FetchUserAttributesOutput | undefined>(undefined)
  const [ready, setReady] = useState(false)
  const [session, setSession] = useState<Awaited<ReturnType<typeof fetchAuthSession>> | null>(null)
  const signingOutRef = useRef(false)

  useEffect(() => {
    fetchAuthSession()
      .then(async s => {
        if (s.tokens) {
          // Whichever page started the Google redirect recorded the account it
          // had committed to. Until this was shared, only the Profile "link
          // Google" flow set it — so on the login pages a mismatch went
          // completely unchecked and you were signed in as the other account.
          // We use peek instead of a destructive take so that React StrictMode
          // double-execution doesn't consume the value on the first pass and
          // cause the second pass to bypass the mismatch check.
          const expectedEmail = peekExpectedEmail()

          let tokenSource = s
          if (expectedEmail) {
            const fresh = await fetchAuthSession({ forceRefresh: true })
            if (fresh.tokens) {
              tokenSource = fresh
            }

            const currentEmail = tokenSource.tokens?.idToken?.payload?.email as string
            if (currentEmail && !emailsMatch(currentEmail, expectedEmail)) {
              // Clearing local storage is NOT signing out. Cognito holds its own
              // managed-login session as a cookie on its own domain, so doing
              // only this left the rejected account still signed in there: the
              // next "Continue with Google" was authorised straight back into it
              // with no prompt at all, and the user had no way out. Bounce
              // through Cognito's /logout so the session actually ends.
              clearLocalAuthStorage()
              sessionStorage.setItem(AUTH_ERROR_KEY, 'email_mismatch')
              const logoutUrl = cognitoLogoutUrl('/')
              window.location.replace(logoutUrl ?? '/login?error=email_mismatch')
              return undefined
            }
            
            // Match successful, clear the expectation.
            clearExpectedEmail()
          }

          setSession(tokenSource)
          setUser({ signInDetails: { loginId: tokenSource.tokens?.idToken?.payload?.email as string || '' } })
          return fetchUserAttributes()
        }
        return undefined
      })
      .then(data => {
        if (data) setAttrs(data as FetchUserAttributesOutput)
      })
      .catch(() => {
        setSession(null)
        setUser(undefined)
      })
      .finally(() => setReady(true))
  }, [])

  const handleSignOut = useCallback(async () => {
    if (signingOutRef.current) return
    signingOutRef.current = true

    try {
      // This used to hand-roll a GlobalSignOut call with `fetch('')` — an empty
      // URL, so it POSTed to the current page, got the app's own HTML back with
      // a 200, and never threw. The revocation never happened and nothing said
      // so: signing out only cleared local storage while the Cognito session and
      // refresh tokens stayed alive server-side.
      //
      // Amplify's signOut does the real thing, and for a federated user it also
      // walks the Cognito logout endpoint (so it may navigate away before the
      // lines below run — that is fine, they are the fallback).
      await signOut({ global: true })
    } catch (err) {
      console.warn('[Auth] Global sign-out failed:', err)
    }

    clearLocalAuthStorage()
    window.location.replace('/')
  }, [])

  if (!ready) {
    return <LoadingOverlay text="Initializing Workspace..." fullScreen={true} />
  }

  // Auth gate. Bypassed in demo mode via VITE_AUTH_DISABLED=true (set in .env.local) so
  // the app runs without Cognito configured. In production (var unset) a missing session
  // redirects to /login — mirrors the backend REQUIRE_AUTH flag.
  if (import.meta.env.VITE_AUTH_DISABLED !== 'true' && !session) {
    return <Navigate to="/login" replace />
  }

  return (
    <AuthContext.Provider value={{ signOut: handleSignOut, user, userAttributes: attrs }}>
      <div className="flex h-screen w-screen bg-background">
        <Outlet />
      </div>
    </AuthContext.Provider>
  )
}

// Amplify Auth and its Cognito user pool are the application's single identity
// system. The pool, Lambda triggers, Google OAuth flow, and backend token
// verifier all depend on the Cognito subject carried in the Amplify ID token.
export default CognitoAuthGuard
