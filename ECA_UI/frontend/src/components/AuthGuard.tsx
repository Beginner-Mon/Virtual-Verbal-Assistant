import { useState, useEffect, useCallback, useRef } from 'react'
import { Outlet, Navigate } from 'react-router-dom'
import { fetchAuthSession, fetchUserAttributes } from 'aws-amplify/auth'
import { AuthContext, type FetchUserAttributesOutput } from '../contexts/AuthContext'

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

export default function AuthGuard() {
  const [user, setUser] = useState<any>(null)
  const [attrs, setAttrs] = useState<FetchUserAttributesOutput | undefined>(undefined)
  const [ready, setReady] = useState(false)
  const [session, setSession] = useState<any>(null)
  const signingOutRef = useRef(false)

  useEffect(() => {
    fetchAuthSession()
      .then(async s => {
        if (s.tokens) {
          const linkingEmail = sessionStorage.getItem('linkingEmail') || localStorage.getItem('linkingEmail')
          sessionStorage.removeItem('linkingEmail')
          localStorage.removeItem('linkingEmail')

          let tokenSource = s
          if (linkingEmail) {
            const fresh = await fetchAuthSession({ forceRefresh: true })
            if (fresh.tokens) {
              tokenSource = fresh
            }

            const currentEmail = tokenSource.tokens?.idToken?.payload?.email as string
            if (currentEmail && currentEmail !== linkingEmail) {
              clearLocalAuthStorage()
              window.location.replace('/login?error=email_mismatch')
              return undefined
            }
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
        setUser(null)
      })
      .finally(() => setReady(true))
  }, [])

  const handleSignOut = useCallback(async () => {
    if (signingOutRef.current) return
    signingOutRef.current = true

    try {
      const s = await fetchAuthSession()
      const accessToken = s.tokens?.accessToken?.toString()
      if (accessToken) {
        await fetch('', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/x-amz-json-1.1',
            'X-Amz-Target': 'AWSCognitoIdentityProviderService.GlobalSignOut',
          },
          body: JSON.stringify({ AccessToken: accessToken }),
        })
      }
    } catch (err) {
      console.warn('[Auth] Global sign-out failed:', err)
    }

    clearLocalAuthStorage()
    window.location.replace('/')
  }, [])

  if (!ready) {
    return (
      <div className="auth-loading-screen">
        <div className="auth-loading-inner">
          <div className="auth-spinner" />
          <span className="auth-loading-text">Loading...</span>
        </div>
      </div>
    )
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
