import { useState, useEffect, useCallback } from 'react'
import { isAmplifyConfigured } from '../config/amplify'
import { Outlet } from 'react-router-dom'
import { fetchUserAttributes } from 'aws-amplify/auth'
import { AuthContext, type FetchUserAttributesOutput } from '../contexts/AuthContext'

function AuthInner({ signOut: _signOut, user }: any) {
  const [attrs, setAttrs] = useState<FetchUserAttributesOutput | undefined>(undefined)

  useEffect(() => {
    fetchUserAttributes()
      .then((data) => {
        console.log('[Auth] User attributes:', data)
        setAttrs(data as FetchUserAttributesOutput)
      })
      .catch((err) => {
        console.warn('[Auth] Failed to fetch attributes:', err)
        setAttrs(undefined)
      })
  }, [user])

  /**
   * Full sign-out that clears the Cognito Hosted UI session cookie.
   *
   * If we don't redirect to `/logout`, the Google session remains active
   * in the browser cookies, causing auto-signin on the next attempt.
   */
  const handleSignOut = useCallback(async () => {
    // ── Step 1: Wipe local tokens ──────────────────────────────
    const purge = (storage: Storage) => {
      const doomed: string[] = []
      for (let i = 0; i < storage.length; i++) {
        const key = storage.key(i)
        if (
          key &&
          (key.startsWith('CognitoIdentityServiceProvider') ||
            key.startsWith('amplify-'))
        ) {
          doomed.push(key)
        }
      }
      doomed.forEach((k) => storage.removeItem(k))
    }
    purge(localStorage)
    purge(sessionStorage)

    // ── Step 2: Redirect to Cognito /logout ────────────────────
    try {
      const { Amplify } = await import('aws-amplify')
      /* eslint-disable @typescript-eslint/no-explicit-any */
      const cfg = Amplify.getConfig() as any
      const oauth = cfg?.Auth?.Cognito?.loginWith?.oauth
      const clientId = cfg?.Auth?.Cognito?.userPoolClientId
      /* eslint-enable @typescript-eslint/no-explicit-any */

      if (oauth?.domain && clientId) {
        // HARDCODE chính xác 100% đường link để loại bỏ mọi rủi ro từ biến môi trường
        const exactLogoutUri = 'http://localhost:5173/'

        const url =
          `https://${oauth.domain}/logout` +
          `?client_id=${clientId}` +
          `&logout_uri=${encodeURIComponent(exactLogoutUri)}`
        
        console.log('[Auth] Redirecting to clear Hosted UI session:', url)
        window.location.href = url
        return
      }
    } catch (err) {
      console.warn('[Auth] Could not build Cognito logout URL:', err)
    }

    // Fallback if no OAuth configured
    window.location.href = '/'
  }, [])

  return (
    <AuthContext.Provider value={{ signOut: handleSignOut, user, userAttributes: attrs }}>
      <div className="flex h-screen w-screen bg-background">
        <Outlet />
      </div>
    </AuthContext.Provider>
  )
}

const formFields = {
  signUp: {
    given_name: {
      order: 1,
      label: 'First Name',
      placeholder: 'Enter your first name',
      isRequired: true,
    },
    family_name: {
      order: 2,
      label: 'Last Name',
      placeholder: 'Enter your last name',
      isRequired: true,
    },
    email: {
      order: 3,
      label: 'Email',
      placeholder: 'Enter your email',
    },
    password: {
      order: 4,
      label: 'Password',
      placeholder: 'Create a password',
    },
    confirm_password: {
      order: 5,
      label: 'Confirm Password',
      placeholder: 'Confirm your password',
    },
  },
}

function ConfirmHumanCheckbox() {
  const [checked, setChecked] = useState(false)

  return (
    <div className="confirm-human-wrapper">
      <label className="confirm-human-label">
        <input
          type="checkbox"
          checked={checked}
          onChange={(e) => setChecked(e.target.checked)}
          className="confirm-human-checkbox"
          id="confirm-human-input"
          name="confirmHuman"
        />
        <span className="confirm-human-checkmark" />
        <span className="confirm-human-text">I'm not a robot</span>
      </label>
    </div>
  )
}

export default function AuthGuard() {
  /* eslint-disable-next-line @typescript-eslint/no-explicit-any */
  const [AuthModule, setAuthModule] = useState<{
    Authenticator: any
    useAuthenticator: () => any
  } | null>(null)
  const [ready, setReady] = useState(false)

  useEffect(() => {
    if (!isAmplifyConfigured) return

    Promise.all([
      import('@aws-amplify/ui-react'),
      import('@aws-amplify/ui-react/styles.css'),
    ]).then(([mod]) => {
      setAuthModule({
        Authenticator: mod.Authenticator,
        useAuthenticator: mod.useAuthenticator,
      })
      setReady(true)
    })
  }, [])

  if (!ready) {
    return (
      <div className="auth-loading-screen">
        <div className="auth-loading-inner">
          <div className="auth-spinner" />
          <span className="auth-loading-text">Loading…</span>
        </div>
      </div>
    )
  }

  if (AuthModule) {
    const { Authenticator } = AuthModule

    const CustomSignUpFormFields = () => (
      <>
        <Authenticator.SignUp.FormFields />
        <ConfirmHumanCheckbox />
      </>
    )

    return (
      <div className="auth-center-container">
        <Authenticator
          socialProviders={['google']}
          signUpAttributes={['given_name', 'family_name']}
          formFields={formFields}
          components={{
            SignUp: {
              FormFields: CustomSignUpFormFields,
            },
          }}
        >
          {({ signOut, user }: any) => (
            <AuthInner signOut={signOut} user={user} />
          )}
        </Authenticator>
      </div>
    )
  }

  // Bypass if not configured
  return (
    <AuthContext.Provider value={{}}>
      <Outlet />
    </AuthContext.Provider>
  )
}
