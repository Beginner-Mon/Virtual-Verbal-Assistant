import { useState, useEffect, useCallback, useRef } from 'react'
import { isAmplifyConfigured } from '../config/amplify'
import { Outlet } from 'react-router-dom'
import { Amplify } from 'aws-amplify'
import { fetchAuthSession, fetchUserAttributes, signUp, signInWithRedirect } from 'aws-amplify/auth'
import { I18n } from 'aws-amplify/utils'
import { AuthContext, type FetchUserAttributesOutput } from '../contexts/AuthContext'

function getCognitoConfig() {
  /* eslint-disable @typescript-eslint/no-explicit-any */
  const cfg = Amplify.getConfig() as any
  const cognito = cfg?.Auth?.Cognito
  /* eslint-enable @typescript-eslint/no-explicit-any */

  return cognito as
    | {
        userPoolId?: string
        userPoolClientId?: string
        userPoolEndpoint?: string
      }
    | undefined
}

function clearLocalAuthStorage() {
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
    doomed.forEach((key) => storage.removeItem(key))
  }

  purge(localStorage)
  purge(sessionStorage)
}

async function globalSignOutFromCognito() {
  const session = await fetchAuthSession()
  const accessToken = session.tokens?.accessToken?.toString()
  const cognito = getCognitoConfig()

  if (!accessToken || !cognito?.userPoolId) return

  const region = cognito.userPoolId.split('_')[0]
  const endpoint = cognito.userPoolEndpoint ?? `https://cognito-idp.${region}.amazonaws.com/`

  await fetch(endpoint, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/x-amz-json-1.1',
      'X-Amz-Target': 'AWSCognitoIdentityProviderService.GlobalSignOut',
    },
    body: JSON.stringify({ AccessToken: accessToken }),
  })
}

function AuthInner({ user }: any) {
  const [attrs, setAttrs] = useState<FetchUserAttributesOutput | undefined>(undefined)
  const signingOutRef = useRef(false)

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

  const handleSignOut = useCallback(async () => {
    if (signingOutRef.current) return
    signingOutRef.current = true

    try {
      await globalSignOutFromCognito().catch((err) => {
        console.warn('[Auth] Cognito global sign-out failed:', err)
      })
      clearLocalAuthStorage()
      window.location.replace('/')
    } catch (err) {
      console.warn('[Auth] Sign-out failed, clearing local auth state:', err)
      clearLocalAuthStorage()
      window.location.replace('/')
    }
  }, [])

  return (
    <AuthContext.Provider value={{ signOut: handleSignOut, user, userAttributes: attrs }}>
      <div className="flex h-screen w-screen bg-background">
        <Outlet />
      </div>
    </AuthContext.Provider>
  )
}

function handleGoogleSignIn() {
  signInWithRedirect({ provider: 'Google', options: { prompt: 'select_account' } })
}

const GoogleSignInFooter = () => (
  <div className="flex flex-col gap-3 mt-4">
    <div className="flex items-center gap-2">
      <div className="h-px flex-1 bg-border/40" />
      <span className="text-xs text-muted-foreground">or</span>
      <div className="h-px flex-1 bg-border/40" />
    </div>
    <button
      type="button"
      onClick={handleGoogleSignIn}
      className="flex items-center justify-center gap-2 w-full px-4 py-2.5 rounded-xl border border-border/60 text-sm font-medium text-foreground hover:bg-secondary/60 transition-colors"
    >
      <svg className="w-5 h-5" viewBox="0 0 24 24">
        <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92a5.06 5.06 0 0 1-2.2 3.32v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.1z" />
        <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" />
        <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" />
        <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" />
      </svg>
      Continue with Google
    </button>
  </div>
)

export default function AuthGuard() {
  /* eslint-disable-next-line @typescript-eslint/no-explicit-any */
  const [AuthModule, setAuthModule] = useState<{
    Authenticator: any
    useAuthenticator: () => any
  } | null>(null)
  const [ready, setReady] = useState(false)

  useEffect(() => {
    if (!isAmplifyConfigured) {
      setReady(true)
      return
    }

    Promise.all([
      import('@aws-amplify/ui-react'),
      import('@aws-amplify/ui-react/styles.css'),
    ]).then(([mod]) => {
      I18n.putVocabularies({
        en: {
          'PreSignUp failed with error EMAIL_EXISTS_USE_GOOGLE.': 'This email already has a Google account. Please sign in with Google instead.',
        },
      })
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
          <span className="auth-loading-text">Loading...</span>
        </div>
      </div>
    )
  }

  if (AuthModule) {
    const { Authenticator } = AuthModule

    const CustomSignUpFormFields = () => (
      <>
        <Authenticator.SignUp.FormFields />
      </>
    )

    return (
      <div className="auth-center-container">
        <Authenticator
          socialProviders={[]}
          signUpAttributes={['preferred_username']}
          formFields={{
            signUp: {
              preferred_username: { order: 1, label: 'Full Name', placeholder: 'Enter your full name', isRequired: true },
              email: { order: 2, label: 'Email', placeholder: 'Enter your email' },
              password: { order: 3, label: 'Password', placeholder: 'Create a password' },
              confirm_password: { order: 4, label: 'Confirm Password', placeholder: 'Confirm your password' },
            },
          }}
          components={{
            SignUp: { FormFields: CustomSignUpFormFields },
            SignIn: { Footer: GoogleSignInFooter },
          }}
          services={{
            async handleSignUp(input: Parameters<typeof signUp>[0]) {
              try {
                return await signUp(input)
              } catch (err: any) {
                const msg: string = err?.message || ''
                if (msg.includes('EMAIL_EXISTS_USE_GOOGLE')) {
                  throw new Error(
                    'This email is already associated with a Google account. Please sign in with Google instead.'
                  )
                }
                throw err
              }
            },
          }}
        >
          {({ user }: any) => (
            <AuthInner user={user} />
          )}
        </Authenticator>
      </div>
    )
  }

  return (
    <AuthContext.Provider value={{}}>
      <Outlet />
    </AuthContext.Provider>
  )
}
