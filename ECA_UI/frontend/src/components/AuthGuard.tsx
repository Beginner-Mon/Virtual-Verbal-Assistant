import { useState, useEffect, useCallback, useRef } from 'react'
import { isAmplifyConfigured } from '../config/amplify'
import { Outlet } from 'react-router-dom'
import { Amplify } from 'aws-amplify'
import { fetchAuthSession, fetchUserAttributes } from 'aws-amplify/auth'
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
    if (!isAmplifyConfigured) {
      setReady(true)
      return
    }

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
