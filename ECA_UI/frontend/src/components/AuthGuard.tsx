import { useState, useEffect } from 'react'
import { isAmplifyConfigured } from '../config/amplify'
import { Outlet } from 'react-router-dom'
import { AuthContext } from '../contexts/AuthContext'

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
  const [ready, setReady] = useState(!isAmplifyConfigured)

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
          signUpAttributes={['given_name', 'family_name']}
          formFields={formFields}
          components={{
            SignUp: {
              FormFields: CustomSignUpFormFields,
            },
          }}
        >
          {({ signOut, user }: any) => (
            <AuthContext.Provider value={{ signOut, user }}>
              <div className="flex h-screen w-screen bg-background">
                <Outlet />
              </div>
            </AuthContext.Provider>
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
