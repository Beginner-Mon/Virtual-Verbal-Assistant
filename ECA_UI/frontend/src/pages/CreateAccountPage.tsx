import { useState } from 'react'
import { useNavigate, useLocation } from 'react-router-dom'
import { signUp, confirmSignUp, signIn, signInWithRedirect } from 'aws-amplify/auth'
import { useRedirectIfAuthenticated } from '../hooks/useRedirectIfAuthenticated'
import { Loader2 } from 'lucide-react'

export default function CreateAccountPage() {
  const navigate = useNavigate()
  const location = useLocation()
  useRedirectIfAuthenticated()
  const email = (location.state as any)?.email || ''

  const [displayName, setDisplayName] = useState('')
  const [password, setPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [step, setStep] = useState<'form' | 'verification' | 'success'>('form')
  const [code, setCode] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleSignUp = async () => {
    setError(null)
    if (password !== confirmPassword) { setError('Passwords do not match'); return }
    if (password.length < 8) { setError('Password must be at least 8 characters'); return }
    if (!displayName.trim()) { setError('Please enter your display name'); return }

    setLoading(true)
    try {
      await signUp({
        username: email,
        password,
        options: { userAttributes: { preferred_username: displayName.trim() } },
      })
      setStep('verification')
    } catch (err: any) {
      setError(err.message || 'Failed to create account')
    } finally {
      setLoading(false)
    }
  }

  const handleConfirmSignUp = async () => {
    setError(null)
    if (!code.trim()) { setError('Please enter the verification code'); return }

    setLoading(true)
    try {
      await confirmSignUp({ username: email, confirmationCode: code.trim() })
      await signIn({ username: email, password })
      setStep('success')
      setTimeout(() => navigate('/'), 1500)
    } catch (err: any) {
      setError(err.message || 'Failed to verify code')
    } finally {
      setLoading(false)
    }
  }

  const handleGoogleSignIn = () => {
    signInWithRedirect({ provider: 'Google', options: { prompt: 'select_account' } })
  }

  return (
    <div className="h-screen w-screen flex items-center justify-center bg-background">
      <div className="w-full max-w-md px-6 space-y-6">
        {step === 'form' && (
          <>
            <div>
              <h1 className="text-2xl font-semibold text-foreground">Create your account</h1>
              <div className="text-sm text-muted-foreground mt-2 bg-secondary/40 rounded-lg px-4 py-3">
                <span className="text-foreground font-medium">{email}</span>
              </div>
            </div>

            {error && (
              <div className="text-sm text-destructive bg-destructive/10 py-2 px-3 rounded-lg">{error}</div>
            )}

            <div className="space-y-4">
              <div>
                <label className="text-xs text-muted-foreground mb-1.5 block">Display Name</label>
                <input
                  type="text"
                  value={displayName}
                  onChange={e => setDisplayName(e.target.value)}
                  placeholder="Your display name"
                  className="w-full text-sm text-foreground bg-secondary/40 rounded-lg px-3 py-2.5 border border-border/30 outline-none focus:border-primary/50 transition-colors"
                />
              </div>
              <div>
                <label className="text-xs text-muted-foreground mb-1.5 block">Password</label>
                <input
                  type="password"
                  value={password}
                  onChange={e => setPassword(e.target.value)}
                  placeholder="Min. 8 characters"
                  className="w-full text-sm text-foreground bg-secondary/40 rounded-lg px-3 py-2.5 border border-border/30 outline-none focus:border-primary/50 transition-colors"
                />
              </div>
              <div>
                <label className="text-xs text-muted-foreground mb-1.5 block">Confirm password</label>
                <input
                  type="password"
                  value={confirmPassword}
                  onChange={e => setConfirmPassword(e.target.value)}
                  placeholder="Re-enter password"
                  className="w-full text-sm text-foreground bg-secondary/40 rounded-lg px-3 py-2.5 border border-border/30 outline-none focus:border-primary/50 transition-colors"
                />
              </div>

              <button
                onClick={handleSignUp}
                disabled={loading}
                className="w-full px-4 py-2.5 rounded-lg text-sm font-medium bg-primary text-primary-foreground hover:bg-primary/90 transition-colors disabled:opacity-50 flex items-center justify-center gap-2"
              >
                {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : null}
                Create Account
              </button>
            </div>

            <div className="flex items-center gap-2">
              <div className="h-px flex-1 bg-border/40" />
              <span className="text-xs text-muted-foreground">or</span>
              <div className="h-px flex-1 bg-border/40" />
            </div>

            <button
              onClick={handleGoogleSignIn}
              className="flex items-center justify-center gap-2.5 w-full px-4 py-3 rounded-xl border border-border/60 text-sm font-medium text-foreground hover:bg-secondary/60 transition-colors"
            >
              <svg className="w-5 h-5" viewBox="0 0 24 24">
                <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92a5.06 5.06 0 0 1-2.2 3.32v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.1z"/>
                <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
                <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/>
                <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
              </svg>
              Continue with Google
            </button>
          </>
        )}

        {step === 'verification' && (
          <div className="space-y-4">
            <div>
              <h1 className="text-2xl font-semibold text-foreground">Check your email</h1>
              <p className="text-sm text-muted-foreground mt-2">
                A verification code has been sent to <strong className="text-foreground">{email}</strong>.
              </p>
            </div>
            {error && (
              <div className="text-sm text-destructive bg-destructive/10 py-2 px-3 rounded-lg">{error}</div>
            )}
            <div>
              <label className="text-xs text-muted-foreground mb-1.5 block">Verification code</label>
              <input
                type="text"
                value={code}
                onChange={e => setCode(e.target.value)}
                placeholder="6-digit code"
                autoFocus
                className="w-full text-sm text-foreground bg-secondary/40 rounded-lg px-3 py-2.5 border border-border/30 outline-none focus:border-primary/50 transition-colors"
              />
            </div>
            <button
              onClick={handleConfirmSignUp}
              disabled={loading}
              className="w-full px-4 py-2.5 rounded-lg text-sm font-medium bg-primary text-primary-foreground hover:bg-primary/90 transition-colors disabled:opacity-50 flex items-center justify-center gap-2"
            >
              {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : null}
              Verify
            </button>
          </div>
        )}

        {step === 'success' && (
          <div className="text-center space-y-4">
            <h1 className="text-2xl font-semibold text-foreground">Account created</h1>
            <p className="text-sm text-muted-foreground">Redirecting...</p>
          </div>
        )}
      </div>
    </div>
  )
}
