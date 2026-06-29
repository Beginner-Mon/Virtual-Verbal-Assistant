import { useState } from 'react'
import { useNavigate, useLocation } from 'react-router-dom'
import { signIn, signInWithRedirect, resetPassword, confirmResetPassword } from 'aws-amplify/auth'
import { useRedirectIfAuthenticated } from '../hooks/useRedirectIfAuthenticated'
import { Loader2 } from 'lucide-react'

export default function EnterPasswordPage() {
  const navigate = useNavigate()
  const location = useLocation()
  useRedirectIfAuthenticated()
  const email = (location.state as any)?.email || ''

  const [password, setPassword] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const [forgotStep, setForgotStep] = useState<'idle' | 'code' | 'success'>('idle')
  const [resetCode, setResetCode] = useState('')
  const [newPassword, setNewPassword] = useState('')
  const [resetError, setResetError] = useState<string | null>(null)
  const [resetLoading, setResetLoading] = useState(false)

  const handleSignIn = async () => {
    if (!password) return
    setLoading(true)
    setError(null)

    try {
      await signIn({ username: email, password })
      navigate('/')
    } catch (err: any) {
      setError(err.message || 'Invalid email or password')
    } finally {
      setLoading(false)
    }
  }

  const handleForgotPassword = async () => {
    setResetLoading(true)
    setResetError(null)

    try {
      await resetPassword({ username: email })
      setForgotStep('code')
    } catch (err: any) {
      setResetError(err.message || 'Failed to send reset code')
    } finally {
      setResetLoading(false)
    }
  }

  const handleConfirmReset = async () => {
    if (!resetCode.trim() || !newPassword || newPassword.length < 8) {
      setResetError('Please enter a valid code and password (min. 8 characters)')
      return
    }
    setResetLoading(true)
    setResetError(null)

    try {
      await confirmResetPassword({ username: email, confirmationCode: resetCode.trim(), newPassword })
      setForgotStep('success')
    } catch (err: any) {
      setResetError(err.message || 'Failed to reset password')
    } finally {
      setResetLoading(false)
    }
  }

  const handleGoogleSignIn = () => {
    signInWithRedirect({ provider: 'Google', options: { prompt: 'select_account' } })
  }

  return (
    <div className="h-screen w-screen flex items-center justify-center bg-background">
      <div className="w-full max-w-md px-6 space-y-6">
        <div>
          <h1 className="text-2xl font-semibold text-foreground">Enter your password</h1>
          <div className="text-sm text-muted-foreground mt-2 bg-secondary/40 rounded-lg px-4 py-3">
            <span className="text-foreground font-medium">{email}</span>
          </div>
        </div>

        {error && (
          <div className="text-sm text-destructive bg-destructive/10 py-2 px-3 rounded-lg">{error}</div>
        )}

        {forgotStep === 'idle' && (
          <div className="space-y-4">
            <div>
              <label className="text-xs text-muted-foreground mb-1.5 block">Password</label>
              <input
                type="password"
                value={password}
                onChange={e => setPassword(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && handleSignIn()}
                placeholder="Enter your password"
                autoFocus
                className="w-full text-sm text-foreground bg-secondary/40 rounded-lg px-3 py-2.5 border border-border/30 outline-none focus:border-primary/50 transition-colors"
              />
            </div>

            <button
              onClick={handleSignIn}
              disabled={loading || !password}
              className="w-full px-4 py-2.5 rounded-lg text-sm font-medium bg-primary text-primary-foreground hover:bg-primary/90 transition-colors disabled:opacity-50 flex items-center justify-center gap-2"
            >
              {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : null}
              {loading ? 'Signing in...' : 'Sign in'}
            </button>

            <div className="flex items-center justify-between text-sm">
              <button
                onClick={handleForgotPassword}
                disabled={resetLoading}
                className="text-muted-foreground hover:text-foreground transition-colors"
              >
                Forgot password?
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
          </div>
        )}

        {forgotStep === 'code' && (
          <div className="space-y-4">
            <p className="text-sm text-muted-foreground">
              A verification code has been sent to <strong className="text-foreground">{email}</strong>.
            </p>
            {resetError && (
              <div className="text-sm text-destructive bg-destructive/10 py-2 px-3 rounded-lg">{resetError}</div>
            )}
            <div>
              <label className="text-xs text-muted-foreground mb-1.5 block">Verification code</label>
              <input
                type="text"
                value={resetCode}
                onChange={e => setResetCode(e.target.value)}
                placeholder="6-digit code"
                className="w-full text-sm text-foreground bg-secondary/40 rounded-lg px-3 py-2.5 border border-border/30 outline-none focus:border-primary/50 transition-colors"
              />
            </div>
            <div>
              <label className="text-xs text-muted-foreground mb-1.5 block">New password</label>
              <input
                type="password"
                value={newPassword}
                onChange={e => setNewPassword(e.target.value)}
                placeholder="Min. 8 characters"
                className="w-full text-sm text-foreground bg-secondary/40 rounded-lg px-3 py-2.5 border border-border/30 outline-none focus:border-primary/50 transition-colors"
              />
            </div>
            <button
              onClick={handleConfirmReset}
              disabled={resetLoading}
              className="w-full px-4 py-2.5 rounded-lg text-sm font-medium bg-primary text-primary-foreground hover:bg-primary/90 transition-colors disabled:opacity-50 flex items-center justify-center gap-2"
            >
              {resetLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : null}
              Reset Password
            </button>
          </div>
        )}

        {forgotStep === 'success' && (
          <div className="space-y-4 text-center">
            <p className="text-sm text-green-600">Password reset successfully.</p>
            <button
              onClick={() => setForgotStep('idle')}
              className="px-4 py-2.5 rounded-lg text-sm font-medium bg-primary text-primary-foreground hover:bg-primary/90 transition-colors"
            >
              Sign in with new password
            </button>
          </div>
        )}
      </div>
    </div>
  )
}
