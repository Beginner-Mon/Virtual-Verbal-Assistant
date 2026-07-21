import { useState, useEffect } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { signInWithRedirect } from 'aws-amplify/auth'
import { customOutputs } from '../config/amplify'
import { useRedirectIfAuthenticated } from '../hooks/useRedirectIfAuthenticated'
import { Loader2 } from 'lucide-react'

export default function LoginPage() {
  const navigate = useNavigate()
  const [searchParams, setSearchParams] = useSearchParams()
  useRedirectIfAuthenticated()
  const [email, setEmail] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<{ hasEmail: boolean; hasGoogle: boolean } | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [mismatchError, setMismatchError] = useState(false)

  useEffect(() => {
    if (searchParams.get('error') === 'email_mismatch') {
      setMismatchError(true)
      setSearchParams({}, { replace: true })
    }
  }, [searchParams, setSearchParams])

  const handleLookup = async () => {
    if (!email.trim()) return
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const apiUrl = customOutputs?.authApiUrl
      if (!apiUrl) throw new Error('API not configured')

      const res = await fetch(`${apiUrl}api/lookup?email=${encodeURIComponent(email.trim())}`)
      if (!res.ok) throw new Error('Failed to look up email')

      const data = await res.json()
      setResult(data)

      if (!data.hasEmail && !data.hasGoogle) {
        navigate('/create-account', { state: { email: email.trim() } })
      } else if (data.hasEmail && !data.hasGoogle) {
        navigate('/enter-password', { state: { email: email.trim() } })
      }
    } catch (err: any) {
      setError(err.message || 'Something went wrong')
    } finally {
      setLoading(false)
    }
  }

  const handleGoogleSignIn = () => {
    signInWithRedirect({ provider: 'Google', options: { prompt: 'SELECT_ACCOUNT' } })
  }

  const showGoogleOnly = result && !result.hasEmail && result.hasGoogle
  const showBoth = result && result.hasEmail && result.hasGoogle

  return (
    <div className="h-screen w-screen flex items-center justify-center bg-background">
      <div className="w-full max-w-md px-6 space-y-6">
        <div className="text-center">
          <h1 className="text-2xl font-semibold text-foreground">Welcome back</h1>
          <p className="text-sm text-muted-foreground mt-2">Enter your email to continue.</p>
        </div>

        {mismatchError && (
          <div className="text-sm text-destructive bg-destructive/10 py-2 px-3 rounded-lg">
            The selected Google account email does not match your current account. Please use the same Google account.
          </div>
        )}

        {error && (
          <div className="text-sm text-destructive bg-destructive/10 py-2 px-3 rounded-lg">{error}</div>
        )}

        <div className="space-y-3">
          <div>
            <label className="text-xs text-muted-foreground mb-1.5 block">Email</label>
            <input
              type="email"
              value={email}
              onChange={e => setEmail(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && handleLookup()}
              placeholder="you@example.com"
              autoFocus
              className="w-full text-sm text-foreground bg-secondary/40 rounded-lg px-3 py-2.5 border border-border/30 outline-none focus:border-primary/50 transition-colors"
            />
          </div>

          <button
            onClick={handleLookup}
            disabled={loading || !email.trim()}
            className="w-full px-4 py-2.5 rounded-lg text-sm font-medium bg-primary text-primary-foreground hover:bg-primary/90 transition-colors disabled:opacity-50 flex items-center justify-center gap-2"
          >
            {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : null}
            Continue
          </button>
        </div>

        {showGoogleOnly && (
          <div className="space-y-3 text-center">
            <p className="text-sm text-muted-foreground">
              <strong className="text-foreground">{email.trim()}</strong> is linked to a Google account.
            </p>
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

        {showBoth && (
          <div className="space-y-3 text-center">
            <p className="text-sm text-muted-foreground">
              <strong className="text-foreground">{email.trim()}</strong> has both email and Google sign-in.
            </p>
            <button
              onClick={() => navigate('/enter-password', { state: { email: email.trim() } })}
              className="w-full px-4 py-2.5 rounded-lg text-sm font-medium bg-primary text-primary-foreground hover:bg-primary/90 transition-colors"
            >
              Sign in with Password
            </button>
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

        {!result && (
          <>
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
      </div>
    </div>
  )
}
