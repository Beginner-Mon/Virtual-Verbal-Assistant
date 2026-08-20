import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { fetchAuthSession } from 'aws-amplify/auth'
import { customOutputs } from '../config/amplify'
import { Loader2, CheckCircle2, ArrowLeft } from 'lucide-react'
import AuthLayout from '../layouts/AuthLayout'
import { PasswordInput } from '../components/ui/password-input'

export default function SetPasswordPage() {
  const navigate = useNavigate()
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [loading, setLoading] = useState(false)
  const [success, setSuccess] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetchAuthSession().then(session => {
      const p = session.tokens?.idToken?.payload as any
      setEmail(p?.['custom:email'] || '')
    })
  }, [])

  const handleSubmit = async () => {
    setError(null)
    if (password !== confirmPassword) { setError('Passwords do not match'); return }
    if (password.length < 8) { setError('Password must be at least 8 characters'); return }

    setLoading(true)
    try {
      const apiUrl = customOutputs?.authApiUrl
      const session = await fetchAuthSession()
      const token = session.tokens?.idToken?.toString()

      if (!apiUrl || !token) throw new Error('Missing API configuration')

      const res = await fetch(`${apiUrl}api/user/set-password`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Authorization: token },
        body: JSON.stringify({ password }),
      })

      if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        throw new Error(body.message || 'Failed to set password')
      }

      setSuccess(true)
      await fetchAuthSession({ forceRefresh: true })
    } catch (err: any) {
      setError(err.message || 'Failed to set password')
    } finally {
      setLoading(false)
    }
  }

  return (
    <AuthLayout>
        {success ? (
          <div className="space-y-6 text-center">
            <div className="flex justify-center">
              <CheckCircle2 className="w-12 h-12 text-green-500" />
            </div>
            <div>
              <h1 className="text-2xl font-semibold text-foreground">Password set</h1>
              <p className="text-sm text-muted-foreground mt-2">You can now sign in with your email and password.</p>
            </div>
            <button
              onClick={() => navigate('/')}
              className="inline-flex items-center gap-2 px-4 py-2.5 rounded-lg text-sm font-medium bg-primary text-primary-foreground hover:bg-primary/90 transition-colors"
            >
              <ArrowLeft className="w-4 h-4" />
              Back to app
            </button>
          </div>
        ) : (
          <>
            <div>
              <h1 className="text-2xl font-semibold text-foreground">Create your password</h1>
              <p className="text-sm text-muted-foreground mt-2">Set a password to sign in with your email.</p>
            </div>

            <div className="text-sm text-muted-foreground bg-secondary/40 rounded-lg px-4 py-3">
              <span className="text-foreground font-medium">{email}</span>
            </div>

            {error && (
              <div className="text-sm text-destructive bg-destructive/10 py-2 px-3 rounded-lg">{error}</div>
            )}

            <div className="space-y-4">
              <div>
                <label className="text-xs text-muted-foreground mb-1.5 block">Password</label>
                <PasswordInput
                  value={password}
                  onChange={e => setPassword(e.target.value)}
                  placeholder="Min. 8 characters"
                />
              </div>
              <div>
                <label className="text-xs text-muted-foreground mb-1.5 block">Confirm password</label>
                <PasswordInput
                  value={confirmPassword}
                  onChange={e => setConfirmPassword(e.target.value)}
                  placeholder="Re-enter password"
                />
              </div>
            </div>

            <button
              onClick={handleSubmit}
              disabled={loading}
              className="w-full px-4 py-2.5 rounded-lg text-sm font-medium bg-primary text-primary-foreground hover:bg-primary/90 transition-colors disabled:opacity-50 flex items-center justify-center gap-2"
            >
              {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : null}
              {loading ? 'Setting password...' : 'Continue'}
            </button>
          </>
        )}
    </AuthLayout>
  )
}
