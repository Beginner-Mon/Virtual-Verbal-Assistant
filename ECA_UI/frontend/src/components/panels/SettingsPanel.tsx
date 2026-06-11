import { useState, useEffect } from 'react'
import { Settings, IdCard, Ellipsis, CreditCard, LogOut, CheckCircle2, Loader2 } from 'lucide-react'
import { useAuth } from '../../contexts/AuthContext'
import { customOutputs, isAmplifyConfigured } from '../../config/amplify'
import { fetchAuthSession } from 'aws-amplify/auth'

interface SettingsPanelProps {
  onOpenModal?: (type: 'profile' | 'settings') => void
}

interface AuthStatus {
  emailSub: string | null
  googleSub: string | null
  displayName: string
  email: string
}

export default function SettingsPanel({ onOpenModal }: SettingsPanelProps) {
  const { signOut } = useAuth()
  const [authStatus, setAuthStatus] = useState<AuthStatus | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [password, setPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [passwordStatus, setPasswordStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle')



  useEffect(() => {
    if (!isAmplifyConfigured) return

    const apiUrl = customOutputs?.authApiUrl
    if (!apiUrl) return

    setLoading(true)

    fetchAuthSession()
      .then(session => {
        const token = session.tokens?.idToken?.toString()
        if (!token) throw new Error('No auth token')
        return fetch(`${apiUrl}api/user/auth-status`, {
          headers: { Authorization: token },
        })
      })
      .then(res => {
        if (!res.ok) throw new Error('Failed to fetch auth status')
        return res.json()
      })
      .then(data => {
        setAuthStatus(data)
        setLoading(false)
      })
      .catch(err => {
        console.warn('[Settings] Failed to fetch auth status:', err)
        setError('Could not load account status')
        setLoading(false)
      })
  }, [])

  const handleSetPassword = async () => {
    setError(null)

    if (password !== confirmPassword) {
      setError('Passwords do not match')
      return
    }
    if (password.length < 8) {
      setError('Password must be at least 8 characters')
      return
    }

    setPasswordStatus('loading')

    const apiUrl = customOutputs?.authApiUrl
    const session = await fetchAuthSession()
    const token = session.tokens?.idToken?.toString()

    try {
      const res = await fetch(`${apiUrl}api/user/set-password`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: token,
        },
        body: JSON.stringify({ password }),
      })

      if (!res.ok) throw new Error('Failed to set password')

      setPasswordStatus('success')
      setPassword('')
      setConfirmPassword('')
      setAuthStatus(prev => prev ? { ...prev, userStatus: 'CONFIRMED' } : null)
    } catch (err) {
      console.error('[Settings] Set password failed:', err)
      setPasswordStatus('error')
      setError('Failed to set password')
    }
  }

  const items = [
    { id: 'profile' as const, icon: IdCard, label: 'Profile' },
    { id: 'settings' as const, icon: Settings, label: 'Settings' },
  ]

  return (
    <div className="p-2">
      {isAmplifyConfigured && (
        <div className="px-3 py-2 mb-2">
          <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">Account</h3>

          {loading && (
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <Loader2 className="w-3 h-3 animate-spin" />
              Loading...
            </div>
          )}

          {error && !loading && (
            <div className="text-sm text-destructive bg-destructive/10 py-1.5 px-3 rounded-lg mb-2">
              {error}
            </div>
          )}

          {authStatus && (
            <>
              {!authStatus?.emailSub && passwordStatus !== 'success' && (
                <div className="space-y-2">
                  <input
                    type="password"
                    placeholder="New password"
                    value={password}
                    onChange={e => setPassword(e.target.value)}
                    className="w-full px-3 py-1.5 rounded-lg border border-border/60 text-sm bg-transparent"
                  />
                  <input
                    type="password"
                    placeholder="Confirm password"
                    value={confirmPassword}
                    onChange={e => setConfirmPassword(e.target.value)}
                    className="w-full px-3 py-1.5 rounded-lg border border-border/60 text-sm bg-transparent"
                  />
                  <button
                    onClick={handleSetPassword}
                    disabled={passwordStatus === 'loading'}
                    className="w-full px-4 py-1.5 rounded-xl bg-primary text-primary-foreground text-sm font-medium hover:opacity-90 transition-opacity disabled:opacity-50"
                  >
                    {passwordStatus === 'loading' ? 'Setting...' : 'Set Password'}
                  </button>
                </div>
              )}

              {passwordStatus === 'success' && (
                <div className="flex items-center gap-2 text-sm text-green-600">
                  <CheckCircle2 className="w-4 h-4" />
                  Password set! You can now sign in with email and password.
                </div>
              )}

              {authStatus?.googleSub && passwordStatus !== 'success' && (
                  <div className="flex items-center gap-2 text-sm text-muted-foreground">
                    <CheckCircle2 className="w-4 h-4 text-green-600" />
                    Google account linked
                  </div>
                )}
            </>
          )}
        </div>
      )}

      <div className="h-px bg-border/40 my-1" />

      {items.map((item) => {
        const Icon = item.icon
        return (
          <button
            key={item.id}
            onClick={() => onOpenModal?.(item.id)}
            className="flex items-center gap-3 w-full px-3 py-2.5 rounded-xl text-sm text-foreground hover:bg-secondary/60 transition-colors"
          >
            <Icon className="w-4 h-4 shrink-0 text-muted-foreground" />
            <span>{item.label}</span>
          </button>
        )
      })}

      <div className="h-px bg-border/40 my-1" />

      <button
        className="flex items-center gap-3 w-full px-3 py-2.5 rounded-xl text-sm text-foreground hover:bg-secondary/60 transition-colors"
      >
        <Ellipsis className="w-4 h-4 shrink-0" />
        <span>Learn More</span>
      </button>
      <button
        className="flex items-center gap-3 w-full px-3 py-2.5 rounded-xl text-sm text-foreground hover:bg-secondary/60 transition-colors"
      >
        <CreditCard className="w-4 h-4 shrink-0" />
        <span>Billing</span>
      </button>

      <div className="h-px bg-border/40 my-1" />

      <button
        onClick={() => signOut?.()}
        className="flex items-center gap-3 w-full px-3 py-2.5 rounded-xl text-sm text-destructive hover:bg-destructive/10 transition-colors font-medium"
      >
        <LogOut className="w-4 h-4 shrink-0" />
        <span>Sign Out</span>
      </button>
    </div>
  )
}
