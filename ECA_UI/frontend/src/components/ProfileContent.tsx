import { useState, useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { IdCard, User, KeyRound, Link2, LogOut } from 'lucide-react'
import { useAuth } from '../contexts/AuthContext'
import { fetchAuthSession } from 'aws-amplify/auth'
import { linkGoogleAccount, mountGoogleLinkButton } from '../lib/googleLink'
import AvatarWithLogo from './AvatarWithLogo'

interface Props {
  onClose?: () => void
}

export default function ProfileContent({ onClose }: Props) {
  const { signOut, user, userAttributes } = useAuth()
  const sectionRefs = useRef<Record<string, HTMLDivElement | null>>({})
  const [activeSection, setActiveSection] = useState<string>('profile')

  const [payload, setPayload] = useState<Record<string, unknown> | null>(null)

  useEffect(() => {
    fetchAuthSession().then(session => {
      setPayload((session.tokens?.idToken?.payload as Record<string, unknown>) ?? null)
    })
  }, [])

  const emailSub = (payload?.['custom:emailSub'] as string | undefined) || null

  /**
   * Is Google actually attached to this account?
   *
   * `custom:googleSub` alone was wrong, which is why the badge stayed on "Not
   * connected" even right after signing in with Google. That claim is injected
   * from the UserMappings row, and the row is only written on certain paths â€”
   * a linked sign-in resolves to the existing user without a PostConfirmation,
   * and an anchor created by AdminCreateUser is an admin action, not a sign-up.
   *
   * The `identities` claim comes from Cognito itself and lists every federated
   * provider on the user, so it is true by construction. The custom claim is
   * kept as a fallback for accounts written before that.
   */
  const identities = (() => {
    const raw = payload?.identities as unknown
    if (!raw) return [] as Array<{ providerName?: string }>
    // Cognito sends an array, but it arrives JSON-encoded in some token shapes.
    if (Array.isArray(raw)) return raw as Array<{ providerName?: string }>
    try {
      const parsed = JSON.parse(String(raw))
      return Array.isArray(parsed) ? parsed : []
    } catch {
      return []
    }
  })()

  const googleLinked =
    identities.some((i) => i?.providerName?.toLowerCase() === 'google') ||
    !!(payload?.['custom:googleSub'] as string | undefined) ||
    (payload?.['custom:googleLinked'] as string | undefined) === 'true'
  const displayName = (payload?.['custom:displayName'] as string | undefined) || null
  const email = (payload?.['custom:email'] as string | undefined) || (user?.signInDetails?.loginId as string | undefined) || (userAttributes?.email as string | undefined) || 'Unknown user'

  const profilePicture = userAttributes?.picture
  const navigate = useNavigate()

  const handleOpenSetPassword = () => {
    onClose?.()
    navigate('/set-password')
  }

  // â”€â”€ Linking Google â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  //
  // Deliberately NOT signInWithRedirect. Through the hosted UI a link is a
  // sign-up, so choosing the wrong account in Google's chooser made Cognito
  // create an account for that address before anything could object. Here the
  // credential goes straight to an authenticated endpoint that refuses a
  // mismatch without writing anything.
  const googleButtonRef = useRef<HTMLDivElement | null>(null)
  const [linkState, setLinkState] = useState<'idle' | 'working' | 'linked'>('idle')
  const [linkError, setLinkError] = useState<string | null>(null)

  useEffect(() => {
    const container = googleButtonRef.current
    if (!container || googleLinked || linkState === 'linked') return

    void mountGoogleLinkButton(container, {
      loginHint: email,
      onError: setLinkError,
      onCredential: async (credential) => {
        setLinkError(null)
        setLinkState('working')
        const result = await linkGoogleAccount(credential)
        if (result.ok) {
          setLinkState('linked')
          // The `identities` claim only changes on a fresh token, and it is what
          // the badge above reads.
          await fetchAuthSession({ forceRefresh: true })
          return
        }
        setLinkState('idle')
        setLinkError(result.message)
      },
    })
  }, [email, googleLinked, linkState])

  const scrollTo = (id: string) => {
    setActiveSection(id)
    sectionRefs.current[id]?.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }

  const setRef = (id: string) => (el: HTMLDivElement | null) => {
    // eslint-disable-next-line react-hooks/refs -- section anchor ref, not render state
    sectionRefs.current[id] = el
  }

  const navClass = (id: string) =>
    `flex items-center gap-2.5 px-3 py-2 rounded-lg text-sm transition-colors ${
      activeSection === id
        ? 'bg-primary/10 text-primary font-medium'
        : 'text-muted-foreground hover:text-foreground hover:bg-secondary/60'
    }`

  const navItems = [
    { id: 'profile', icon: IdCard, label: 'Profile' },
    { id: 'display-name', icon: User, label: 'Display Name' },
    { id: 'security', icon: KeyRound, label: 'Security' },
    { id: 'account-linked', icon: Link2, label: 'Account Linked' },
  ]

  return (
    <div className="flex flex-1 overflow-hidden">
      <nav className="hidden md:flex w-1/4 border-r border-border/40 p-3 flex-col justify-between shrink-0">
        <div className="space-y-1">
          {navItems.map((item) => {
            const Icon = item.icon
            return (
              <button
                key={item.id}
                onClick={() => scrollTo(item.id)}
                className={`w-full ${navClass(item.id)}`}
              >
                <Icon className="w-4 h-4" />
                {item.label}
              </button>
            )
          })}
        </div>

        <button
          onClick={() => signOut?.()}
          className="flex items-center gap-2.5 w-full px-3 py-2 rounded-lg text-sm text-destructive hover:bg-destructive/10 transition-colors font-medium"
        >
          <LogOut className="w-4 h-4" />
          Sign Out
        </button>
      </nav>

      <div className="flex-1 md:w-3/4 overflow-y-auto p-6">
        <div ref={setRef('profile')} className="scroll-mt-6 pb-5">
          <h3 className="text-base font-semibold text-foreground mb-5">Profile</h3>
          <div className="flex items-center gap-4">
            <AvatarWithLogo size="md" profilePicture={profilePicture} />
            <div>
              <p className="text-xs text-muted-foreground">Signed in as</p>
              <p className="text-sm font-medium text-foreground">{displayName ?? email.split('@')[0]}</p>
              <p className="text-sm text-muted-foreground">{email}</p>
            </div>
          </div>
        </div>

        <div className="border-b border-border/40" />

        <div ref={setRef('display-name')} className="scroll-mt-6 py-5 space-y-4">
          <h3 className="text-base font-semibold text-foreground">Display Name</h3>
          <p className="text-sm text-muted-foreground">This is your public display name. It can be your real name or a pseudonym.</p>
          <div>
            <label className="text-xs text-muted-foreground mb-1 block">Display name</label>
            <input
              type="text"
              defaultValue={displayName ?? ''}
              placeholder="Enter your display name"
              className="w-full text-sm text-foreground bg-secondary/40 rounded-lg px-3 py-2 border border-border/30 outline-none focus:border-primary/50 transition-colors"
            />
          </div>
          <button className="px-4 py-2 rounded-lg text-sm font-medium bg-primary text-primary-foreground hover:bg-primary/90 transition-colors">
            Save Changes
          </button>
        </div>

        <div className="border-b border-border/40" />

        <div ref={setRef('security')} className="scroll-mt-6 py-5 space-y-4">
          <h3 className="text-base font-semibold text-foreground">Security</h3>
          <p className="text-sm text-muted-foreground">Set a password for email sign-in.</p>
          <button
            onClick={handleOpenSetPassword}
            disabled={!!emailSub}
            className="px-4 py-2 rounded-lg text-sm font-medium bg-primary text-primary-foreground hover:bg-primary/90 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Set my Password
          </button>
        </div>

        <div className="border-b border-border/40" />

        <div ref={setRef('account-linked')} className="scroll-mt-6 py-5 space-y-4">
          <h3 className="text-base font-semibold text-foreground">Account Linked</h3>
          <p className="text-sm text-muted-foreground">Connect your accounts for seamless sign-in across platforms.</p>
          <div className="space-y-3">
            <div className="flex items-center justify-between p-3 rounded-xl border border-border/40 bg-card">
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 rounded-full bg-secondary/60 flex items-center justify-center">
                  <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none">
                    <path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92a5.06 5.06 0 0 1-2.2 3.32v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.1z" fill="#4285F4"/>
                    <path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" fill="#34A853"/>
                    <path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" fill="#FBBC05"/>
                    <path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" fill="#EA4335"/>
                  </svg>
                </div>
                <div>
                  <p className="text-sm font-medium text-foreground">Google</p>
                  <p className="text-xs text-muted-foreground">
                    {googleLinked || linkState === 'linked' ? 'Connected' : 'Not connected'}
                  </p>
                </div>
              </div>
              {googleLinked || linkState === 'linked' ? (
                <span className="px-3 py-1.5 rounded-lg text-xs font-medium bg-primary/10 text-primary">
                  Linked
                </span>
              ) : linkState === 'working' ? (
                <span className="px-3 py-1.5 text-xs text-muted-foreground">Linkingâ€¦</span>
              ) : (
                // Google renders its own button in here. Its markup is fixed by
                // Google and cannot be restyled, so it sits in a plain box
                // rather than being made to imitate the buttons around it.
                <div ref={googleButtonRef} className="shrink-0" />
              )}
            </div>

            {linkError && (
              <p className="text-xs text-destructive px-1" role="alert">
                {linkError}
              </p>
            )}

          </div>
        </div>
      </div>
    </div>
  )
}
