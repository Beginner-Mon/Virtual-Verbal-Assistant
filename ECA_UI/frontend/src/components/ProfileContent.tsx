import { useState, useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { IdCard, User, KeyRound, Link2, LogOut } from 'lucide-react'
import { useAuth } from '../contexts/AuthContext'
import { fetchAuthSession } from 'aws-amplify/auth'
import { linkGoogleAccount, mountGoogleLinkButton } from '../lib/googleLink'
import { Avatar, AvatarImage, AvatarFallback } from './ui/avatar'

interface Props {
  onClose?: () => void
}

export default function ProfileContent({ onClose }: Props) {
  const { signOut, user, userAttributes } = useAuth()
  const sectionRefs = useRef<Record<string, HTMLDivElement | null>>({})
  const [activeSection, setActiveSection] = useState<string>('profile')

  const [payload, setPayload] = useState<Record<string, any> | null>(null)

  useEffect(() => {
    fetchAuthSession().then(session => {
      setPayload(session.tokens?.idToken?.payload as any)
    })
  }, [])

  const emailSub = payload?.['custom:emailSub'] || null

  /**
   * Is Google actually attached to this account?
   *
   * `custom:googleSub` alone was wrong, which is why the badge stayed on "Not
   * connected" even right after signing in with Google. That claim is injected
   * from the UserMappings row, and the row is only written on certain paths —
   * a linked sign-in resolves to the existing user without a PostConfirmation,
   * and an anchor created by AdminCreateUser is an admin action, not a sign-up.
   *
   * The `identities` claim comes from Cognito itself and lists every federated
   * provider on the user, so it is true by construction. The custom claim is
   * kept as a fallback for accounts written before that.
   */
  const identities = (() => {
    const raw = payload?.identities
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
    !!payload?.['custom:googleSub'] ||
    payload?.['custom:googleLinked'] === 'true'
  const displayName = payload?.['custom:displayName'] || null
  const email = payload?.['custom:email'] || user?.signInDetails?.loginId || userAttributes?.email || 'Unknown user'

  const profilePicture = userAttributes?.picture
  const navigate = useNavigate()

  const handleOpenSetPassword = () => {
    onClose?.()
    navigate('/set-password')
  }

  // ── Linking Google ─────────────────────────────────────────────────────
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
            <Avatar className="w-14 h-14">
              <AvatarImage src={profilePicture} alt={displayName ?? ''} referrerPolicy="no-referrer" />
              <AvatarFallback className="bg-primary/10 text-primary">
                <svg className="w-15 h-15" viewBox="0 0 638 543" fill="none" stroke="currentColor" strokeWidth="24">
                  <g transform="translate(0,543) scale(0.1,-0.1)">
                  <path d="M2505 5096 c-105 -34 -149 -62 -187 -117 -68 -101 -85 -264 -42 -409 23 -79 89 -207 129 -251 19 -22 35 -41 35 -44 0 -3 -19 -24 -42 -45 -39 -38 -42 -39 -59 -24 -10 9 -49 25 -86 35 l-67 19 -45 -27 c-53 -30 -205 -180 -272 -268 -111 -145 -184 -295 -244 -505 -134 -465 -149 -891 -49 -1350 84 -383 294 -815 484 -994 70 -65 153 -126 174 -126 5 0 -21 32 -59 70 -37 39 -65 73 -63 76 13 13 117 -74 141 -118 36 -64 73 -82 151 -74 105 11 140 18 154 32 25 26 31 14 34 -68 l3 -83 56 53 56 52 18 -45 c9 -26 16 -47 14 -49 -2 -2 -30 -22 -61 -45 -32 -23 -58 -43 -58 -44 -1 -28 -30 -219 -34 -226 -9 -14 14 -25 25 -14 6 6 16 59 24 118 8 59 16 109 18 112 2 2 26 19 52 38 37 26 52 32 62 23 7 -5 10 -14 7 -19 -9 -15 6 -49 22 -49 13 0 18 15 16 50 0 8 5 18 11 22 6 4 12 21 13 38 3 50 84 111 105 79 5 -8 9 -33 9 -56 0 -23 7 -48 15 -57 8 -8 15 -24 15 -37 0 -18 2 -20 16 -8 8 7 28 36 42 63 l27 51 3 -83 3 -83 28 21 c16 11 32 20 36 20 4 0 4 -41 0 -92 -6 -79 -5 -90 8 -85 44 17 199 122 229 155 34 38 92 162 105 222 9 46 20 45 44 -5 37 -79 43 -69 36 64 -6 124 6 159 19 54 3 -31 13 -68 20 -82 8 -14 14 -37 14 -51 0 -32 41 -160 51 -160 4 0 11 4 14 10 3 5 11 10 17 10 19 0 -1 -63 -33 -102 -38 -48 -21 -48 50 -2 60 40 95 83 106 135 10 46 26 69 48 69 10 0 27 4 37 10 13 7 19 6 24 -6 6 -16 -19 -137 -35 -166 -11 -22 -12 -38 0 -38 9 0 65 47 129 109 l35 35 33 -140 c18 -76 41 -178 50 -227 13 -61 22 -87 32 -87 12 0 13 10 8 46 -9 68 -85 396 -97 417 -8 14 -1 30 32 75 l41 57 -6 -52 c-7 -58 7 -78 31 -44 8 12 37 47 65 79 28 31 57 71 66 89 15 31 32 36 32 10 0 -7 -25 -48 -56 -90 -31 -43 -54 -80 -51 -83 3 -3 22 9 43 27 85 72 77 72 71 -1 -3 -36 -8 -76 -13 -89 l-7 -24 43 24 c59 33 142 117 176 178 16 28 32 51 36 51 4 0 34 39 66 88 33 48 67 94 76 101 16 13 17 10 10 -40 -10 -79 -53 -163 -129 -252 -37 -43 -64 -81 -61 -85 6 -6 69 37 151 102 50 40 205 229 237 289 71 133 77 488 12 842 -28 155 -83 420 -118 565 -13 58 -41 186 -61 285 -53 261 -70 328 -110 444 -80 230 -191 440 -307 580 -145 176 -390 339 -619 414 -184 60 -412 72 -714 37 -200 -23 -262 -37 -387 -86 l-92 -36 -24 29 c-44 53 -101 158 -124 228 -19 59 -23 91 -23 200 0 161 11 187 120 276 41 34 72 65 69 70 -3 5 -6 9 -7 8 -1 0 -20 -6 -42 -13z m1703 -1033 c54 -46 165 -173 158 -180 -2 -2 -56 48 -121 112 -118 116 -139 155 -37 68z m-143 -11 c24 -21 65 -62 91 -91 44 -48 45 -53 30 -70 -15 -17 -21 -13 -110 78 -84 84 -92 96 -78 108 22 17 18 18 67 -25z m-277 -54 c95 -97 107 -115 91 -134 -10 -12 -8 -22 9 -53 l22 -39 -47 -40 c-60 -49 -108 -74 -136 -70 -14 2 -50 46 -114 141 l-93 137 31 0 c16 0 43 7 58 15 30 15 91 85 91 104 0 22 18 10 88 -61z m-306 -220 c65 -100 83 -138 66 -138 -7 0 -29 -17 -50 -37 -21 -20 -39 -35 -42 -32 -2 2 -33 51 -68 109 -35 58 -72 117 -82 133 l-18 27 53 0 c30 0 61 5 69 10 8 5 16 10 17 10 1 0 26 -37 55 -82z m-305 -156 c63 -134 65 -142 48 -142 -17 0 -129 225 -120 240 12 19 28 -3 72 -98z m-169 -34 c22 -57 43 -120 46 -141 l7 -38 -81 7 c-78 7 -82 6 -109 -20 l-28 -27 -11 28 c-15 41 -40 123 -47 161 -6 27 -4 32 12 32 27 0 92 34 133 69 19 17 36 30 37 31 1 0 19 -46 41 -102z m-419 -68 c26 -124 27 -154 5 -146 -9 3 -22 6 -29 6 -20 0 -32 26 -59 129 l-25 91 29 0 c16 0 32 5 35 10 14 23 23 5 44 -90z m-184 -163 c-8 -17 -15 -40 -15 -51 0 -23 -40 -56 -40 -34 -1 7 -9 62 -19 122 l-18 109 36 33 36 33 18 -91 c15 -82 16 -94 2 -121z m-276 -78 c1 -33 -4 -51 -18 -63 -17 -16 -18 -15 -25 30 -10 74 -7 104 12 129 l17 23 7 -37 c4 -20 7 -57 7 -82z m-229 -471 c13 -35 31 -79 40 -98 10 -19 23 -62 30 -95 6 -33 16 -78 22 -100 5 -22 13 -58 17 -80 4 -23 11 -46 15 -53 4 -7 11 -34 14 -59 13 -89 50 -53 55 53 2 54 19 70 25 23 4 -33 22 -79 43 -110 10 -14 21 -38 25 -55 7 -29 84 -195 95 -204 6 -5 13 -33 25 -107 8 -43 -19 -156 -53 -223 -18 -35 -42 -64 -66 -79 -22 -15 -34 -28 -28 -34 14 -14 92 12 120 42 33 35 91 75 97 68 12 -12 26 -313 22 -479 -4 -172 -3 -177 14 -151 26 40 74 158 99 242 26 89 33 82 50 -57 11 -84 11 -109 0 -135 -7 -18 -20 -53 -29 -78 -16 -45 -18 -46 -72 -57 -112 -23 -146 -12 -185 60 -15 27 -33 52 -41 55 -8 3 -14 12 -14 20 0 20 -58 99 -86 118 -12 8 -30 34 -39 59 -13 33 -25 48 -46 55 -17 6 -29 17 -29 28 0 10 -14 36 -30 57 -17 22 -30 46 -30 53 0 15 -35 53 -49 53 -14 0 -61 59 -61 76 0 8 14 23 31 32 17 9 50 34 72 54 23 21 49 41 59 44 10 3 18 13 18 23 0 9 5 22 11 28 7 7 6 35 -6 94 -32 167 -54 343 -46 353 10 12 9 92 -2 116 -3 8 -29 76 -58 150 -48 127 -52 142 -56 248 -3 61 -3 112 0 112 2 0 15 -28 27 -62z" />
                  </g>
                </svg>
              </AvatarFallback>
            </Avatar>
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
                <span className="px-3 py-1.5 text-xs text-muted-foreground">Linking…</span>
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
