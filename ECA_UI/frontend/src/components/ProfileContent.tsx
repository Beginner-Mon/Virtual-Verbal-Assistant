import { useState, useRef } from 'react'
import { IdCard, User, Link2, LogOut } from 'lucide-react'
import { useAuth } from '../contexts/AuthContext'
import { Avatar, AvatarImage, AvatarFallback } from './ui/avatar'

export default function ProfileContent() {
  const { signOut, user, userAttributes } = useAuth()
  const email = user?.signInDetails?.loginId ?? userAttributes?.email ?? 'Unknown user'
  const displayName = userAttributes?.given_name
    ? `${userAttributes.given_name} ${userAttributes.family_name ?? ''}`.trim()
    : email?.split('@')[0] ?? 'User'
  const profilePicture = userAttributes?.picture
  const isGoogleLinked = !!profilePicture

  const [activeSection, setActiveSection] = useState<string>('profile')
  const sectionRefs = useRef<Record<string, HTMLDivElement | null>>({})

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
    { id: 'account-linked', icon: Link2, label: 'Account Linked' },
  ]

  return (
    <div className="flex flex-1 overflow-hidden">
      {/* Sidebar nav — hidden on mobile, shown on desktop */}
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

      {/* Content — full width on mobile, 3/4 on desktop */}
      <div className="flex-1 md:w-3/4 overflow-y-auto p-6">
        {/* Profile */}
        <div ref={setRef('profile')} className="scroll-mt-6 pb-5">
          <h3 className="text-base font-semibold text-foreground mb-5">Profile</h3>
          <div className="flex items-center gap-4">
            <Avatar className="w-14 h-14">
              <AvatarImage src={profilePicture} alt={displayName} referrerPolicy="no-referrer" />
              <AvatarFallback className="bg-primary/10" />
            </Avatar>
            <div>
              <p className="text-xs text-muted-foreground">Signed in as</p>
              <p className="text-sm font-medium text-foreground">{displayName}</p>
              <p className="text-sm text-muted-foreground">{email}</p>
            </div>
          </div>
        </div>

        <div className="border-b border-border/40" />

        {/* Display Name */}
        <div ref={setRef('display-name')} className="scroll-mt-6 py-5 space-y-4">
          <h3 className="text-base font-semibold text-foreground">Display Name</h3>
          <p className="text-sm text-muted-foreground">This is your public display name. It can be your real name or a pseudonym.</p>
          <div>
            <label className="text-xs text-muted-foreground mb-1 block">Display name</label>
            <input
              type="text"
              defaultValue={displayName}
              placeholder="Enter your display name"
              className="w-full text-sm text-foreground bg-secondary/40 rounded-lg px-3 py-2 border border-border/30 outline-none focus:border-primary/50 transition-colors"
            />
          </div>
          <button className="px-4 py-2 rounded-lg text-sm font-medium bg-primary text-primary-foreground hover:bg-primary/90 transition-colors">
            Save Changes
          </button>
        </div>

        <div className="border-b border-border/40" />

        {/* Account Linked */}
        <div ref={setRef('account-linked')} className="scroll-mt-6 py-5 space-y-4">
          <h3 className="text-base font-semibold text-foreground">Account Linked</h3>
          <p className="text-sm text-muted-foreground">Connect your accounts for seamless sign-in across platforms.</p>
          <div className="space-y-3">
            {/* Google */}
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
                  <p className="text-xs text-muted-foreground">{isGoogleLinked ? 'Connected' : 'Not connected'}</p>
                </div>
              </div>
              {isGoogleLinked ? (
                <span className="px-3 py-1.5 rounded-lg text-xs font-medium bg-primary/10 text-primary">
                  Linked
                </span>
              ) : (
                <button className="px-3 py-1.5 rounded-lg text-xs font-medium border border-border/40 text-foreground hover:bg-secondary/60 transition-colors">
                  Link
                </button>
              )}
            </div>

            {/* GitHub */}
            <div className="flex items-center justify-between p-3 rounded-xl border border-border/40 bg-card">
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 rounded-full bg-secondary/60 flex items-center justify-center">
                  <svg className="w-4 h-4" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0 0 24 12c0-6.63-5.37-12-12-12z"/>
                  </svg>
                </div>
                <div>
                  <p className="text-sm font-medium text-foreground">GitHub</p>
                  <p className="text-xs text-muted-foreground">Not connected</p>
                </div>
              </div>
              <button className="px-3 py-1.5 rounded-lg text-xs font-medium border border-border/40 text-foreground hover:bg-secondary/60 transition-colors">
                Link
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}