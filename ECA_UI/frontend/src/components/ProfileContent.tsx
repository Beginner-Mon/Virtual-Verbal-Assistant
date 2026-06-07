import { useState, useRef } from 'react'
import { IdCard, User, Link2, LogOut } from 'lucide-react'
import { useAuth } from '../contexts/AuthContext'

export default function ProfileContent() {
  const { signOut, user } = useAuth()
  const email = user?.signInDetails?.loginId ?? 'Unknown user'

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
      {/* Sidebar nav — 1/4 width */}
      <nav className="w-1/4 border-r border-border/40 p-3 flex flex-col justify-between shrink-0">
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

      {/* Content — 3/4 width */}
      <div className="w-3/4 overflow-y-auto p-6 space-y-10">
        <div ref={setRef('profile')} className="scroll-mt-6 space-y-5">
          <h3 className="text-base font-semibold text-foreground border-b border-border/20 pb-2">Profile</h3>
          <div className="space-y-4">
            <div>
              <label className="text-xs text-muted-foreground mb-1 block">Username</label>
              <p className="text-sm text-foreground bg-secondary/40 rounded-lg px-3 py-2 border border-border/30">{email}</p>
            </div>
            <div>
              <label className="text-xs text-muted-foreground mb-1 block">Email</label>
              <p className="text-sm text-foreground bg-secondary/40 rounded-lg px-3 py-2 border border-border/30">{email}</p>
            </div>
          </div>
        </div>

        <div ref={setRef('display-name')} className="scroll-mt-6 space-y-5">
          <h3 className="text-base font-semibold text-foreground border-b border-border/20 pb-2">Display Name</h3>
          <div>
            <label className="text-xs text-muted-foreground mb-1 block">Display Name</label>
            <input
              type="text"
              placeholder="Enter your display name"
              className="w-full text-sm text-foreground bg-secondary/40 rounded-lg px-3 py-2 border border-border/30 outline-none focus:border-primary/50 transition-colors"
            />
          </div>
        </div>

        <div ref={setRef('account-linked')} className="scroll-mt-6 space-y-5">
          <h3 className="text-base font-semibold text-foreground border-b border-border/20 pb-2">Account Linked</h3>
          <div className="text-sm text-muted-foreground">No linked accounts yet.</div>
        </div>
      </div>
    </div>
  )
}
