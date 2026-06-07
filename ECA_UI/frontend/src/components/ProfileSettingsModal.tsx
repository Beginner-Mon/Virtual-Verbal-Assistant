import { useState, useEffect, useCallback, useRef } from 'react'
import { X, IdCard, User, Link2, LogOut, KeyRound } from 'lucide-react'
import { useAuth } from '../contexts/AuthContext'

interface ProfileSettingsModalProps {
  type: 'profile' | 'settings'
  onClose: () => void
}

export default function ProfileSettingsModal({ type, onClose }: ProfileSettingsModalProps) {
  const { signOut, user } = useAuth()
  const email = user?.signInDetails?.loginId ?? 'Unknown user'

  const [activeSection, setActiveSection] = useState<string>(
    type === 'profile' ? 'profile' : 'providers'
  )

  const contentRef = useRef<HTMLDivElement>(null)
  const sectionRefs = useRef<Record<string, HTMLDivElement | null>>({})

  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    },
    [onClose]
  )

  useEffect(() => {
    document.addEventListener('keydown', handleKeyDown)
    document.body.style.overflow = 'hidden'
    return () => {
      document.removeEventListener('keydown', handleKeyDown)
      document.body.style.overflow = ''
    }
  }, [handleKeyDown])

  const scrollTo = (id: string) => {
    setActiveSection(id)
    sectionRefs.current[id]?.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }

  const setRef = (id: string) => (el: HTMLDivElement | null) => {
    sectionRefs.current[id] = el
  }

  const navClass = (id: string) =>
    `flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm transition-colors whitespace-nowrap ${
      activeSection === id
        ? 'bg-primary/10 text-primary font-medium'
        : 'text-muted-foreground hover:text-foreground hover:bg-secondary/60'
    }`

  return (
    <div className="fixed inset-0 z-[10000] flex items-center justify-center">
      <div
        className="absolute inset-0 bg-black/40 backdrop-blur-sm"
        onPointerDown={(e) => {
          if (e.target === e.currentTarget) onClose()
        }}
      />

      <div className="relative w-[800px] h-[600px] bg-card rounded-2xl border border-border/50 shadow-[0_32px_80px_rgba(0,0,0,0.5)] animate-panel-in flex flex-col overflow-hidden">
        {/* Header with close button */}
        <div className="flex items-center justify-between px-6 py-3 border-b border-border/40 shrink-0">
          <h2 className="text-base font-semibold text-foreground">
            {type === 'profile' ? 'User Profile' : 'Settings'}
          </h2>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-muted-foreground hover:text-foreground hover:bg-secondary/60 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Navigation bar */}
        <div className="flex items-center gap-1 px-4 py-2 border-b border-border/40 shrink-0 overflow-x-auto">
          {type === 'profile' && (
            <>
              <button onClick={() => scrollTo('profile')} className={navClass('profile')}>
                <IdCard className="w-4 h-4" />
                Profile
              </button>
              <button onClick={() => scrollTo('display-name')} className={navClass('display-name')}>
                <User className="w-4 h-4" />
                Display Name
              </button>
              <button onClick={() => scrollTo('account-linked')} className={navClass('account-linked')}>
                <Link2 className="w-4 h-4" />
                Account Linked
              </button>
            </>
          )}
          {type === 'settings' && (
            <button onClick={() => scrollTo('providers')} className={navClass('providers')}>
              <KeyRound className="w-4 h-4" />
              Providers
            </button>
          )}
        </div>

        {/* Scrollable content */}
        <div ref={contentRef} className="flex-1 overflow-y-auto p-6 space-y-10">
          {type === 'profile' && (
            <>
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

              <div className="h-px bg-border/40" />

              <button
                onClick={() => signOut?.()}
                className="flex items-center gap-2.5 text-sm text-muted-foreground hover:text-foreground transition-colors"
              >
                <LogOut className="w-4 h-4" />
                Sign Out
              </button>
            </>
          )}

          {type === 'settings' && (
            <div ref={setRef('providers')} className="scroll-mt-6 space-y-5">
              <h3 className="text-base font-semibold text-foreground border-b border-border/20 pb-2">Providers</h3>
              <div className="text-sm text-muted-foreground">Configure your LLM API keys here.</div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
