import { useState, useEffect, type ReactNode } from 'react'
import { isAmplifyConfigured } from './config/amplify'
import CharacterViewer from './components/CharacterViewer'
import ChatPanel from './components/ChatPanel'
import './App.css'

/* ──────────────────────────── Auth Gate ──────────────────────────── */

interface AuthGateProps {
  children: (ctx: { signOut?: () => void; user?: { signInDetails?: { loginId?: string } } }) => ReactNode
}

export function AuthGate({ children }: AuthGateProps) {
  const [Authenticator, setAuthenticator] = useState<React.ComponentType<any> | null>(null)
  const [ready, setReady] = useState(!isAmplifyConfigured)

  useEffect(() => {
    if (!isAmplifyConfigured) return

    Promise.all([
      import('@aws-amplify/ui-react'),
      import('@aws-amplify/ui-react/styles.css'),
    ]).then(([mod]) => {
      setAuthenticator(() => mod.Authenticator)
      setReady(true)
    })
  }, [])

  if (!ready) {
    return (
      <div className="h-screen w-screen bg-background flex items-center justify-center">
        <div className="flex flex-col items-center gap-3">
          <div className="w-8 h-8 rounded-full border-2 border-primary border-t-transparent animate-spin" />
          <span className="text-sm text-muted-foreground">Loading…</span>
        </div>
      </div>
    )
  }

  if (Authenticator) {
    return (
      <div className="flex h-screen w-screen items-center justify-center bg-background">
        <Authenticator>{children as any}</Authenticator>
      </div>
    )
  }

  return <>{children({})}</>
}

/* ──────────────────────── App Content Layout ─────────────────────── */

interface AppContentProps {
  signOut?: () => void
  user?: { signInDetails?: { loginId?: string } }
}

import Sidebar from './components/Sidebar'

/* ... previous imports remain unchanged ... */

function AppContent({ signOut, user }: AppContentProps) {
  // We can pass the user prop down to the Sidebar later.
  void user;
  return (
    <main className="flex h-screen w-screen overflow-hidden bg-background">
      {/* Far Left — Sidebar */}
      <Sidebar />

      {/* Middle Left — Chat panel (flex: 1) */}
      <section className="flex-[1] min-w-[340px] max-w-[480px]">
        <ChatPanel />
      </section>

      {/* Right — 3D Character (flex: 2) */}
      <section className="flex-[2] relative min-w-0 border-l border-border/40">
        <CharacterViewer />

        {/* Temporary Sign-out Button for demo/auth testing */}
        {signOut && (
          <div className="absolute top-4 right-4 z-20">
            <button
              onClick={signOut}
              className="px-3 py-1.5 rounded-lg bg-card/60 backdrop-blur-md border border-border/30 shadow-lg text-xs font-medium text-muted-foreground hover:text-destructive transition-colors"
            >
              Sign out
            </button>
          </div>
        )}
      </section>
    </main>
  )
}

/* ────────────────────────────── App ───────────────────────────────── */

export default function App() {
  // Bypassing Auth for dev testing
  return <AppContent />
  
  /*
  return (
    <AuthGate>
      {({ signOut, user }) => <AppContent signOut={signOut} user={user} />}
    </AuthGate>
  )
  */
}
