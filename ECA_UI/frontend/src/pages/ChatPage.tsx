import ChatPanel from '../components/ChatPanel'
import CharacterViewer from '../components/CharacterViewer'
import { useAuth } from '../components/AuthGuard'

export default function ChatPage() {
  const { signOut } = useAuth()

  return (
    <>
      {/* Middle Left — Chat panel (flex: 1) */}
      <section className="flex-[1] min-w-[340px] max-w-[480px]">
        <ChatPanel />
      </section>

      {/* Right — 3D Character (flex: 2) */}
      <section className="flex-[2] relative min-w-0 border-l border-border/40">
        <CharacterViewer />

        {/* Sign-out button */}
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
    </>
  )
}
