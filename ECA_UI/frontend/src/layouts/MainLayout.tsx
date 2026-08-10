import { Link, Outlet } from 'react-router-dom'
import { Music2 } from 'lucide-react'
import FloatingNavBar from '../components/FloatingNavBar'
import ChatPanel from '../components/ChatPanel'
import { MotionProvider, useMotion } from '../contexts/MotionContext'
import { ChatProvider } from '../contexts/ChatContext'
import { GraphicsProvider } from '../contexts/GraphicsContext'

function AudioToggle() {
  const { isMusicPlaying, toggleMusic } = useMotion()

  return (
    <button
      onClick={toggleMusic}
      title={isMusicPlaying ? 'Pause Music' : 'Play Music'}
      className="relative w-7 h-7 flex items-center justify-center text-muted-foreground transition-all duration-200 shrink-0"
    >
      <Music2 className="w-[14px] h-[14px]" />
      {!isMusicPlaying && (
        <div className="absolute top-1/2 left-1/2 w-4 h-[1px] bg-muted-foreground -translate-x-1/2 -translate-y-1/2 -rotate-[45deg]" />
      )}
    </button>
  )
}

export default function MainLayout() {
  return (
    <MotionProvider>
      <ChatProvider>
        <GraphicsProvider>
        <main className="relative h-screen w-screen overflow-hidden bg-background">
        <div className="fixed top-5 left-5 z-[9990] flex items-center opacity-80 transition-opacity hover:opacity-100">
          <Link to="/" aria-label="Go to home" className="flex items-center">
            <img src="/eca-logo.svg" alt="" className="w-14 h-14" />
            <h1 className="text-2xl font-semibold tracking-[0.18em] text-foreground">ECA</h1>
          </Link>
          <AudioToggle />
        </div>

        {/* Full-screen content */}
        <div className="w-full h-full">
          <Outlet />
        </div>

        {/* Mobile chat — fixed at bottom */}
        <div className="block md:hidden fixed bottom-0 inset-x-0 z-40 h-[40vh] max-h-[40vh] rounded-t-2xl border-t border-border/30 shadow-2xl overflow-hidden backdrop-blur-md">
          <ChatPanel />
        </div>

        {/* Floating navigation overlay */}
        <FloatingNavBar />
      </main>
        </GraphicsProvider>
      </ChatProvider>
    </MotionProvider>
  )
}
