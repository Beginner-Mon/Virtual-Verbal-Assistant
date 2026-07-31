import { Link, Outlet } from 'react-router-dom'
import FloatingNavBar from '../components/FloatingNavBar'
import ChatPanel from '../components/ChatPanel'
import { MotionProvider } from '../contexts/MotionContext'
import { ChatProvider } from '../contexts/ChatContext'

export default function MainLayout() {
  return (
    <MotionProvider>
      <ChatProvider>
        <main className="relative h-screen w-screen overflow-hidden bg-background">
        <Link
          to="/"
          aria-label="Go to home"
          className="fixed top-5 left-5 z-[9990] opacity-80 transition-opacity hover:opacity-100"
        >
          <h1 className="text-2xl font-semibold tracking-[0.18em] text-foreground">ECA</h1>
        </Link>

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
      </ChatProvider>
    </MotionProvider>
  )
}
