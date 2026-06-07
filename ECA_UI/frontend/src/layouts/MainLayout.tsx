import { Outlet } from 'react-router-dom'
import FloatingNavBar from '../components/FloatingNavBar'
import ChatPanel from '../components/ChatPanel'
import { MotionProvider } from '../contexts/MotionContext'
import ecaLogo from '../asset/eca-logo.svg'

export default function MainLayout() {
  return (
    <MotionProvider>
      <main className="relative h-screen w-screen overflow-hidden bg-background">
        {/* Brand logo — top left */}
        <img
          src={ecaLogo}
          alt="ECA"
          className="fixed top-5 left-5 z-[10000] h-14 w-auto opacity-80 hover:opacity-100 transition-opacity"
        />

        {/* Full-screen content */}
        <div className="w-full h-full">
          <Outlet />
        </div>

        {/* Mobile chat — fixed at bottom */}
        <div className="block md:hidden fixed bottom-0 inset-x-0 z-40 h-[40vh] max-h-[40vh] rounded-t-2xl border-t border-border/30 shadow-2xl overflow-hidden backdrop-blur-md" style={{ background: 'rgba(0,0,0,0.15)' }}>
          <ChatPanel />
        </div>

        {/* Floating navigation overlay */}
        <FloatingNavBar />
      </main>
    </MotionProvider>
  )
}
