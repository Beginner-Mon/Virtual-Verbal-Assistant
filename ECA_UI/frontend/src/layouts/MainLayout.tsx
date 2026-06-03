import { Outlet } from 'react-router-dom'
import FloatingNavBar from '../components/FloatingNavBar'

export default function MainLayout() {
  return (
    <main className="relative h-screen w-screen overflow-hidden bg-background">
      {/* Full-screen content */}
      <div className="w-full h-full">
        <Outlet />
      </div>

      {/* Floating navigation overlay */}
      <FloatingNavBar />
    </main>
  )
}
