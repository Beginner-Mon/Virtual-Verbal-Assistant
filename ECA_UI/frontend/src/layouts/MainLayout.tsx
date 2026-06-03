import { Outlet } from 'react-router-dom'
import Sidebar from '../components/Sidebar'

export default function MainLayout() {
  return (
    <main className="flex h-screen w-screen overflow-hidden bg-background">
      {/* Far Left — Sidebar */}
      <Sidebar />

      {/* Main Content Area */}
      <div className="flex-1 flex min-w-0 overflow-hidden">
        <Outlet />
      </div>
    </main>
  )
}
