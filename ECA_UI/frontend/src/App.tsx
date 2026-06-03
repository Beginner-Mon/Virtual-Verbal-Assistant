import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import AuthGuard from './components/AuthGuard'
import MainLayout from './layouts/MainLayout'
import ChatPage from './pages/ChatPage'
import SettingsPage from './pages/SettingsPage'
import './App.css'

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        {/* AuthGuard protects everything inside it */}
        <Route element={<AuthGuard />}>
          {/* MainLayout provides the Sidebar */}
          <Route element={<MainLayout />}>
            <Route path="/" element={<ChatPage />} />
            <Route path="/settings" element={<SettingsPage />} />
            
            {/* Catch-all route to redirect back to home */}
            <Route path="*" element={<Navigate to="/" replace />} />
          </Route>
        </Route>
      </Routes>
    </BrowserRouter>
  )
}
