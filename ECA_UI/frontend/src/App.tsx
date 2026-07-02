import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import AuthGuard from './components/AuthGuard'
import MainLayout from './layouts/MainLayout'
import ChatPage from './pages/ChatPage'
import LoginPage from './pages/LoginPage'
import EnterPasswordPage from './pages/EnterPasswordPage'
import CreateAccountPage from './pages/CreateAccountPage'
import SetPasswordPage from './pages/SetPasswordPage'
import './App.css'

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        {/* Public */}
        <Route path="/login" element={<LoginPage />} />
        <Route path="/enter-password" element={<EnterPasswordPage />} />
        <Route path="/create-account" element={<CreateAccountPage />} />

        {/* Protected */}
        <Route element={<AuthGuard />}>
          <Route element={<MainLayout />}>
            <Route path="/" element={<ChatPage />} />
          </Route>
          <Route path="/set-password" element={<SetPasswordPage />} />
        </Route>

        {/* Catch-all */}
        <Route path="*" element={<Navigate to="/login" replace />} />
      </Routes>
    </BrowserRouter>
  )
}
