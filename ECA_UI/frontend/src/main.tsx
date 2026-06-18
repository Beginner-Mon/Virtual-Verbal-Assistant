import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { initializeAmplify } from './config/amplify'
import { ThemeProvider } from './contexts/ThemeContext'
import './index.css'
import App from './App.tsx'

initializeAmplify().then(() => {
  createRoot(document.getElementById('root')!).render(
    <StrictMode>
      <ThemeProvider defaultTheme="light" storageKey="eca-theme">
        <App />
      </ThemeProvider>
    </StrictMode>,
  )
})
