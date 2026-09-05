import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { initializeAmplify } from './config/amplify'
import { ThemeProvider } from './contexts/ThemeContext'
import { wakeCrudApi } from './lib/api'
// Before the first render: components read translations during their initial
// render, and i18next.init is synchronous with bundled resources, so importing
// it here means there is never a frame rendered against an empty catalogue.
import './i18n'
import './index.css'
import App from './App.tsx'

// Before the first render, not after: the CRUD Lambda and Neon's compute both
// start cold, and this spends that wait while the user is still looking at the
// shell rather than on their first click. Fire-and-forget by design — it needs
// no token and its failure is not the app's problem.
wakeCrudApi()

initializeAmplify().then(() => {
  createRoot(document.getElementById('root')!).render(
    <StrictMode>
      <ThemeProvider defaultTheme="light" storageKey="eca-theme">
        <App />
      </ThemeProvider>
    </StrictMode>,
  )
})
