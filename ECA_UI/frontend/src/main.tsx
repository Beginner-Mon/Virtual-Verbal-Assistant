import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { initializeAmplify } from './config/amplify'
import './index.css'
import App from './App.tsx'

initializeAmplify().then(() => {
  createRoot(document.getElementById('root')!).render(
    <StrictMode>
      <App />
    </StrictMode>,
  )
})
