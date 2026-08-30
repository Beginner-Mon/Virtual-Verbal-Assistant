import { createContext, useContext } from 'react'

export interface GraphicsSettings {
  ssao: boolean
  particles: boolean
  vignette: boolean
  showGrid: boolean
  showAxes: boolean
  mtoon: boolean
}

interface GraphicsContextValue {
  settings: GraphicsSettings
  setSetting: <K extends keyof GraphicsSettings>(key: K, value: GraphicsSettings[K]) => void
}

export const GraphicsContext = createContext<GraphicsContextValue | null>(null)

export function useGraphics(): GraphicsContextValue {
  const ctx = useContext(GraphicsContext)
  if (!ctx) throw new Error('useGraphics must be used within GraphicsProvider')
  return ctx
}
