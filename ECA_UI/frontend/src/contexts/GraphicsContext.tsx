import { createContext, useContext, useSyncExternalStore, useCallback, type ReactNode } from 'react'
import { ENV_CONFIG } from '../config/environmentConfig'

export interface GraphicsSettings {
  ssao: boolean
  particles: boolean
  vignette: boolean
  showGrid: boolean
  showAxes: boolean
  mtoon: boolean
}

const DEFAULTS: GraphicsSettings = {
  // The NormalPass it needs is wired up now, so the toggle finally does
  // something. Still off by default: nobody has yet seen how it treats MToon
  // outlines (see the note in ScenePostProcessing).
  ssao: false,
  particles: ENV_CONFIG.particles.enabled,
  vignette: true,
  showGrid: ENV_CONFIG.debug.showGrid,
  showAxes: ENV_CONFIG.debug.showAxes,
  mtoon: ENV_CONFIG.mtoon.enabled,
}

const STORAGE_KEY = 'vva-graphics-settings'

function load(): GraphicsSettings {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (raw) return { ...DEFAULTS, ...JSON.parse(raw) }
  } catch { /* corrupted — use defaults */ }
  return DEFAULTS
}

function save(settings: GraphicsSettings): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(settings))
  } catch { /* quota exceeded — silently ignore */ }
}

type Listener = () => void

/**
 * Module-level reactive store.
 *
 * React context does NOT cross the R3F <Canvas> boundary (it creates a separate
 * reconciler). A plain pub/sub store bypasses that limit — both the UI provider
 * (outside Canvas) and the scene provider (inside Canvas) subscribe to the same
 * object, so toggles from the settings panel instantly reflect in the 3D scene.
 */
let state: GraphicsSettings = load()
const listeners = new Set<Listener>()

export const graphicsStore = {
  get(): GraphicsSettings {
    return state
  },
  set<K extends keyof GraphicsSettings>(key: K, value: GraphicsSettings[K]): void {
    state = { ...state, [key]: value }
    save(state)
    listeners.forEach((fn) => fn())
  },
  subscribe(fn: Listener): () => void {
    listeners.add(fn)
    return () => { listeners.delete(fn) }
  },
}

interface GraphicsContextValue {
  settings: GraphicsSettings
  setSetting: <K extends keyof GraphicsSettings>(key: K, value: GraphicsSettings[K]) => void
}

const GraphicsContext = createContext<GraphicsContextValue | null>(null)

export function GraphicsProvider({ children }: { children: ReactNode }) {
  const settings = useSyncExternalStore(
    graphicsStore.subscribe,
    graphicsStore.get,
    () => DEFAULTS,
  )

  const setSetting = useCallback(<K extends keyof GraphicsSettings>(key: K, value: GraphicsSettings[K]) => {
    graphicsStore.set(key, value)
  }, [])

  return (
    <GraphicsContext.Provider value={{ settings, setSetting }}>
      {children}
    </GraphicsContext.Provider>
  )
}

export function useGraphics(): GraphicsContextValue {
  const ctx = useContext(GraphicsContext)
  if (!ctx) throw new Error('useGraphics must be used within GraphicsProvider')
  return ctx
}
