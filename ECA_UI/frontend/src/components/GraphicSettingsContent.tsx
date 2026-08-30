import { MonitorCog } from 'lucide-react'
import { Switch } from './ui/switch'
import { useGraphics, type GraphicsSettings } from '../hooks/useGraphics'

interface Section {
  title: string
  items: {
    key: keyof GraphicsSettings
    label: string
    description: string
  }[]
}

const SECTIONS: Section[] = [
  {
    title: 'Performance',
    items: [
      { key: 'ssao', label: 'SSAO', description: 'Screen-space ambient occlusion for depth' },
      { key: 'particles', label: 'Floating Particles', description: 'Animated background particles' },
    ],
  },
  {
    title: 'Visual',
    items: [
      { key: 'vignette', label: 'Vignette', description: 'Subtle dark edges around the viewport' },
      { key: 'mtoon', label: 'MToon Shading', description: 'Anime-style toon shading on materials' },
    ],
  },
  {
    title: 'Debug',
    items: [
      { key: 'showGrid', label: 'Show Grid', description: 'Reference grid on the ground plane' },
      { key: 'showAxes', label: 'Show Axes', description: 'Colored X/Y/Z axis indicators' },
    ],
  },
]

export default function GraphicSettingsContent() {
  const { settings, setSetting } = useGraphics()

  return (
    <div className="flex-1 overflow-y-auto p-6">
      <div className="flex items-center gap-2 mb-6">
        <MonitorCog className="w-5 h-5 text-muted-foreground" />
        <h2 className="text-lg font-semibold text-foreground">Graphic Settings</h2>
      </div>

      <div className="space-y-6">
        {SECTIONS.map((section) => (
          <div key={section.title}>
            <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-3">
              {section.title}
            </h3>
            <div className="space-y-1">
              {section.items.map((item) => (
                <div
                  key={item.key}
                  className="flex justify-between items-center px-4 py-3 rounded-xl hover:bg-secondary/40 transition-colors"
                >
                  <div>
                    <span className="text-sm font-medium text-foreground">{item.label}</span>
                    <p className="text-xs text-muted-foreground mt-0.5">{item.description}</p>
                  </div>
                  <Switch
                    checked={settings[item.key]}
                    onCheckedChange={(v) => setSetting(item.key, v)}
                  />
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

