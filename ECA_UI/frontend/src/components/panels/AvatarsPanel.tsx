import { UserRound, Check } from 'lucide-react'
import { ScrollArea } from '../ui/scroll-area'
import { useMotion } from '../../contexts/MotionContext'

const AVATAR_COLORS = [
  'from-violet-500 to-purple-600',
  'from-cyan-500 to-blue-600',
  'from-rose-500 to-pink-600',
  'from-amber-500 to-orange-600',
  'from-emerald-500 to-teal-600',
  'from-fuchsia-500 to-pink-600',
]

function getAvatarMeta(label: string, index: number) {
  const name = label
    .replace(/\.vrm$/i, '')
    .replace(/[_/-]/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase())
  const initial = name[0]?.toUpperCase() ?? '?'
  const color = AVATAR_COLORS[index % AVATAR_COLORS.length]
  return { name, initial, color }
}

export default function AvatarsPanel() {
  const { selectedVrmId, setSelectedVrmId, vrmOptions } = useMotion()

  return (
    <div className="flex flex-col h-full">
      <div className="px-4 py-3 border-b border-border/40 shrink-0">
        <h2 className="text-sm font-semibold text-foreground tracking-tight flex items-center gap-2">
          <UserRound className="w-4 h-4 text-primary" />
          Characters
        </h2>
        <p className="text-[11px] text-muted-foreground mt-0.5">Choose a 3D avatar</p>
      </div>

      <ScrollArea className="flex-1 min-h-0 p-3">
        <div className="space-y-2">
          {vrmOptions.length === 0 && (
            <p className="text-xs text-muted-foreground text-center py-4">No VRM avatars found</p>
          )}
          {vrmOptions.map((option, index) => {
            const { name, initial, color } = getAvatarMeta(option.label, index)
            return (
              <button
                key={option.id}
                onClick={() => setSelectedVrmId(option.id)}
                className={`
                  w-full flex items-center gap-3 p-3 rounded-xl text-left transition-all duration-200
                  ${selectedVrmId === option.id
                    ? 'bg-primary/15 ring-1 ring-primary/40'
                    : 'hover:bg-secondary/60'
                  }
                `}
              >
                <div className={`w-10 h-10 rounded-xl bg-gradient-to-br ${color} flex items-center justify-center shrink-0 shadow-lg`}>
                  <span className="text-white text-sm font-bold">{initial}</span>
                </div>

                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-foreground truncate">{name}</p>
                  <p className="text-[11px] text-muted-foreground">VRM avatar</p>
                </div>

                {selectedVrmId === option.id && (
                  <div className="w-5 h-5 rounded-full bg-primary flex items-center justify-center shrink-0">
                    <Check className="w-3 h-3 text-primary-foreground" />
                  </div>
                )}
              </button>
            )
          })}
        </div>
      </ScrollArea>
    </div>
  )
}
