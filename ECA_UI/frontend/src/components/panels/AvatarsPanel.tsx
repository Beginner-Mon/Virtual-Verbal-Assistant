import { useState } from 'react'
import { UserRound } from 'lucide-react'
import { ScrollArea } from '../ui/scroll-area'
import { useMotion } from '../../contexts/MotionContext'

const AVATAR_COLORS = [
  'from-violet-500 to-purple-600',
  'from-cyan-500 to-blue-600',
  'from-rose-500 to-pink-600',
]

const DISPLAY_NAMES: Record<string, string> = {
  bronya: 'SilverWing',
  'hatsune-miku': 'Hatsune Miku',
  miki: 'Miki',
}

function getAvatarMeta(label: string, index: number) {
  const name = label
    .replace(/\.vrm$/i, '')
    .replace(/[_/-]/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase())
  const key = label.replace(/\.vrm$/i, '').split('/').pop()?.toLowerCase() ?? ''
  const displayName = DISPLAY_NAMES[key] ?? name
  const initial = name[0]?.toUpperCase() ?? '?'
  const color = AVATAR_COLORS[index % AVATAR_COLORS.length]
  return { name, initial, color, displayName }
}

function AvatarCard({
  initial,
  color,
  displayName,
  isSelected,
  onClick,
}: {
  initial: string
  color: string
  displayName: string
  isSelected: boolean
  onClick: () => void
}) {
  const [hovered, setHovered] = useState(false)

  return (
    <button
      onClick={onClick}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      className={`
        group relative flex flex-col rounded-lg overflow-hidden
        transition-all duration-200 w-[140px] shrink-0
        ${isSelected
          ? 'ring-2 ring-primary'
          : 'hover:ring-1 hover:ring-border/60'
        }
      `}
    >
      <div className={`w-full h-[170px] bg-gradient-to-br ${color} flex items-center justify-center relative`}>
        <span className="text-white text-6xl font-bold opacity-30 select-none">{initial}</span>

        <div className={`
          absolute inset-x-0 bottom-0 bg-black/60 backdrop-blur-sm py-2 px-2 text-center
          transition-all duration-200 ease-out
          ${hovered ? 'translate-y-0 opacity-100' : 'translate-y-full opacity-0'}
        `}>
          <span className="text-xs font-medium text-white truncate">
            {displayName}
          </span>
        </div>
      </div>
    </button>
  )
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

      <ScrollArea className="flex-1 min-h-0 p-4">
        {vrmOptions.length === 0 && (
          <p className="text-xs text-muted-foreground text-center py-4">No VRM avatars found</p>
        )}
        <div className="grid grid-cols-2 gap-4 pt-0.5 justify-items-center">
          {vrmOptions.map((option, index) => {
            const { initial, color, displayName } = getAvatarMeta(option.label, index)
            return (
              <AvatarCard
                key={option.id}
                initial={initial}
                color={color}
                displayName={displayName}
                isSelected={selectedVrmId === option.id}
                onClick={() => setSelectedVrmId(option.id)}
              />
            )
          })}
        </div>
      </ScrollArea>
    </div>
  )
}
