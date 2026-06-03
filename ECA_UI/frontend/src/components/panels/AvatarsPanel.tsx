import { useState } from 'react'
import { UserRound, Check } from 'lucide-react'

interface AvatarOption {
  id: string
  name: string
  description: string
  color: string
}

const AVATARS: AvatarOption[] = [
  { id: 'seele', name: 'Seele', description: 'Default character', color: 'from-violet-500 to-purple-600' },
  { id: 'avatar-2', name: 'Nova', description: 'Coming soon', color: 'from-cyan-500 to-blue-600' },
  { id: 'avatar-3', name: 'Aria', description: 'Coming soon', color: 'from-rose-500 to-pink-600' },
  { id: 'avatar-4', name: 'Echo', description: 'Coming soon', color: 'from-amber-500 to-orange-600' },
]

export default function AvatarsPanel() {
  const [selected, setSelected] = useState('seele')

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="px-4 py-3 border-b border-border/40 shrink-0">
        <h2 className="text-sm font-semibold text-foreground tracking-tight flex items-center gap-2">
          <UserRound className="w-4 h-4 text-primary" />
          Characters
        </h2>
        <p className="text-[11px] text-muted-foreground mt-0.5">Choose a 3D avatar</p>
      </div>

      {/* Avatar list */}
      <div className="flex-1 overflow-y-auto scrollbar-thin p-3 space-y-2">
        {AVATARS.map((avatar) => (
          <button
            key={avatar.id}
            onClick={() => setSelected(avatar.id)}
            className={`
              w-full flex items-center gap-3 p-3 rounded-xl text-left transition-all duration-200
              ${selected === avatar.id
                ? 'bg-primary/15 ring-1 ring-primary/40'
                : 'hover:bg-secondary/60'
              }
            `}
          >
            {/* Avatar preview */}
            <div className={`w-10 h-10 rounded-xl bg-gradient-to-br ${avatar.color} flex items-center justify-center shrink-0 shadow-lg`}>
              <span className="text-white text-sm font-bold">{avatar.name[0]}</span>
            </div>

            {/* Info */}
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium text-foreground truncate">{avatar.name}</p>
              <p className="text-[11px] text-muted-foreground">{avatar.description}</p>
            </div>

            {/* Selected check */}
            {selected === avatar.id && (
              <div className="w-5 h-5 rounded-full bg-primary flex items-center justify-center shrink-0">
                <Check className="w-3 h-3 text-primary-foreground" />
              </div>
            )}
          </button>
        ))}
      </div>
    </div>
  )
}
