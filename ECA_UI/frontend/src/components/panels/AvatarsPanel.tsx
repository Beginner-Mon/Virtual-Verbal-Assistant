import { useState } from 'react'
import { UserRound } from 'lucide-react'
import { ScrollArea } from '../ui/scroll-area'
import { useMotion } from '../../contexts/MotionContext'
import { incompatibilityReason, type Character } from '../../lib/characters'

const AVATAR_COLORS = [
  'from-violet-500 to-purple-600',
  'from-cyan-500 to-blue-600',
  'from-rose-500 to-pink-600',
]

/**
 * Display names used to come from a hardcoded map here, which is why `anne`
 * had none and fell through to a title-cased filename. They live on the
 * character record now, so adding a character no longer means editing this file.
 */
function AvatarCard({
  color,
  displayName,
  thumbnailUrl,
  disabledReason,
  isSelected,
  onClick,
}: {
  color: string
  displayName: string
  thumbnailUrl: string | null
  disabledReason: string | null
  isSelected: boolean
  onClick: () => void
}) {
  const [hovered, setHovered] = useState(false)
  const disabled = disabledReason !== null

  return (
    <button
      onClick={disabled ? undefined : onClick}
      disabled={disabled}
      title={disabledReason ?? displayName}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      className={`
        group relative flex flex-col rounded-lg overflow-hidden
        transition-all duration-200 w-[140px] shrink-0
        ${disabled ? 'opacity-40 cursor-not-allowed grayscale' : ''}
        ${isSelected
          ? 'ring-2 ring-primary'
          : disabled ? '' : 'hover:ring-1 hover:ring-border/60'
        }
      `}
    >
      <div className={`w-full h-[170px] bg-gradient-to-br ${color} flex items-center justify-center relative`}>
        {thumbnailUrl ? (
          <img
            src={thumbnailUrl}
            alt=""
            className="w-full h-full object-cover select-none"
            loading="lazy"
          />
        ) : (
          // No thumbnails exist yet — characters.thumbnail_url is null for
          // every row, so the original logo-on-gradient placeholder stands.
          <img
            src="/eca-logo.svg"
            alt=""
            className="w-40 h-40 opacity-30 select-none"
            style={{ filter: 'brightness(0) invert(1)' }}
          />
        )}

        <div className={`
          absolute inset-x-0 bottom-0 bg-black/60 backdrop-blur-sm py-2 px-2 text-center
          transition-all duration-200 ease-out
          ${hovered || disabled ? 'translate-y-0 opacity-100' : 'translate-y-full opacity-0'}
        `}>
          <span className="text-xs font-medium text-white truncate block">
            {displayName}
          </span>
          {disabledReason && (
            <span className="text-[10px] text-white/70 line-clamp-2">{disabledReason}</span>
          )}
        </div>
      </div>
    </button>
  )
}

export default function AvatarsPanel() {
  const {
    selectedVrmId,
    setSelectedVrmId,
    vrmOptions,
    vrmOptionsLoading,
    vrmOptionsError,
  } = useMotion()

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
        {vrmOptionsLoading && (
          <p className="text-xs text-muted-foreground text-center py-4">Loading characters…</p>
        )}

        {/* Named, not swallowed: an empty grid and a failed fetch look identical
            otherwise, and the difference decides whether anyone goes looking. */}
        {vrmOptionsError && (
          <p className="text-xs text-destructive text-center py-4 px-2">
            Could not load characters.
            <br />
            <span className="text-muted-foreground">{vrmOptionsError}</span>
          </p>
        )}

        {!vrmOptionsLoading && !vrmOptionsError && vrmOptions.length === 0 && (
          <p className="text-xs text-muted-foreground text-center py-4">No VRM avatars found</p>
        )}

        <div className="grid grid-cols-2 gap-4 pt-0.5 justify-items-center">
          {vrmOptions.map((option, index) => {
            const character: Character | undefined = option.character
            return (
              <AvatarCard
                key={option.id}
                color={AVATAR_COLORS[index % AVATAR_COLORS.length]}
                displayName={option.label}
                thumbnailUrl={character?.thumbnail_url ?? null}
                disabledReason={character ? incompatibilityReason(character) : null}
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
