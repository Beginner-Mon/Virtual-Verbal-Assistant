import { useEffect, useCallback } from 'react'
import { X, Check } from 'lucide-react'
import AvatarWithLogo from './AvatarWithLogo'
import { AVATAR_BG_OPTIONS, type AvatarBgId } from '@/lib/avatarPalette'

interface Props {
  currentColorId: AvatarBgId
  onSelect: (id: AvatarBgId) => void
  onClose: () => void
}

export default function AvatarPickerModal({ currentColorId, onSelect, onClose }: Props) {
  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    },
    [onClose]
  )

  useEffect(() => {
    document.addEventListener('keydown', handleKeyDown)
    return () => document.removeEventListener('keydown', handleKeyDown)
  }, [handleKeyDown])

  const currentBg = AVATAR_BG_OPTIONS.find((o) => o.id === currentColorId) ?? AVATAR_BG_OPTIONS[0]

  return (
    <div className="fixed inset-0 z-[10001] flex items-center justify-center p-4">
      {/* overlay */}
      <div className="absolute inset-0 bg-black/50 backdrop-blur-sm" onClick={onClose} />
      {/* card */}
      <div className="relative w-full max-w-sm rounded-2xl bg-card border border-border/50 shadow-[0_16px_64px_rgba(0,0,0,0.4)] flex flex-col overflow-hidden animate-panel-in">
        {/* header */}
        <div className="flex items-center gap-2 px-4 py-3 border-b border-border/40 shrink-0">
          <h3 className="text-sm font-semibold text-foreground flex-1">Choose avatar background</h3>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-muted-foreground hover:text-foreground hover:bg-secondary/60 transition-colors"
            aria-label="Close"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* preview on top */}
        <div className="flex flex-col items-center justify-center py-6 gap-3 border-b border-border/30">
          <AvatarWithLogo size="lg" bgClassName={currentBg.className} logoClassName={currentBg.logoClassName} />
          <p className="text-xs text-muted-foreground">Preview</p>
        </div>

        {/* options below */}
        <div className="p-4">
          <p className="text-xs font-medium text-muted-foreground mb-3">Background color</p>
          <div className="grid grid-cols-4 gap-3">
            {AVATAR_BG_OPTIONS.map((opt) => {
              const isSelected = opt.id === currentColorId
              const checkColor = opt.id === 'slate' ? 'text-muted-foreground' : opt.id === 'amber' ? 'text-amber-950' : 'text-white'
              return (
                <button
                  key={opt.id}
                  onClick={() => onSelect(opt.id)}
                  aria-label={opt.id}
                  title={opt.id}
                  className={`
                    relative w-12 h-12 rounded-full flex items-center justify-center cursor-pointer
                    transition-all duration-150
                    ${opt.className}
                    ${isSelected ? 'ring-2 ring-primary ring-offset-2 ring-offset-card' : 'hover:scale-105'}
                  `}
                >
                  {isSelected && <Check className={`w-5 h-5 drop-shadow ${checkColor}`} strokeWidth={3} />}
                </button>
              )
            })}
          </div>
        </div>
      </div>
    </div>
  )
}
