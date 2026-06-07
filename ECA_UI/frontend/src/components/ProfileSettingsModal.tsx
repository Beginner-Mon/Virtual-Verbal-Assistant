import { useEffect, useCallback } from 'react'
import { X, ChevronLeft } from 'lucide-react'
import ProfileContent from './ProfileContent'
import SettingsContent from './SettingsContent'

interface ProfileSettingsModalProps {
  type: 'profile' | 'settings'
  onClose: () => void
  onBack?: () => void
}

export default function ProfileSettingsModal({ type, onClose, onBack }: ProfileSettingsModalProps) {
  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    },
    [onClose]
  )

  useEffect(() => {
    document.addEventListener('keydown', handleKeyDown)
    document.body.style.overflow = 'hidden'
    return () => {
      document.removeEventListener('keydown', handleKeyDown)
      document.body.style.overflow = ''
    }
  }, [handleKeyDown])

  return (
    <div className="fixed inset-0 z-[10000] flex items-end justify-center">
      <div
        className="absolute inset-0 bg-black/40 backdrop-blur-sm"
        onPointerDown={(e) => {
          if (e.target === e.currentTarget) onClose()
        }}
      />

      <div className="relative w-full max-w-2xl h-full bg-card border border-border/50 border-b-0 shadow-[0_-8px_40px_rgba(0,0,0,0.4)] animate-slide-up flex flex-col overflow-hidden">
        <div className="flex items-center gap-2 px-4 py-3 border-b border-border/40 shrink-0">
          <button
            onClick={onBack ?? onClose}
            className="p-1.5 -ml-1 rounded-lg text-muted-foreground hover:text-foreground hover:bg-secondary/60 transition-colors"
          >
            <ChevronLeft className="w-5 h-5" />
          </button>
          <h2 className="text-base font-semibold text-foreground flex-1">
            {type === 'profile' ? 'User Profile' : 'Settings'}
          </h2>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-muted-foreground hover:text-foreground hover:bg-secondary/60 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {type === 'profile' ? <ProfileContent /> : <SettingsContent />}
      </div>
    </div>
  )
}
