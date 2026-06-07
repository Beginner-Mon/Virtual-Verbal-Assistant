import { useState, useEffect, useCallback } from 'react'
import { X, ChevronLeft } from 'lucide-react'
import ProfileContent from './ProfileContent'
import SettingsContent from './SettingsContent'

interface ProfileSettingsModalProps {
  type: 'profile' | 'settings'
  onClose: () => void
  onBack?: () => void
}

export default function ProfileSettingsModal({ type, onClose, onBack }: ProfileSettingsModalProps) {
  const [settingsView, setSettingsView] = useState<'main' | 'providers' | 'provider-detail'>('main')
  const [selectedProvider, setSelectedProvider] = useState<{ id: string; name: string } | undefined>()

  useEffect(() => {
    setSettingsView('main')
    setSelectedProvider(undefined)
  }, [type])

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

  const handleBack = () => {
    if (type === 'settings' && settingsView === 'provider-detail') {
      setSettingsView('providers')
      setSelectedProvider(undefined)
    } else if (type === 'settings' && settingsView === 'providers') {
      setSettingsView('main')
    } else {
      onBack ?? onClose()
    }
  }

  const handleSelectProvider = (provider: { id: string; name: string }) => {
    setSelectedProvider(provider)
    setSettingsView('provider-detail')
  }

  const title =
    type === 'profile'
      ? 'User Profile'
      : settingsView === 'provider-detail'
        ? selectedProvider?.name ?? 'Settings'
        : 'Settings'

  return (
    <div className="fixed inset-0 z-[10000] flex items-end md:items-center justify-center">
      <div
        className="absolute inset-0 bg-black/40 backdrop-blur-sm"
        onPointerDown={(e) => {
          if (e.target === e.currentTarget) onClose()
        }}
      />

      <div className="relative w-full max-w-2xl md:max-w-none h-full md:w-[900px] md:h-[600px] bg-card md:rounded-2xl border border-border/50 border-b-0 md:border-b shadow-[0_-8px_40px_rgba(0,0,0,0.4)] md:shadow-[0_32px_80px_rgba(0,0,0,0.5)] animate-slide-up md:animate-panel-in flex flex-col overflow-hidden">
        <div className="flex items-center gap-2 px-4 py-3 border-b border-border/40 shrink-0">
          <button
            onClick={handleBack}
            className="p-1.5 -ml-1 rounded-lg text-muted-foreground hover:text-foreground hover:bg-secondary/60 transition-colors"
          >
            <ChevronLeft className="w-5 h-5" />
          </button>
          <h2 className="text-base font-semibold text-foreground flex-1">{title}</h2>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-muted-foreground hover:text-foreground hover:bg-secondary/60 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {type === 'profile' ? (
          <ProfileContent />
        ) : settingsView === 'providers' ? (
          <SettingsContent
            view="providers"
            onSelectProvider={handleSelectProvider}
          />
        ) : settingsView === 'provider-detail' ? (
          <SettingsContent
            view="provider-detail"
            selectedProvider={selectedProvider}
          />
        ) : (
          <SettingsContent onNavigateToProviders={() => setSettingsView('providers')} />
        )}
      </div>
    </div>
  )
}