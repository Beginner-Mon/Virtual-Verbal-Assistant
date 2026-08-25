import { useState, useEffect, useCallback, lazy, Suspense } from 'react'
import { X, ChevronLeft } from 'lucide-react'

const ProfileContent = lazy(() => import('./ProfileContent'))
const SettingsContent = lazy(() => import('./SettingsContent'))
const NotificationsContent = lazy(() => import('./NotificationsContent'))
const GraphicSettingsContent = lazy(() => import('./GraphicSettingsContent'))
const AboutUsContent = lazy(() => import('./AboutUsContent'))
const BillingContent = lazy(() => import('./BillingContent'))

interface ProfileSettingsModalProps {
  type: 'profile' | 'settings' | 'notifications' | 'graphics' | 'about' | 'billing'
  onClose: () => void
  onBack?: () => void
}

const ROOT_VIEW = 'main' as const

export default function ProfileSettingsModal({ type, onClose, onBack }: ProfileSettingsModalProps) {
  const [settingsView, setSettingsView] = useState<'main' | 'providers' | 'provider-detail' | 'graphics' | 'about'>(ROOT_VIEW)
  const [selectedProvider, setSelectedProvider] = useState<{ id: string; name: string } | undefined>()

  useEffect(() => {
    setSettingsView(ROOT_VIEW)
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
      setSettingsView(ROOT_VIEW)
    } else if (settingsView === 'graphics' || settingsView === 'about') {
      setSettingsView(ROOT_VIEW)
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
      : type === 'notifications'
        ? 'Notifications'
        : type === 'billing'
          ? 'Billing Sandbox'
        : type === 'graphics'
          ? 'Graphic Settings'
          : type === 'about'
            ? 'About Us'
            : settingsView === 'graphics'
              ? 'Graphic Settings'
              : settingsView === 'about'
                ? 'About Us'
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
          {type === 'settings' && settingsView !== ROOT_VIEW && (
            <button
              onClick={handleBack}
              className="p-1.5 -ml-1 rounded-lg text-muted-foreground hover:text-foreground hover:bg-secondary/60 transition-colors"
              aria-label="Back"
            >
              <ChevronLeft className="w-5 h-5" />
            </button>
          )}
          <h2 className="text-base font-semibold text-foreground flex-1">{title}</h2>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-muted-foreground hover:text-foreground hover:bg-secondary/60 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        <Suspense fallback={<div className="flex-1 flex items-center justify-center text-sm text-muted-foreground">Loading...</div>}>
          {type === 'profile' ? (
            <ProfileContent onClose={onClose} />
          ) : type === 'notifications' ? (
            <NotificationsContent />
          ) : type === 'billing' ? (
            <BillingContent />
          ) : type === 'graphics' ? (
            <GraphicSettingsContent />
          ) : type === 'about' ? (
            <AboutUsContent />
          ) : settingsView === 'graphics' ? (
            <GraphicSettingsContent />
          ) : settingsView === 'about' ? (
            <AboutUsContent />
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
            <SettingsContent
              onNavigateToProviders={() => setSettingsView('providers')}
              onNavigateToGraphics={() => setSettingsView('graphics')}
              onNavigateToAbout={() => setSettingsView('about')}
            />
          )}
        </Suspense>
      </div>
    </div>
  )
}
