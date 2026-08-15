import { Settings, IdCard, Bell, CreditCard, LogOut } from 'lucide-react'
import { useAuth } from '../../contexts/AuthContext'

interface SettingsPanelProps {
  onOpenModal?: (type: 'profile' | 'settings' | 'notifications' | 'billing') => void
}

export default function SettingsPanel({ onOpenModal }: SettingsPanelProps) {
  const { signOut } = useAuth()

  const items = [
    { id: 'profile' as const, icon: IdCard, label: 'Profile' },
    { id: 'settings' as const, icon: Settings, label: 'Settings' },
  ]

  return (
    <div className="p-2">
      {items.map((item) => {
        const Icon = item.icon
        return (
          <button
            key={item.id}
            onClick={() => onOpenModal?.(item.id)}
            className="flex items-center gap-3 w-full px-3 py-2.5 rounded-xl text-sm text-foreground hover:bg-secondary/60 transition-colors"
          >
            <Icon className="w-4 h-4 shrink-0 text-muted-foreground" />
            <span>{item.label}</span>
          </button>
        )
      })}

      <button
        onClick={() => onOpenModal?.('notifications')}
        className="flex items-center gap-3 w-full px-3 py-2.5 rounded-xl text-sm text-foreground hover:bg-secondary/60 transition-colors"
      >
        <Bell className="w-4 h-4 shrink-0 text-muted-foreground" />
        <span>Notifications</span>
      </button>

      <div className="h-px bg-border/40 my-1" />
      <button
        onClick={() => onOpenModal?.('billing')}
        className="flex items-center gap-3 w-full px-3 py-2.5 rounded-xl text-sm text-foreground hover:bg-secondary/60 transition-colors"
      >
        <CreditCard className="w-4 h-4 shrink-0" />
        <span>Billing</span>
      </button>

      <div className="h-px bg-border/40 my-1" />

      <button
        onClick={() => signOut?.()}
        className="flex items-center gap-3 w-full px-3 py-2.5 rounded-xl text-sm text-destructive hover:bg-destructive/10 transition-colors font-medium"
      >
        <LogOut className="w-4 h-4 shrink-0" />
        <span>Sign Out</span>
      </button>
    </div>
  )
}
