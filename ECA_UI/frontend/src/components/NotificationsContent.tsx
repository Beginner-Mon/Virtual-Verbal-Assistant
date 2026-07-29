import { Bell, BellOff, Volume2, MessageCircle, Activity } from 'lucide-react'
import { ScrollArea } from './ui/scroll-area'

interface Props {
  onClose?: () => void
}

export default function NotificationsContent({ onClose: _onClose }: Props) {
  const settings = [
    {
      id: 'messages',
      icon: MessageCircle,
      label: 'Messages',
      description: 'Notifications for new chat messages and responses',
      enabled: true,
    },
    {
      id: 'motion',
      icon: Activity,
      label: 'Motion Complete',
      description: 'Notify when 3D motion rendering finishes',
      enabled: true,
    },
    {
      id: 'sound',
      icon: Volume2,
      label: 'Sound Alerts',
      description: 'Play a sound when receiving notifications',
      enabled: false,
    },
    {
      id: 'silent',
      icon: BellOff,
      label: 'Silent Mode',
      description: 'Pause all notifications temporarily',
      enabled: false,
    },
  ]

  return (
    <ScrollArea className="flex-1 min-h-0">
      <div className="p-6 flex flex-col gap-4">
        <div className="flex items-center gap-3 mb-2">
          <div className="w-10 h-10 rounded-full bg-primary/10 flex items-center justify-center">
            <Bell className="w-5 h-5 text-primary" />
          </div>
          <div>
            <h3 className="text-sm font-semibold text-foreground">Notification Preferences</h3>
            <p className="text-xs text-muted-foreground">Manage how and when you receive alerts</p>
          </div>
        </div>

        <div className="flex flex-col gap-2">
          {settings.map((setting) => {
            const Icon = setting.icon
            return (
              <div
                key={setting.id}
                className="flex items-center justify-between p-3 rounded-xl bg-secondary/20 border border-border/10"
              >
                <div className="flex items-center gap-3">
                  <Icon className="w-4 h-4 text-muted-foreground shrink-0" />
                  <div>
                    <p className="text-sm font-medium text-foreground">{setting.label}</p>
                    <p className="text-xs text-muted-foreground">{setting.description}</p>
                  </div>
                </div>
                <div
                  role="switch"
                  aria-checked={setting.enabled}
                  tabIndex={0}
                  className={`relative inline-flex h-5 w-9 shrink-0 cursor-pointer items-center rounded-full transition-colors ${
                    setting.enabled ? 'bg-primary' : 'bg-muted-foreground/30'
                  }`}
                >
                  <span
                    className={`inline-block h-3.5 w-3.5 rounded-full bg-white shadow-sm transition-transform ${
                      setting.enabled ? 'translate-x-[18px]' : 'translate-x-[4px]'
                    }`}
                  />
                </div>
              </div>
            )
          })}
        </div>
      </div>
    </ScrollArea>
  )
}
