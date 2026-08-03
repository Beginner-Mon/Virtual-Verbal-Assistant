import { Bell, Package } from 'lucide-react'

export default function NotificationsContent() {
  return (
    <div className="flex-1 overflow-y-auto p-6">
      <div className="flex items-center gap-2 mb-6">
        <Bell className="w-5 h-5 text-muted-foreground" />
        <h2 className="text-lg font-semibold text-foreground">Notifications</h2>
      </div>

      <div className="flex flex-col items-center justify-center flex-1 py-16 text-center">
        <Package className="w-12 h-12 text-muted-foreground/40 mb-4" />
        <h3 className="text-sm font-semibold text-foreground">Project ECA v0.0</h3>
        <p className="text-xs text-muted-foreground mt-1">No updates available yet. Check back later.</p>
      </div>
    </div>
  )
}
