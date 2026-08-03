import { Package } from 'lucide-react'

export default function NotificationsContent() {
  return (
    <div className="flex-1 overflow-y-auto flex items-center justify-center p-6">
      <div className="flex flex-col items-center justify-center text-center">
        <Package className="w-12 h-12 text-muted-foreground/40 mb-4" />
        <h3 className="text-sm font-semibold text-foreground">Project ECA v0.0</h3>
        <p className="text-xs text-muted-foreground mt-1">No updates available yet. Check back later.</p>
      </div>
    </div>
  )
}
