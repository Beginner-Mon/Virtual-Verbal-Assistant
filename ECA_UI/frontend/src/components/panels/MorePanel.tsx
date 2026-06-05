import { MoreHorizontal, Sparkles } from 'lucide-react'
import { ScrollArea } from '../ui/scroll-area'

export default function MorePanel() {
  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="px-4 py-3 border-b border-border/40 shrink-0">
        <h2 className="text-sm font-semibold text-foreground tracking-tight flex items-center gap-2">
          <MoreHorizontal className="w-4 h-4 text-primary" />
          More
        </h2>
      </div>

      {/* Placeholder */}
      <ScrollArea className="flex-1">
        <div className="flex flex-col items-center justify-center px-6 py-8 text-center h-full min-h-[300px]">
          <div className="w-14 h-14 rounded-2xl bg-primary/10 flex items-center justify-center mb-4">
            <Sparkles className="w-7 h-7 text-primary/60" />
          </div>
          <p className="text-sm font-medium text-foreground/80 mb-1">Coming soon</p>
          <p className="text-xs text-muted-foreground leading-relaxed">
            More features will be added here in future updates.
          </p>
        </div>
      </ScrollArea>
    </div>
  )
}
