import { MessageSquare, Settings, User, SquarePen } from 'lucide-react'
import { cn } from '@/lib/utils'

export default function Sidebar() {
  return (
    <div className="w-16 h-full bg-card/80 backdrop-blur-xl border-r border-border/40 flex flex-col items-center py-4 z-10">
      {/* Top: Logo */}
      <div className="mb-6">
        <span className="font-black text-xl tracking-tighter bg-gradient-to-br from-primary to-accent bg-clip-text text-transparent select-none">
          ECA
        </span>
      </div>

      {/* New Session Action */}
      <div className="mb-4 pb-4 border-b border-border/40 w-full flex justify-center">
        <button 
          title="New session"
          className="w-10 h-10 rounded-xl flex items-center justify-center text-muted-foreground hover:bg-secondary hover:text-foreground transition-colors"
        >
          <SquarePen className="w-5 h-5" />
        </button>
      </div>

      {/* Middle: Conversations */}
      <div className="flex-1 flex flex-col gap-4 w-full items-center overflow-y-auto scrollbar-thin py-2">
        {/* Placeholder conversation icons */}
        <button className="w-10 h-10 rounded-xl flex items-center justify-center bg-primary/10 text-primary hover:bg-primary/20 transition-colors">
          <MessageSquare className="w-5 h-5" />
        </button>
        <button className="w-10 h-10 rounded-xl flex items-center justify-center text-muted-foreground hover:bg-secondary transition-colors">
          <MessageSquare className="w-5 h-5" />
        </button>
        <button className="w-10 h-10 rounded-xl flex items-center justify-center text-muted-foreground hover:bg-secondary transition-colors">
          <MessageSquare className="w-5 h-5" />
        </button>
      </div>

      {/* Bottom: Settings & Avatar */}
      <div className="mt-auto flex flex-col gap-4 items-center pt-4 border-t border-border/40 w-full">
        <button className="w-10 h-10 rounded-xl flex items-center justify-center text-muted-foreground hover:bg-secondary hover:text-foreground transition-colors">
          <Settings className="w-5 h-5" />
        </button>
        
        <div className="w-10 h-10 rounded-full bg-secondary flex items-center justify-center text-muted-foreground ring-2 ring-transparent hover:ring-primary/30 transition-all cursor-pointer">
          <User className="w-5 h-5" />
        </div>
      </div>
    </div>
  )
}
