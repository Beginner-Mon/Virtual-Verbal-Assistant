import { cn } from '@/lib/utils'

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: Date
}

interface ChatMessageProps {
  message: Message
}

export default function ChatMessage({ message }: ChatMessageProps) {
  const isUser = message.role === 'user'

  if (!isUser) {
    // Agent message: no bubble, no avatar, direct markdown styling
    return (
      <div className="px-5 py-3 animate-message-in">
        <div className="prose prose-invert prose-p:leading-relaxed prose-pre:bg-secondary/50 prose-pre:border prose-pre:border-border/40 max-w-none text-foreground/90">
          <p className="whitespace-pre-wrap break-words">{message.content}</p>
        </div>
      </div>
    )
  }

  // User message: keep the styled bubble on the right
  return (
    <div className="flex gap-3 px-4 py-2 animate-message-in flex-row-reverse">
      {/* Avatar */}
      <div className="flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center text-[11px] font-semibold select-none bg-primary/20 text-primary ring-1 ring-primary/30">
        U
      </div>

      {/* Bubble */}
      <div className="max-w-[80%] rounded-2xl px-4 py-2.5 text-sm leading-relaxed bg-primary text-primary-foreground rounded-tr-md">
        <p className="whitespace-pre-wrap break-words">{message.content}</p>
        <span className="block mt-1 text-[10px] tabular-nums text-primary-foreground/50 text-right">
          {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
        </span>
      </div>
    </div>
  )
}
