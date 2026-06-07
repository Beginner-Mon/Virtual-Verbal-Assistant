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
    return (
      <div className="px-3 md:px-5 py-1 md:py-3 animate-message-in w-full max-w-full">
        <div className="prose prose-invert prose-p:leading-relaxed prose-pre:bg-secondary/50 prose-pre:border prose-pre:border-border/40 max-w-none text-xs md:text-sm text-foreground/90">
          <p className="whitespace-pre-wrap break-words">{message.content}</p>
        </div>
      </div>
    )
  }

  return (
    <div className="flex gap-1.5 md:gap-3 px-2 md:px-4 py-1 md:py-2 animate-message-in flex-row-reverse w-full max-w-full">
      <div className="flex-shrink-0 w-6 h-6 md:w-8 md:h-8 rounded-full flex items-center justify-center text-[10px] md:text-[11px] font-semibold select-none bg-primary/20 text-primary ring-1 ring-primary/30">
        U
      </div>
      <div className="max-w-[80%] min-w-0 rounded-2xl px-3 md:px-4 py-1.5 md:py-2.5 text-xs md:text-sm leading-relaxed bg-primary text-primary-foreground rounded-tr-md">
        <p className="whitespace-pre-wrap break-words">{message.content}</p>
        <span className="block mt-1 text-[10px] tabular-nums text-primary-foreground/50 text-right">
          {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
        </span>
      </div>
    </div>
  )
}
