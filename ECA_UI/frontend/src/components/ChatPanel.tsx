import { useState, useRef, useEffect, type KeyboardEvent } from 'react'
import { ArrowUp, Mic, Sparkles } from 'lucide-react'
import TextareaAutosize from 'react-textarea-autosize'
import { ScrollArea } from './ui/scroll-area'
import ChatMessage from './ChatMessage'
import type { Message } from './ChatMessage'

/* ─── Demo data ─── */
const INITIAL_MESSAGES: Message[] = [
  {
    id: '1',
    role: 'assistant',
    content: "Hello! I'm your Virtual Verbal Assistant. How can I help you today? 🎙️\n\nI can format text like **bold**, *italics*, and lists once we integrate a Markdown parser. For now, I'm just displaying raw text with nice typography spacing.",
    timestamp: new Date(),
  },
]

const DEMO_RESPONSES = [
  "That's a great question! Let me think about that for a moment… 🤔",
  "I'd be happy to help you with that! Here's what I think…",
  'Interesting! Let me process that and give you my best answer.',
  "Thanks for sharing that. Here's my perspective on it…",
  `I understand what you're looking for. Let me explain…`,
]

/* ─── ChatPanel ─── */
export default function ChatPanel() {
  const [messages, setMessages] = useState<Message[]>(INITIAL_MESSAGES)
  const [input, setInput] = useState('')
  const [isTyping, setIsTyping] = useState(false)
  const bottomRef = useRef<HTMLDivElement>(null)

  /* auto-scroll on new messages */
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, isTyping])

  /* send handler */
  const handleSend = () => {
    const text = input.trim()
    if (!text) return

    const userMsg: Message = {
      id: crypto.randomUUID(),
      role: 'user',
      content: text,
      timestamp: new Date(),
    }

    setMessages((prev) => [...prev, userMsg])
    setInput('')
    setIsTyping(true)

    // Simulated response
    const delay = 1200 + Math.random() * 800
    setTimeout(() => {
      const reply: Message = {
        id: crypto.randomUUID(),
        role: 'assistant',
        content: DEMO_RESPONSES[Math.floor(Math.random() * DEMO_RESPONSES.length)],
        timestamp: new Date(),
      }
      setMessages((prev) => [...prev, reply])
      setIsTyping(false)
    }, delay)
  }

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <div className="flex flex-col h-full bg-card/50 backdrop-blur-xl border-r border-border/40 relative z-10">
      {/* ── Header ── */}
      <header className="flex items-center gap-3 px-5 py-4 border-b border-border/40 bg-card/80 backdrop-blur-sm shrink-0">
        <div className="flex-1 min-w-0">
          <h1 className="text-sm font-semibold text-foreground tracking-tight">
            Virtual Assistant
          </h1>
          <p className="text-xs text-muted-foreground flex items-center gap-1">
            <Sparkles className="w-3 h-3" />
            Online · Ready to chat
          </p>
        </div>
      </header>

      {/* ── Messages ── */}
      <ScrollArea className="flex-1 px-2">
        <div className="py-4 space-y-2">
          {messages.map((msg) => (
            <ChatMessage key={msg.id} message={msg} />
          ))}

          {/* typing indicator */}
          {isTyping && (
            <div className="px-5 py-3 animate-message-in">
              <div className="flex gap-1.5">
                <span className="w-1.5 h-1.5 rounded-full bg-muted-foreground/50 animate-bounce [animation-delay:-0.3s]" />
                <span className="w-1.5 h-1.5 rounded-full bg-muted-foreground/50 animate-bounce [animation-delay:-0.15s]" />
                <span className="w-1.5 h-1.5 rounded-full bg-muted-foreground/50 animate-bounce" />
              </div>
            </div>
          )}
          <div ref={bottomRef} />
        </div>
      </ScrollArea>

      {/* ── Input ── */}
      <div className="p-4 bg-card/80 backdrop-blur-sm shrink-0">
        <div className="flex flex-wrap items-end gap-2 bg-secondary/40 border border-border/40 rounded-2xl p-2 focus-within:ring-1 focus-within:ring-primary/50 focus-within:border-primary/50 transition-all">
          <TextareaAutosize
            minRows={1}
            maxRows={6}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder="Type your message…"
            disabled={isTyping}
            className="flex-1 bg-transparent px-3 py-2 text-sm text-foreground placeholder:text-muted-foreground/70 resize-none focus:outline-none disabled:opacity-50 min-w-[200px]"
          />
          
          <div className="flex items-center gap-1 shrink-0 pb-1">
            <button
              title="Record audio"
              className="p-2 text-muted-foreground hover:text-foreground transition-colors disabled:opacity-50"
              disabled={isTyping}
            >
              <Mic className="w-5 h-5" />
            </button>
            <button
              onClick={handleSend}
              disabled={!input.trim() || isTyping}
              className="w-8 h-8 rounded-full bg-primary hover:bg-primary/90 disabled:opacity-30 disabled:cursor-not-allowed flex items-center justify-center transition-all text-primary-foreground"
            >
              <ArrowUp className="w-4 h-4" />
            </button>
          </div>
        </div>
        <p className="text-[10px] text-muted-foreground/50 mt-2 text-center select-none">
          Press Enter to send · Shift+Enter for new line
        </p>
      </div>
    </div>
  )
}
