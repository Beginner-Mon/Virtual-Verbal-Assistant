import { useState, useRef, useEffect, type KeyboardEvent } from 'react'
import { ArrowUp, Mic, Sparkles, Square, Plus, Globe, Paperclip } from 'lucide-react'
import TextareaAutosize from 'react-textarea-autosize'
import { ScrollArea } from './ui/scroll-area'
import ChatMessage from './ChatMessage'
import { useChat } from '../contexts/ChatContext'

/* ─── ChatPanel ─── */
export default function ChatPanel() {
  const {
    messages,
    input,
    setInput,
    isTyping,
    isGenerating,
    stageLabel,
    webSearch,
    setWebSearch,
    handleSend,
    handleStop,
  } = useChat()

  const [showAddMenu, setShowAddMenu] = useState(false)

  const bottomRef = useRef<HTMLDivElement>(null)
  const addMenuRef = useRef<HTMLDivElement>(null)

  /* close add menu on outside click */
  useEffect(() => {
    if (!showAddMenu) return
    const handler = (e: MouseEvent) => {
      if (addMenuRef.current && !addMenuRef.current.contains(e.target as Node)) {
        setShowAddMenu(false)
      }
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [showAddMenu])

  /* auto-scroll on new messages */
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, isTyping, stageLabel])

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <div className="flex flex-col h-full bg-transparent md:backdrop-blur-xl border-r border-border/40 relative z-10">
      {/* ── Header ── */}
      <header className="hidden md:flex items-center gap-3 px-5 py-4 border-b border-border/40 bg-card/80 backdrop-blur-sm shrink-0">
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
      <ScrollArea className="flex-1 min-h-0 px-1 md:px-2">
        <div className="py-2 md:py-4 space-y-1 md:space-y-2 max-w-full overflow-x-hidden">
          {messages.map((msg) => (
            <ChatMessage key={msg.id} message={msg} />
          ))}

          {/* typing / stage indicator */}
          {(isTyping || stageLabel) && (
            <div className="px-3 md:px-5 py-2 md:py-3 animate-message-in">
              {stageLabel ? (
                <p className="text-xs text-muted-foreground italic">{stageLabel}</p>
              ) : (
                <div className="flex gap-1.5">
                  <span className="w-1.5 h-1.5 rounded-full bg-muted-foreground/50 animate-bounce [animation-delay:-0.3s]" />
                  <span className="w-1.5 h-1.5 rounded-full bg-muted-foreground/50 animate-bounce [animation-delay:-0.15s]" />
                  <span className="w-1.5 h-1.5 rounded-full bg-muted-foreground/50 animate-bounce" />
                </div>
              )}
            </div>
          )}
          <div ref={bottomRef} />
        </div>
      </ScrollArea>

      {/* ── Input ── */}
      <div className="p-2 md:p-4 bg-transparent md:bg-card/80 md:backdrop-blur-sm shrink-0">
        <div className="flex flex-col bg-card border border-border/40 rounded-2xl p-1.5 md:p-2 focus-within:ring-1 focus-within:ring-primary/50 focus-within:border-primary/50 transition-all relative">
          <TextareaAutosize
            minRows={1}
            maxRows={6}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder="Type your message…"
            disabled={isGenerating}
            className="w-full bg-transparent px-3 py-2 text-xs md:text-sm text-foreground placeholder:text-muted-foreground/70 resize-none focus:outline-none disabled:opacity-50"
          />
          
          <div className="flex items-center justify-between gap-1 shrink-0 pt-2 px-1 pb-1">
            {webSearch && (
              <button
                onClick={() => setWebSearch(false)}
                title="Web search on — click to turn off"
                className="flex items-center gap-1 bg-secondary rounded-lg px-2 py-1 text-xs text-muted-foreground"
              >
                <Globe className="w-3 h-3" />
                Web
              </button>
            )}
            <div className="relative" ref={addMenuRef}>
              <button
                onClick={() => setShowAddMenu((prev) => !prev)}
                title="Add"
                className="p-1.5 rounded-lg text-muted-foreground hover:text-foreground hover:bg-secondary/60 transition-colors"
              >
                <Plus className="w-5 h-5" />
              </button>

              {showAddMenu && (
                <div className="absolute bottom-full left-0 mb-2 w-48 bg-card border border-border/50 rounded-xl shadow-xl overflow-hidden z-50 animate-panel-in">
                  <button
                    onClick={() => { setWebSearch(!webSearch); setShowAddMenu(false) }}
                    className="w-full flex items-center gap-3 px-4 py-3 text-sm text-foreground hover:bg-secondary/60 transition-colors"
                  >
                    <Globe className="w-4 h-4 text-muted-foreground" />
                    Search online
                    {webSearch && <span className="ml-auto text-xs text-primary">On</span>}
                  </button>
                  <div className="h-px bg-border/40 mx-3" />
                  <button
                    onClick={() => { setShowAddMenu(false) }}
                    className="w-full flex items-center gap-3 px-4 py-3 text-sm text-foreground hover:bg-secondary/60 transition-colors"
                  >
                    <Paperclip className="w-4 h-4 text-muted-foreground" />
                    Files
                  </button>
                </div>
              )}
            </div>

            <div className="flex items-center gap-1">
              <button
                title="Record audio"
                className="p-2 text-muted-foreground hover:text-foreground transition-colors disabled:opacity-50"
                disabled={isGenerating}
              >
                <Mic className="w-5 h-5" />
              </button>
              {isGenerating ? (
                <button
                  onClick={handleStop}
                  className="w-8 h-8 rounded-full bg-primary hover:bg-primary/90 flex items-center justify-center transition-all text-primary-foreground"
                >
                  <Square className="w-4 h-4 fill-current" />
                </button>
              ) : (
                <button
                  onClick={handleSend}
                  disabled={!input.trim()}
                  className="w-8 h-8 rounded-full bg-primary hover:bg-primary/90 disabled:opacity-30 disabled:cursor-not-allowed flex items-center justify-center transition-all text-primary-foreground"
                >
                  <ArrowUp className="w-4 h-4" />
                </button>
              )}
            </div>
          </div>
        </div>
        <p className="hidden md:block text-[10px] text-muted-foreground/50 mt-2 text-center select-none">
          Press Enter to send · Shift+Enter for new line
        </p>
      </div>
    </div>
  )
}
