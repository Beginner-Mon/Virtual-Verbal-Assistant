import { useState, useRef, useEffect, type KeyboardEvent } from 'react'
import { ArrowUp, Mic, Sparkles, Square, Plus, Globe, Image, X, Volume2, SquarePen } from 'lucide-react'
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
    voiceReply,
    setVoiceReply,
    isRestoring,
    startNewSession,
    handleSend,
    handleStop,
    imageUrls,
    addImage,
    removeImage,
  } = useChat()

  const [showAddMenu, setShowAddMenu] = useState(false)

  const bottomRef = useRef<HTMLDivElement>(null)
  const addMenuRef = useRef<HTMLDivElement>(null)
  const imageInputRef = useRef<HTMLInputElement>(null)

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
            {isRestoring ? 'Đang tải hội thoại trước...' : 'Online · Ready to chat'}
          </p>
        </div>
        {/* Without this the conversation restored on load is the only one the
            user can ever be in — there is no other way out of it yet. */}
        <button
          onClick={startNewSession}
          title="Cuộc trò chuyện mới"
          className="p-1.5 rounded-lg text-muted-foreground hover:text-foreground hover:bg-secondary/60 transition-colors shrink-0"
        >
          <SquarePen className="w-4 h-4" />
        </button>
      </header>

      {/* ── Messages ── */}
      <ScrollArea className="flex-1 min-h-0 px-1 md:px-2">
        <div className="py-2 md:py-4 space-y-1 md:space-y-2 max-w-full overflow-x-hidden">
          {messages.map((msg, i) => (
            <ChatMessage key={msg.id} message={msg} isStreaming={isGenerating && i === messages.length - 1} />
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
        <div className="flex flex-col gap-3 bg-card border border-border/40 rounded-2xl p-1.5 md:p-2 focus-within:ring-1 focus-within:ring-primary/50 focus-within:border-primary/50 transition-all relative">
          {imageUrls.length > 0 && (
            <div className="flex gap-1.5 overflow-x-auto mx-1 p-2">
              {imageUrls.map((url, i) => (
                <div key={url} className="relative group shrink-0">
                  <img
                    src={url}
                    alt="Preview"
                    className="w-20 h-20 rounded-lg object-cover"
                  />
                  <button
                    onClick={() => removeImage(i)}
                    className="absolute -top-1.5 -left-1.5 w-5 h-5 rounded-full bg-black/70 hover:bg-black flex items-center justify-center transition-colors opacity-0 group-hover:opacity-100 cursor-pointer"
                  >
                    <X className="w-3 h-3 text-white" />
                  </button>
                </div>
              ))}
            </div>
          )}
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
          
          <div className="flex items-center gap-1 shrink-0 pt-2 px-1 pb-1">
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
                    onClick={() => { setVoiceReply(!voiceReply); setShowAddMenu(false) }}
                    className="w-full flex items-center gap-3 px-4 py-3 text-sm text-foreground hover:bg-secondary/60 transition-colors"
                  >
                    <Volume2 className="w-4 h-4 text-muted-foreground" />
                    Trả lời bằng giọng nói
                    {voiceReply && <span className="ml-auto text-xs text-primary">On</span>}
                  </button>
                  <div className="h-px bg-border/40 mx-3" />
                  <button
                    onClick={() => { imageInputRef.current?.click(); setShowAddMenu(false) }}
                    disabled={imageUrls.length >= 10}
                    className="w-full flex items-center gap-3 px-4 py-3 text-sm text-foreground hover:bg-secondary/60 transition-colors disabled:opacity-50"
                  >
                    <Image className="w-4 h-4 text-muted-foreground" />
                    Media
                  </button>
                  {/* <div className="h-px bg-border/40 mx-3" />
                  <button
                    onClick={() => { setShowAddMenu(false) }}
                    className="w-full flex items-center gap-3 px-4 py-3 text-sm text-foreground hover:bg-secondary/60 transition-colors"
                  >
                    <Paperclip className="w-4 h-4 text-muted-foreground" />
                    Files
                  </button> */}
                </div>
              )}
            </div>

            {webSearch && (
              <div className="flex items-center gap-1 bg-secondary rounded-lg px-2 py-1 text-xs text-muted-foreground">
                <Globe className="w-3 h-3" />
                Web
                <button
                  onClick={() => setWebSearch(false)}
                  className="hover:text-foreground transition-colors cursor-pointer"
                >
                  <X className="w-3 h-3" />
                </button>
              </div>
            )}

            {voiceReply && (
              <div className="flex items-center gap-1 bg-secondary rounded-lg px-2 py-1 text-xs text-muted-foreground">
                <Volume2 className="w-3 h-3" />
                Giọng nói
                <button
                  onClick={() => setVoiceReply(false)}
                  className="hover:text-foreground transition-colors cursor-pointer"
                >
                  <X className="w-3 h-3" />
                </button>
              </div>
            )}

            <div className="flex-1" />

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
        <input
          ref={imageInputRef}
          type="file"
          accept="image/*"
          multiple
          className="hidden"
          onChange={(e) => {
            const files = e.target.files
            if (files) {
              const remaining = 10 - imageUrls.length
              for (let i = 0; i < Math.min(files.length, remaining); i++) {
                addImage(files[i])
              }
            }
            e.target.value = ''
          }}
        />
        <p className="hidden md:block text-[10px] text-muted-foreground/50 mt-2 text-center select-none">
          Press Enter to send · Shift+Enter for new line
        </p>
      </div>
    </div>
  )
}
