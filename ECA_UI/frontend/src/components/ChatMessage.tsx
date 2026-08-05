import { useState, useRef, useCallback, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { Copy, ThumbsUp, ThumbsDown, Volume2, Check, Pause } from 'lucide-react'

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: Date
  audioUrl?: string
}

interface ChatMessageProps {
  message: Message
}

function useCopy(text: string) {
  const [copied, setCopied] = useState(false)
  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(text)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch {
      // clipboard API not available
    }
  }
  return { copied, handleCopy }
}

export default function ChatMessage({ message }: ChatMessageProps) {
  const isUser = message.role === 'user'

  if (!isUser) {
    const cleaned = message.content.replace(/<\/?evidence_citation>/g, '')
    return (
      <div className="px-3 md:px-5 py-1 md:py-3 animate-message-in w-full max-w-full">
        <div className="prose dark:prose-invert prose-p:leading-relaxed prose-strong:text-foreground prose-headings:text-foreground prose-pre:bg-secondary/50 prose-pre:border prose-pre:border-border/40 max-w-none text-[clamp(0.75rem,0.68rem+0.3vw,0.875rem)] text-foreground/90">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{cleaned}</ReactMarkdown>
        </div>
        {message.content && <AssistantActions content={message.content} audioUrl={message.audioUrl} />}
      </div>
    )
  }

  return (
    <div className="group flex px-2 md:px-4 py-0.5 md:py-1 animate-message-in flex-row-reverse w-full max-w-full">
      <div className="w-fit max-w-[80%]">
        <div className="min-w-0 rounded-2xl px-2.5 md:px-3 py-1 md:py-2 text-[clamp(0.75rem,0.68rem+0.3vw,0.875rem)] leading-relaxed bg-primary text-primary-foreground">
          <p className="whitespace-pre-wrap break-words">{message.content}</p>
        </div>
        <UserCopyAction content={message.content} />
      </div>
    </div>
  )
}

function AssistantActions({ content, audioUrl }: { content: string; audioUrl?: string }) {
  const { copied, handleCopy } = useCopy(content)
  const [liked, setLiked] = useState(false)
  const [disliked, setDisliked] = useState(false)

  const handleLike = () => {
    setLiked((v) => !v)
    if (!liked) setDisliked(false)
  }

  const handleDislike = () => {
    setDisliked((v) => !v)
    if (!disliked) setLiked(false)
  }

  const btnClass =
    'p-1 rounded-md text-muted-foreground hover:text-foreground hover:bg-secondary/60'

  const iconSize = 'size-4'

  return (
    <div className="flex items-center gap-1 mt-1.5">
      <button className={btnClass} onClick={handleCopy} title="Copy">
        {copied ? <Check className={iconSize} /> : <Copy className={iconSize} />}
      </button>
      <button className={btnClass} onClick={handleLike} title="Thích">
        <ThumbsUp className={`${iconSize} ${liked ? 'text-green-500' : ''}`} />
      </button>
      <button className={btnClass} onClick={handleDislike} title="Không thích">
        <ThumbsDown className={`${iconSize} ${disliked ? 'text-blue-500' : ''}`} />
      </button>
      {audioUrl && <AudioButton audioUrl={audioUrl} btnClass={btnClass} iconSize={iconSize} />}
    </div>
  )
}

function AudioButton({ audioUrl, btnClass, iconSize }: { audioUrl: string; btnClass: string; iconSize: string }) {
  const audioRef = useRef<HTMLAudioElement | null>(null)
  const barRef = useRef<HTMLDivElement | null>(null)
  const rafRef = useRef<number>(0)
  const [playing, setPlaying] = useState(false)
  const [paused, setPaused] = useState(false)
  const [progress, setProgress] = useState(0)

  const tick = useCallback(() => {
    const a = audioRef.current
    if (!a) return
    setProgress((a.currentTime / a.duration) * 100 || 0)
    rafRef.current = requestAnimationFrame(tick)
  }, [])

  const reset = useCallback(() => {
    const a = audioRef.current
    if (!a) return
    a.pause()
    a.currentTime = 0
    setPlaying(false)
    setPaused(false)
    setProgress(0)
    cancelAnimationFrame(rafRef.current)
  }, [])

  const handleEnded = useCallback(() => {
    setPlaying(false)
    setPaused(false)
    setProgress(0)
    cancelAnimationFrame(rafRef.current)
    if (audioRef.current) audioRef.current.currentTime = 0
  }, [])

  useEffect(() => {
    return () => {
      cancelAnimationFrame(rafRef.current)
      audioRef.current?.pause()
    }
  }, [])

  const handleToggle = () => {
    if (!audioRef.current) {
      audioRef.current = new Audio(audioUrl)
      audioRef.current.addEventListener('ended', handleEnded)
    }
    const a = audioRef.current
    if (playing) {
      a.pause()
      setPlaying(false)
      setPaused(true)
      cancelAnimationFrame(rafRef.current)
    } else {
      a.play().catch(() => {})
      setPlaying(true)
      setPaused(false)
      rafRef.current = requestAnimationFrame(tick)
    }
  }

  const handleBarClick = (e: React.MouseEvent<HTMLDivElement>) => {
    const a = audioRef.current
    const bar = barRef.current
    if (!a || !bar) return
    const rect = bar.getBoundingClientRect()
    const pct = ((e.clientX - rect.left) / rect.width) * 100
    a.currentTime = (pct / 100) * a.duration
    setProgress(pct)
  }

  const isActive = playing || paused

  return (
    <div className="group flex items-center gap-1">
      <button
        className={`${btnClass} ${isActive ? 'text-foreground' : ''}`}
        onClick={handleToggle}
        title={playing ? 'Dừng' : paused ? 'Tiếp tục' : 'Nghe'}
        onDoubleClick={(e) => e.preventDefault()}
      >
        {playing ? <Pause className={iconSize} /> : <Volume2 className={iconSize} />}
      </button>
      <div
        ref={barRef}
        className={`h-1.5 bg-secondary rounded-full cursor-pointer flex-shrink-0 overflow-hidden transition-all duration-300 ease-out ${isActive ? 'w-24 opacity-100' : 'w-0 opacity-0'}`}
        onClick={handleBarClick}
      >
        <div
          className="h-full bg-primary rounded-full"
          style={{ width: `${progress}%` }}
        />
      </div>
    </div>
  )
}

function UserCopyAction({ content }: { content: string }) {
  const { copied, handleCopy } = useCopy(content)

  return (
    <div className="flex justify-start mt-0.5 opacity-0 group-hover:opacity-100">
      <button
        className="p-1 rounded-md text-muted-foreground hover:text-foreground hover:bg-secondary/60"
        onClick={handleCopy}
        title="Copy"
      >
        {copied ? <Check className="size-4" /> : <Copy className="size-4" />}
      </button>
    </div>
  )
}
