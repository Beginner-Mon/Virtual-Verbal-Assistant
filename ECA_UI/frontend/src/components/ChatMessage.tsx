import { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { Copy, ThumbsUp, ThumbsDown, Volume2, Check } from 'lucide-react'

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
        <AssistantActions content={message.content} audioUrl={message.audioUrl} />
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

  const handlePlayAudio = () => {
    if (!audioUrl) return
    const audio = new Audio(audioUrl)
    audio.play().catch(() => {})
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
      {audioUrl && (
        <button className={btnClass} onClick={handlePlayAudio} title="Nghe">
          <Volume2 className={iconSize} />
        </button>
      )}
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
