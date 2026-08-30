import { useState, useRef, useCallback, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { Copy, ThumbsUp, ThumbsDown, Volume2, Check, Pause, Square, Loader2 } from 'lucide-react'
import { speakText } from '../lib/api'
import { createSpeechAudio, startSpeaking, stopSpeaking } from '../lib/speechLipSync'
import { useMotion } from '../hooks/useMotion'

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: Date
  audioUrl?: string
  /** Voice mode: audio for this reply is being synthesised right now. Drives the
   *  spinner, which is the only thing that tells the user the toggle did
   *  anything during the 30-45s wait. */
  speechPending?: boolean
  /** Voice mode: speak as soon as the audio lands, without waiting for a click.
   *  "Tráº£ lá»i báº±ng giá»ng nÃ³i" promises exactly this â€” attaching a silent audio
   *  file looks identical to the toggle being off. */
  autoplay?: boolean
  /** One line about this reply's 3D motion: rendering, or why there is none.
   *  Cleared once the clip plays, because the avatar is then saying it. The
   *  GPU worker is off by default, so "unavailable" is the ordinary case and a
   *  user who asked to SEE a movement needs telling â€” silence reads as the
   *  request having been misunderstood. */
  motionNotice?: string
  /** Restored turns only: the motion this reply rendered, and when it dies.
   *  A live turn plays its motion straight from the SSE event and needs
   *  neither. */
  motionJobId?: string
  motionExpiresAt?: string
  /** What the user asked for, so the replay picker can label it. */
  motionLabel?: string
}

interface ChatMessageProps {
  message: Message
  isStreaming?: boolean
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

export default function ChatMessage({ message, isStreaming }: ChatMessageProps) {
  const isUser = message.role === 'user'

  if (!isUser) {
    /* Assistant message Ä‘Æ°á»£c táº¡o rá»—ng ngay lÃºc user gá»­i (ChatContext.tsx:364)
     * Ä‘á»ƒ id cá»§a nÃ³ báº¯t token tá»« stream. ChÆ°a cÃ³ text thÃ¬ khÃ´ng render â€” cÃ¡i
     * shell rá»—ng váº«n Äƒn py-3 + space-y-2 cá»§a container, Ä‘áº©y loading indicator
     * xuá»‘ng hÆ¡n 30px so vá»›i user message. */
    if (!message.content) return null

    const cleaned = message.content.replace(/<\/?evidence_citation>/g, '')
    return (
      <div className="px-3 md:px-5 py-1 md:py-3 animate-message-in w-full max-w-full">
        <div className="prose dark:prose-invert prose-p:leading-relaxed prose-strong:text-foreground prose-headings:text-foreground prose-pre:bg-secondary/50 prose-pre:border prose-pre:border-border/40 max-w-none text-[clamp(0.75rem,0.68rem+0.3vw,0.875rem)] text-foreground/90">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{cleaned}</ReactMarkdown>
        </div>
        {message.motionNotice && (
          // Muted and small: an aside about the 3D clip, not part of the
          // answer. It disappears on its own once the avatar starts moving.
          <p className="mt-1 text-[0.7rem] italic text-muted-foreground">
            {message.motionNotice}
          </p>
        )}
        <AssistantActions
          content={message.content}
          audioUrl={message.audioUrl}
          speechPending={message.speechPending}
          autoplay={message.autoplay}
          isStreaming={isStreaming}
        />
      </div>
    )
  }

  return (
    <div className="group flex px-2 md:px-4 py-0.5 md:py-1 animate-message-in flex-row-reverse w-full max-w-full">
      <div className="w-fit max-w-[80%]">
        {(message.content || !message.audioUrl) && (
          <div className="min-w-0 rounded-2xl px-2.5 md:px-3 py-1 md:py-2 text-[clamp(0.75rem,0.68rem+0.3vw,0.875rem)] leading-relaxed bg-primary text-primary-foreground">
            <p className="whitespace-pre-wrap break-words">{message.content}</p>
          </div>
        )}
        {message.audioUrl && (
          <div className="mt-1 rounded-xl overflow-hidden bg-secondary/40 p-1">
            <audio controls src={message.audioUrl} className="w-full h-8" preload="metadata" />
          </div>
        )}
        <UserCopyAction content={message.content} />
      </div>
    </div>
  )
}

function AssistantActions({
  content,
  audioUrl,
  speechPending,
  autoplay,
  isStreaming,
}: {
  content: string
  audioUrl?: string
  speechPending?: boolean
  autoplay?: boolean
  isStreaming?: boolean
}) {
  const { copied, handleCopy } = useCopy(content)
  const [liked, setLiked] = useState(false)
  const [disliked, setDisliked] = useState(false)

  if (isStreaming) return null

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
      <button className={btnClass} onClick={handleLike} title="ThÃ­ch">
        <ThumbsUp className={`${iconSize} ${liked ? 'text-green-500' : ''}`} />
      </button>
      <button className={btnClass} onClick={handleDislike} title="KhÃ´ng thÃ­ch">
        <ThumbsDown className={`${iconSize} ${disliked ? 'text-blue-500' : ''}`} />
      </button>
      <AudioButton
        audioUrl={audioUrl}
        text={content}
        speechPending={speechPending}
        autoplay={autoplay}
        btnClass={btnClass}
        iconSize={iconSize}
      />
    </div>
  )
}

/**
 * Speaker control for one assistant message.
 *
 * `audioUrl` is present only when the reply was voiced automatically (the
 * "Tráº£ lá»i báº±ng giá»ng nÃ³i" toggle). Otherwise the first click synthesises on
 * demand â€” which is also the only way a restored conversation can be heard,
 * since the WAV files are not persisted with the transcript.
 */
function AudioButton({
  audioUrl,
  text,
  speechPending,
  autoplay,
  btnClass,
  iconSize,
}: {
  audioUrl?: string
  text: string
  speechPending?: boolean
  autoplay?: boolean
  btnClass: string
  iconSize: string
}) {
  const { avatarRef } = useMotion()
  const audioRef = useRef<HTMLAudioElement | null>(null)
  const barRef = useRef<HTMLDivElement | null>(null)
  const rafRef = useRef<number>(0)
  const abortRef = useRef<AbortController | null>(null)
  const autoplayedRef = useRef(false)
  /** Non-zero while this clip owns the avatar's mouth. */
  const speakerRef = useRef(0)
  const [playing, setPlaying] = useState(false)
  const [paused, setPaused] = useState(false)
  const [progress, setProgress] = useState(0)
  const [url, setUrl] = useState<string | undefined>(audioUrl)
  const [synthesizing, setSynthesizing] = useState(false)
  const [failed, setFailed] = useState(false)

  /* An automatically-voiced reply arrives *after* the message is first rendered
   * (speech_ready lands seconds later), so the prop has to be adopted when it
   * changes â€” and the old <audio> discarded, or the element keeps the stale
   * source and plays the previous answer. */
  useEffect(() => {
    if (!audioUrl || audioUrl === url) return
    audioRef.current?.pause()
    audioRef.current = null
    setUrl(audioUrl)
    setPlaying(false)
    setPaused(false)
    setProgress(0)
  }, [audioUrl, url])

  const tick = useCallback(() => {
    const a = audioRef.current
    if (!a) return
    setProgress((a.currentTime / a.duration) * 100 || 0)
    rafRef.current = requestAnimationFrame(tick)
  }, [])

  /** Release the avatar's mouth. Safe to call when it was never taken. */
  const releaseMouth = useCallback(() => {
    stopSpeaking(speakerRef.current, avatarRef.current)
    speakerRef.current = 0
  }, [avatarRef])

  const handleEnded = useCallback(() => {
    setPlaying(false)
    setPaused(false)
    setProgress(0)
    cancelAnimationFrame(rafRef.current)
    releaseMouth()
    if (audioRef.current) audioRef.current.currentTime = 0
  }, [releaseMouth])

  /** Stop for good: unlike the play/pause toggle this also rewinds, so the next
   *  click on the speaker starts from the beginning instead of resuming. */
  const handleStop = useCallback(() => {
    const a = audioRef.current
    if (!a) return
    a.pause()
    a.currentTime = 0
    setPlaying(false)
    setPaused(false)
    setProgress(0)
    cancelAnimationFrame(rafRef.current)
    releaseMouth()
  }, [releaseMouth])

  useEffect(() => {
    return () => {
      cancelAnimationFrame(rafRef.current)
      audioRef.current?.pause()
      abortRef.current?.abort()
      releaseMouth()
    }
  }, [releaseMouth])

  const play = useCallback(
    async (src: string) => {
      if (!audioRef.current) {
        // createSpeechAudio, not `new Audio(src)`: the element has to be
        // CORS-clean *before* it loads or the lip-sync analyser reads silence.
        audioRef.current = createSpeechAudio(src)
        audioRef.current.addEventListener('ended', handleEnded)
      }
      const el = audioRef.current

      // Tap the element BEFORE playing. Once it is routed through Web Audio its
      // sound leaves via the graph, so doing this mid-playback would drop the
      // first moments of speech.
      speakerRef.current = await startSpeaking(el, avatarRef.current)

      el.play().then(
        () => {
          setPlaying(true)
          setPaused(false)
          rafRef.current = requestAnimationFrame(tick)
        },
        () => {
          // Browsers refuse to start audio without a recent user gesture. Voice
          // mode can land 40s after the click that sent the message, so this is
          // a normal outcome, not a failure: leave the speaker idle and the user
          // can press it. Never surface it as an error.
          setPlaying(false)
          releaseMouth()
        },
      )
    },
    [handleEnded, tick, avatarRef, releaseMouth],
  )

  /* Voice mode: speak the moment the audio lands. Guarded by a ref so a later
   * re-render (a sibling message updating, say) cannot replay the same clip. */
  useEffect(() => {
    if (!autoplay || !url || autoplayedRef.current) return
    autoplayedRef.current = true
    play(url)
  }, [autoplay, url, play])

  const handleToggle = async () => {
    if (playing) {
      audioRef.current?.pause()
      setPlaying(false)
      setPaused(true)
      cancelAnimationFrame(rafRef.current)
      // The mouth must stop with the sound, not keep moving over a paused clip.
      releaseMouth()
      return
    }

    if (url) {
      play(url)
      return
    }

    // Voice mode already has a job running for this message â€” clicking must not
    // queue a second synthesis of the same text.
    if (speechPending) return

    // Nothing to play yet â€” ask the server to read this message aloud.
    if (synthesizing) return
    setSynthesizing(true)
    setFailed(false)
    const controller = new AbortController()
    abortRef.current = controller
    try {
      const fresh = await speakText(text, { signal: controller.signal })
      if (controller.signal.aborted) return
      setUrl(fresh)
      play(fresh)
    } catch (e) {
      if ((e as Error).name !== 'AbortError') {
        console.warn('[TTS] on-demand synthesis failed:', e)
        setFailed(true)
      }
    } finally {
      if (!controller.signal.aborted) setSynthesizing(false)
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

  // `speechPending` is voice mode synthesising in the background; `synthesizing`
  // is this button's own on-demand request. Both mean "wait, audio is coming".
  const busy = synthesizing || !!speechPending
  const isActive = playing || paused

  return (
    <div className="group flex items-center gap-1">
      <button
        className={`${btnClass} ${isActive ? 'text-foreground' : ''} ${failed ? 'text-destructive' : ''}`}
        onClick={handleToggle}
        disabled={busy}
        title={
          busy
            ? 'Äang táº¡o giá»ng Ä‘á»c... (CPU, cÃ³ thá»ƒ máº¥t 30-45s)'
            : failed
              ? 'KhÃ´ng táº¡o Ä‘Æ°á»£c giá»ng Ä‘á»c â€” báº¥m Ä‘á»ƒ thá»­ láº¡i'
              : playing
                ? 'Táº¡m dá»«ng'
                : paused
                  ? 'Tiáº¿p tá»¥c'
                  : url
                    ? 'Nghe'
                    : 'Äá»c tin nháº¯n nÃ y'
        }
        onDoubleClick={(e) => e.preventDefault()}
      >
        {busy ? (
          <Loader2 className={`${iconSize} animate-spin`} />
        ) : playing ? (
          <Pause className={iconSize} />
        ) : (
          <Volume2 className={iconSize} />
        )}
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
      {/* Kept mounted and collapsed rather than unmounted, so appearing does not
          shove the progress bar sideways â€” same trick the bar itself uses. */}
      <button
        className={`${btnClass} overflow-hidden transition-all duration-300 ease-out ${
          isActive ? 'opacity-100' : 'w-0 p-0 opacity-0 pointer-events-none'
        }`}
        onClick={handleStop}
        title="Dá»«ng háº³n"
        tabIndex={isActive ? 0 : -1}
        aria-hidden={!isActive}
      >
        <Square className={iconSize} />
      </button>
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

