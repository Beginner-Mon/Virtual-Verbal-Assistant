import {
  createContext,
  useContext,
  useRef,
  useState,
  useCallback,
  useEffect,
  useMemo,
  type ReactNode,
} from 'react'
import type { Message } from '../components/ChatMessage'
import { getSession, listSessions, deleteSession, streamChat, type SessionMessage } from '../lib/api'
import { useMotion } from './MotionContext'
import testAudio from '../asset/audio/test.wav'

/** Key holding the *pointer* to the conversation, never the conversation.
 *
 *  Postgres stays the source of truth, so this survives a move to a hosted
 *  database untouched — and the day Cognito is switched on, `currentUserId()`
 *  starts returning the real `sub` and the same session simply belongs to a
 *  real account instead of a per-browser demo id. */
const SESSION_KEY = 'vva_session_id'

function loadOrCreateSessionId(): string {
  const existing = localStorage.getItem(SESSION_KEY)
  if (existing) return existing
  const fresh = crypto.randomUUID()
  localStorage.setItem(SESSION_KEY, fresh)
  return fresh
}

function buildInitialMessages(): Message[] {
  const hour = new Date().getHours()
  const greeting = hour < 12 ? 'Good morning' : hour < 18 ? 'Good afternoon' : 'Good evening'
  return [
    {
      id: '1',
      role: 'assistant',
      content: `${greeting}! My name is ECA, your Virtual Verbal Assistant. How can I help you today? \uD83C\uDF99\uFE0F`,
      timestamp: new Date(),
      audioUrl: testAudio,
    },
  ]
}

export interface SessionItem {
  session_id: string
  created_at: string
  updated_at: string
  first_user_message_preview: string
  message_count: number
}

export interface ChatContextType {
  messages: Message[]
  setMessages: React.Dispatch<React.SetStateAction<Message[]>>
  input: string
  setInput: (value: string) => void
  isTyping: boolean
  isGenerating: boolean
  stageLabel: string | null
  webSearch: boolean
  setWebSearch: (value: boolean) => void
  voiceReply: boolean
  setVoiceReply: (value: boolean) => void
  isRestoring: boolean
  startNewSession: () => void
  handleSend: () => Promise<void>
  handleStop: () => void
  imageUrls: string[]
  addImage: (file: File) => void
  removeImage: (index: number) => void
  sessionList: SessionItem[]
  sessionsDirty: boolean
  activeSessionId: string
  refreshSessions: () => Promise<void>
  switchToSession: (sessionId: string) => Promise<void>
  deleteSessionAction: (sessionId: string) => Promise<void>
  markSessionsClean: () => void
}

const ChatContext = createContext<ChatContextType | null>(null)

export function ChatProvider({ children }: { children: ReactNode }) {
  const { transitionTo } = useMotion()

  const [messages, setMessages] = useState<Message[]>(() => buildInitialMessages())
  const [input, setInput] = useState('')
  const [isTyping, setIsTyping] = useState(false)
  const [isGenerating, setIsGenerating] = useState(false)
  const [stageLabel, setStageLabel] = useState<string | null>(null)
  const [webSearch, setWebSearch] = useState(false)
  // Off by default: VieNeu runs on CPU and takes ~10-20s for a long Vietnamese
  // answer, and the backend holds the SSE stream open until it finishes.
  const [voiceReply, setVoiceReply] = useState(false)
  const [isRestoring, setIsRestoring] = useState(true)
  const [imageUrls, setImageUrls] = useState<string[]>([])
  const imageUrlsRef = useRef<string[]>([])

  const [sessionList, setSessionList] = useState<SessionItem[]>([])
  const [sessionsDirty, setSessionsDirty] = useState(true)
  const switchingRef = useRef(false)

  const inputRef = useRef(input)
  inputRef.current = input
  const isGeneratingRef = useRef(isGenerating)
  isGeneratingRef.current = isGenerating

  const sessionIdRef = useRef<string>(loadOrCreateSessionId())
  const [activeSessionId, setActiveSessionId] = useState<string>(() => loadOrCreateSessionId())
  const abortControllerRef = useRef<AbortController | null>(null)
  const thinkingRef = useRef(false)

  const addImage = useCallback((file: File) => {
    const url = URL.createObjectURL(file)
    imageUrlsRef.current = [...imageUrlsRef.current, url]
    setImageUrls((prev) => [...prev, url])
  }, [])

  const removeImage = useCallback((index: number) => {
    const urls = imageUrlsRef.current
    URL.revokeObjectURL(urls[index])
    const next = [...urls]
    next.splice(index, 1)
    imageUrlsRef.current = next
    setImageUrls(next)
  }, [])

  /* ── Restore the conversation this browser was last in ──────────────────
   *
   * Runs once on mount. A 404 is the normal case, not an error: a brand-new
   * session id has no row until the first turn is written, so we simply keep
   * the greeting. Anything else is logged and also falls back to the greeting —
   * a failed restore must never leave the user staring at an empty panel. */
  useEffect(() => {
    let cancelled = false

    ;(async () => {
      try {
        const data = await getSession(sessionIdRef.current)
        const history = (data?.messages ?? []) as SessionMessage[]
        if (cancelled || history.length === 0) return
        setMessages(
          history.map((m, i) => ({
            id: `restored-${i}`,
            role: m.role,
            content: m.content,
            timestamp: new Date(m.timestamp),
            // Audio is not persisted — the WAV lives on the TTS box under a
            // random name. The per-message speaker button can re-synthesise it.
          })),
        )
      } catch (e) {
        const status = (e as { response?: { status?: number } }).response?.status
        if (status !== 404) console.warn('[session] restore failed:', e)
      } finally {
        if (!cancelled) setIsRestoring(false)
      }
    })()

    return () => {
      cancelled = true
    }
  }, [])

  /** Abandon this conversation and start a clean one.
   *
   * The old one is not deleted — it stays in Postgres and will be reachable
   * once the Sessions panel is wired up. */
  const startNewSession = useCallback(() => {
    abortControllerRef.current?.abort()
    const fresh = crypto.randomUUID()
    localStorage.setItem(SESSION_KEY, fresh)
    sessionIdRef.current = fresh
    setActiveSessionId(fresh)
    setMessages(buildInitialMessages())
    setInput('')
    setIsTyping(false)
    setStageLabel(null)
    setIsGenerating(false)
    setSessionsDirty(true)
  }, [])

  const refreshSessions = useCallback(async () => {
    try {
      const data = await listSessions()
      const sessions = (data?.sessions ?? []) as SessionItem[]
      setSessionList(sessions)
      setSessionsDirty(false)
    } catch (e) {
      console.warn('[sessions] refresh failed:', e)
    }
  }, [])

  const markSessionsClean = useCallback(() => {
    setSessionsDirty(false)
  }, [])

  const endThinking = useCallback(() => {
    if (!thinkingRef.current) return
    thinkingRef.current = false
    void transitionTo('thinking_outro')
  }, [transitionTo])

  const switchToSession = useCallback(async (sessionId: string) => {
    if (sessionId === sessionIdRef.current || switchingRef.current) return
    switchingRef.current = true
    abortControllerRef.current?.abort()

    try {
      const data = await getSession(sessionId)
      const history = (data?.messages ?? []) as SessionMessage[]
      sessionIdRef.current = sessionId
      setActiveSessionId(sessionId)
      localStorage.setItem(SESSION_KEY, sessionId)
      setInput('')
      setIsTyping(false)
      setStageLabel(null)
      setIsGenerating(false)
      endThinking()

      if (history.length > 0) {
        setMessages(
          history.map((m, i) => ({
            id: `switched-${i}`,
            role: m.role,
            content: m.content,
            timestamp: new Date(m.timestamp),
          }))
        )
      } else {
        setMessages(buildInitialMessages())
      }
    } catch (e) {
      console.warn('[session] switch failed:', e)
    } finally {
      switchingRef.current = false
    }
  }, [endThinking])

  const deleteSessionAction = useCallback(async (sessionId: string) => {
    try {
      await deleteSession(sessionId)
      setSessionList((prev) => prev.filter((s) => s.session_id !== sessionId))
      if (sessionId === sessionIdRef.current) {
        startNewSession()
      }
    } catch (e) {
      console.warn('[session] delete failed:', e)
    }
  }, [startNewSession])

  const handleStop = useCallback(() => {
    abortControllerRef.current?.abort()
    setIsTyping(false)
    setStageLabel(null)
    setIsGenerating(false)
    endThinking()
  }, [endThinking])

  const handleSend = useCallback(async () => {
    const text = inputRef.current.trim()
    if (!text || isGeneratingRef.current) return

    const userMsg: Message = {
      id: crypto.randomUUID(),
      role: 'user',
      content: text,
      timestamp: new Date(),
    }

    setMessages((prev) => [...prev, userMsg])
    setInput('')
    setIsTyping(true)
    setIsGenerating(true)
    thinkingRef.current = true
    void transitionTo('thinking_intro')

    const assistantMsgId = crypto.randomUUID()
    setMessages((prev) => [
      ...prev,
      { id: assistantMsgId, role: 'assistant', content: '', timestamp: new Date() },
    ])

    const controller = new AbortController()
    abortControllerRef.current = controller
    /** True while this stream is still the newest one.
     *
     *  With voice on, the backend keeps the stream open long after the text is
     *  done, and we release the composing UI at `speech_pending` — so the user
     *  can send a second message while this one is still streaming. Without this
     *  check the old stream's `done` would clear `isGenerating` for the NEW
     *  reply. Message updates are exempt: they target their own `assistantMsgId`
     *  and stay correct however late they land. */
    const isCurrent = () => abortControllerRef.current === controller

    const STAGE_SEARCHING = '\uD83D\uDD0D \u0110ang t\xECm ki\u1EBFm th\xF4ng tin...'
    const STAGE_COMPOSING = '\u270D\uFE0F \u0110ang so\u1EA1n c\xE2u tr\u1EA3 l\u1EDDi...'

    try {
      await streamChat(
        {
          query: text,
          sessionId: sessionIdRef.current,
          webSearch,
          outputMode: voiceReply ? 'both' : 'text',
        },
        (type, data) => {
          if (type === 'stage') {
            const { node, status } = data as { node: string; status: string }
            if (node === 'planner' && status === 'complete') {
              setIsTyping(false)
              setStageLabel(STAGE_SEARCHING)
            } else if (node === 'retriever_agent' && status === 'complete') {
              setStageLabel(STAGE_COMPOSING)
            } else if (node === 'synthesizer' && status === 'started') {
              setStageLabel(null)
            }
          } else if (type === 'token') {
            setIsTyping(false)
            setStageLabel(null)
            endThinking()
            const content = (data as { content: string }).content
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === assistantMsgId ? { ...msg, content: msg.content + content } : msg
              )
            )
          } else if (type === 'speech_pending') {
            // Spin the speaker on this message. Without it the toggle has no
            // visible effect at all during the 30-45s wait, which reads exactly
            // like it is broken.
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === assistantMsgId ? { ...msg, speechPending: true } : msg
              )
            )
            // The answer text is already complete here — the backend only holds
            // the stream open to poll Redis for the audio (up to 130s). Release
            // the composing UI now, otherwise the stop button lingers for two
            // minutes after the reply is fully readable.
            if (!isCurrent()) return
            setStageLabel(null)
            setIsGenerating(false)
            endThinking()
          } else if (type === 'speech_ready') {
            const { url } = data as { url?: string }
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === assistantMsgId
                  // `autoplay` is what makes voice mode audible. Setting only
                  // `audioUrl` attaches a file nobody plays, which is why the
                  // toggle looked identical whether it was on or off.
                  ? { ...msg, audioUrl: url, speechPending: false, autoplay: !!url }
                  : msg
              )
            )
          } else if (type === 'speech_failed') {
            // Text is still there and readable — a missing voice is not worth
            // an error bubble. Surface it in the console for diagnosis only.
            console.warn('[TTS]', (data as { error?: string }).error ?? 'speech failed')
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === assistantMsgId ? { ...msg, speechPending: false } : msg
              )
            )
          } else if (type === 'done') {
            if (!isCurrent()) return
            setStageLabel(null)
            setIsGenerating(false)
            setSessionsDirty(true)
          }
        },
        controller.signal,
      )
    } catch (e) {
      if ((e as Error).name !== 'AbortError') {
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === assistantMsgId
              ? { ...msg, content: 'Sorry, something went wrong. Please try again.' }
              : msg
          )
        )
      }
    } finally {
      // Same reason as `done`: a superseded stream must not reset the UI that
      // now belongs to a newer message.
      if (isCurrent()) {
        setIsTyping(false)
        setStageLabel(null)
        setIsGenerating(false)
        endThinking()
      }
    }
  }, [webSearch, voiceReply, transitionTo, endThinking])

  useEffect(() => {
    return () => {
      imageUrlsRef.current.forEach((url) => URL.revokeObjectURL(url))
      imageUrlsRef.current = []
    }
  }, [])

  const value = useMemo<ChatContextType>(
    () => ({
      messages,
      setMessages,
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
      sessionList,
      sessionsDirty,
      activeSessionId,
      refreshSessions,
      switchToSession,
      deleteSessionAction,
      markSessionsClean,
    }),
    [messages, input, isTyping, isGenerating, stageLabel, webSearch, voiceReply, isRestoring, startNewSession, handleSend, handleStop, imageUrls, addImage, removeImage, sessionList, sessionsDirty, activeSessionId, refreshSessions, switchToSession, deleteSessionAction, markSessionsClean],
  )

  return <ChatContext.Provider value={value}>{children}</ChatContext.Provider>
}

export function useChat(): ChatContextType {
  const ctx = useContext(ChatContext)
  if (!ctx) throw new Error('useChat must be used within ChatProvider')
  return ctx
}
