import {
  createContext,
  useContext,
  useRef,
  useState,
  useCallback,
  useMemo,
  type ReactNode,
} from 'react'
import type { Message } from '../components/ChatMessage'
import { streamChat } from '../lib/api'
import { useMotion } from './MotionContext'

function buildInitialMessages(): Message[] {
  const hour = new Date().getHours()
  const greeting = hour < 12 ? 'Good morning' : hour < 18 ? 'Good afternoon' : 'Good evening'
  return [
    {
      id: '1',
      role: 'assistant',
      content: `${greeting}! My name is ECA, your Virtual Verbal Assistant. How can I help you today? \uD83C\uDF99\uFE0F`,
      timestamp: new Date(),
    },
  ]
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
  handleSend: () => Promise<void>
  handleStop: () => void
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

  const inputRef = useRef(input)
  inputRef.current = input
  const isGeneratingRef = useRef(isGenerating)
  isGeneratingRef.current = isGenerating

  const sessionIdRef = useRef<string>(crypto.randomUUID())
  const abortControllerRef = useRef<AbortController | null>(null)
  const thinkingRef = useRef(false)

  const endThinking = useCallback(() => {
    if (!thinkingRef.current) return
    thinkingRef.current = false
    void transitionTo('thinking_outro')
  }, [transitionTo])

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

    abortControllerRef.current = new AbortController()

    const STAGE_SEARCHING = '\uD83D\uDD0D \u0110ang t\xECm ki\u1EBFm th\xF4ng tin...'
    const STAGE_COMPOSING = '\u270D\uFE0F \u0110ang so\u1EA1n c\xE2u tr\u1EA3 l\u1EDDi...'

    try {
      await streamChat(
        { query: text, sessionId: sessionIdRef.current, webSearch },
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
          } else if (type === 'done') {
            setStageLabel(null)
            setIsGenerating(false)
          }
        },
        abortControllerRef.current.signal,
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
      setIsTyping(false)
      setStageLabel(null)
      setIsGenerating(false)
      endThinking()
    }
  }, [webSearch, transitionTo, endThinking])

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
      handleSend,
      handleStop,
    }),
    [messages, input, isTyping, isGenerating, stageLabel, webSearch, handleSend, handleStop],
  )

  return <ChatContext.Provider value={value}>{children}</ChatContext.Provider>
}

export function useChat(): ChatContextType {
  const ctx = useContext(ChatContext)
  if (!ctx) throw new Error('useChat must be used within ChatProvider')
  return ctx
}
