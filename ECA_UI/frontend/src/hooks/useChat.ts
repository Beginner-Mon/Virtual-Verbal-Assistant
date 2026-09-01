import { createContext, useContext } from 'react'
import type { Message } from '../components/ChatMessage'
import type { UiStrings } from '../lib/characterCopy'

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
  /** Chat-surface copy for the selected character (greeting, placeholder,
   *  stage labels, error line). Always fully populated — see uiStringsFor. */
  ui: UiStrings
  webSearch: boolean
  setWebSearch: (value: boolean) => void
  voiceReply: boolean
  setVoiceReply: (value: boolean) => void
  isRestoring: boolean
  isSwitching: boolean
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
  // Audio recording (frontend only, click toggle)
  isRecording: boolean
  recordingDuration: number
  recordingError: string | null
  previewAudioUrl: string | null
  startRecord: () => Promise<void>
  stopRecord: () => void
  cancelRecord: () => void
  sendAudio: () => void
}

export const ChatContext = createContext<ChatContextType | null>(null)

export function useChat(): ChatContextType {
  const ctx = useContext(ChatContext)
  if (!ctx) throw new Error('useChat must be used within ChatProvider')
  return ctx
}
