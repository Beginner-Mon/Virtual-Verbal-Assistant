/* eslint-disable react-hooks/refs, react-hooks/set-state-in-effect */
import {
  useRef,
  useState,
  useCallback,
  useEffect,
  useMemo,
  type ReactNode,
} from 'react'
import type { Message } from '../components/ChatMessage'
import {
  getSession, listSessions, deleteSession, streamChat, fetchMotionStatus,
  type SessionMessage,
} from '../lib/api'
import { pollMotionJob } from '../lib/motionJob'
import { clearSessionPointer, readSessionPointer, stampSessionPointer } from '../lib/chatSession'
import { useMotion } from '../hooks/useMotion'
import { ChatContext, type ChatContextType, type SessionItem } from '../hooks/useChat'
import { uiStringsFor, getGreeting, type UiStrings } from '../lib/characterCopy'
import { withGreeting } from '../lib/greeting'
import { useLocale } from '../hooks/useLocale'

export type { SessionItem, ChatContextType } from '../hooks/useChat'
import { useAudioRecorder } from '../hooks/useAudioRecorder'

/* The pointer to the conversation lives in lib/chatSession.ts, along with the
 * reason it is written late and expires. Nothing here mints an id: a
 * conversation gets one when its first message is sent, and until then there is
 * nothing to point at and nothing to restore. */

/** Id of the opening message, so the greeting can be recognised and replaced
 *  when the user switches character before saying anything. */
const GREETING_ID = '1'

/**
 * The character's own opening line.
 *
 * Previously this was a time-of-day English greeting from "ECA" with a
 * microphone emoji, identical whichever avatar was on screen \u2014 the first thing
 * a user read contradicted the character they had just picked.
 *
 * The old version also attached `testAudio`, a bundled test WAV, so the speaker
 * button on the first message played a fixed clip that had nothing to do with
 * the text. That was debug scaffolding and is gone.
 */
function buildInitialMessages(ui: UiStrings): Message[] {
  return [
    {
      id: GREETING_ID,
      role: 'assistant',
      content: getGreeting(ui),
      timestamp: new Date(),
    },
  ]
}



export function ChatProvider({ children }: { children: ReactNode }) {
  const { transitionTo, selectedVrmId, vrmOptions, playMotionFile, registerSessionMotion } =
    useMotion()

  /** Copy for whoever is on screen, in the language the site is being read in.
   *  Falls back to neutral strings until the catalog resolves, so nothing
   *  renders "undefined" on a cold load. */
  const { locale } = useLocale()
  const ui = useMemo(
    () => uiStringsFor(vrmOptions.find((o) => o.id === selectedVrmId)?.character, locale),
    [vrmOptions, selectedVrmId, locale],
  )
  const uiRef = useRef<UiStrings>(ui)
  uiRef.current = ui

  // Neutral copy, not this character's — the catalog has not resolved on the
  // first render and a wrong name is worse than no name. The effect below swaps
  // in the character's own greeting once `selectedVrmId` settles. Passing no
  // character is what selects that neutral set, now per locale.
  const [messages, setMessages] = useState<Message[]>(() =>
    buildInitialMessages(uiStringsFor(null, locale)),
  )
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
  const [isSwitching, setIsSwitching] = useState(false)
  const switchingRef = useRef(false)

  // Audio recording — frontend only (no backend)
  const { isRecording, duration: recordingDuration, audioUrl: previewAudioUrl, audioBlob: previewAudioBlob, error: recordingError, start: startRecord, stop: stopRecord, cancel: cancelRecord } = useAudioRecorder()

  const sendAudio = useCallback(() => {
    if (!previewAudioBlob || !previewAudioUrl) return
    const url = URL.createObjectURL(previewAudioBlob)
    const msg: Message = {
      id: crypto.randomUUID(),
      role: 'user',
      content: '',
      timestamp: new Date(),
      audioUrl: url,
    }
    setMessages((prev) => [...prev, msg])
    // clear preview but keep sent message url
    // useAudioRecorder's cancel would revoke previewUrl — do it manually
    // we already created a new url for the message, so revoke preview
    URL.revokeObjectURL(previewAudioUrl)
    // reset recorder preview (call cancel without revoking again)
    // hack: clear via internal state by calling cancel then restoring? simpler: just let hook clear on next start
    // Instead, we manually clear by calling cancel and then set new preview to null via effect
    // For now, just clear preview url via hook's cancel (will revoke again harmlessly if already revoked)
    cancelRecord()
  }, [previewAudioBlob, previewAudioUrl, cancelRecord])

  const inputRef = useRef(input)
  inputRef.current = input
  const isGeneratingRef = useRef(isGenerating)
  isGeneratingRef.current = isGenerating

  // Read once, not twice: this used to call a minting function in both places.
  // Null means "no conversation yet" — either a first visit, or the last one
  // aged out. Neither is an error and neither is worth a request.
  const initialSessionId = useRef(readSessionPointer()).current
  const sessionIdRef = useRef<string | null>(initialSessionId)
  const [activeSessionId, setActiveSessionId] = useState<string | null>(initialSessionId)

  /** The id to send this turn under, minting one if this is the first message.
   *
   *  This is the only place an id is created. Doing it here rather than at load
   *  is what closes the window in which a pointer existed and the row behind it
   *  did not — the window every clean page load used to spend a 404 in. */
  const ensureSessionId = useCallback((): string => {
    const id = sessionIdRef.current ?? crypto.randomUUID()
    sessionIdRef.current = id
    stampSessionPointer(id)
    setActiveSessionId(id)
    return id
  }, [])
  const abortControllerRef = useRef<AbortController | null>(null)
  const thinkingRef = useRef(false)

  // Keep the initial greeting immutable after first hydration.
  // - First real character after mount ('' -> 'anne' after catalog) may adopt
  //   once if still only the fallback greeting is shown.
  // - Subsequent switches do NOT mutate nor append — greeting gốc giữ nguyên.
  //   Dấu hiệu đổi character và việc đổi backend role 'assistant' -> tên model
  //   sẽ làm ở phase sau (hiện giữ nguyên).
  const prevVrmRef = useRef<string>('')
  const hydratedRef = useRef<boolean>(false)

  useEffect(() => {
    if (isRestoring) return
    if (!selectedVrmId) return

    if (!hydratedRef.current) {
      hydratedRef.current = true
      if (prevVrmRef.current === '') {
        setMessages((prev) => withGreeting(prev, GREETING_ID, getGreeting(uiRef.current)))
        prevVrmRef.current = selectedVrmId
        return
      }
    }

    // Subsequent character switches: intentionally no-op — keep original greeting.
    // Only track for future use (e.g., when we switch to storing model name).
    if (prevVrmRef.current !== selectedVrmId) {
      prevVrmRef.current = selectedVrmId
    }
  }, [selectedVrmId, isRestoring])

  /**
   * Re-say the opening line when the site language changes.
   *
   * The greeting is built in the `useState` initialiser above, which runs once
   * per mount — so before this, switching language left the assistant's first
   * message in the old one until a reload. Every other string on the page
   * changed instantly, which made that one bubble look broken.
   *
   * `withGreeting` decides whether it may be rewritten at all: only while it is
   * the only message. Once a real conversation sits below it, the screen is a
   * record and the top of it is left alone — the same rule the character switch
   * above already follows.
   */
  useEffect(() => {
    setMessages((prev) => withGreeting(prev, GREETING_ID, getGreeting(uiRef.current)))
  }, [locale])

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

    // No pointer, nothing to restore. Note the setIsRestoring(false): the flag
    // is cleared in the `finally` below, so returning without it would leave it
    // true forever and the greeting effect, which waits on it, would never run.
    if (!sessionIdRef.current) {
      setIsRestoring(false)
      return
    }
    const sessionId = sessionIdRef.current

    ;(async () => {
      try {
        const data = await getSession(sessionId)
        const history = (data?.messages ?? []) as SessionMessage[]
        if (cancelled || history.length === 0) return
        setMessages([
          // The greeting is always the first message — it is not persisted in
          // the backend, so we prepend it client-side on every restore.
          ...buildInitialMessages(uiRef.current),
          ...history.map((m, i) => ({
            id: `restored-${i}`,
            role: m.role,
            content: m.content,
            timestamp: new Date(m.timestamp),
            // Audio is not persisted — the WAV lives on the TTS box under a
            // random name. The per-message speaker button can re-synthesise it.
            //
            // Motion IS: the job id rides on the message row, so a refresh
            // does not lose a render the GPU already paid for. Carried here
            // and acted on by the effect below.
            motionJobId: m.motion_job_id,
            motionExpiresAt: m.motion_expires_at,
            // The user's own words for this turn — the picker lists motions by
            // what was asked for, and on a restore the question is the message
            // immediately before the answer.
            motionLabel: m.role === 'assistant' ? history[i - 1]?.content : undefined,
          })),
        ])

        // Motions the GPU already rendered for this conversation. Two outcomes
        // and no third: still fetchable, or gone.
        //
        // Nothing is played. A restore is not the moment to start animating —
        // the avatar may not even have loaded yet, and replaying an answer the
        // user read yesterday is not what they came back for. They go into the
        // replay picker, and the user chooses.
        history.forEach((m, i) => {
          if (m.role !== 'assistant' || !m.motion_job_id) return
          // Deadline, not a stored verdict: this payload may have been sitting
          // in the tab for hours. Compare against the clock NOW. No expiry at
          // all means assume gone — see session_store.motion_expires_at.
          const alive = m.motion_expires_at
            ? new Date(m.motion_expires_at) > new Date()
            : false
          if (alive) {
            registerSessionMotion({
              jobId: m.motion_job_id,
              // Deliberately no url. A signed URL lives five minutes, and this
              // page has no cached clip — fetching one now would hand the
              // picker a dead link. It resolves a fresh one when picked.
              label: history[i - 1]?.content ?? '',
            })
          } else {
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === `restored-${i}`
                  ? { ...msg, motionNotice: uiRef.current.motion_gone }
                  : msg
              )
            )
          }
        })
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
  }, [registerSessionMotion])

  /** Abandon this conversation and start a clean one.
   *
   * The old one is not deleted — it stays in Postgres and will be reachable
   * once the Sessions panel is wired up. */
  const startNewSession = useCallback(() => {
    abortControllerRef.current?.abort()
    // Clear the pointer rather than mint a new one. Minting here is what left
    // an id behind for anyone who opened a new chat and never typed in it —
    // and that id 404'd on every load from then on, permanently. The next
    // message creates one.
    clearSessionPointer()
    sessionIdRef.current = null
    setActiveSessionId(null)
    setMessages(buildInitialMessages(uiRef.current))
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

    // Đổi UI ngay: không hiện session cũ trong lúc load session mới
    setIsSwitching(true)
    setActiveSessionId(sessionId)
    sessionIdRef.current = sessionId
    // Picking a conversation out of the list makes it the one you are in, so
    // its clock starts now however old the conversation itself is.
    stampSessionPointer(sessionId)
    setMessages([])
    setInput('')
    setIsTyping(false)
    setStageLabel(null)
    setIsGenerating(false)
    endThinking()

    try {
      const data = await getSession(sessionId)
      // Race: user đã bấm session khác trong lúc await
      if (sessionId !== sessionIdRef.current) return
      const history = (data?.messages ?? []) as SessionMessage[]

      if (history.length > 0) {
        setMessages([
          // Prepend the greeting — it is not stored in the backend.
          ...buildInitialMessages(uiRef.current),
          ...history.map((m, i) => ({
            id: `switched-${i}`,
            role: m.role,
            content: m.content,
            timestamp: new Date(m.timestamp),
          })),
        ])
      } else {
        setMessages(buildInitialMessages(uiRef.current))
      }
    } catch (e) {
      const isAbort = (e as { name?: string })?.name === 'AbortError' || (e as { code?: string })?.code === 'ERR_CANCELED'
      if (isAbort) return
      console.warn('[session] switch failed:', e)
    } finally {
      // Chỉ clear nếu vẫn đang ở session này (tránh đè isSwitching của session mới)
      if (sessionId === sessionIdRef.current) setIsSwitching(false)
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

    /* Captured once per send rather than read per event: switching character
     * mid-stream must not swap the label under a reply already being written,
     * and the error line below has to match the character who greeted the
     * user. */
    const copy = uiRef.current
    const STAGE_SEARCHING = copy.stage_searching
    const STAGE_COMPOSING = copy.stage_composing

    try {
      await streamChat(
        {
          query: text,
          sessionId: ensureSessionId(),
          webSearch,
          outputMode: voiceReply ? 'both' : 'text',
          // The selected character IS the persona: characters.slug is what the
          // backend caches personas under, and ChatRequest.persona_id already
          // accepts exactly this shape. Picking an avatar changes how the
          // assistant speaks, which until now it did not.
          personaId: selectedVrmId || undefined,
          // Same locale that renders this page. It selects the character's voice
          // and any safety warning the backend inserts — both are text a person
          // reads verbatim, so a declared choice beats a detected one.
          locale,
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
          } else if (type === 'motion') {
            // The agent enqueues a render and returns straight away; the GPU
            // takes a few seconds. This event carries the job id (or the reason
            // there is none), so the browser can poll for it and play the clip
            // on the avatar when it lands.
            //
            // Fire-and-forget on purpose: this callback is synchronous and the
            // rest of the reply must keep streaming while the render runs.
            // Nothing here touches `isGenerating` or `stageLabel` — a motion is
            // an extra on top of an answer, and must never hold the composing
            // UI open or block `done`.
            const m = data as {
              state: string
              job_id?: string
              retry_after_seconds?: number
            }
            const notice = (text: string | undefined) =>
              setMessages((prev) =>
                prev.map((msg) =>
                  msg.id === assistantMsgId ? { ...msg, motionNotice: text } : msg
                )
              )

            if (m.state === 'unavailable') {
              notice(copy.motion_unavailable)
            } else if (m.state === 'busy') {
              notice(copy.motion_busy)
            } else if (m.job_id) {
              const jobId = m.job_id
              notice(copy.motion_rendering)
              void (async () => {
                try {
                  const url = await pollMotionJob(
                    jobId,
                    (id) => fetchMotionStatus(id, controller.signal),
                    { signal: controller.signal },
                  )
                  // job_id is the cache key: the URL is a CloudFront signature
                  // that differs on every fetch, so keying on it would re-fetch
                  // and re-retarget the same clip each replay.
                  // `text` is what the user typed, so the motion picker lists
                  // "động tác squat" rather than a hash nobody can read.
                  await playMotionFile(url, jobId, text)
                  notice(undefined)
                } catch (e) {
                  if ((e as Error).name === 'AbortError') return
                  console.warn('[motion]', e)
                  notice(copy.motion_failed)
                }
              })()
            }
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
              ? { ...msg, content: copy.error_stream }
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
    // `locale` belongs here: switching language mid-session must reach the next
    // request. Without it the closure keeps whichever locale was active when the
    // callback was created, and the backend would keep serving the old
    // character voice and the old safety-warning language.
  }, [webSearch, voiceReply, selectedVrmId, locale, transitionTo, endThinking, playMotionFile, ensureSessionId])

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
      ui,
      webSearch,
      setWebSearch,
      voiceReply,
      setVoiceReply,
      isRestoring,
      isSwitching,
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
      isRecording,
      recordingDuration,
      recordingError,
      previewAudioUrl,
      startRecord,
      stopRecord,
      cancelRecord,
      sendAudio,
    }),
    [messages, input, isTyping, isGenerating, stageLabel, ui, webSearch, voiceReply, isRestoring, isSwitching, startNewSession, handleSend, handleStop, imageUrls, addImage, removeImage, sessionList, sessionsDirty, activeSessionId, refreshSessions, switchToSession, deleteSessionAction, markSessionsClean, isRecording, recordingDuration, recordingError, previewAudioUrl, startRecord, stopRecord, cancelRecord, sendAudio],
  )

  return <ChatContext.Provider value={value}>{children}</ChatContext.Provider>
}

