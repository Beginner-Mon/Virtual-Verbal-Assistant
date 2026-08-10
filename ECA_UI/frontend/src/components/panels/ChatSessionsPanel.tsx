import { useState, useEffect, useCallback, useRef } from 'react'
import { MessageSquare, Trash2, RefreshCw, Loader2 } from 'lucide-react'
import { ScrollArea } from '../ui/scroll-area'
import ConfirmDialog from '../ui/confirm-dialog'
import { useChat, type SessionItem } from '../../contexts/ChatContext'

let _cacheLoaded = false

export default function ChatSessionsPanel({ onSessionSelected }: { onSessionSelected?: () => void }) {
  const {
    sessionList,
    sessionsDirty,
    activeSessionId,
    refreshSessions,
    switchToSession,
    deleteSessionAction,
  } = useChat()

  const [loading, setLoading] = useState(false)
  const [deleteTarget, setDeleteTarget] = useState<SessionItem | null>(null)
  const [refreshing, setRefreshing] = useState(false)
  const didInitialLoad = useRef(false)

  const load = useCallback(async () => {
    setLoading(true)
    await refreshSessions()
    _cacheLoaded = true
    setLoading(false)
  }, [refreshSessions])

  useEffect(() => {
    if (!didInitialLoad.current && !_cacheLoaded) {
      didInitialLoad.current = true
      load()
    }
  }, [load])

  useEffect(() => {
    if (didInitialLoad.current && sessionsDirty) {
      load()
    }
  }, [sessionsDirty, load])

  const handleRefresh = async () => {
    setRefreshing(true)
    await refreshSessions()
    _cacheLoaded = true
    setRefreshing(false)
  }

  const handleDelete = (s: SessionItem) => {
    setDeleteTarget(s)
  }

  const confirmDelete = async () => {
    if (!deleteTarget) return
    await deleteSessionAction(deleteTarget.session_id)
    setDeleteTarget(null)
    await refreshSessions()
    _cacheLoaded = true
  }

  return (
    <div className="flex flex-col h-full">
      <div className="px-4 py-3 border-b border-border/40 shrink-0 flex items-center justify-between">
        <div>
          <h2 className="text-base font-semibold text-foreground tracking-tight flex items-center gap-2">
            <MessageSquare className="w-4 h-4 text-primary" />
            Chat Sessions
          </h2>
          <p className="text-xs text-muted-foreground mt-0.5">Your conversation history</p>
        </div>
        <button
          onClick={handleRefresh}
          disabled={refreshing}
          className="p-1.5 rounded-lg text-muted-foreground hover:text-foreground hover:bg-secondary/60 transition-colors"
          title="Refresh"
        >
          <RefreshCw className={`w-4 h-4 ${refreshing ? 'animate-spin' : ''}`} />
        </button>
      </div>

      <ScrollArea className="flex-1 min-h-0">
        {loading ? (
          <div className="flex items-center justify-center py-16">
            <Loader2 className="w-5 h-5 animate-spin text-muted-foreground" />
          </div>
        ) : sessionList.length === 0 ? (
          <div className="flex flex-col items-center justify-center px-6 py-8 text-center h-full min-h-[300px]">
            <div className="w-14 h-14 rounded-2xl bg-primary/10 flex items-center justify-center mb-4">
              <MessageSquare className="w-7 h-7 text-primary/60" />
            </div>
            <p className="text-sm font-medium text-foreground/80 mb-1">No sessions yet</p>
            <p className="text-xs text-muted-foreground leading-relaxed">
              Start a conversation to see your chat history here.
            </p>
          </div>
        ) : (
          <div className="px-2 py-1">
            <p className="px-3 py-2 text-xs font-medium text-muted-foreground">Recents</p>
            {sessionList.map((s) => {
              const isActive = s.session_id === activeSessionId
              return (
                <button
                  key={s.session_id}
                  onClick={() => {
                    switchToSession(s.session_id)
                    onSessionSelected?.()
                  }}
                  className={`
                    w-full flex items-center justify-between px-3 py-2 rounded-lg
                    transition-colors duration-150 group
                    ${isActive ? 'bg-secondary/60' : 'bg-card hover:bg-secondary/60'}
                  `}
                >
                  <span
                    className="text-sm font-medium text-foreground/80 truncate text-left pr-2"
                    title={s.first_user_message_preview || '(empty)'}
                  >
                    {s.first_user_message_preview || '(empty)'}
                  </span>
                  <span
                    onClick={(e) => {
                      e.stopPropagation()
                      handleDelete(s)
                    }}
                    className="shrink-0 p-1 rounded-md text-muted-foreground/0 group-hover:text-muted-foreground/60 hover:text-destructive hover:bg-destructive/10"
                    title="Delete session"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </span>
                </button>
              )
            })}
          </div>
        )}
      </ScrollArea>

      <ConfirmDialog
        open={deleteTarget !== null}
        title="Delete session?"
        message={deleteTarget?.first_user_message_preview ?? ''}
        confirmLabel="Delete"
        cancelLabel="Cancel"
        variant="danger"
        onConfirm={confirmDelete}
        onCancel={() => setDeleteTarget(null)}
      />
    </div>
  )
}
