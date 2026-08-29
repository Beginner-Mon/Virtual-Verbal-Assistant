import { useState, useRef, useCallback, useEffect } from 'react'
import {
  DndContext,
  useDraggable,
  type DragEndEvent,
  PointerSensor,
  useSensor,
  useSensors,
  type Modifier,
} from '@dnd-kit/core'
import {
  useFloating,
  offset,
  shift,
  autoUpdate,
  type Placement,
} from '@floating-ui/react'
import {
  SquarePen,
  MessageSquare,
  UserRound,
  Settings2,
  GripVertical,
} from 'lucide-react'
// LogOut and LucideIcon were imported here "reserved for auth feature
// (commented out)" behind @ts-expect-error. Nothing references them, and an
// unused import is not a reservation — git remembers. Reinstate when the
// feature lands.
import { useMediaQuery } from '../lib/use-media-query'
import { useAuth } from '../contexts/AuthContext'
import { useMotion } from '../contexts/MotionContext'
import ChatPanel from './ChatPanel'
import ChatSessionsPanel from './panels/ChatSessionsPanel'
import AvatarsPanel from './panels/AvatarsPanel'
import SettingsPanel from './panels/SettingsPanel'
import MotionControlPanel from './panels/MotionControlPanel'
import ProfileSettingsModal from './ProfileSettingsModal'
import MobileNavBar from './MobileNavBar'
import AvatarWithLogo from './AvatarWithLogo'

/* ─── Types ─── */
type DockedEdge = 'left' | 'right' | 'top' | 'bottom'
export type PanelId = 'chat' | 'sessions' | 'avatars' | 'motion' | 'settings' | null

export interface NavItem {
  id: PanelId
  icon: React.ComponentType<{ className?: string }>
  label: string
}

const NAV_ITEMS: NavItem[] = [
  { id: 'chat', icon: SquarePen, label: 'Chat' },
  { id: 'sessions', icon: MessageSquare, label: 'Sessions' },
  { id: 'avatars', icon: UserRound, label: 'Avatars' },
  { id: 'motion', icon: Settings2, label: 'Motion' },
]

/* ─── Panel content map ─── */
function PanelContent({
  panelId,
  onOpenModal,
  onSessionSelected,
}: {
  panelId: PanelId
  onOpenModal?: (type: 'profile' | 'settings' | 'notifications' | 'billing') => void
  onSessionSelected?: () => void
}) {
  switch (panelId) {
    case 'chat':
      return <ChatPanel />
    case 'sessions':
      return <ChatSessionsPanel onSessionSelected={onSessionSelected} />
    case 'avatars':
      return <AvatarsPanel />
    case 'settings':
      return <SettingsPanel onOpenModal={onOpenModal} />
    case 'motion':
      return <MotionControlPanel />
    default:
      return null
  }
}

/* ─── Helpers ─── */
function getPlacementFromEdge(edge: DockedEdge): Placement {
  switch (edge) {
    case 'left':
      return 'right-start'
    case 'right':
      return 'left-start'
    case 'top':
      return 'bottom-start'
    case 'bottom':
      return 'top-start'
  }
}

function getSnapPosition(edge: DockedEdge, x: number, y: number, barWidth: number, barHeight: number) {
  const padding = 16
  const clampedX = Math.max(padding, Math.min(x, window.innerWidth - barWidth - padding))
  const clampedY = Math.max(padding, Math.min(y, window.innerHeight - barHeight - padding))

  switch (edge) {
    case 'left':
      return { x: padding, y: clampedY }
    case 'right':
      return { x: window.innerWidth - barWidth - padding, y: clampedY }
    case 'top':
      return { x: clampedX, y: padding }
    case 'bottom':
      return { x: clampedX, y: window.innerHeight - barHeight - padding }
  }
}

function findNearestEdge(x: number, y: number, barWidth: number, barHeight: number): DockedEdge {
  const centerX = x + barWidth / 2
  const centerY = y + barHeight / 2
  const distLeft = centerX
  const distRight = window.innerWidth - centerX
  const distTop = centerY
  const distBottom = window.innerHeight - centerY

  const min = Math.min(distLeft, distRight, distTop, distBottom)
  if (min === distLeft) return 'left'
  if (min === distRight) return 'right'
  if (min === distTop) return 'top'
  return 'bottom'
}

/* ═══════════════════════════════════════════════
   Draggable Nav Bar (inner component)
   ═══════════════════════════════════════════════ */

function DraggableBar({
  dockedEdge,
  position,
  activePanel,
  onIconClick,
  isDragging,
}: {
  dockedEdge: DockedEdge
  position: { x: number; y: number }
  activePanel: PanelId
  onIconClick: (id: PanelId) => void
  isDragging: boolean
}) {
  const { userAttributes } = useAuth()
  const profilePicture = userAttributes?.picture

  const { attributes, listeners, setNodeRef, transform } = useDraggable({
    id: 'floating-nav-bar',
  })

  const isHorizontal = dockedEdge === 'top' || dockedEdge === 'bottom'

  const currentX = position.x + (transform?.x ?? 0)
  const currentY = position.y + (transform?.y ?? 0)

  const style: React.CSSProperties = {
    position: 'fixed',
    left: 0,
    top: 0,
    transform: `translate3d(${currentX}px, ${currentY}px, 0)`,
    transition: isDragging ? 'none' : 'transform 0.35s cubic-bezier(0.16, 1, 0.3, 1)',
    zIndex: 9999,
    touchAction: 'none',
  }

  return (
    <div
      ref={setNodeRef}
      style={style}
      className={`
        floating-nav-bar
        flex ${isHorizontal ? 'flex-row' : 'flex-col'} items-center gap-1
        p-1.5 rounded-2xl
        bg-card/70 backdrop-blur-2xl
        border border-border/50
        shadow-[0_8px_40px_rgba(0,0,0,0.3),0_0_0_1px_rgba(255,255,255,0.05)_inset]
      `}
    >
      {/* Drag handle */}
      <div
        {...attributes}
        {...listeners}
        className={`
          flex items-center justify-center cursor-grab active:cursor-grabbing
          text-muted-foreground/40 hover:text-muted-foreground/70 transition-colors
          ${isHorizontal ? 'px-1 py-2' : 'py-1 px-2'}
        `}
        title="Drag to reposition"
      >
        <GripVertical className={`w-4 h-4 ${isHorizontal ? 'rotate-90' : ''}`} />
      </div>

      {/* Separator */}
      <div className={`${isHorizontal ? 'w-px h-6' : 'h-px w-6'} bg-border/40`} />

      {/* Nav icons */}
      {NAV_ITEMS.map((item) => {
        const Icon = item.icon
        const isActive = activePanel === item.id
        return (
          <button
            key={item.id}
            onClick={() => onIconClick(item.id)}
            title={item.label}
            className={`
              nav-icon-btn
              w-10 h-10 rounded-xl flex items-center justify-center cursor-pointer
              transition-all duration-200 relative
              ${isActive
                ? 'bg-primary/20 text-primary shadow-[0_0_12px_rgba(var(--primary-rgb,139,92,246),0.3)]'
                : 'text-muted-foreground hover:bg-secondary/60 hover:text-foreground'
              }
            `}
          >
            <Icon className="w-[18px] h-[18px]" />
            {isActive && (
              <span className="absolute bottom-0.5 left-1/2 -translate-x-1/2 w-1 h-1 rounded-full bg-primary" />
            )}
          </button>
        )
      })}

      {/* Separator before avatar */}
      <div className={`${isHorizontal ? 'w-px h-6' : 'h-px w-6'} bg-border/40`} />

      {/* Avatar circle (replaces settings icon) */}
      <button
        onClick={() => onIconClick('settings')}
        title="Profile & Settings"
        className={`
          shrink-0 ${isHorizontal ? 'ml-2' : 'mt-2'}
          rounded-full cursor-pointer
        `}
      >
        <AvatarWithLogo size="sm" profilePicture={profilePicture} />
      </button>
    </div>
  )
}

/* ═══════════════════════════════════════════════
   FloatingNavBar (main export)
   ═══════════════════════════════════════════════ */

export default function FloatingNavBar() {
  const isMobile = useMediaQuery('(max-width: 767px)')
  const { isMusicPlaying, toggleMusic } = useMotion()

  const [dockedEdge, setDockedEdge] = useState<DockedEdge>('left')
  const [activePanel, setActivePanel] = useState<PanelId>(null)
  const [isDragging, setIsDragging] = useState(false)
  const [modalType, setModalType] = useState<'profile' | 'settings' | 'notifications' | 'billing' | null>(null)
  const prevPanelRef = useRef<PanelId>(null)
  const barRef = useRef<HTMLDivElement>(null)

  // We need to measure the bar to calculate snap positions
  const [barSize, setBarSize] = useState({ width: 56, height: 320 })

  // Position state (pixels from top-left corner of viewport)
  const [position, setPosition] = useState({ x: 16, y: 0 })

  // Initialize vertical center on mount
  useEffect(() => {
    setPosition({ x: 16, y: Math.round((window.innerHeight - barSize.height) / 2) })
  }, [])

  // Re-measure bar size when docked edge changes (orientation flips)
  useEffect(() => {
    const el = barRef.current?.querySelector('.floating-nav-bar')
    if (el) {
      const rect = el.getBoundingClientRect()
      setBarSize({ width: rect.width, height: rect.height })
    }
  }, [dockedEdge])

  // Floating UI for the panel

  const { refs, floatingStyles, isPositioned } = useFloating({
    placement: getPlacementFromEdge(dockedEdge),
    middleware: [
      offset(12),
      shift({ padding: 16 }),
    ],
    whileElementsMounted: autoUpdate,
  })

  // Sync reference to the bar element (stable ref, run once)
  useEffect(() => {
    const el = barRef.current?.querySelector('.floating-nav-bar')
    if (el) {
      refs.setReference(el as HTMLElement)
    }
  }, [])

  // Plan A1: keep position after viewport resize — re-measure bar, clamp position, keep floating ref fresh
  useEffect(() => {
    let timer: ReturnType<typeof setTimeout> | null = null
    const handleResize = () => {
      if (timer) clearTimeout(timer)
      timer = setTimeout(() => {
        // Mobile uses MobileNavBar; nothing to re-measure here
        if (isMobile) return
        const el = barRef.current?.querySelector('.floating-nav-bar') as HTMLElement | null
        if (el) {
          refs.setReference(el)
          const rect = el.getBoundingClientRect()
          const newSize = { width: rect.width, height: rect.height }
          setBarSize((prev) => (prev.width === newSize.width && prev.height === newSize.height ? prev : newSize))
          const padding = 16
          setPosition((prev) => ({
            x: Math.max(padding, Math.min(prev.x, window.innerWidth - newSize.width - padding)),
            y: Math.max(padding, Math.min(prev.y, window.innerHeight - newSize.height - padding)),
          }))
        } else {
          // Fallback: clamp with last known barSize if DOM not yet painted
          const padding = 16
          setPosition((prev) => ({
            x: Math.max(padding, Math.min(prev.x, window.innerWidth - barSize.width - padding)),
            y: Math.max(padding, Math.min(prev.y, window.innerHeight - barSize.height - padding)),
          }))
        }
      }, 100)
    }
    window.addEventListener('resize', handleResize)
    return () => {
      window.removeEventListener('resize', handleResize)
      if (timer) clearTimeout(timer)
    }
  }, [isMobile, refs, barSize.width, barSize.height])

  // Plan A1: when breakpoint flips (mobile ↔ desktop) re-anchor floating-ui and re-measure
  useEffect(() => {
    if (isMobile) return
    const raf = requestAnimationFrame(() => {
      const el = barRef.current?.querySelector('.floating-nav-bar') as HTMLElement | null
      if (!el) {
        // DOM may not be painted yet — retry next frame
        requestAnimationFrame(() => {
          const retry = barRef.current?.querySelector('.floating-nav-bar') as HTMLElement | null
          if (!retry) return
          refs.setReference(retry)
          const rect = retry.getBoundingClientRect()
          const newSize = { width: rect.width, height: rect.height }
          setBarSize((prev) => (prev.width === newSize.width && prev.height === newSize.height ? prev : newSize))
          const padding = 16
          setPosition((prev) => ({
            x: Math.max(padding, Math.min(prev.x, window.innerWidth - newSize.width - padding)),
            y: Math.max(padding, Math.min(prev.y, window.innerHeight - newSize.height - padding)),
          }))
        })
        return
      }
      refs.setReference(el)
      const rect = el.getBoundingClientRect()
      const newSize = { width: rect.width, height: rect.height }
      setBarSize((prev) => (prev.width === newSize.width && prev.height === newSize.height ? prev : newSize))
      const padding = 16
      setPosition((prev) => ({
        x: Math.max(padding, Math.min(prev.x, window.innerWidth - newSize.width - padding)),
        y: Math.max(padding, Math.min(prev.y, window.innerHeight - newSize.height - padding)),
      }))
    })
    return () => cancelAnimationFrame(raf)
  }, [isMobile, refs])

  // DnD sensors — increase activation distance so clicks don't trigger drag
  const sensors = useSensors(
    useSensor(PointerSensor, {
      activationConstraint: {
        distance: 8,
      },
    })
  )

  // No parameter: dnd-kit passes a DragStartEvent, this handler does not read
  // it, and the leading-underscore convention is not configured here.
  const handleDragStart = useCallback(() => {
    setIsDragging(true)
    setActivePanel(null)
  }, [])

  const handleDragEnd = useCallback((event: DragEndEvent) => {
    setIsDragging(false)

    const { delta } = event
    const newX = position.x + delta.x
    const newY = position.y + delta.y

    // Find nearest edge and snap
    const nearestEdge = findNearestEdge(newX, newY, barSize.width, barSize.height)
    setDockedEdge(nearestEdge)

    // Calculate expected size if orientation flips
    const isNowHorizontal = nearestEdge === 'top' || nearestEdge === 'bottom'
    const wasHorizontal = dockedEdge === 'top' || dockedEdge === 'bottom'
    
    let expectedWidth = barSize.width
    let expectedHeight = barSize.height
    
    if (isNowHorizontal !== wasHorizontal) {
      expectedWidth = barSize.height
      expectedHeight = barSize.width
    }

    const snapPos = getSnapPosition(nearestEdge, newX, newY, expectedWidth, expectedHeight)
    setPosition(snapPos)
  }, [position, barSize, dockedEdge])

  const handleIconClick = useCallback((id: PanelId) => {
    setActivePanel((prev) => (prev === id ? null : id))
  }, [])

  // Close panel when clicking outside
  useEffect(() => {
    if (!activePanel || isMobile) return

    const handleClickOutside = (e: MouseEvent) => {
      const target = e.target as HTMLElement
      if (
        target.closest('.floating-nav-bar') ||
        target.closest('.floating-panel')
      ) {
        return
      }
      setActivePanel(null)
    }

    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [activePanel, isMobile])

  // Calculate dynamic dimensions
  const isHorizontal = dockedEdge === 'top' || dockedEdge === 'bottom'
  
  const navBarCenterX = position.x + (barSize.width / 2)
  const isMiddleThird = navBarCenterX >= window.innerWidth / 3 && navBarCenterX <= (window.innerWidth * 2) / 3
  
  let panelDimensionsClass = isHorizontal ? 'w-[360px] h-[480px]' : 'w-[360px] h-[520px]'
  if (activePanel === 'chat') {
    panelDimensionsClass = isMiddleThird ? 'w-[600px] h-[400px]' : 'w-[360px] h-[600px]'
  }
  if (activePanel === 'avatars') {
    panelDimensionsClass = isHorizontal ? 'w-[400px] h-[540px]' : 'w-[400px] h-[580px]'
  }

  // Custom drag constraint
  const restrictToScreen: Modifier = ({ transform }) => {
    const padding = 16
    const minX = padding - position.x
    const maxX = window.innerWidth - barSize.width - padding - position.x
    const minY = padding - position.y
    const maxY = window.innerHeight - barSize.height - padding - position.y

    return {
      ...transform,
      x: Math.max(minX, Math.min(maxX, transform.x)),
      y: Math.max(minY, Math.min(maxY, transform.y)),
    }
  }

  /* ─── Mobile: draggable hamburger + icon-only dropdown ─── */
  if (isMobile) {
    return (
      <>
        <MobileNavBar
          activePanel={activePanel}
          onIconClick={handleIconClick}
          onOpenModal={(type) => {
            prevPanelRef.current = activePanel
            setModalType(type)
            setActivePanel(null)
          }}
          navItems={NAV_ITEMS}
          isMusicPlaying={isMusicPlaying}
          toggleMusic={toggleMusic}
          panelContent={
            activePanel && activePanel !== 'chat' ? (
            <PanelContent
              panelId={activePanel}
              onOpenModal={(type) => {
                prevPanelRef.current = activePanel
                setModalType(type)
                setActivePanel(null)
              }}
              onSessionSelected={() => setActivePanel('chat')}
            />
            ) : undefined
          }
        />
        {modalType && (
          <ProfileSettingsModal
            type={modalType}
            onClose={() => setModalType(null)}
            onBack={() => {
              setModalType(null)
              setActivePanel(prevPanelRef.current)
            }}
          />
        )}
      </>
    )
  }

  /* ─── Desktop: full draggable nav bar + floating panels ─── */
  return (
    <div ref={barRef}>
      <DndContext
        sensors={sensors}
        onDragStart={handleDragStart}
        onDragEnd={handleDragEnd}
        modifiers={[restrictToScreen]}
      >
        <DraggableBar
          dockedEdge={dockedEdge}
          position={position}
          activePanel={activePanel}
          onIconClick={handleIconClick}
          isDragging={isDragging}
        />
      </DndContext>

      {/* Non-settings floating panels (use floating UI) */}
      {activePanel && activePanel !== 'settings' && (
        <div
          ref={refs.setFloating}
          style={{
            ...floatingStyles,
            zIndex: 9998,
            visibility: isPositioned ? 'visible' : 'hidden',
          }}
        >
          <div
            className={`
              floating-panel
              ${panelDimensionsClass}
              rounded-2xl overflow-hidden
              bg-card/80 backdrop-blur-2xl
              border border-border/50
              shadow-[0_16px_64px_rgba(0,0,0,0.4),0_0_0_1px_rgba(255,255,255,0.05)_inset]
              ${isPositioned ? 'animate-panel-in' : ''}
            `}
          >
            <PanelContent
              panelId={activePanel}
              onOpenModal={(type) => {
                setModalType(type)
                setActivePanel(null)
              }}
              onSessionSelected={() => setActivePanel('chat')}
            />
          </div>
        </div>
      )}

      {/* Settings dropdown — manually positioned at bottom of sidebar */}
      {activePanel === 'settings' && (
        <div
          style={{
            position: 'fixed',
            zIndex: 9998,
            ...(dockedEdge === 'left' && {
              left: position.x + barSize.width + 10,
              top: position.y + barSize.height,
              transform: 'translateY(-100%)',
            }),
            ...(dockedEdge === 'right' && {
              right: window.innerWidth - position.x + 10,
              top: position.y + barSize.height,
              transform: 'translateY(-100%)',
            }),
            ...(dockedEdge === 'top' && {
              left: position.x + barSize.width,
              top: position.y + barSize.height + 10,
              transform: 'translateX(-100%)',
            }),
            ...(dockedEdge === 'bottom' && {
              left: position.x + barSize.width,
              bottom: window.innerHeight - position.y + 10,
              transform: 'translateX(-100%)',
            }),
          }}
        >
          <div
            className="
              floating-panel
              min-w-[200px]
              rounded-2xl overflow-hidden
              bg-card/80 backdrop-blur-2xl
              border border-border/50
              shadow-[0_16px_64px_rgba(0,0,0,0.4),0_0_0_1px_rgba(255,255,255,0.05)_inset]
              animate-panel-in
            "
          >
            <PanelContent
              panelId="settings"
              onOpenModal={(type) => {
                prevPanelRef.current = activePanel
                setModalType(type)
                setActivePanel(null)
              }}
            />
          </div>
        </div>
      )}

      {/* Centered modal for Profile / Settings */}
      {modalType && (
        <ProfileSettingsModal
          type={modalType}
          onClose={() => setModalType(null)}
          onBack={() => {
            setModalType(null)
            setActivePanel(prevPanelRef.current)
          }}
        />
      )}
    </div>
  )
}
