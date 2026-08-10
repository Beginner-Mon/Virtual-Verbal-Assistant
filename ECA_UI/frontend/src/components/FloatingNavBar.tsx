import { useState, useRef, useCallback, useEffect } from 'react'
import {
  DndContext,
  useDraggable,
  type DragEndEvent,
  type DragStartEvent,
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
// @ts-expect-error - reserved for auth feature (commented out)
import { LogOut } from 'lucide-react'
// @ts-expect-error - reserved for auth feature (commented out)
import type { LucideIcon } from 'lucide-react'
import { useMediaQuery } from '../lib/use-media-query'
import { useAuth } from '../contexts/AuthContext'
import { useMotion } from '../contexts/MotionContext'
import { Avatar, AvatarImage, AvatarFallback } from './ui/avatar'

import ChatPanel from './ChatPanel'
import ChatSessionsPanel from './panels/ChatSessionsPanel'
import AvatarsPanel from './panels/AvatarsPanel'
import SettingsPanel from './panels/SettingsPanel'
import MotionControlPanel from './panels/MotionControlPanel'
import ProfileSettingsModal from './ProfileSettingsModal'
import MobileNavBar from './MobileNavBar'

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
  onOpenModal?: (type: 'profile' | 'settings' | 'notifications') => void
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
              w-10 h-10 rounded-xl flex items-center justify-center
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
          rounded-full
        `}
      >
        <Avatar className="w-9 h-9">
          <AvatarImage src={profilePicture} alt="Avatar" referrerPolicy="no-referrer" />
          <AvatarFallback className="bg-muted-foreground/20 text-muted-foreground">
            <svg className="w-14 h-14" viewBox="0 0 638 543" fill="none" stroke="currentColor" strokeWidth="40">
              <g transform="translate(0,543) scale(0.1,-0.1)">
              <path d="M2505 5096 c-105 -34 -149 -62 -187 -117 -68 -101 -85 -264 -42 -409 23 -79 89 -207 129 -251 19 -22 35 -41 35 -44 0 -3 -19 -24 -42 -45 -39 -38 -42 -39 -59 -24 -10 9 -49 25 -86 35 l-67 19 -45 -27 c-53 -30 -205 -180 -272 -268 -111 -145 -184 -295 -244 -505 -134 -465 -149 -891 -49 -1350 84 -383 294 -815 484 -994 70 -65 153 -126 174 -126 5 0 -21 32 -59 70 -37 39 -65 73 -63 76 13 13 117 -74 141 -118 36 -64 73 -82 151 -74 105 11 140 18 154 32 25 26 31 14 34 -68 l3 -83 56 53 56 52 18 -45 c9 -26 16 -47 14 -49 -2 -2 -30 -22 -61 -45 -32 -23 -58 -43 -58 -44 -1 -28 -30 -219 -34 -226 -9 -14 14 -25 25 -14 6 6 16 59 24 118 8 59 16 109 18 112 2 2 26 19 52 38 37 26 52 32 62 23 7 -5 10 -14 7 -19 -9 -15 6 -49 22 -49 13 0 18 15 16 50 0 8 5 18 11 22 6 4 12 21 13 38 3 50 84 111 105 79 5 -8 9 -33 9 -56 0 -23 7 -48 15 -57 8 -8 15 -24 15 -37 0 -18 2 -20 16 -8 8 7 28 36 42 63 l27 51 3 -83 3 -83 28 21 c16 11 32 20 36 20 4 0 4 -41 0 -92 -6 -79 -5 -90 8 -85 44 17 199 122 229 155 34 38 92 162 105 222 9 46 20 45 44 -5 37 -79 43 -69 36 64 -6 124 6 159 19 54 3 -31 13 -68 20 -82 8 -14 14 -37 14 -51 0 -32 41 -160 51 -160 4 0 11 4 14 10 3 5 11 10 17 10 19 0 -1 -63 -33 -102 -38 -48 -21 -48 50 -2 60 40 95 83 106 135 10 46 26 69 48 69 10 0 27 4 37 10 13 7 19 6 24 -6 6 -16 -19 -137 -35 -166 -11 -22 -12 -38 0 -38 9 0 65 47 129 109 l35 35 33 -140 c18 -76 41 -178 50 -227 13 -61 22 -87 32 -87 12 0 13 10 8 46 -9 68 -85 396 -97 417 -8 14 -1 30 32 75 l41 57 -6 -52 c-7 -58 7 -78 31 -44 8 12 37 47 65 79 28 31 57 71 66 89 15 31 32 36 32 10 0 -7 -25 -48 -56 -90 -31 -43 -54 -80 -51 -83 3 -3 22 9 43 27 85 72 77 72 71 -1 -3 -36 -8 -76 -13 -89 l-7 -24 43 24 c59 33 142 117 176 178 16 28 32 51 36 51 4 0 34 39 66 88 33 48 67 94 76 101 16 13 17 10 10 -40 -10 -79 -53 -163 -129 -252 -37 -43 -64 -81 -61 -85 6 -6 69 37 151 102 50 40 205 229 237 289 71 133 77 488 12 842 -28 155 -83 420 -118 565 -13 58 -41 186 -61 285 -53 261 -70 328 -110 444 -80 230 -191 440 -307 580 -145 176 -390 339 -619 414 -184 60 -412 72 -714 37 -200 -23 -262 -37 -387 -86 l-92 -36 -24 29 c-44 53 -101 158 -124 228 -19 59 -23 91 -23 200 0 161 11 187 120 276 41 34 72 65 69 70 -3 5 -6 9 -7 8 -1 0 -20 -6 -42 -13z m1703 -1033 c54 -46 165 -173 158 -180 -2 -2 -56 48 -121 112 -118 116 -139 155 -37 68z m-143 -11 c24 -21 65 -62 91 -91 44 -48 45 -53 30 -70 -15 -17 -21 -13 -110 78 -84 84 -92 96 -78 108 22 17 18 18 67 -25z m-277 -54 c95 -97 107 -115 91 -134 -10 -12 -8 -22 9 -53 l22 -39 -47 -40 c-60 -49 -108 -74 -136 -70 -14 2 -50 46 -114 141 l-93 137 31 0 c16 0 43 7 58 15 30 15 91 85 91 104 0 22 18 10 88 -61z m-306 -220 c65 -100 83 -138 66 -138 -7 0 -29 -17 -50 -37 -21 -20 -39 -35 -42 -32 -2 2 -33 51 -68 109 -35 58 -72 117 -82 133 l-18 27 53 0 c30 0 61 5 69 10 8 5 16 10 17 10 1 0 26 -37 55 -82z m-305 -156 c63 -134 65 -142 48 -142 -17 0 -129 225 -120 240 12 19 28 -3 72 -98z m-169 -34 c22 -57 43 -120 46 -141 l7 -38 -81 7 c-78 7 -82 6 -109 -20 l-28 -27 -11 28 c-15 41 -40 123 -47 161 -6 27 -4 32 12 32 27 0 92 34 133 69 19 17 36 30 37 31 1 0 19 -46 41 -102z m-419 -68 c26 -124 27 -154 5 -146 -9 3 -22 6 -29 6 -20 0 -32 26 -59 129 l-25 91 29 0 c16 0 32 5 35 10 14 23 23 5 44 -90z m-184 -163 c-8 -17 -15 -40 -15 -51 0 -23 -40 -56 -40 -34 -1 7 -9 62 -19 122 l-18 109 36 33 36 33 18 -91 c15 -82 16 -94 2 -121z m-276 -78 c1 -33 -4 -51 -18 -63 -17 -16 -18 -15 -25 30 -10 74 -7 104 12 129 l17 23 7 -37 c4 -20 7 -57 7 -82z m-229 -471 c13 -35 31 -79 40 -98 10 -19 23 -62 30 -95 6 -33 16 -78 22 -100 5 -22 13 -58 17 -80 4 -23 11 -46 15 -53 4 -7 11 -34 14 -59 13 -89 50 -53 55 53 2 54 19 70 25 23 4 -33 22 -79 43 -110 10 -14 21 -38 25 -55 7 -29 84 -195 95 -204 6 -5 13 -33 25 -107 8 -43 -19 -156 -53 -223 -18 -35 -42 -64 -66 -79 -22 -15 -34 -28 -28 -34 14 -14 92 12 120 42 33 35 91 75 97 68 12 -12 26 -313 22 -479 -4 -172 -3 -177 14 -151 26 40 74 158 99 242 26 89 33 82 50 -57 11 -84 11 -109 0 -135 -7 -18 -20 -53 -29 -78 -16 -45 -18 -46 -72 -57 -112 -23 -146 -12 -185 60 -15 27 -33 52 -41 55 -8 3 -14 12 -14 20 0 20 -58 99 -86 118 -12 8 -30 34 -39 59 -13 33 -25 48 -46 55 -17 6 -29 17 -29 28 0 10 -14 36 -30 57 -17 22 -30 46 -30 53 0 15 -35 53 -49 53 -14 0 -61 59 -61 76 0 8 14 23 31 32 17 9 50 34 72 54 23 21 49 41 59 44 10 3 18 13 18 23 0 9 5 22 11 28 7 7 6 35 -6 94 -32 167 -54 343 -46 353 10 12 9 92 -2 116 -3 8 -29 76 -58 150 -48 127 -52 142 -56 248 -3 61 -3 112 0 112 2 0 15 -28 27 -62z" />
              </g>
            </svg>
          </AvatarFallback>
        </Avatar>
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
  const [modalType, setModalType] = useState<'profile' | 'settings' | 'notifications' | null>(null)
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

  // DnD sensors — increase activation distance so clicks don't trigger drag
  const sensors = useSensors(
    useSensor(PointerSensor, {
      activationConstraint: {
        distance: 8,
      },
    })
  )

  const handleDragStart = useCallback((_event: DragStartEvent) => {
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
