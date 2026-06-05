import { useState, useRef, useCallback, useEffect } from 'react'
import {
  DndContext,
  useDraggable,
  type DragEndEvent,
  type DragStartEvent,
  PointerSensor,
  useSensor,
  useSensors,
} from '@dnd-kit/core'
import { CSS } from '@dnd-kit/utilities'
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
  MoreHorizontal,
  Settings,
  GripVertical,
  type LucideIcon,
} from 'lucide-react'

import ChatPanel from './ChatPanel'
import ChatSessionsPanel from './panels/ChatSessionsPanel'
import AvatarsPanel from './panels/AvatarsPanel'
import SettingsPanel from './panels/SettingsPanel'
import MorePanel from './panels/MorePanel'

/* ─── Types ─── */
type DockedEdge = 'left' | 'right' | 'top' | 'bottom'
type PanelId = 'chat' | 'sessions' | 'avatars' | 'more' | 'settings' | null

interface NavItem {
  id: PanelId
  icon: LucideIcon
  label: string
}

const NAV_ITEMS: NavItem[] = [
  { id: 'chat', icon: SquarePen, label: 'Chat' },
  { id: 'sessions', icon: MessageSquare, label: 'Sessions' },
  { id: 'avatars', icon: UserRound, label: 'Avatars' },
  { id: 'more', icon: MoreHorizontal, label: 'More' },
  { id: 'settings', icon: Settings, label: 'Settings' },
]

/* ─── Panel content map ─── */
function PanelContent({ panelId }: { panelId: PanelId }) {
  switch (panelId) {
    case 'chat':
      return <ChatPanel />
    case 'sessions':
      return <ChatSessionsPanel />
    case 'avatars':
      return <AvatarsPanel />
    case 'settings':
      return <SettingsPanel />
    case 'more':
      return <MorePanel />
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

function getSnapPosition(edge: DockedEdge, barWidth: number, barHeight: number) {
  const padding = 16
  switch (edge) {
    case 'left':
      return { x: padding, y: Math.round((window.innerHeight - barHeight) / 2) }
    case 'right':
      return { x: window.innerWidth - barWidth - padding, y: Math.round((window.innerHeight - barHeight) / 2) }
    case 'top':
      return { x: Math.round((window.innerWidth - barWidth) / 2), y: padding }
    case 'bottom':
      return { x: Math.round((window.innerWidth - barWidth) / 2), y: window.innerHeight - barHeight - padding }
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
  const { attributes, listeners, setNodeRef, transform } = useDraggable({
    id: 'floating-nav-bar',
  })

  const isHorizontal = dockedEdge === 'top' || dockedEdge === 'bottom'

  const style: React.CSSProperties = {
    position: 'fixed',
    left: position.x,
    top: position.y,
    transform: transform ? CSS.Translate.toString(transform) : undefined,
    transition: isDragging ? 'none' : 'left 0.35s cubic-bezier(0.16, 1, 0.3, 1), top 0.35s cubic-bezier(0.16, 1, 0.3, 1)',
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
    </div>
  )
}

/* ═══════════════════════════════════════════════
   FloatingNavBar (main export)
   ═══════════════════════════════════════════════ */

export default function FloatingNavBar() {
  const [dockedEdge, setDockedEdge] = useState<DockedEdge>('left')
  const [activePanel, setActivePanel] = useState<PanelId>(null)
  const [isDragging, setIsDragging] = useState(false)
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
  const { refs, floatingStyles } = useFloating({
    placement: getPlacementFromEdge(dockedEdge),
    middleware: [
      offset(12),
      shift({ padding: 16 }),
    ],
    whileElementsMounted: autoUpdate,
  })

  // Sync reference to the bar element
  useEffect(() => {
    const el = barRef.current?.querySelector('.floating-nav-bar')
    if (el) {
      refs.setReference(el as HTMLElement)
    }
  }, [refs, position, dockedEdge])

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
  }, [])

  const handleDragEnd = useCallback((event: DragEndEvent) => {
    setIsDragging(false)

    const { delta } = event
    const newX = position.x + delta.x
    const newY = position.y + delta.y

    // Find nearest edge and snap
    const nearestEdge = findNearestEdge(newX, newY, barSize.width, barSize.height)
    setDockedEdge(nearestEdge)

    // We need to recalc bar size for the new orientation before snapping
    // For now, use current barSize; the useEffect above will re-measure
    const snapPos = getSnapPosition(nearestEdge, barSize.width, barSize.height)
    setPosition(snapPos)
  }, [position, barSize])

  const handleIconClick = useCallback((id: PanelId) => {
    setActivePanel((prev) => (prev === id ? null : id))
  }, [])

  // Close panel when clicking outside
  useEffect(() => {
    if (!activePanel) return

    const handleClickOutside = (e: MouseEvent) => {
      const target = e.target as HTMLElement
      // Don't close if clicking inside the nav bar or the panel
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
  }, [activePanel])

  const isHorizontal = dockedEdge === 'top' || dockedEdge === 'bottom'

  return (
    <div ref={barRef}>
      <DndContext
        sensors={sensors}
        onDragStart={handleDragStart}
        onDragEnd={handleDragEnd}
      >
        <DraggableBar
          dockedEdge={dockedEdge}
          position={position}
          activePanel={activePanel}
          onIconClick={handleIconClick}
          isDragging={isDragging}
        />
      </DndContext>

      {/* Floating panel */}
      {activePanel && (
        <div
          ref={refs.setFloating}
          style={{
            ...floatingStyles,
            zIndex: 9998,
          }}
          className={`
            floating-panel
            ${isHorizontal ? 'w-[360px] h-[480px]' : 'w-[360px] h-[520px]'}
            rounded-2xl overflow-hidden
            bg-card/80 backdrop-blur-2xl
            border border-border/50
            shadow-[0_16px_64px_rgba(0,0,0,0.4),0_0_0_1px_rgba(255,255,255,0.05)_inset]
            animate-panel-in
          `}
        >
          <PanelContent panelId={activePanel} />
        </div>
      )}
    </div>
  )
}
