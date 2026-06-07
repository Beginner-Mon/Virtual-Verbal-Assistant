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
  Menu,
  X,
  LogOut,
  type LucideIcon,
} from 'lucide-react'
import { useMediaQuery } from '../lib/use-media-query'

import ChatPanel from './ChatPanel'
import ChatSessionsPanel from './panels/ChatSessionsPanel'
import AvatarsPanel from './panels/AvatarsPanel'
import SettingsPanel from './panels/SettingsPanel'
import MorePanel from './panels/MorePanel'
import ProfileSettingsModal from './ProfileSettingsModal'

/* ─── Types ─── */
type DockedEdge = 'left' | 'right' | 'top' | 'bottom'
type PanelId = 'chat' | 'sessions' | 'avatars' | 'more' | 'settings' | null

interface NavItem {
  id: PanelId
  icon: React.ComponentType<{ className?: string }>
  label: string
}

const NAV_ITEMS: NavItem[] = [
  { id: 'chat', icon: SquarePen, label: 'Chat' },
  { id: 'sessions', icon: MessageSquare, label: 'Sessions' },
  { id: 'avatars', icon: UserRound, label: 'Avatars' },
  { id: 'more', icon: Settings2, label: 'More' },
]

/* ─── Panel content map ─── */
function PanelContent({
  panelId,
  onOpenModal,
}: {
  panelId: PanelId
  onOpenModal?: (type: 'profile' | 'settings') => void
}) {
  switch (panelId) {
    case 'chat':
      return <ChatPanel />
    case 'sessions':
      return <ChatSessionsPanel />
    case 'avatars':
      return <AvatarsPanel />
    case 'settings':
      return <SettingsPanel onOpenModal={onOpenModal} />
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
          w-8 h-8 rounded-full
          transition-all duration-200 shrink-0 ${isHorizontal ? 'ml-2' : 'mt-2'}
          ${activePanel === 'settings'
            ? 'bg-primary/20 ring-2 ring-primary'
            : 'bg-muted-foreground/20 hover:bg-muted-foreground/30'
          }
        `}
      />
    </div>
  )
}

/* ═══════════════════════════════════════════════
   FloatingNavBar (main export)
   ═══════════════════════════════════════════════ */

export default function FloatingNavBar() {
  const isMobile = useMediaQuery('(max-width: 767px)')

  const [dockedEdge, setDockedEdge] = useState<DockedEdge>('left')
  const [activePanel, setActivePanel] = useState<PanelId>(null)
  const [isDragging, setIsDragging] = useState(false)
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)
  const [modalType, setModalType] = useState<'profile' | 'settings' | null>(null)
  const barRef = useRef<HTMLDivElement>(null)
  const mobileMenuRef = useRef<HTMLDivElement>(null)

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
    setMobileMenuOpen(false)
  }, [])

  // Close mobile menu on outside pointer
  useEffect(() => {
    if (!mobileMenuOpen) return
    const handler = (e: PointerEvent) => {
      if (mobileMenuRef.current && !mobileMenuRef.current.contains(e.target as Node)) {
        setMobileMenuOpen(false)
      }
    }
    document.addEventListener('pointerdown', handler)
    return () => document.removeEventListener('pointerdown', handler)
  }, [mobileMenuOpen])

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

  /* ─── Mobile drag state ─── */
  const btnSize = 36
  const [menuX, setMenuX] = useState(0)
  const [menuY, setMenuY] = useState(80)
  const [isMenuDragging, setIsMenuDragging] = useState(false)
  const menuXRef = useRef(0)
  const menuYRef = useRef(80)
  const menuIsDraggingRef = useRef(false)
  const menuDragStart = useRef<{ x: number; y: number; startMenuX: number; startMenuY: number } | null>(null)
  const menuInitRef = useRef(false)

  useEffect(() => {
    if (!isMobile || menuInitRef.current) return
    menuInitRef.current = true
    const x = window.innerWidth - btnSize - 8
    menuXRef.current = x
    setMenuX(x)
  }, [isMobile])

  const getMenuYBounds = useCallback(() => {
    const chatTop = window.innerHeight * 0.6
    return { minY: 80, maxY: chatTop - btnSize - 8 }
  }, [])

  const handleMenuPointerDown = useCallback((e: React.PointerEvent) => {
    const el = e.currentTarget as HTMLElement
    el.setPointerCapture(e.pointerId)
    menuDragStart.current = { x: e.clientX, y: e.clientY, startMenuX: menuXRef.current, startMenuY: menuYRef.current }
    menuIsDraggingRef.current = false
    setIsMenuDragging(false)
  }, [])

  const handleMenuPointerMove = useCallback((e: React.PointerEvent) => {
    if (!menuDragStart.current) return
    const dx = e.clientX - menuDragStart.current.x
    const dy = e.clientY - menuDragStart.current.y

    if (!menuIsDraggingRef.current && Math.abs(dx) < 5 && Math.abs(dy) < 5) return

    menuIsDraggingRef.current = true
    setIsMenuDragging(true)

    const newX = Math.max(8, Math.min(window.innerWidth - btnSize - 8, menuDragStart.current.startMenuX + dx))
    const { minY, maxY } = getMenuYBounds()
    const newY = Math.max(minY, Math.min(maxY, menuDragStart.current.startMenuY + dy))

    menuXRef.current = newX
    menuYRef.current = newY
    setMenuX(newX)
    setMenuY(newY)
  }, [getMenuYBounds])

  const handleMenuPointerUp = useCallback(() => {
    const wasDragging = menuIsDraggingRef.current
    menuDragStart.current = null
    menuIsDraggingRef.current = false
    setIsMenuDragging(false)

    if (wasDragging) {
      const center = menuXRef.current + btnSize / 2
      const snapX = center < window.innerWidth / 2 ? 8 : window.innerWidth - btnSize - 8
      menuXRef.current = snapX
      setMenuX(snapX)
    }
  }, [])

  const handleMenuPointerCancel = useCallback(() => {
    menuDragStart.current = null
    menuIsDraggingRef.current = false
    setIsMenuDragging(false)
  }, [])

  /* ─── Split nav items for mobile (no Chat) ─── */
  const mobileNavItems = NAV_ITEMS.filter((item) => item.id !== 'chat')

  /* ─── Mobile: draggable hamburger + icon-only dropdown ─── */
  if (isMobile) {
    return (
      <>
        <div
          ref={mobileMenuRef}
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            transform: `translate(${menuX}px, ${menuY}px)`,
            transition: isMenuDragging ? 'none' : 'transform 0.3s cubic-bezier(0.16, 1, 0.3, 1)',
            zIndex: 10000,
            touchAction: 'none',
          }}
          onPointerDown={handleMenuPointerDown}
          onPointerMove={handleMenuPointerMove}
          onPointerUp={handleMenuPointerUp}
          onPointerCancel={handleMenuPointerCancel}
        >
          <div
            className="flex flex-col items-center backdrop-blur-md border border-border/30 shadow-lg rounded-xl overflow-hidden transition-all duration-200"
            style={{ background: 'rgba(0,0,0,0.15)' }}
          >
            {/* Menu button */}
            <button
              onClick={() => setMobileMenuOpen((prev) => !prev)}
              className="flex items-center justify-center"
              style={{ width: btnSize, height: btnSize, touchAction: 'none' }}
            >
              <div className="relative w-4 h-4">
                <Menu
                  className="absolute inset-0 w-4 h-4 transition-all duration-200 text-muted-foreground"
                  style={{
                    opacity: mobileMenuOpen ? 0 : 1,
                    transform: `rotate(${mobileMenuOpen ? 45 : 0}deg)`,
                  }}
                />
                <X
                  className="absolute inset-0 w-4 h-4 transition-all duration-200 text-muted-foreground"
                  style={{
                    opacity: mobileMenuOpen ? 1 : 0,
                    transform: `rotate(${mobileMenuOpen ? 0 : -45}deg)`,
                  }}
                />
              </div>
            </button>

            {/* Icon-only dropdown — slides down/up with transition */}
            <div
              className="flex flex-col items-center gap-1 transition-all duration-200"
              style={{
                maxHeight: mobileMenuOpen ? '200px' : '0px',
                opacity: mobileMenuOpen ? 1 : 0,
                paddingTop: mobileMenuOpen ? '6px' : '0',
                paddingBottom: mobileMenuOpen ? '6px' : '0',
                pointerEvents: mobileMenuOpen ? 'auto' : 'none',
              }}
            >
              {mobileNavItems.map((item) => {
                const Icon = item.icon
                return (
                  <button
                    key={item.id}
                    onClick={() => handleIconClick(item.id)}
                    className={`flex items-center justify-center rounded-lg transition-colors ${activePanel === item.id ? 'bg-primary/20 text-primary' : 'text-muted-foreground hover:text-foreground hover:bg-white/10'}`}
                    style={{ width: btnSize - 4, height: btnSize - 4 }}
                    title={item.label}
                  >
                    <Icon className="w-4 h-4" />
                  </button>
                )
              })}

              {/* Mobile avatar */}
              <div className="h-px w-6 bg-border/40" />
              <button
                onClick={() => handleIconClick('settings')}
                className={`rounded-full transition-colors ${activePanel === 'settings' ? 'bg-primary/20 ring-2 ring-primary' : 'bg-muted-foreground/20 hover:bg-muted-foreground/30'}`}
                style={{ width: 20, height: 20 }}
                title="Profile & Settings"
              />
            </div>
          </div>
        </div>

        {/* Mobile full-screen panel overlay */}
        {activePanel && activePanel !== 'chat' && (
          <div className="fixed inset-0 z-[9999] bg-background animate-panel-in">
            <div className="flex flex-col h-full">
              <div className="flex items-center px-4 h-14 border-b border-border/40 shrink-0 bg-card">
                <button
                  onClick={() => setActivePanel(null)}
                  className="p-2 -ml-2 rounded-lg text-muted-foreground hover:text-foreground hover:bg-secondary/60 transition-colors"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>
              <div className="flex-1 min-h-0 overflow-hidden">
                <PanelContent
                  panelId={activePanel}
                  onOpenModal={(type) => {
                    setModalType(type)
                    setActivePanel(null)
                  }}
                />
              </div>
            </div>
          </div>
        )}

        {/* Centered modal for Profile / Settings on mobile */}
        {modalType && (
          <ProfileSettingsModal
            type={modalType}
            onClose={() => setModalType(null)}
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
        />
      )}
    </div>
  )
}
