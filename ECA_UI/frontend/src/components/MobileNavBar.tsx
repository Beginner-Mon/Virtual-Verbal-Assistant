import { useState, useRef, useCallback, useEffect } from 'react'
import { Menu, X, Music2 } from 'lucide-react'
import type { PanelId, NavItem } from './FloatingNavBar'
import { useAvatarBg } from '../hooks/useAvatarBg'
import AvatarWithLogo from './AvatarWithLogo'

interface MobileNavBarProps {
  activePanel: PanelId
  onIconClick: (id: PanelId) => void
  onOpenModal: (type: 'profile' | 'settings') => void
  navItems: NavItem[]
  panelContent?: React.ReactNode
  isMusicPlaying: boolean
  toggleMusic: () => void
}

export default function MobileNavBar({
  activePanel,
  onIconClick,
  navItems,
  panelContent,
  isMusicPlaying,
  toggleMusic,
}: MobileNavBarProps) {
  const { bg } = useAvatarBg()

  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)
  const mobileMenuRef = useRef<HTMLDivElement>(null)
  const btnSize = 36

  const menuXRef = useRef(0)
  const menuYRef = useRef(80)
  const menuIsDraggingRef = useRef(false)
  const menuDragStart = useRef<{ x: number; y: number; startMenuX: number; startMenuY: number; target?: HTMLElement } | null>(null)
  const menuInitRef = useRef(false)

  useEffect(() => {
    if (menuInitRef.current) return
    menuInitRef.current = true
    const x = window.innerWidth - btnSize - 8
    const y = 80
    menuXRef.current = x
    menuYRef.current = y
    if (mobileMenuRef.current) {
      mobileMenuRef.current.style.transform = `translate(${x}px, ${y}px)`
    }
  }, [])

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

  const getMenuYBounds = useCallback(() => {
    const chatTop = window.innerHeight * 0.6
    return { minY: 80, maxY: chatTop - btnSize - 8 }
  }, [])

  // Plan A1 (mobile companion): keep hamburger inside viewport after window resize
  useEffect(() => {
    let timer: ReturnType<typeof setTimeout> | null = null
    const handleResize = () => {
      if (timer) clearTimeout(timer)
      timer = setTimeout(() => {
        const newX = Math.max(8, Math.min(menuXRef.current, window.innerWidth - btnSize - 8))
        const { minY, maxY } = getMenuYBounds()
        const newY = Math.max(minY, Math.min(menuYRef.current, maxY))
        menuXRef.current = newX
        menuYRef.current = newY
        if (mobileMenuRef.current) {
          mobileMenuRef.current.style.transform = `translate(${newX}px, ${newY}px)`
        }
      }, 100)
    }
    window.addEventListener('resize', handleResize)
    return () => {
      window.removeEventListener('resize', handleResize)
      if (timer) clearTimeout(timer)
    }
  }, [getMenuYBounds])

  const handleMenuPointerDown = useCallback((e: React.PointerEvent) => {
    const el = e.currentTarget as HTMLElement
    el.setPointerCapture(e.pointerId)
    el.style.transition = 'none'
    menuDragStart.current = { 
      x: e.clientX, 
      y: e.clientY, 
      startMenuX: menuXRef.current, 
      startMenuY: menuYRef.current,
      target: e.target as HTMLElement
    }
    menuIsDraggingRef.current = false
  }, [])

  const handleMenuPointerMove = useCallback((e: React.PointerEvent) => {
    if (!menuDragStart.current) return
    const dx = e.clientX - menuDragStart.current.x
    const dy = e.clientY - menuDragStart.current.y

    if (!menuIsDraggingRef.current && Math.abs(dx) < 5 && Math.abs(dy) < 5) return

    menuIsDraggingRef.current = true

    const newX = Math.max(8, Math.min(window.innerWidth - btnSize - 8, menuDragStart.current.startMenuX + dx))
    const { minY, maxY } = getMenuYBounds()
    const newY = Math.max(minY, Math.min(maxY, menuDragStart.current.startMenuY + dy))

    menuXRef.current = newX
    menuYRef.current = newY

    const el = e.currentTarget as HTMLElement
    el.style.transform = `translate(${newX}px, ${newY}px)`
  }, [getMenuYBounds])

  const handleMenuPointerUp = useCallback((e: React.PointerEvent) => {
    const wasDragging = menuIsDraggingRef.current
    const target = menuDragStart.current?.target
    const el = e.currentTarget as HTMLElement
    menuDragStart.current = null
    menuIsDraggingRef.current = false

    el.style.transition = ''

    if (wasDragging) {
      const center = menuXRef.current + btnSize / 2
      const snapX = center < window.innerWidth / 2 ? 8 : window.innerWidth - btnSize - 8
      menuXRef.current = snapX
      el.style.transform = `translate(${snapX}px, ${menuYRef.current}px)`
    } else {
      // It was a click, not a drag. Because we captured the pointer, the native
      // click event won't fire on the original child target. We simulate it here.
      if (target && typeof target.click === 'function') {
        target.click()
      }
    }
    
    if (el.hasPointerCapture(e.pointerId)) {
      el.releasePointerCapture(e.pointerId)
    }
  }, [])

  const handleMenuPointerCancel = useCallback(() => {
    menuDragStart.current = null
    menuIsDraggingRef.current = false
    if (mobileMenuRef.current) {
      mobileMenuRef.current.style.transition = ''
    }
  }, [])

  const mobileNavItems = navItems.filter((item) => item.id !== 'chat')

  return (
    <>
      <div
        ref={mobileMenuRef}
        style={{
          position: 'fixed',
          top: 0,
          left: 0,
          zIndex: 10000,
          touchAction: 'none',
          willChange: 'transform',
          transition: 'transform 0.3s cubic-bezier(0.16, 1, 0.3, 1)',
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
                  onClick={() => {
                    onIconClick(item.id)
                    setMobileMenuOpen(false)
                  }}
                  className={`flex items-center justify-center rounded-lg transition-colors ${activePanel === item.id ? 'bg-primary/20 text-primary' : 'text-muted-foreground hover:text-foreground hover:bg-white/10'}`}
                  style={{ width: btnSize - 4, height: btnSize - 4 }}
                  title={item.label}
                >
                  <Icon className="w-4 h-4" />
                </button>
              )
            })}

            {/* Background Music Toggle */}
            <button
              onClick={toggleMusic}
              title={isMusicPlaying ? 'Pause Music' : 'Play Music'}
              className="relative flex items-center justify-center text-muted-foreground transition-all duration-200"
              style={{ width: btnSize - 4, height: btnSize - 4 }}
            >
              <Music2 className="w-4 h-4" />
              {!isMusicPlaying && (
                <div className="absolute top-1/2 left-1/2 w-4 h-[1.5px] bg-muted-foreground -translate-x-1/2 -translate-y-1/2 -rotate-[45deg]" />
              )}
            </button>

            <div className="h-px w-6 bg-border/40" />
            <button
              onClick={() => {
                onIconClick('settings')
                setMobileMenuOpen(false)
              }}
              className={`rounded-full transition-colors`}
              title="Profile & Settings"
            >
              <AvatarWithLogo size="xs" bgClassName={bg.className} logoClassName={bg.logoClassName} />
            </button>
          </div>
        </div>
      </div>

      {activePanel && activePanel !== 'chat' && (
        <div className="fixed inset-0 z-[9999] flex items-end">
          <div
            className="absolute inset-0 bg-black/40 backdrop-blur-sm"
            onClick={() => onIconClick(activePanel)}
          />
          <div className="relative w-full h-auto max-h-[80vh] bg-card rounded-t-2xl animate-slide-up flex flex-col overflow-hidden shadow-[0_-8px_40px_rgba(0,0,0,0.4)]">
            <div className="flex items-center px-4 h-14 border-b border-border/40 shrink-0">
              <button
                onClick={() => onIconClick(activePanel)}
                className="p-2 -ml-2 rounded-lg text-muted-foreground hover:text-foreground hover:bg-secondary/60 transition-colors"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            <div className="flex-1 min-h-0 overflow-hidden">
              {panelContent}
            </div>
          </div>
        </div>
      )}
    </>
  )
}
