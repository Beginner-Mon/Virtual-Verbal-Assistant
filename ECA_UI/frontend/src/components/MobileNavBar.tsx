import { useState, useRef, useCallback, useEffect } from 'react'
import { Menu, X, Music2 } from 'lucide-react'
import type { PanelId, NavItem } from './FloatingNavBar'
import { useAuth } from '../contexts/AuthContext'
import { Avatar, AvatarImage, AvatarFallback } from './ui/avatar'

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
  const { userAttributes } = useAuth()
  const profilePicture = userAttributes?.picture

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
              <Avatar style={{ width: 24, height: 24 }}>
                <AvatarImage src={profilePicture} alt="Avatar" referrerPolicy="no-referrer" />
                <AvatarFallback className="bg-muted-foreground/20 text-muted-foreground">
                  <svg className="w-5 h-5" viewBox="0 0 638 543" fill="none" stroke="currentColor" strokeWidth="24">
                    <g transform="translate(0,543) scale(0.1,-0.1)">
                    <path d="M2505 5096 c-105 -34 -149 -62 -187 -117 -68 -101 -85 -264 -42 -409 23 -79 89 -207 129 -251 19 -22 35 -41 35 -44 0 -3 -19 -24 -42 -45 -39 -38 -42 -39 -59 -24 -10 9 -49 25 -86 35 l-67 19 -45 -27 c-53 -30 -205 -180 -272 -268 -111 -145 -184 -295 -244 -505 -134 -465 -149 -891 -49 -1350 84 -383 294 -815 484 -994 70 -65 153 -126 174 -126 5 0 -21 32 -59 70 -37 39 -65 73 -63 76 13 13 117 -74 141 -118 36 -64 73 -82 151 -74 105 11 140 18 154 32 25 26 31 14 34 -68 l3 -83 56 53 56 52 18 -45 c9 -26 16 -47 14 -49 -2 -2 -30 -22 -61 -45 -32 -23 -58 -43 -58 -44 -1 -28 -30 -219 -34 -226 -9 -14 14 -25 25 -14 6 6 16 59 24 118 8 59 16 109 18 112 2 2 26 19 52 38 37 26 52 32 62 23 7 -5 10 -14 7 -19 -9 -15 6 -49 22 -49 13 0 18 15 16 50 0 8 5 18 11 22 6 4 12 21 13 38 3 50 84 111 105 79 5 -8 9 -33 9 -56 0 -23 7 -48 15 -57 8 -8 15 -24 15 -37 0 -18 2 -20 16 -8 8 7 28 36 42 63 l27 51 3 -83 3 -83 28 21 c16 11 32 20 36 20 4 0 4 -41 0 -92 -6 -79 -5 -90 8 -85 44 17 199 122 229 155 34 38 92 162 105 222 9 46 20 45 44 -5 37 -79 43 -69 36 64 -6 124 6 159 19 54 3 -31 13 -68 20 -82 8 -14 14 -37 14 -51 0 -32 41 -160 51 -160 4 0 11 4 14 10 3 5 11 10 17 10 19 0 -1 -63 -33 -102 -38 -48 -21 -48 50 -2 60 40 95 83 106 135 10 46 26 69 48 69 10 0 27 4 37 10 13 7 19 6 24 -6 6 -16 -19 -137 -35 -166 -11 -22 -12 -38 0 -38 9 0 65 47 129 109 l35 35 33 -140 c18 -76 41 -178 50 -227 13 -61 22 -87 32 -87 12 0 13 10 8 46 -9 68 -85 396 -97 417 -8 14 -1 30 32 75 l41 57 -6 -52 c-7 -58 7 -78 31 -44 8 12 37 47 65 79 28 31 57 71 66 89 15 31 32 36 32 10 0 -7 -25 -48 -56 -90 -31 -43 -54 -80 -51 -83 3 -3 22 9 43 27 85 72 77 72 71 -1 -3 -36 -8 -76 -13 -89 l-7 -24 43 24 c59 33 142 117 176 178 16 28 32 51 36 51 4 0 34 39 66 88 33 48 67 94 76 101 16 13 17 10 10 -40 -10 -79 -53 -163 -129 -252 -37 -43 -64 -81 -61 -85 6 -6 69 37 151 102 50 40 205 229 237 289 71 133 77 488 12 842 -28 155 -83 420 -118 565 -13 58 -41 186 -61 285 -53 261 -70 328 -110 444 -80 230 -191 440 -307 580 -145 176 -390 339 -619 414 -184 60 -412 72 -714 37 -200 -23 -262 -37 -387 -86 l-92 -36 -24 29 c-44 53 -101 158 -124 228 -19 59 -23 91 -23 200 0 161 11 187 120 276 41 34 72 65 69 70 -3 5 -6 9 -7 8 -1 0 -20 -6 -42 -13z m1703 -1033 c54 -46 165 -173 158 -180 -2 -2 -56 48 -121 112 -118 116 -139 155 -37 68z m-143 -11 c24 -21 65 -62 91 -91 44 -48 45 -53 30 -70 -15 -17 -21 -13 -110 78 -84 84 -92 96 -78 108 22 17 18 18 67 -25z m-277 -54 c95 -97 107 -115 91 -134 -10 -12 -8 -22 9 -53 l22 -39 -47 -40 c-60 -49 -108 -74 -136 -70 -14 2 -50 46 -114 141 l-93 137 31 0 c16 0 43 7 58 15 30 15 91 85 91 104 0 22 18 10 88 -61z m-306 -220 c65 -100 83 -138 66 -138 -7 0 -29 -17 -50 -37 -21 -20 -39 -35 -42 -32 -2 2 -33 51 -68 109 -35 58 -72 117 -82 133 l-18 27 53 0 c30 0 61 5 69 10 8 5 16 10 17 10 1 0 26 -37 55 -82z m-305 -156 c63 -134 65 -142 48 -142 -17 0 -129 225 -120 240 12 19 28 -3 72 -98z m-169 -34 c22 -57 43 -120 46 -141 l7 -38 -81 7 c-78 7 -82 6 -109 -20 l-28 -27 -11 28 c-15 41 -40 123 -47 161 -6 27 -4 32 12 32 27 0 92 34 133 69 19 17 36 30 37 31 1 0 19 -46 41 -102z m-419 -68 c26 -124 27 -154 5 -146 -9 3 -22 6 -29 6 -20 0 -32 26 -59 129 l-25 91 29 0 c16 0 32 5 35 10 14 23 23 5 44 -90z m-184 -163 c-8 -17 -15 -40 -15 -51 0 -23 -40 -56 -40 -34 -1 7 -9 62 -19 122 l-18 109 36 33 36 33 18 -91 c15 -82 16 -94 2 -121z m-276 -78 c1 -33 -4 -51 -18 -63 -17 -16 -18 -15 -25 30 -10 74 -7 104 12 129 l17 23 7 -37 c4 -20 7 -57 7 -82z m-229 -471 c13 -35 31 -79 40 -98 10 -19 23 -62 30 -95 6 -33 16 -78 22 -100 5 -22 13 -58 17 -80 4 -23 11 -46 15 -53 4 -7 11 -34 14 -59 13 -89 50 -53 55 53 2 54 19 70 25 23 4 -33 22 -79 43 -110 10 -14 21 -38 25 -55 7 -29 84 -195 95 -204 6 -5 13 -33 25 -107 8 -43 -19 -156 -53 -223 -18 -35 -42 -64 -66 -79 -22 -15 -34 -28 -28 -34 14 -14 92 12 120 42 33 35 91 75 97 68 12 -12 26 -313 22 -479 -4 -172 -3 -177 14 -151 26 40 74 158 99 242 26 89 33 82 50 -57 11 -84 11 -109 0 -135 -7 -18 -20 -53 -29 -78 -16 -45 -18 -46 -72 -57 -112 -23 -146 -12 -185 60 -15 27 -33 52 -41 55 -8 3 -14 12 -14 20 0 20 -58 99 -86 118 -12 8 -30 34 -39 59 -13 33 -25 48 -46 55 -17 6 -29 17 -29 28 0 10 -14 36 -30 57 -17 22 -30 46 -30 53 0 15 -35 53 -49 53 -14 0 -61 59 -61 76 0 8 14 23 31 32 17 9 50 34 72 54 23 21 49 41 59 44 10 3 18 13 18 23 0 9 5 22 11 28 7 7 6 35 -6 94 -32 167 -54 343 -46 353 10 12 9 92 -2 116 -3 8 -29 76 -58 150 -48 127 -52 142 -56 248 -3 61 -3 112 0 112 2 0 15 -28 27 -62z" />
                    </g>
                  </svg>
                </AvatarFallback>
              </Avatar>
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
