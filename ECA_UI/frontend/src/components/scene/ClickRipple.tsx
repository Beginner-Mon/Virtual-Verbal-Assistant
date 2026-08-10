import { useEffect } from 'react'

interface ClickRippleProps {
  x: number
  y: number
  theme: 'light' | 'dark'
  onDone: () => void
}

export default function ClickRipple({ x, y, theme, onDone }: ClickRippleProps) {
  useEffect(() => {
    const timer = setTimeout(onDone, 700)
    return () => clearTimeout(timer)
  }, [onDone])

  const glow = theme === 'dark'
    ? 'rgba(96,165,250,0.8)'
    : 'rgba(37,99,235,0.6)'

  return (
    <>
      <style>{`
        @keyframes click-ripple {
          0% { transform: translate(-50%, -50%) scale(0); opacity: 0.6; }
          100% { transform: translate(-50%, -50%) scale(1); opacity: 0; }
        }
      `}</style>
      <div
        className="absolute pointer-events-none"
        style={{ left: x, top: y, zIndex: 50 }}
      >
        <div
          className="rounded-full border-2"
          style={{
            width: 60,
            height: 60,
            borderColor: glow,
            boxShadow: `0 0 14px ${glow}`,
            animation: 'click-ripple 0.6s ease-out forwards',
          }}
        />
      </div>
    </>
  )
}
