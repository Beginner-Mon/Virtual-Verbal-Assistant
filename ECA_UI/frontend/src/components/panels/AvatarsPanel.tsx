import { useState, type CSSProperties } from 'react'
import { UserRound, Check, TriangleAlert } from 'lucide-react'
import { ScrollArea } from '../ui/scroll-area'
import { useMotion } from '../../contexts/MotionContext'
import { incompatibilityReason, type Character } from '../../lib/characters'

/** [nền A, nền B + blob phụ, accent blob] */
const MESH_PALETTE: Array<[string, string, string]> = [
  ['#7c3aed', '#a855f7', '#e879f9'], // violet
  ['#0891b2', '#2563eb', '#22d3ee'], // cyan → blue
  ['#e11d48', '#ec4899', '#fb7185'], // rose
  ['#d97706', '#ea580c', '#fbbf24'], // amber
  ['#059669', '#0d9488', '#34d399'], // emerald
  ['#4f46e5', '#9333ea', '#818cf8'], // indigo
]

/** FNV-1a. Colours hang off the slug, not the list position — reordering the
 *  catalog used to repaint every character. */
function hashSlug(s: string): number {
  let h = 2166136261
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i)
    h = Math.imul(h, 16777619)
  }
  return h >>> 0
}

/** Two radial blobs over a full-cover base gradient, so every character gets a
 *  distinct grain even while they all share the same placeholder logo. Inline
 *  because Tailwind cannot emit per-slug gradient positions. */
function meshStyle(slug: string): CSSProperties {
  const h = hashSlug(slug)
  const [a, b, c] = MESH_PALETTE[h % MESH_PALETTE.length]
  // Blobs are forced into opposite halves; left to themselves they overlap into
  // a single smear and every card looks the same again. `flip` decides which
  // half the bright accent takes, otherwise every card lights up top-left.
  const flip = (h >>> 27) & 1
  const near = { x: 8 + ((h >>> 3) % 34), y: 4 + ((h >>> 9) % 36) } //  8–42% / 4–40%
  const far = { x: 58 + ((h >>> 15) % 34), y: 56 + ((h >>> 21) % 36) } // 58–92% / 56–92%
  const accent = flip ? far : near
  const wash = flip ? near : far
  return {
    backgroundImage: [
      `radial-gradient(95% 75% at ${accent.x}% ${accent.y}%, ${c} 0%, transparent 58%)`,
      `radial-gradient(110% 85% at ${wash.x}% ${wash.y}%, ${b} 0%, transparent 62%)`,
      `linear-gradient(${flip ? 320 : 140}deg, ${a} 0%, ${b} 100%)`,
    ].join(', '),
  }
}

/**
 * Display names used to come from a hardcoded map here, which is why `anne`
 * had none and fell through to a title-cased filename. They live on the
 * character record now, so adding a character no longer means editing this file.
 */
function AvatarCard({
  slug,
  displayName,
  thumbnailUrl,
  disabledReason,
  isSelected,
  onClick,
}: {
  slug: string
  displayName: string
  thumbnailUrl: string | null
  disabledReason: string | null
  isSelected: boolean
  onClick: () => void
}) {
  const [imgFailed, setImgFailed] = useState(false)
  const disabled = disabledReason !== null

  return (
    <button
      type="button"
      onClick={disabled ? undefined : onClick}
      disabled={disabled}
      aria-pressed={isSelected}
      title={disabledReason ?? displayName}
      className="
        group relative w-full aspect-[5/6] rounded-xl cursor-pointer
        disabled:cursor-not-allowed
        transition-shadow duration-200
        focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring
        focus-visible:ring-offset-2 focus-visible:ring-offset-card
      "
    >
      {/* Clip layer — anything painted inside is guaranteed to follow the radius.
          The old label sat directly on the button with `backdrop-blur`, which
          Chromium refuses to clip against an ancestor's border-radius; that is
          what made the name bar bleed out of the bottom corners. */}
      <div className="absolute inset-0 rounded-[inherit] overflow-hidden">
        <div className={`absolute inset-0 ${disabled ? 'grayscale opacity-60' : ''}`}>
          {/* Mesh + logo: the placeholder layer, always present. A real
              thumbnail is painted on top of it, so it doubles as the
              while-loading state without any extra bookkeeping. */}
          <div
            style={meshStyle(slug)}
            className="
              absolute inset-0 flex items-center justify-center
              transition-transform duration-500 ease-out group-hover:scale-110
            "
          >
            {/* eca-logo.svg is 638×543 with `meet`, so in a square box it fits by
                width and pads top/bottom. 114% of the card width reproduces the
                old `w-40` on a 140px card at any card size — do not "fix" it. */}
            <img
              src="/eca-logo.svg"
              alt=""
              className="
                w-[114%] h-auto max-w-none select-none
                opacity-30 group-hover:opacity-45 transition-opacity duration-200
              "
              style={{ filter: 'brightness(0) invert(1)' }}
            />
          </div>

          {/* No thumbnails exist yet — characters.thumbnail_url is null for every
              row — but when they land they cover the placeholder. */}
          {thumbnailUrl && !imgFailed && (
            <img
              src={thumbnailUrl}
              alt=""
              loading="lazy"
              onError={() => setImgFailed(true)}
              className="absolute inset-0 w-full h-full object-cover select-none"
            />
          )}
        </div>

        {/* Scrim + name. Gradient, not a blurred bar: see the clip-layer note. */}
        <div
          className={`
            absolute inset-x-0 bottom-0 pt-6 pb-2 px-2 text-center
            bg-gradient-to-t from-black/85 via-black/60 to-transparent
            transition-all duration-200 ease-out
            ${disabled
              ? 'translate-y-0 opacity-100'
              : `translate-y-full opacity-0
                 group-hover:translate-y-0 group-hover:opacity-100
                 group-focus-visible:translate-y-0 group-focus-visible:opacity-100`
            }
          `}
        >
          <span className="text-xs font-medium text-white truncate block">
            {displayName}
          </span>
          {disabledReason && (
            <span className="mt-0.5 flex items-start justify-center gap-1 text-[10px] text-white/70">
              <TriangleAlert className="w-3 h-3 shrink-0 mt-px" />
              <span className="line-clamp-2 text-left">{disabledReason}</span>
            </span>
          )}
        </div>
      </div>

      {/* Rings live on top of the artwork and are inset, so they track the radius
          instead of floating outside the border box the way `ring-2` did. */}
      <span
        className={`
          absolute inset-0 rounded-[inherit] pointer-events-none transition-all duration-200
          ${isSelected
            ? 'ring-2 ring-inset ring-primary'
            : disabled ? '' : 'ring-0 ring-inset ring-white/40 group-hover:ring-1'
          }
        `}
      />

      {isSelected && (
        <span
          className="
            absolute top-2 right-2 w-5 h-5 rounded-full
            bg-primary text-primary-foreground
            flex items-center justify-center shadow-sm pointer-events-none
          "
        >
          <Check className="w-3 h-3" strokeWidth={3} />
        </span>
      )}
    </button>
  )
}

export default function AvatarsPanel() {
  const {
    selectedVrmId,
    setSelectedVrmId,
    vrmOptions,
    vrmOptionsLoading,
    vrmOptionsError,
  } = useMotion()

  return (
    <div className="flex flex-col h-full">
      <div className="px-4 py-3 border-b border-border/40 shrink-0">
        <h2 className="text-sm font-semibold text-foreground tracking-tight flex items-center gap-2">
          <UserRound className="w-4 h-4 text-primary" />
          Characters
        </h2>
        <p className="text-[11px] text-muted-foreground mt-0.5">Choose a 3D avatar</p>
      </div>

      <ScrollArea className="flex-1 min-h-0">
        <div className="p-4">
          {/* Skeletons in the real grid, so nothing jumps when the catalog lands. */}
          {vrmOptionsLoading && (
            <div className="grid grid-cols-2 gap-3">
              {Array.from({ length: 4 }, (_, i) => (
                <div key={i} className="w-full aspect-[5/6] rounded-xl bg-secondary/60 animate-pulse" />
              ))}
            </div>
          )}

          {/* Named, not swallowed: an empty grid and a failed fetch look identical
              otherwise, and the difference decides whether anyone goes looking. */}
          {vrmOptionsError && (
            <p className="text-xs text-destructive text-center py-4 px-2">
              Could not load characters.
              <br />
              <span className="text-muted-foreground">{vrmOptionsError}</span>
            </p>
          )}

          {!vrmOptionsLoading && !vrmOptionsError && vrmOptions.length === 0 && (
            <p className="text-xs text-muted-foreground text-center py-4">No VRM avatars found</p>
          )}

          {!vrmOptionsLoading && vrmOptions.length > 0 && (
            <div className="grid grid-cols-2 gap-3">
              {vrmOptions.map((option) => {
                const character: Character | undefined = option.character
                return (
                  <AvatarCard
                    key={option.id}
                    slug={option.id}
                    displayName={option.label}
                    thumbnailUrl={character?.thumbnail_url ?? null}
                    disabledReason={character ? incompatibilityReason(character) : null}
                    isSelected={selectedVrmId === option.id}
                    onClick={() => setSelectedVrmId(option.id)}
                  />
                )
              })}
            </div>
          )}
        </div>
      </ScrollArea>
    </div>
  )
}
