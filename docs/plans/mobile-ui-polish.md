# Mobile UI Polish Plan

## 1. Chat Panel: Backdrop-only, Chat Box White

### Problem
- Mobile chat container uses `background: rgba(0,0,0,0.15)` (MainLayout.tsx:24) plus `backdrop-blur-md` — background color + blur
- ChatPanel itself uses `bg-transparent md:bg-card/50 md:backdrop-blur-xl` (ChatPanel.tsx:151) — adds background on desktop
- Input area uses `bg-secondary/40` (ChatPanel.tsx:188)
- User wants: panel = backdrop only (transparent), the typing area = white background

### Changes

**MainLayout.tsx:24** — Remove `background: rgba(0,0,0,0.15)` from mobile chat container, keep `backdrop-blur-md`

**ChatPanel.tsx:151** — Remove `md:bg-card/50`, keep only `md:backdrop-blur-xl`

**ChatPanel.tsx:187-188** — Add white background to the input container div. For dark mode, use a subtle dark card color instead.

---

## 2. Responsive Text Sizing (iPhone ↔ iPad)

### Problem
- ChatMessage.tsx uses `text-xs md:text-sm` (line 18, 30)
- `text-xs` = 0.75rem (12px) looks good on small iPhone (~375px)
- `text-sm` = 0.875rem (14px) at `md:` breakpoint (768px)
- Between 376–767px, text stays 12px which looks too small on larger screens (iPad, landscape)
- User wants fluid scaling based on viewport width

### Solution
Replace `text-xs md:text-sm` with `text-[clamp(0.75rem,0.5rem+1.2cqi,0.875rem)]` using container query inline size (`cqi`) or viewport width (`vw`). This fluidly scales between 12px–14px as the container/viewport grows.

Apply the same clamp to other `text-xs md:text-sm` patterns throughout the component (ChatMessage.tsx, ChatPanel.tsx).

Alternatively, use `sm:text-sm` (640px) as an intermediate step if a simpler solution is preferred:
- `text-xs sm:text-sm` would jump to 14px at 640px instead of 768px

---

## 3. Mobile Menu: Pop Up When Near Chat Panel

### Problem
- FloatingNavBar's mobile menu always expands downward (icons appear below hamburger)
- When menu is dragged near the bottom (close to chat panel at 60vh), the downward expansion overlaps the chat panel or goes off-screen
- User wants: when near the display message area (bottom), the menu should pop UP instead of down

### Solution
In FloatingNavBar.tsx, modify the mobile menu icon list (lines 520-553):

1. Calculate `isNearBottom` — when `menuY + btnSize + dropdownHeight > chatPanelTop` (approximately > 55-60vh)
2. When `isNearBottom`:
   - Reverse the flex order so icons appear ABOVE the hamburger button
   - Change the `transform-origin` to `bottom center` so it "grows up"
   - Add a CSS slide-up animation (translateY from +8px to 0) instead of slide-down
3. When not near bottom:
   - Keep current behavior (icons below hamburger, top-to-bottom expansion)

---

## 4. Modals as Bottom Sheets + Logo Z-Index Fix

### Problem
- ProfileSettingsModal and ModalOverlay use centered positioning (`items-center justify-center`) with `animate-panel-in` (scale + blur)
- Logo uses `z-[10000]` (MainLayout.tsx:13), same as modals `z-[10000]`
- User wants: modals to slide up from bottom like bottom sheets, and overlay ALL content including logo

### Solution

**Z-index fix** (MainLayout.tsx:13):
- Lower logo to `z-[9990]` so modals at `z-[10000]` overlay it

**Animation** (App.css):
- Add new `slide-up` keyframe: `translateY(100%)` → `translateY(0)`, no scale/blur
- Add `animate-slide-up` class

**ProfileSettingsModal.tsx:29**:
- Change `items-center justify-center` to `items-end justify-center`
- Replace `animate-panel-in` with `animate-slide-up`
- Remove `h-[600px]` constraint; let it be `max-h-[85vh]` with auto height
- Add rounded top corners only (`rounded-t-2xl rounded-b-none`)
- Keep same width constraint

**ModalOverlay.tsx:28,36**:
- Same treatment: bottom sheet style, `items-end`, `animate-slide-up`
- `max-h-[85vh]`, `rounded-t-2xl rounded-b-none`, `w-full` or full-width

---

## Files to Modify

| File | Changes |
|------|---------|
| `src/layouts/MainLayout.tsx` | Remove background from mobile chat container; lower logo z-index |
| `src/components/ChatPanel.tsx` | Remove md:bg-card/50, add white bg to input area |
| `src/components/ChatMessage.tsx` | Fluid text sizing with clamp() |
| `src/components/FloatingNavBar.tsx` | Mobile menu directional logic (up/down based on position) |
| `src/components/ProfileSettingsModal.tsx` | Bottom sheet styling |
| `src/components/ModalOverlay.tsx` | Bottom sheet styling |
| `src/App.css` | Add `slide-up` keyframe |

## Priority & Sequence

1. Bottom sheet modals + logo z-index (visual fix, unblocks other work)
2. Chat panel backdrop cleanup
3. Responsive text sizing
4. Mobile menu direction logic
