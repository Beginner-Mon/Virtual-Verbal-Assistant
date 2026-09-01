# Plan: Avatar Background Customization — Edit Badge + Nested Picker Modal

> Owner: Product Owner | K: Senior Solution Architect | N: Senior Developer
> Branch: `feature/langgraph-rewrite` | Ngày: 31-08-2026
> Trạng thái: Draft — chờ duyệt trước khi code

## 1) Mục tiêu — Mr. Senryuu

Cho phép user đổi màu nền avatar cá nhân ngay trong `ProfileContent` (trang Profile mở từ avatar tròn trên `FloatingNavBar`).

Scope đợt này **chỉ UI, không đụng backend/DB/Cognito/DynamoDB** — pure frontend. Bấm icon edit **hình tròn nhỏ góc dưới-phải avatar (icon `Pen` lucide-react)** -> mở modal lồng bên trong `ProfileContent`, modal hiển thị avatar lớn ở trên + các option màu ở dưới. Màu đổi phải **đồng bộ mọi chỗ hiển thị avatar** (FloatingNavBar desktop + MobileNavBar + ProfileContent). Avatar hiện tại **chỉ là màu nền + logo ECA**, không dùng ảnh Google nữa — đổi màu chính là đổi cái đang thấy.

## 2) Phạm vi

### In-scope (CHỈ UI — lần này)
- Icon edit **hình tròn nhỏ góc dưới-phải avatar** trong `ProfileContent` (`Profile` section) — dùng `Pen` lucide-react, bấm mới hiện modal.
- State `showAvatarPicker` + modal lồng `AvatarPickerModal` (skeleton, chưa save).
- `AvatarWithLogo` nhận `bgClassName` để đổi nền — **bỏ prop `profilePicture`**, chỉ đổi màu `AvatarFallback` (hiện tại chính là avatar).
- Palette màu cố định (8 màu) định nghĩa tập trung.
- **Đồng bộ toàn app**: đổi 1 lần -> FloatingNavBar + MobileNavBar + ProfileContent cùng đổi ngay (qua React Context).

### Out-of-scope (KHÔNG làm)
- Backend, DB, Cognito, DynamoDB, API — bỏ hết.
- Animation / upload ảnh custom.
- Đồng bộ màu sang 3D scene / VRM.
- Nút Save/Cancel trong modal (chỉ chọn là đổi ngay, modal skeleton).

## 3) Hiện trạng (As-Is)

```
FloatingNavBar (avatar tròn nhỏ) ──click──> SettingsPanel ──Profile──> ProfileSettingsModal (z-[10000])
                                                                             └── ProfileContent
                                                                                  ├── AvatarWithLogo size="md"  ← chỉ còn AvatarFallback bg-muted-foreground/20 + <EcaLogo> (không còn AvatarImage)
                                                                                  ├── Display Name (input chưa gắn handler)
                                                                                  ├── Security (Set my Password)
                                                                                  └── Account Linked (Google)
```

- `ECA_UI/frontend/src/components/AvatarWithLogo.tsx:16-25` — hiện nhận `size` + `profilePicture`, nhưng **bạn đã bỏ dùng ảnh Google** -> đợt này sẽ **bỏ prop `profilePicture`**, component chỉ render `AvatarFallback` (màu nền + logo). Cái "fallback" trước đây **chính là avatar hiện tại** — hiểu đúng rồi, đổi màu fallback = đổi luôn avatar đang thấy.
- `ECA_UI/frontend/src/components/ProfileContent.tsx:159-166` — avatar render trần, không có wrapper relative / nút edit.
- `ProfileSettingsModal` đã là modal `fixed inset-0 z-[10000]` với `backdrop-blur` + `overflow-hidden` (`ProfileSettingsModal.tsx:87-95`), modal lồng phải xử lý z-index + scroll lock.

### Fallback là gì? (làm rõ theo yêu cầu mới của bạn)
- Trước đây `AvatarWithLogo` có 2 lớp: `AvatarImage` (ảnh Google) + `AvatarFallback` (màu + logo). Fallback chỉ hiện khi không có ảnh.
- **Nay bạn đã bỏ ảnh Google** -> `AvatarImage` không còn, **chỉ còn `AvatarFallback`**. Nó không còn là "dự phòng" nữa mà là **avatar chính**. Đổi `bgClassName` của nó là đổi trực tiếp màu nền avatar ở mọi chỗ — đúng như bạn nói.

## 4) Thiết kế (To-Be)

```
ProfileContent
 ├── div.relative (wrap avatar)
 │     ├── AvatarWithLogo size="md" bgClassName={bg.className}  ← đổi màu trực tiếp, không còn profilePicture
 │     └── button.edit-badge absolute -right-1 -bottom-1 rounded-full w-7 h-7 bg-card border shadow-md  ← hình tròn nhỏ, icon Pen
 │           └── onClick => setShowPicker(true)
 └── {showPicker && <AvatarPickerModal />}   ← mới, fixed overlay z-[10001]
           ├── overlay bg-black/50
           ├── card: avatar preview lớn (AvatarWithLogo size="lg" bgClassName) ở trên
           └── grid palette 4x2 (option buttons) ở dưới — chọn xong set Context ngay
```

### 4.1 Quyết định kiến trúc — CHỈ UI, đồng bộ toàn app

**Bắt buộc đồng bộ toàn app** (FloatingNavBar + MobileNavBar + ProfileContent):

- Tạo `contexts/AvatarBgContext.tsx` (React Context + Provider) bọc ở `MainLayout` hoặc `App.tsx`.
- Context lưu `colorId` trong `useState`. Có 2 lựa chọn lưu:
  - **Option A (khuyên dùng nếu bạn muốn giữ sau reload):** + `localStorage vva_avatar_bg` — đọc 1 lần lúc mount, ghi mỗi khi đổi.
  - **Option B (pure memory, theo đúng "chỉ UI không lưu gì"):** chỉ `useState`, reload là về mặc định.
- Mặc định plan viết theo **Option A**, nếu bạn không muốn lưu gì thì bỏ 2 dòng `localStorage` là xong — Context vẫn đảm bảo đổi 1 chỗ re-render cả 3 chỗ ngay.
- Mọi nơi hiển thị avatar đều `useAvatarBg()` để lấy `bg.className` -> đổi 1 chỗ, cả 3 chỗ đổi ngay.
- Không dùng Cognito/DynamoDB/Postgres — bỏ khỏi scope.

```
App
 └── AvatarBgProvider (new — holds colorId + localStorage)
      ├── FloatingNavBar -> AvatarWithLogo bgClassName={bg.className}
      ├── MobileNavBar   -> AvatarWithLogo bgClassName={bg.className}
      └── ProfileSettingsModal -> ProfileContent -> AvatarWithLogo bgClassName={bg.className} + edit badge
```

### 4.2 Palette

File mới `ECA_UI/frontend/src/lib/avatarPalette.ts`:

```ts
export const AVATAR_BG_OPTIONS = [
  { id: 'slate',  className: 'bg-slate-500',    value: '#64748b' },
  { id: 'violet', className: 'bg-violet-500',   value: '#8b5cf6' },
  { id: 'blue',   className: 'bg-sky-500',      value: '#0ea5e9' },
  { id: 'emerald',className: 'bg-emerald-500',  value: '#10b981' },
  { id: 'amber',  className: 'bg-amber-500',    value: '#f59e0b' },
  { id: 'rose',   className: 'bg-rose-500',     value: '#f43f5e' },
  { id: 'cyan',   className: 'bg-cyan-500',     value: '#06b6d4' },
  { id: 'indigo', className: 'bg-indigo-500',   value: '#6366f1' },
] as const
export type AvatarBgId = typeof AVATAR_BG_OPTIONS[number]['id']
```

Mặc định `slate` trùng `bg-muted` hiện tại để không đổi visual với user cũ.

## 5) Chi tiết thay đổi từng file

### 5.1 `AvatarWithLogo.tsx` — XÓA SẠCH code ảnh Google, chỉ đổi màu nền

**Hiện trạng cần xóa:**
- `AvatarWithLogo.tsx:1` `import { Avatar, AvatarImage, AvatarFallback }`
- `AvatarWithLogo.tsx:13` `profilePicture?: string`
- `AvatarWithLogo.tsx:20` `<AvatarImage src={profilePicture} ... />`

**Sau khi xóa:**

```tsx
import { Avatar, AvatarFallback } from './ui/avatar' // bỏ AvatarImage
import { cn } from '@/lib/utils'
interface AvatarWithLogoProps {
  size: keyof typeof SIZES
  bgClassName?: string  // ví dụ "bg-violet-500" — màu nền chính
}
export default function AvatarWithLogo({ size, bgClassName }: AvatarWithLogoProps) {
  const s = SIZES[size]
  return (
    <Avatar className={s.avatar}>
      <AvatarFallback className={cn("bg-muted-foreground/20", bgClassName)}>
        <EcaLogo className={s.logo} />
      </AvatarFallback>
    </Avatar>
  )
}
```

- **Xóa sạch `profilePicture` + `<AvatarImage>` + import** — avatar giờ chỉ là `AvatarFallback` (màu nền + logo). Đổi `bgClassName` là đổi trực tiếp cái user thấy ở mọi chỗ.
- `bgClassName` optional, default `bg-muted-foreground/20`.
- Đồng thời xóa ở 3 nơi dùng:
  - `FloatingNavBar.tsx:147-148` `const profilePicture = userAttributes?.picture` + `useAuth` import nếu không còn dùng
  - `FloatingNavBar.tsx:238` `profilePicture={profilePicture}`
  - `MobileNavBar.tsx:4,25-26,255` tương tự
  - `ProfileContent.tsx:61,160` tương tự (xóa `userAttributes`, `profilePicture`)

### 5.2 `lib/avatarPalette.ts` + `contexts/AvatarBgContext.tsx` — mới (UI-only, global sync)

```ts
// lib/avatarPalette.ts — 8 màu, export AVATAR_BG_OPTIONS

// contexts/AvatarBgContext.tsx
const AvatarBgContext = createContext<{ colorId: AvatarBgId; bg: typeof AVATAR_BG_OPTIONS[number]; setColorId: (id: AvatarBgId) => void } | null>(null)

export function AvatarBgProvider({ children }: { children: React.ReactNode }) {
  // Option A: có localStorage (giữ sau reload) — nếu bạn không muốn thì xóa 2 dòng localStorage
  const [colorId, setColorId] = useState<AvatarBgId>(() => 
    (localStorage.getItem('vva_avatar_bg') as AvatarBgId) || 'slate'
  )
  useEffect(() => { localStorage.setItem('vva_avatar_bg', colorId) }, [colorId])
  // Option B: pure memory — const [colorId, setColorId] = useState<AvatarBgId>('slate')
  const bg = AVATAR_BG_OPTIONS.find(o => o.id === colorId)!
  return <AvatarBgContext.Provider value={{ colorId, bg, setColorId }}>{children}</AvatarBgContext.Provider>
}
export const useAvatarBg = () => { const ctx = useContext(AvatarBgContext); if (!ctx) throw new Error(...); return ctx }
```

- Provider bọc ở `layouts/MainLayout.tsx` (hoặc `App.tsx` trong `<AuthGuard>`), đảm bảo FloatingNavBar + MobileNavBar + ProfileContent cùng đọc 1 source.
- Đổi `colorId` ở ProfileContent -> Context re-render -> cả 3 avatar đổi ngay.
- **localStorage để làm gì?** Chỉ để **giữ màu sau khi reload trang**. Nếu không dùng, `useState` sẽ reset về `slate` mỗi lần reload. Bạn không thích thì bỏ 2 dòng `localStorage` là thành pure UI memory — vẫn đồng bộ toàn app trong phiên.

### 5.3 `ProfileContent.tsx` — thêm edit badge + nested modal trigger

Vị trí `ProfileContent.tsx:159-166` hiện:

```tsx
<div className="flex items-center gap-4">
  <AvatarWithLogo size="md" profilePicture={profilePicture} />
  ...
</div>
```

Đổi thành:

```tsx
const { colorId, setColorId, bg } = useAvatarBg() // global, sync toàn app
const [showPicker, setShowPicker] = useState(false)

<div className="flex items-center gap-4">
  <div className="relative shrink-0">
    <AvatarWithLogo size="md" bgClassName={bg.className} />
    <button
      onClick={() => setShowPicker(true)}
      aria-label="Edit avatar background"
      className="absolute -right-1 -bottom-1 w-7 h-7 rounded-full bg-card border border-border shadow-md flex items-center justify-center text-muted-foreground hover:text-foreground hover:bg-secondary transition-colors"
    >
      <Pen className="w-3.5 h-3.5" /> {/* lucide-react Pen, hình tròn nhỏ góc dưới-phải */}
    </button>
  </div>
  ...
</div>

{showPicker && (
  <AvatarPickerModal
    currentColorId={colorId}
    onSelect={(id) => setColorId(id)} // đổi là sync toàn app ngay (Context)
    onClose={() => setShowPicker(false)}
  />
)}
```

- Badge dùng `Pen` (lucide-react) theo yêu cầu — hình tròn nhỏ `w-7 h-7 rounded-full` bám góc dưới-phải.
- `absolute -right-1 -bottom-1` + `border` + `shadow-md` để nổi trên avatar.

### 5.4 `components/AvatarPickerModal.tsx` — mới, skeleton theo yêu cầu

Yêu cầu: modal bên trong ProfileContent, hiển thị hình avatar ở trên và các option ở dưới.

```tsx
interface Props {
  currentColorId: AvatarBgId
  onSelect: (id: AvatarBgId) => void
  onClose: () => void
}
```

Layout:

```
[overlay fixed inset-0 bg-black/50 z-[10001] onClick=onClose]
[card centered max-w-sm rounded-2xl bg-card border shadow-xl]
  ├── header: "Choose avatar background" + X button
  ├── preview: AvatarWithLogo size="lg" bgClassName={current.bg} (căn giữa, py-6) — chính là avatar đang đổi
  └── grid: 4 cột, mỗi option là button tròn w-12 h-12 với bg tương ứng + ring khi selected + Check icon
```

- Dùng `fixed inset-0` riêng, không phụ thuộc `ProfileSettingsModal` overflow — tránh bị clip.
- `z-[10001]` > `z-[10000]` của `ProfileSettingsModal`.
- Không có nút Save/Cancel đợt này (theo yêu cầu "tới đó thôi") — `onSelect` đổi local ngay, `onClose` chỉ đóng.
- Keyboard: `Escape` đóng, `onPointerDown` ngoài card đóng (copy pattern `ProfileSettingsModal.tsx:90-92`).
- Scroll lock: `ProfileSettingsModal` đã lock `body.overflow=hidden`, modal lồng không cần thêm.

### 5.5 `FloatingNavBar.tsx` / `MobileNavBar.tsx` — BẮT BUỘC (đồng bộ toàn app)

- `FloatingNavBar.tsx:238` đổi từ `<AvatarWithLogo size="sm" profilePicture={profilePicture} />` thành `<AvatarWithLogo size="sm" bgClassName={bg.className} />` — lấy `bg` từ `useAvatarBg()`, bỏ `profilePicture`
- `MobileNavBar.tsx:26` tương tự — bỏ `profilePicture`, chỉ dùng `bgClassName`
- `layouts/MainLayout.tsx` — bọc `AvatarBgProvider` quanh `{children}` (hoặc `App.tsx` trong `<AuthGuard>`).

## 6) Luồng tương tác (UI-only, global sync)

1. User mở Profile: click avatar tròn `FloatingNavBar`/`MobileNavBar` -> `SettingsPanel` -> `Profile`.
2. Trong `ProfileContent`, thấy avatar `md` với **badge hình tròn nhỏ góc dưới-phải, icon `Pen`** .
3. Click badge -> `showPicker=true` -> `AvatarPickerModal` overlay `z-[10001]` hiện (avatar lớn preview ở trên, grid 8 màu ở dưới).
4. Click màu -> `setColorId` trong `AvatarBgContext` -> **cả 3 nơi cùng re-render ngay**: ProfileContent preview + FloatingNavBar avatar nhỏ + MobileNavBar avatar nhỏ đổi màu đồng thời (vì avatar giờ chính là `AvatarFallback` màu nền).
5. Click overlay / X / Escape -> `onClose` -> modal biến mất, màu đã chọn vẫn giữ.
6. Nếu dùng localStorage: Refresh trang -> `AvatarBgProvider` đọc lại `vva_avatar_bg` -> khôi phục màu. Nếu pure memory: reload về `slate`.

## 7) Rủi ro & Mitigation (UI-only)

| Rủi ro | Mitigation |
|--------|------------|
| Nested modal bị clip bởi `ProfileSettingsModal overflow-hidden` | Dùng `fixed` portal + `z-[10001]`, không render như child với `absolute` |
| Bỏ `profilePicture` làm mất ảnh Google (nếu sau này muốn lại) | Đã theo yêu cầu bỏ hẳn — code cũ còn trong git, khôi phục được nếu cần. Phase này `AvatarWithLogo` chỉ còn `AvatarFallback` |
| Màu không đủ contrast với `EcaLogo` trắng | Palette chọn màu saturated vừa, test `brightness(0) invert(1)` logo trên từng bg |
| Mobile: badge quá nhỏ khó bấm | `w-7 h-7` + `touch-manipulation`, test trên `MobileNavBar` |
| Context không bọc đủ cao -> FloatingNavBar không sync | Bọc `AvatarBgProvider` ở `MainLayout` (cha của cả FloatingNavBar + ProfileContent), verify bằng test thủ công đổi màu -> check cả 3 avatar |

## 8) Tiêu chí nghiệm thu (UI-only)

- [ ] `ProfileContent` hiển thị **badge hình tròn nhỏ góc dưới-phải avatar, icon `Pen`**, hover/click được, a11y `aria-label`.
- [ ] Click badge mở `AvatarPickerModal` với avatar preview ở trên, grid 8 option ở dưới.
- [ ] Chọn option đổi preview **và đồng bộ ngay** FloatingNavBar (desktop) + MobileNavBar + ProfileContent (vì avatar giờ chỉ là màu nền fallback — đúng thứ đang đổi).
- [ ] Đóng modal bằng overlay / X / Escape, không ảnh hưởng scroll của `ProfileSettingsModal`.
- [ ] Không regression: `AvatarsPanel` (3D characters) không đổi. `AvatarWithLogo` đã bỏ `profilePicture`.
- [ ] `tsc --noEmit` pass.

## 9) Bước triển khai (UI-only, đảm bảo global sync)

**Slice 1 — Palette + AvatarWithLogo (1h)**
- Tạo `lib/avatarPalette.ts`, sửa `AvatarWithLogo.tsx` nhận `bgClassName`.

**Slice 2 — Context global + localStorage (1h)**
- Tạo `contexts/AvatarBgContext.tsx` (Provider + `useAvatarBg`), bọc ở `layouts/MainLayout.tsx`, test đổi state là cả 3 avatar cùng đổi.

**Slice 3 — Edit badge trong ProfileContent (1h)**
- Wrap avatar `relative`, thêm button `Pencil`, state `showPicker`, lấy `bg` từ `useAvatarBg()`.

**Slice 4 — AvatarPickerModal skeleton (2h)**
- Tạo `components/AvatarPickerModal.tsx` (preview `AvatarWithLogo size="lg"` ở trên + grid 8 màu ở dưới), handle `onClose`/`Escape`/`overlay`, `z-[10001]`.

**Slice 5 — Nối toàn app + polish (1h)**
- Sửa `FloatingNavBar.tsx:238` + `MobileNavBar.tsx` dùng `useAvatarBg()`, verify sync desktop/mobile, responsive, a11y, log worklog `docs/worklogs/31-08-2026.md`.

Tổng ~6h, thuần UI.

## 10) Ghi chú — trả lời 3 câu hỏi của bạn

**Icon Pen + hình tròn nhỏ:** Đã chốt `lucide-react` `Pen` (không phải `Pencil`), badge `w-7 h-7 rounded-full bg-card border shadow-md` ở `absolute -right-1 -bottom-1`, bấm mới hiện modal — đúng spec.

**Fallback giờ là gì?** Bạn hiểu đúng rồi. Trước fallback là dự phòng khi không có ảnh Google. Nay bạn đã bỏ ảnh Google -> xóa `<AvatarImage>` luôn, chỉ còn `AvatarFallback` (màu nền + `<EcaLogo>`). Nó chính là avatar hiện tại, đổi `bgClassName` là đổi trực tiếp cái user thấy ở mọi chỗ.

**localStorage dùng để làm gì?** Chỉ để **giữ màu sau khi reload**. Nếu không dùng, `Context useState` sẽ reset về `slate` mỗi lần reload nhưng vẫn đồng bộ toàn app trong phiên. Bạn không thích lưu thì bảo tôi bỏ 2 dòng `localStorage` trong `AvatarBgContext.tsx` là thành pure memory UI — vẫn đạt yêu cầu đồng bộ FloatingNavBar/Mobile/ProfileContent.

---

## Appendix: File map (UI-only)

- Mới: `ECA_UI/frontend/src/lib/avatarPalette.ts`
- Mới: `ECA_UI/frontend/src/contexts/AvatarBgContext.tsx`
- Mới: `ECA_UI/frontend/src/components/AvatarPickerModal.tsx`
- Sửa: `ECA_UI/frontend/src/components/AvatarWithLogo.tsx`
- Sửa: `ECA_UI/frontend/src/components/ProfileContent.tsx`
- Sửa (bắt buộc): `ECA_UI/frontend/src/components/FloatingNavBar.tsx`, `MobileNavBar.tsx`
- Sửa: `ECA_UI/frontend/src/layouts/MainLayout.tsx` (bọc Provider)

> Sau khi duyệt plan này, N làm Slice 1-5 thuần UI, K review `docs/worklogs/31-08-2026.md` trước khi merge. Mọi thứ backend/DB/Cognito đã bỏ khỏi scope.
