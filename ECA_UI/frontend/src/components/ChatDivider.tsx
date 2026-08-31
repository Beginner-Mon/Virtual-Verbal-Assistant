export default function ChatDivider({ toLabel }: { toLabel?: string }) {
  return (
    <div className="flex items-center gap-3 py-2 px-3 md:px-5 animate-message-in">
      <hr className="flex-1 border-border/40" />
      {toLabel ? (
        <span className="text-[11px] text-muted-foreground whitespace-nowrap">
          Đã chuyển sang {toLabel}
        </span>
      ) : null}
      <hr className="flex-1 border-border/40" />
    </div>
  )
}
