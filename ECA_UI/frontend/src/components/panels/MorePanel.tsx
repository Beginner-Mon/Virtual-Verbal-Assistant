import { Play, Pause, RotateCcw, Activity, Sliders, MoreHorizontal, Sparkles } from 'lucide-react'
import { ScrollArea } from '../ui/scroll-area'
import { useMotion } from '../../contexts/MotionContext'

export default function MorePanel() {
  const {
    cameraMode,
    setCameraMode,
    selectedMotionId,
    setSelectedMotionId,
    isPlaying,
    setIsPlaying,
    speed,
    setSpeed,
    clipInfo,
    handleReset,
    motionOptions,
  } = useMotion()

  return (
    <div className="flex flex-col h-full">
      <div className="px-4 py-3 border-b border-border/40 shrink-0">
        <h2 className="text-sm font-semibold text-foreground tracking-tight flex items-center gap-2">
          <MoreHorizontal className="w-4 h-4 text-primary" />
          Motion Controls
        </h2>
        <p className="text-[11px] text-muted-foreground mt-0.5">Animation source & playback</p>
      </div>

      <ScrollArea className="flex-1 min-h-0 p-4">
        <div className="flex flex-col gap-4">
          <div className="flex flex-col gap-1.5 p-3 rounded-xl bg-secondary/20 border border-border/10">
            <span className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground flex items-center gap-1.5">
              <MoreHorizontal className="w-3 h-3" />
              Camera mode
            </span>
            <select
              value={cameraMode}
              onChange={(e) => setCameraMode(e.target.value as 'head' | 'hips')}
              className="w-full bg-transparent text-xs text-foreground font-medium border-none outline-none cursor-pointer mt-0.5"
            >
              <option value="head" className="bg-card text-foreground">Head - close face view</option>
              <option value="hips" className="bg-card text-foreground">Hips - wider body view</option>
            </select>
          </div>

          <div className="flex flex-col gap-1.5 p-3 rounded-xl bg-secondary/20 border border-border/10">
            <span className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground flex items-center gap-1.5">
              <Activity className="w-3 h-3" />
              Motion source
            </span>
            <select
              value={selectedMotionId}
              onChange={(e) => setSelectedMotionId(e.target.value)}
              className="w-full bg-transparent text-xs text-foreground font-medium border-none outline-none cursor-pointer mt-0.5"
            >
              {motionOptions.map((option) => (
                <option key={option.id} value={option.id} className="bg-card text-foreground">
                  {option.label}
                </option>
              ))}
            </select>
          </div>

          <div className="flex items-center justify-between gap-2 p-3 rounded-xl bg-secondary/20 border border-border/10">
            <div className="flex items-center gap-2">
              <button
                onClick={() => setIsPlaying(!isPlaying)}
                className="w-9 h-9 rounded-lg flex items-center justify-center bg-primary text-primary-foreground shadow-lg hover:bg-primary/95 transition-all active:scale-95"
              >
                {isPlaying ? (
                  <Pause className="w-4 h-4 fill-current" />
                ) : (
                  <Play className="w-4 h-4 fill-current ml-0.5" />
                )}
              </button>
              <button
                onClick={handleReset}
                title="Reset animation"
                className="w-9 h-9 rounded-lg flex items-center justify-center border border-border/40 hover:bg-secondary/40 text-foreground transition-all active:scale-95"
              >
                <RotateCcw className="w-4 h-4" />
              </button>
            </div>

            <div className="flex items-center gap-1.5 bg-secondary/30 border border-border/20 px-2 py-1 rounded-lg">
              <Sliders className="w-3 h-3 text-muted-foreground" />
              <select
                value={speed}
                onChange={(e) => setSpeed(parseFloat(e.target.value))}
                className="bg-transparent text-xs text-foreground font-medium border-none outline-none cursor-pointer"
              >
                <option value="0.5" className="bg-card text-foreground">0.5x</option>
                <option value="1.0" className="bg-card text-foreground">1.0x</option>
                <option value="1.5" className="bg-card text-foreground">1.5x</option>
                <option value="2.0" className="bg-card text-foreground">2.0x</option>
              </select>
            </div>
          </div>

          {clipInfo && (
            <div className="flex flex-col gap-1 p-3 rounded-xl bg-secondary/20 border border-border/10">
              <span className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground flex items-center gap-1.5">
                <Sparkles className="w-3 h-3 text-purple-400" />
                Clip Info
              </span>
              <div className="flex justify-between text-[11px] text-muted-foreground mt-0.5">
                <span>Bones: {clipInfo.tracks} tracks</span>
                <span>Duration: {clipInfo.duration.toFixed(2)}s</span>
              </div>
            </div>
          )}
        </div>
      </ScrollArea>
    </div>
  )
}
