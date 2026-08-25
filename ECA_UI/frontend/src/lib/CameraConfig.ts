export interface CameraConfig {
  /** OrbitControls: allow mouse drag to pan camera up/down/left/right */
  enablePan: boolean
  /** OrbitControls: allow scroll wheel zoom */
  enableZoom: boolean
  /** Min orbit radius (zoom in limit) */
  minDistance: number
  /** Max orbit radius (zoom out limit) */
  maxDistance: number
  /** Auto-follow the VRM bone target (lookAt). When false, camera stays put. */
  followTarget: boolean
  /** Offset X (left/right) from the default camera position */
  offsetX: number
  /** Offset Y (up/down) from the default camera position */
  offsetY: number
  /** Offset Z (forward/backward) from the default camera position */
  offsetZ: number
  /** Lock per-axis: when true, right-drag/pan does not change that coordinate. */
  lockX: boolean
  lockY: boolean
  lockZ: boolean
}

export const DEFAULT_CAMERA_CONFIG: CameraConfig = {
  enablePan: false,
  enableZoom: true,
  minDistance: 1,
  maxDistance: 20,
  followTarget: true,
  offsetX: 0,
  offsetY: 0,
  offsetZ: 0,
  lockX: false,
  lockY: false,
  lockZ: false,
}

export interface CameraResponsivePreset {
  wideFraming: [number, number, number]
  narrowFraming: [number, number, number]
  /** Target Z shift on mobile (negative = target lower = model appears higher in frame).
   *  Lerped from 0 at desktop to this value at mobile. */
  narrowTargetZ: number
}

/** Aspect ratio thresholds for responsive camera offset interpolation.
 *  aspect ≥ wide → t=0 (full wide framing, desktop)
 *  aspect ≤ narrow → t=1 (full narrow framing, mobile portrait)
 *  Between → lerp */
export const ASPECT_RANGE = {
  narrow: 0.6,
  wide: 1.5,
} as const
