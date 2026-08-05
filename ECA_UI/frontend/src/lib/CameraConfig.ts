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
}
