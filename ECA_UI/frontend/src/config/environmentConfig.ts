/**
 * Centralized environment configuration.
 *
 * Every numeric / color value that affects the 3D rendering environment lives
 * here. Scene components import from this file — nothing is hardcoded in JSX.
 * To tune the look: edit values here, hot-reload picks them up immediately.
 */

import * as THREE from 'three'

export const ENV_CONFIG = {
  // ── Debug ─────────────────────────────────────────────────────────────
  debug: {
    showGrid: true,
    showAxes: true,
  },

  // ── Renderer & Color Pipeline ─────────────────────────────────────────
  renderer: {
    toneMapping: THREE.NoToneMapping, // Chuẩn cho MToon Anime, bảo toàn màu gốc
    toneMappingExposure: 1.0,         // Trả về mặc định
    outputColorSpace: THREE.SRGBColorSpace,
  },

  // ── Lighting ──────────────────────────────────────────────────────────
  // RULE: only ONE directional light influences MToon NdotL toon-shading.
  // The hemisphere light is ambient-only (no directional influence).
  lighting: {
    main: {
      color: '#fffaf0',                       // warm white — avoids blue cast on skin
      intensity: 2.0,                         // Đèn chính chuẩn
      position: [0, 5, 4] as [number, number, number], // front, slightly above — flattering for anime faces
      castShadow: true,
    },
    ambient: {
      skyColor: '#b4c7e0',                    // cool sky fill
      groundColor: '#4a3728',                 // warm ground bounce
      intensity: 0.6,                         // Không làm phai mất bóng mờ
    },
  },

  // ── Shadows ───────────────────────────────────────────────────────────
  shadows: {
    type: THREE.PCFSoftShadowMap as THREE.ShadowMapType,
    mapSize: 1024,
    bias: -0.001, // increased negative bias to prevent shadow acne on neck/hair
    normalBias: 0.02,
    // Shadow camera frustum — tight around a standing humanoid
    cameraSize: 2.5,
    cameraNear: 0.1,
    cameraFar: 10,
  },

  // ── Ground & Contact Shadow ───────────────────────────────────────────
  ground: {
    y: -1.5,                                  // matches model group position.y
    // Real shadow-catching ground plane
    planeSize: 15,
    shadowMaterialOpacity: 0.35,
    // drei ContactShadows (screen-space, for extra softness at feet)
    contactShadow: {
      opacity: 0.8, // Soft puddle under feet
      blur: 2.0,
      scale: 5,
      far: 3,
      color: '#1a1020',
    },
  },

  // ── Environment / Background ──────────────────────────────────────────
  environment: {
    useGradient: true,                        // true = shader gradient; false = HDRI
    gradient: {
      // Unified colors — same lighting for both themes; only bg changes.
      dark:  { top: '#1a1a2e', bottom: '#2a2040' },
      light: { top: '#f0f2f8', bottom: '#e0e2ec' },
    },
    hdri: {
      preset: 'studio' as const,
      intensity: 0.3,                         // low for MToon — avoids over-reflection
    },
    iblResolution: 64,                        // low to avoid GPU memory issues (D3D11)
    showStars: {
      dark: true,
      light: false,
    },
  },

  // ── Post Processing ───────────────────────────────────────────────────
  // Subtle enhancements only — no cinematic look. Goal: depth, not drama.
  postProcessing: {
    enabled: true,
    bloom: {
      intensity: 0.15,
      luminanceThreshold: 0.85,
      luminanceSmoothing: 0.4,
    },
    ssao: {
      enabled: true, // Bình thường là true
      intensity: 0.5,
      radius: 0.05,
      samples: 16,
    },
    vignette: {
      offset: 0.35,
      darkness: 0.45,
    },
  },

  // ── Floating Particles ────────────────────────────────────────────────
  particles: {
    enabled: true,                            // default ON
    count: 150,
    size: 0.02,
    color: '#a78bfa',
    opacity: 0.4,
  },
} as const
