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
      position: [0, 3, 8] as [number, number, number], // front, slightly above — flattering for anime faces
      castShadow: true, // true = harsh shadows on face; false = soft fill from ambient
    },
    ambient: {
      skyColor: '#b4c7e0',                    // cool sky fill
      groundColor: '#4a3728',                 // warm ground bounce
      intensity: 1,                         // Không làm phai mất bóng mờ
    },
  },

  // ── Shadows ───────────────────────────────────────────────────────────
  shadows: {
    // MUST NOT be PCFSoftShadowMap — that constant is deprecated and three.js
    // silently rewrites it: `WebGLShadowMap.render()` warns and does
    // `this.type = PCFShadowMap`. R3F then re-applies our value on every render
    // of <Canvas> and flags `shadowMap.needsUpdate = true` because the type
    // "changed" — so each React re-render forced a FULL shadow-map rebuild,
    // which is the black flash. Naming the value three.js actually uses breaks
    // that ping-pong; the rendered result is identical.
    type: THREE.PCFShadowMap as THREE.ShadowMapType,
    mapSize: 1024,
    bias: -0.001, // increased negative bias to prevent shadow acne on neck/hair
    normalBias: 0.02,
    // Frustum extents are AUTO-FITTED to the character every frame — see
    // lib/shadowFit.ts. The old fixed box (cameraSize 2.5 centred on the world
    // origin) left the subject 0.43 units from the edge while wasting ~92% of
    // the shadow map, which is how shadows got sliced off in a straight line.
    // Enlarging the box would only waste more map and blur the result.
    /** Metres of slack around the skeleton.
     *  NOT cosmetic: the fit is computed from ~50 JOINTS, but what casts the
     *  shadow is the MESH — skirt, hair, ribbons, wings reach well past any
     *  bone. A joint-only fit reports "nothing outside the frustum" while the
     *  silhouette is visibly sliced (that is exactly how the fixed-box version
     *  passed a bone-based check yet clipped on screen). */
    fitPadding: 0.35,
    /** World height of the shadow-receiving floor (ground plane sits at z=0). */
    fitGroundZ: 0,
  },

  // ── Ground & Contact Shadow ───────────────────────────────────────────
  ground: {
    y: -1.5,                                  // matches model group position.y
    // Real shadow-catching ground plane
    planeSize: 200,
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
      // OFF — Bloom is what caused the "màn hình chớp đen" report. Measured
      // with a CDP screencast at ~59fps over 19.5s, counting frames whose mean
      // luma fell below 75% of the median (a real black frame reads 17.8 vs a
      // normal 220):
      //
      //   everything on ................ 4 black frames
      //   SSAO off ..................... 3   (not SSAO)
      //   ContactShadows off ........... 5   (not ContactShadows)
      //   whole EffectComposer off ..... 0
      //   Bloom off, rest on ........... 0   ← isolated
      //   Bloom on with mipmapBlur ..... 6   (alternate blur path doesn't help)
      //
      // One isolated frame goes fully black roughly every 3-5s, unrelated to
      // any app event. At intensity 0.15 / threshold 0.85 the effect was barely
      // perceptible, so the trade is not close. Flip back to true only if the
      // underlying @react-three/postprocessing issue is fixed — and re-run the
      // screencast check before trusting it.
      enabled: false,
      intensity: 0.15,
      luminanceThreshold: 0.85,
      luminanceSmoothing: 0.4,
    },
    ssao: {
      enabled: true, // Bình thường là true
      intensity: 0.3,
      radius: 0.05,
      samples: 16,
    },
    vignette: {
      offset: 0.35,
      darkness: 0.1,
    },
  },

  // ── MToon Material Overrides ──────────────────────────────────────────
  // Only applies if `enabled: true`. Tweak shading at the material level
  // BEFORE the shader runs — cleaner than post-processing.
  mtoon: {
    enabled: false,
    shadingShiftFactor:0.85,    // 0-1: higher = less dark shadow on face
    shadeColorHex: '#1a1020',    // color of shaded area (hex, usually dark)
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
