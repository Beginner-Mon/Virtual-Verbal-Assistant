/**
 * The rest pose snapshot stashed on `vrm.scene.userData.restPoses`.
 *
 * CharacterViewer captures it on first render, before any effect and before
 * the mixer exists, so retargeting has a bind pose to measure against for a
 * model that does not expose one itself (HeadController.ts:134-137 explains
 * the three-way fallback that reads it).
 *
 * `userData` is `Record<string, any>` in three.js, so every reader has to
 * assert what it finds there. This is that assertion, written once — the
 * alternative was `as Map<string, any>` repeated in four files, which asserts
 * only that something is a Map and lets a typo in `.quaternion` through.
 */
import type * as THREE from 'three'

export interface RestPose {
  position: THREE.Vector3
  quaternion: THREE.Quaternion
}

/** `vrm.scene.userData.restPoses`, or undefined on a model that never got one. */
export type RestPoseMap = Map<string, RestPose>
