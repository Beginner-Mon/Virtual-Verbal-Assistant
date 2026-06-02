import { Canvas, useFrame } from '@react-three/fiber'
import {
  Float,
  MeshDistortMaterial,
  Sphere,
  Environment,
  ContactShadows,
  Stars,
} from '@react-three/drei'
import { useRef, Suspense, useMemo } from 'react'
import type * as THREE from 'three'

/* ───────────────────────────── AI Head ───────────────────────────── */

function AIHead() {
  const groupRef = useRef<THREE.Group>(null!)
  const leftEyeRef = useRef<THREE.Mesh>(null!)
  const rightEyeRef = useRef<THREE.Mesh>(null!)

  useFrame((state) => {
    const t = state.clock.getElapsedTime()

    // Gentle breathing / bobbing
    groupRef.current.position.y = Math.sin(t * 0.6) * 0.08
    groupRef.current.rotation.y = Math.sin(t * 0.3) * 0.15
    groupRef.current.rotation.x = Math.sin(t * 0.4) * 0.05

    // Eye glow pulse
    const intensity = 1.5 + Math.sin(t * 2) * 0.5
    const leftMat = leftEyeRef.current?.material as THREE.MeshStandardMaterial | undefined
    const rightMat = rightEyeRef.current?.material as THREE.MeshStandardMaterial | undefined
    if (leftMat) leftMat.emissiveIntensity = intensity
    if (rightMat) rightMat.emissiveIntensity = intensity
  })

  return (
    <Float speed={1.5} rotationIntensity={0.2} floatIntensity={0.5}>
      <group ref={groupRef}>
        {/* Main head — distorted metallic sphere */}
        <Sphere args={[1.2, 128, 128]}>
          <MeshDistortMaterial
            color="#6d28d9"
            roughness={0.15}
            metalness={0.9}
            distort={0.25}
            speed={2}
          />
        </Sphere>

        {/* Inner glow layer */}
        <Sphere args={[1.15, 64, 64]}>
          <meshStandardMaterial
            color="#7c3aed"
            emissive="#7c3aed"
            emissiveIntensity={0.3}
            transparent
            opacity={0.3}
          />
        </Sphere>

        {/* Left eye */}
        <Sphere ref={leftEyeRef} args={[0.1, 32, 32]} position={[-0.35, 0.2, 1.0]}>
          <meshStandardMaterial
            color="#e9d5ff"
            emissive="#c4b5fd"
            emissiveIntensity={2}
          />
        </Sphere>

        {/* Right eye */}
        <Sphere ref={rightEyeRef} args={[0.1, 32, 32]} position={[0.35, 0.2, 1.0]}>
          <meshStandardMaterial
            color="#e9d5ff"
            emissive="#c4b5fd"
            emissiveIntensity={2}
          />
        </Sphere>

        {/* Eye point lights */}
        <pointLight position={[-0.35, 0.2, 1.3]} color="#a78bfa" intensity={0.5} distance={2} />
        <pointLight position={[0.35, 0.2, 1.3]} color="#a78bfa" intensity={0.5} distance={2} />
      </group>
    </Float>
  )
}

/* ───────────────────────── Floating Particles ────────────────────── */

function FloatingParticles() {
  const count = 200
  const pointsRef = useRef<THREE.Points>(null!)

  const positions = useMemo(() => {
    const arr = new Float32Array(count * 3)
    for (let i = 0; i < count; i++) {
      arr[i * 3 + 0] = (Math.random() - 0.5) * 10
      arr[i * 3 + 1] = (Math.random() - 0.5) * 10
      arr[i * 3 + 2] = (Math.random() - 0.5) * 10
    }
    return arr
  }, [])

  useFrame((state) => {
    const t = state.clock.getElapsedTime()
    pointsRef.current.rotation.y = t * 0.02
    pointsRef.current.rotation.x = t * 0.01
  })

  return (
    <points ref={pointsRef}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          count={count}
          array={positions}
          itemSize={3}
        />
      </bufferGeometry>
      <pointsMaterial
        size={0.02}
        color="#a78bfa"
        transparent
        opacity={0.6}
        sizeAttenuation
      />
    </points>
  )
}

/* ────────────────────────────── Scene ─────────────────────────────── */

function Scene() {
  return (
    <>
      <ambientLight intensity={0.15} />
      <directionalLight position={[5, 5, 5]} intensity={0.5} color="#e9d5ff" />
      <directionalLight position={[-5, 3, -5]} intensity={0.3} color="#7c3aed" />
      <spotLight position={[0, 5, 0]} angle={0.4} penumbra={1} intensity={0.5} color="#a78bfa" />

      <AIHead />
      <FloatingParticles />

      <ContactShadows
        position={[0, -2, 0]}
        opacity={0.4}
        scale={8}
        blur={2.5}
        far={4}
        color="#4c1d95"
      />

      <Stars radius={50} depth={50} count={1000} factor={2} saturation={0.5} fade speed={0.5} />
      <Environment preset="night" />
    </>
  )
}

/* ───────────────────────── Exported Component ────────────────────── */

export default function CharacterViewer() {
  return (
    <div
      className="relative w-full h-full"
      style={{ background: 'radial-gradient(ellipse at center, #1a0533 0%, #0a0a12 70%)' }}
    >
      <Canvas camera={{ position: [0, 0, 4], fov: 45 }} gl={{ antialias: true, alpha: true }}>
        <Suspense fallback={null}>
          <Scene />
        </Suspense>
      </Canvas>

      {/* Right-edge gradient for blending with the chat panel */}
      <div
        className="absolute inset-0 pointer-events-none"
        style={{ background: 'linear-gradient(to right, transparent 80%, rgba(10,10,18,0.8) 100%)' }}
      />

      {/* Bottom gradient */}
      <div
        className="absolute bottom-0 left-0 right-0 h-24 pointer-events-none"
        style={{ background: 'linear-gradient(to top, rgba(10,10,18,0.6), transparent)' }}
      />
    </div>
  )
}
