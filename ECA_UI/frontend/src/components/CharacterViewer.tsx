import { Canvas, useFrame, useLoader } from '@react-three/fiber'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'
import { VRMLoaderPlugin } from '@pixiv/three-vrm'
import seeleUrl from '../asset/seele.vrm'
import { useTheme } from '../contexts/ThemeContext'
import {
  Environment,
  ContactShadows,
  Stars,
} from '@react-three/drei'
import { useRef, Suspense, useMemo } from 'react'
import type * as THREE from 'three'

function VRMCharacter() {
  const gltf = useLoader(GLTFLoader, seeleUrl, (loader) => {
    loader.register((parser) => new VRMLoaderPlugin(parser))
  })

  const vrm = gltf.userData.vrm

  useFrame((state, delta) => {
    if (vrm) {
      vrm.update(delta)
    }
  })

  // Adjust Y position if the character floats or sinks below the ground plane
  return <primitive object={vrm.scene} position={[0, -1.5, 0]} />
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

  useFrame(({ clock }) => {
    const t = clock.getElapsedTime()
    pointsRef.current.rotation.y = t * 0.02
    pointsRef.current.rotation.x = t * 0.01
  })

  return (
    <points ref={pointsRef}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          args={[positions, 3]}
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

function Scene({ theme }: { theme: 'light' | 'dark' }) {
  return (
    <>
      <ambientLight intensity={theme === 'dark' ? 0.15 : 0.6} />
      <directionalLight position={[5, 5, 5]} intensity={0.5} color={theme === 'dark' ? "#e9d5ff" : "#ffffff"} />
      <directionalLight position={[-5, 3, -5]} intensity={0.3} color="#7c3aed" />
      <spotLight position={[0, 5, 0]} angle={0.4} penumbra={1} intensity={0.5} color="#a78bfa" />

      <VRMCharacter />
      <FloatingParticles />

      <ContactShadows
        position={[0, -2, 0]}
        opacity={theme === 'dark' ? 0.4 : 0.15}
        scale={8}
        blur={2.5}
        far={4}
        color={theme === 'dark' ? "#4c1d95" : "#000000"}
      />

      {theme === 'dark' && <Stars radius={50} depth={50} count={1000} factor={2} saturation={0.5} fade speed={0.5} />}
      <Environment preset={theme === 'dark' ? 'night' : 'city'} />
    </>
  )
}

/* ───────────────────────── Exported Component ────────────────────── */

export default function CharacterViewer() {
  const { theme } = useTheme()

  return (
    <div
      className="relative w-full h-full"
      style={{ 
        background: theme === 'dark' 
          ? 'radial-gradient(ellipse at center, #1a0533 0%, #0a0a12 70%)'
          : 'radial-gradient(ellipse at center, #f3e8ff 0%, #ffffff 70%)'
      }}
    >
      <Canvas camera={{ position: [0, 0, 4], fov: 45 }} gl={{ antialias: true, alpha: true }}>
        <Suspense fallback={null}>
          <Scene theme={theme} />
        </Suspense>
      </Canvas>

      {/* Bottom gradient */}
      <div
        className="absolute bottom-0 left-0 right-0 h-24 pointer-events-none bg-gradient-to-t from-background/80 to-transparent"
      />
    </div>
  )
}
