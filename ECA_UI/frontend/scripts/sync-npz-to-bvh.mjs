import { existsSync, mkdirSync, readFileSync, readdirSync, statSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { spawnSync } from 'node:child_process'
import { fileURLToPath } from 'node:url'

const scriptDir = dirname(fileURLToPath(import.meta.url))
const frontendRoot = resolve(scriptDir, '..')
const assetRoot = resolve(frontendRoot, 'src', 'asset')
const generatedRoot = resolve(assetRoot, 'motions', 'generated')
const dartConverter = resolve(frontendRoot, '..', '..', 'scripts', 'smplx_to_bvh.py')
const kimodoConverter = resolve(frontendRoot, '..', '..', 'scripts', 'kimodo_npz_to_bvh.py')

// ── Detect Kimodo NPZ (has local_rot_mats.npy in ZIP) ─────────────

function isKimodoNpz(filePath) {
  try {
    const buf = readFileSync(filePath)
    if (buf[0] !== 0x50 || buf[1] !== 0x4b) return false
    let off = buf.length - 22
    while (off >= 0) {
      if (buf.readUInt32LE(off) === 0x06054b50) {
        const cdOff = buf.readUInt32LE(off + 16)
        const cdEntries = buf.readUInt16LE(off + 8)
        let c = cdOff
        for (let i = 0; i < cdEntries; i++) {
          const nameLen = buf.readUInt16LE(c + 28)
          let name = ''
          for (let j = 0; j < nameLen; j++) name += String.fromCharCode(buf[c + 46 + j])
          if (name === 'local_rot_mats.npy') return true
          const extraLen = buf.readUInt16LE(c + 30)
          const commentLen = buf.readUInt16LE(c + 32)
          c += 46 + nameLen + extraLen + commentLen
        }
        return false
      }
      off--
    }
    return false
  } catch { return false }
}

function walkNpzs(dir) {
  const entries = readdirSync(dir, { withFileTypes: true })
  const files = []

  for (const entry of entries) {
    const fullPath = resolve(dir, entry.name)
    if (entry.isDirectory()) {
      files.push(...walkNpzs(fullPath))
      continue
    }

    if (entry.isFile() && entry.name.toLowerCase().endsWith('.npz')) {
      files.push(fullPath)
    }
  }

  return files
}

function ensurePythonConverter(npzPath, bvhPath, converterScript) {
  const candidates = [
    { command: 'py', args: ['-3', converterScript, npzPath, bvhPath] },
    { command: 'python', args: [converterScript, npzPath, bvhPath] },
  ]

  let lastError = null

  for (const candidate of candidates) {
    const result = spawnSync(candidate.command, candidate.args, {
      cwd: frontendRoot,
      stdio: 'inherit',
    })

    if (!result.error && result.status === 0) {
      return true
    }

    lastError = result.error ?? new Error(`${candidate.command} exited with code ${result.status}`)
  }

  console.warn(`[sync-npz-to-bvh] Skipped ${npzPath}: ${lastError?.message ?? 'unknown error'}`)
  return false
}

if (!existsSync(assetRoot)) {
  console.log(`[sync-npz-to-bvh] Asset folder not found: ${assetRoot}`)
  process.exit(0)
}

mkdirSync(generatedRoot, { recursive: true })

const npzFiles = walkNpzs(assetRoot)
let converted = 0

for (const npzPath of npzFiles) {
  const bvhPath = resolve(generatedRoot, `${npzPath.split(/[/\\]/).pop()?.replace(/\.npz$/i, '.bvh') ?? 'motion.bvh'}`)

  const npzStat = statSync(npzPath)
  const bvhStat = existsSync(bvhPath) ? statSync(bvhPath) : null
  const shouldConvert = !bvhStat || npzStat.mtimeMs > bvhStat.mtimeMs

  if (!shouldConvert) continue

  const isKimodo = isKimodoNpz(npzPath)
  const converter = isKimodo ? kimodoConverter : dartConverter
  const label = isKimodo ? 'kimodo' : 'smplx'

  console.log(`[sync-npz-to-bvh] Converting [${label}] ${npzPath} -> ${bvhPath}`)
  if (ensurePythonConverter(npzPath, bvhPath, converter)) {
    converted += 1
  }
}

console.log(`[sync-npz-to-bvh] Converted ${converted} NPZ file(s) into ${generatedRoot}`)