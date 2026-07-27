import { existsSync, mkdirSync, readdirSync, statSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { spawnSync } from 'node:child_process'
import { fileURLToPath } from 'node:url'

const scriptDir = dirname(fileURLToPath(import.meta.url))
const frontendRoot = resolve(scriptDir, '..')
const assetRoot = resolve(frontendRoot, 'src', 'asset')
const generatedRoot = resolve(assetRoot, 'motions', 'generated')
const converterScript = resolve(frontendRoot, '..', '..', 'scripts', 'smplx_to_bvh.py')

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

function ensurePythonConverter(npzPath, bvhPath) {
  const candidates = [
    { command: 'python', args: [converterScript, npzPath, bvhPath] },
    { command: 'py', args: ['-3', converterScript, npzPath, bvhPath] },
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

  console.log(`[sync-npz-to-bvh] Converting ${npzPath} -> ${bvhPath}`)
  if (ensurePythonConverter(npzPath, bvhPath)) {
    converted += 1
  }
}

console.log(`[sync-npz-to-bvh] Converted ${converted} NPZ file(s) into ${generatedRoot}`)