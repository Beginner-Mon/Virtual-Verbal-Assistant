// API base resolution priority:
// 1) Query param: ?api_base=https://your-api-host
// 2) Global override: window.TEST_UI_API_BASE_URL
// 3) If page itself is served from :8000, use current origin
// 4) Local default: http://localhost:8000
function resolveApiBase() {
    const params = new URLSearchParams(window.location.search);
    const fromQuery = params.get('api_base');
    const fromGlobal = window.TEST_UI_API_BASE_URL;
    const isApiPort = window.location.port === '8000';

    const base = fromQuery || fromGlobal || (isApiPort ? window.location.origin : 'http://localhost:8000');
    return String(base).replace(/\/$/, '');
}

const API_BASE = resolveApiBase();
console.log('[test-ui] API_BASE =', API_BASE);

function siblingBaseFromApi(defaultPort, fallbackUrl) {
    try {
        const u = new URL(API_BASE);
        u.port = String(defaultPort);
        return u.origin;
    } catch {
        return fallbackUrl;
    }
}

function resolveServiceBase(paramName, globalName, fallbackUrl) {
    const params = new URLSearchParams(window.location.search);
    const fromQuery = params.get(paramName);
    const fromGlobal = window[globalName];
    return String(fromQuery || fromGlobal || fallbackUrl).replace(/\/$/, '');
}

const ORCHESTRATOR_BASE = resolveServiceBase(
    'orchestrator_base',
    'TEST_UI_ORCHESTRATOR_BASE_URL',
    siblingBaseFromApi(8080, 'http://localhost:8080')
);
const DART_BASE = resolveServiceBase(
    'dart_base',
    'TEST_UI_DART_BASE_URL',
    'http://localhost:5001'
);
const SPEECH_BASE = resolveServiceBase(
    'speech_base',
    'TEST_UI_SPEECH_BASE_URL',
    siblingBaseFromApi(5000, 'http://localhost:5000')
);

const SERVICES = {
    rag: {
        name: 'AgenticRAG',
        baseUrl: API_BASE,
        healthUrl: `${API_BASE}/health`,
    },
    dart: {
        name: 'DART Motion',
        baseUrl: DART_BASE,
        healthUrl: `${DART_BASE}/health`,
    },
    orchestrator: {
        name: 'Orchestrator',
        baseUrl: ORCHESTRATOR_BASE,
        healthUrl: `${ORCHESTRATOR_BASE}/health`,
    },
    speechllm: {
        name: 'SpeechLLm',
        baseUrl: SPEECH_BASE,
        healthUrl: `${SPEECH_BASE}/health`,
    },
};

function joinUrl(baseUrl, pathOrUrl) {
    if (!pathOrUrl) return null;
    if (/^https?:\/\//i.test(pathOrUrl)) return pathOrUrl;
    const normalizedBase = String(baseUrl || '').replace(/\/$/, '');
    const normalizedPath = String(pathOrUrl).startsWith('/') ? pathOrUrl : `/${pathOrUrl}`;
    return `${normalizedBase}${normalizedPath}`;
}

function setServiceBaseUrl(serviceKey, baseUrl) {
    const normalizedBase = String(baseUrl || '').replace(/\/$/, '');
    SERVICES[serviceKey].baseUrl = normalizedBase;
    SERVICES[serviceKey].healthUrl = `${normalizedBase}/health`;
}

async function initializeServiceEndpoints() {
    try {
        const orchestratorInfoUrl = `${SERVICES.orchestrator.baseUrl}/info`;
        const response = await fetch(orchestratorInfoUrl, { cache: 'no-store' });
        if (!response.ok) return;

        const info = await response.json();
        const upstream = info?.upstream_services || {};

        if (upstream.dart) {
            const dartBase = String(upstream.dart).replace(/\/generate\/?$/, '');
            setServiceBaseUrl('dart', dartBase);
        }
        if (upstream.tts) {
            const ttsBase = String(upstream.tts).replace(/\/synthesize\/?$/, '');
            setServiceBaseUrl('speechllm', ttsBase);
        }
        console.log('[test-ui] Resolved service endpoints from orchestrator /info', {
            dart: SERVICES.dart.baseUrl,
            speechllm: SERVICES.speechllm.baseUrl,
        });
    } catch (err) {
        console.warn('[test-ui] Could not resolve dynamic service endpoints, using defaults.', err);
    }
}

// ==============================
// Health Checks
// ==============================

async function checkHealth(serviceKey) {
    const service = SERVICES[serviceKey];
    const dot = document.getElementById(`dot-${serviceKey}`);
    const label = document.getElementById(`label-${serviceKey}`);

    dot.className = 'status-dot checking';
    label.textContent = 'Checking...';

    try {
        const start = performance.now();
        const response = await fetch(service.healthUrl);
        const elapsed = Math.round(performance.now() - start);
        const data = await response.json();

        const isOk = data.status === 'healthy' || data.status === 'ok';
        if (isOk) {
            dot.className = 'status-dot online';
            label.textContent = `✅ Healthy (${elapsed}ms)`;
            addLog('success', service.name, `Health check passed in ${elapsed}ms`);
        } else {
            dot.className = 'status-dot offline';
            label.textContent = `⚠️ Unexpected: ${JSON.stringify(data)}`;
            addLog('error', service.name, `Unexpected response: ${JSON.stringify(data)}`);
        }
    } catch (err) {
        // When opened via file://, browsers often block cross-origin JSON reads.
        // Fallback to no-cors to detect basic reachability.
        const isFileProtocol = window.location.protocol === 'file:';
        if (isFileProtocol) {
            try {
                await fetch(service.healthUrl, { mode: 'no-cors' });
                dot.className = 'status-dot checking';
                label.textContent = '⚠️ Reachable — CORS blocked in file:// mode';
                addLog('error', service.name, 'Endpoint reachable, but browser blocked response in file:// mode (CORS). Serve test-ui over http:// for full health details.');
                return;
            } catch {
                // Fall through to offline if even no-cors cannot reach endpoint.
            }
        }

        dot.className = 'status-dot offline';
        label.textContent = `❌ Offline — ${err.message}`;
        addLog('error', service.name, `Health check failed: ${err.message}`);
    }
}

async function checkAllHealth() {
    await Promise.all([
        checkHealth('rag'),
        checkHealth('dart'),
        checkHealth('orchestrator'),
        checkHealth('speechllm'),
    ]);
}

// Auto-check on load
window.addEventListener('DOMContentLoaded', async () => {
    await initializeServiceEndpoints();
    checkAllHealth();
});

// ==============================
// Tab Switching
// ==============================

function switchTab(tabId) {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.test-panel').forEach(p => p.classList.remove('active'));
    document.getElementById(`tab-${tabId}`).classList.add('active');
    document.getElementById(`panel-${tabId}`).classList.add('active');
}

// ==============================
// API Tests
// ==============================

async function testAgenticRAG() {
    const query = document.getElementById('rag-query').value;
    const userId = document.getElementById('rag-user-id').value;
    const btn = document.getElementById('btn-rag');
    const container = document.getElementById('result-rag');
    const timer = document.getElementById('timer-rag');

    if (!query) return;

    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Sending...';
    container.innerHTML = '<div class="result-placeholder">Processing query...</div>';

    const start = performance.now();
    updateTimer(timer, start);

    try {
        const response = await fetch(`${API_BASE}/query`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query, user_id: userId, conversation_history: [] }),
        });

        const elapsed = Math.round(performance.now() - start);
        clearTimerInterval();

        if (!response.ok) throw new Error(`HTTP ${response.status}: ${await response.text()}`);

        const data = await response.json();
        showRagResult(container, data, elapsed, response.status);
        addLog('success', 'AgenticRAG', `Query processed in ${elapsed}ms`);

    } catch (err) {
        clearTimerInterval();
        showError(container, err.message, Math.round(performance.now() - start));
        addLog('error', 'AgenticRAG', err.message);
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<span class="btn-icon">▶</span> Send Query';
    }
}

async function testDART() {
    const prompt = document.getElementById('dart-prompt').value.trim();
    const primitives = Math.max(1, parseInt(document.getElementById('dart-primitives').value, 10) || 1);
    const guidance = parseFloat(document.getElementById('dart-guidance').value) || 5.0;
    const seedStr = document.getElementById('dart-seed').value.trim();
    const respacing = document.getElementById('dart-respacing').value.trim();
    const gender = document.getElementById('dart-gender').value;

    const btn = document.getElementById('btn-dart');
    const container = document.getElementById('result-dart');
    const timer = document.getElementById('timer-dart');

    if (!prompt) return;

    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Generating...';
    container.innerHTML = '<div class="result-placeholder">Generating motion (this can take 30–120s on GPU)...</div>';

    const start = performance.now();
    updateTimer(timer, start);

    try {
        // Keep advanced prompt syntax untouched, e.g. "walk*5,jump*3".
        // If user typed a plain action, use the Primitives field as fallback.
        const textPrompt = (prompt.includes('*') || prompt.includes(','))
            ? prompt
            : `${prompt}*${primitives}`;

        const body = { text_prompt: textPrompt, guidance_scale: guidance, num_steps: 50, gender };
        if (respacing !== '') body.respacing = respacing;
        if (seedStr !== '' && !isNaN(Number(seedStr))) body.seed = Number(seedStr);

        const response = await fetch(`${SERVICES.dart.baseUrl}/generate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });

        const elapsed = Math.round(performance.now() - start);
        clearTimerInterval();

        if (!response.ok) throw new Error(`HTTP ${response.status}: ${await response.text()}`);

        const data = await response.json();
        showDartResult(container, data, elapsed, response.status);
        addLog('success', 'DART Motion', `Generated ${data.num_frames} frames in ${elapsed}ms`);

    } catch (err) {
        clearTimerInterval();
        showError(container, err.message, Math.round(performance.now() - start));
        addLog('error', 'DART Motion', err.message);
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<span class="btn-icon">▶</span> Generate Motion';
    }
}

async function testPipeline() {
    const query = document.getElementById('pipeline-query').value;
    const userId = document.getElementById('pipeline-user-id').value;
    const btn = document.getElementById('btn-pipeline');
    const container = document.getElementById('result-pipeline');
    const timer = document.getElementById('timer-pipeline');

    if (!query) return;

    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Running Pipeline...';
    container.innerHTML = '<div class="result-placeholder">Calling AgenticRAG (:8000) + DART (:5001) in parallel…</div>';

    const start = performance.now();
    updateTimer(timer, start);

    try {
        // main_api.py fans out to AgenticRAG + DART simultaneously
        const response = await fetch(`${API_BASE}/answer`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query, user_id: userId, conversation_history: [] }),
        });

        if (!response.ok) {
            clearTimerInterval();
            throw new Error(`HTTP ${response.status}: ${await response.text()}`);
        }

        let data = await response.json();
        
        // Poll if async enrichment is enabled
        while (data.status === 'processing') {
            await new Promise(r => setTimeout(r, 2000));
            const statusResp = await fetch(`${API_BASE}/answer/status/${data.request_id}`);
            if (statusResp.ok) {
                data = await statusResp.json();
                container.innerHTML = `<div class="result-placeholder">Waiting for background services... (Pending: ${data.pending_services.join(', ')})</div>`;
            } else {
                break; // stop polling on error
            }
        }

        const elapsed = Math.round(performance.now() - start);
        clearTimerInterval();

        showPipelineResult(container, data, elapsed, response.status);

        const motionInfo = data.motion
            ? ` | 🏃 Motion: ${data.motion.num_frames} frames (${data.motion.duration_seconds}s)`
            : ' | No motion';
        const errorsInfo = data.errors ? ` | ⚠️ Errors: ${Object.keys(data.errors).join(', ')}` : '';
        addLog('success', 'Pipeline', `Completed in ${elapsed}ms${motionInfo}${errorsInfo}`);

    } catch (err) {
        clearTimerInterval();
        showError(container, err.message, Math.round(performance.now() - start));
        addLog('error', 'Pipeline', err.message);
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<span class="btn-icon">🚀</span> Run Full Pipeline';
    }
}

async function testSpeechLLm() {
    const text = document.getElementById('speechllm-text').value.trim();
    const userId = document.getElementById('speechllm-user-id').value.trim();
    const emotion = document.getElementById('speechllm-emotion').value;
    const btn = document.getElementById('btn-speechllm');
    const container = document.getElementById('result-speechllm');
    const timer = document.getElementById('timer-speechllm');

    if (!text) {
        alert('Please enter some text to synthesize.');
        return;
    }

    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Generating Speech...';
    container.innerHTML = '<div class="result-placeholder">Processing text and generating audio...</div>';

    const start = performance.now();
    updateTimer(timer, start);

    try {
        const payload = {
            text: text,
            user_id: userId || 'test-user',
        };

        if (emotion) {
            payload.emotion = emotion;
        }

        const response = await fetch(`${SERVICES.speechllm.baseUrl}/synthesize`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(payload),
        });

        const elapsed = Math.round(performance.now() - start);
        clearTimerInterval();

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || `HTTP ${response.status}`);
        }

        const data = await response.json();

        showSpeechLLmResult(container, data, elapsed, response.status);

        addLog('success', 'SpeechLLm', `Generated speech for "${text.substring(0, 50)}${text.length > 50 ? '...' : ''}"`);

    } catch (err) {
        clearTimerInterval();
        showError(container, err.message, Math.round(performance.now() - start));
        addLog('error', 'SpeechLLm', err.message);
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<span class="btn-icon">🎤</span> Generate Speech';
    }
}

// ==============================
// Specialised Result Renderers
// ==============================

/** Render AgenticRAG /query response with text_answer highlight. */
function showRagResult(container, data, elapsed, status) {
    const textAnswer = data.text_answer || '';
    const decision = data.orchestrator_decision || {};
    const motion = data.motion_prompt;

    container.className = 'result-container result-success';
    container.innerHTML = `
        <div class="result-card">
            <div class="result-card-section">
                <div class="result-card-label">💬 Text Answer</div>
                <div class="result-card-value result-text">${escapeHtml(textAnswer)}</div>
            </div>
            <div class="result-card-section">
                <div class="result-card-label">🧠 Orchestrator Decision</div>
                <div class="result-card-value">
                    <span class="chip">${escapeHtml(decision.action || 'unknown')}</span>
                    <span class="chip">Confidence: ${decision.confidence}</span>
                    <br><span style="color:var(--text-muted); font-size:0.9em;">Reasoning: ${escapeHtml(decision.reasoning || '')}</span>
                </div>
            </div>
            ${motion ? `<div class="result-card-section"><div class="result-card-label">🏃 Motion Prompt</div><div class="result-card-value"><code>${escapeHtml(motion)}</code></div></div>` : ''}
            <details>
                <summary style="cursor:pointer;color:var(--text-muted);font-size:0.8rem;margin-top:8px">Full JSON</summary>
                <pre class="result-json">${syntaxHighlight(JSON.stringify(data, null, 2))}</pre>
            </details>
        </div>
        <div class="result-meta">
            <span>✅ Status: ${status}</span>
            <span>⏱️ ${elapsed}ms</span>
            <span>📦 ${JSON.stringify(data).length} bytes</span>
        </div>
    `;
}

/** Render DART /generate response with download link. */
function showDartResult(container, data, elapsed, status) {
    const fileUrl = joinUrl(SERVICES.dart.baseUrl, data.motion_file_url);

    container.className = 'result-container result-success';
    container.innerHTML = `
        <div class="result-card">
            <div class="result-card-row">
                <div class="result-card-section">
                    <div class="result-card-label">🎬 Frames</div>
                    <div class="result-card-value">${data.num_frames || '—'} @ ${data.fps || 30} fps</div>
                </div>
                <div class="result-card-section">
                    <div class="result-card-label">⏱ Duration</div>
                    <div class="result-card-value">${data.duration_seconds || '—'}s</div>
                </div>
                <div class="result-card-section">
                    <div class="result-card-label">🆔 Request ID</div>
                    <div class="result-card-value"><code>${data.request_id || '—'}</code></div>
                </div>
            </div>
            <div class="result-card-section">
                <div class="result-card-label">📝 Prompt</div>
                <div class="result-card-value"><code>${escapeHtml(data.text_prompt || '—')}</code></div>
            </div>
            ${fileUrl ? `<div class="result-card-section">
                <div class="result-card-label">📥 Download NPZ</div>
                <div class="result-card-value">
                    <a href="${fileUrl}" target="_blank" style="color:var(--accent)">${fileUrl}</a>
                </div>
            </div>` : ''}
            <details>
                <summary style="cursor:pointer;color:var(--text-muted);font-size:0.8rem;margin-top:8px">Full JSON</summary>
                <pre class="result-json">${syntaxHighlight(JSON.stringify(data, null, 2))}</pre>
            </details>
        </div>
        <div class="result-meta">
            <span>✅ Status: ${status}</span>
            <span>⏱️ ${elapsed}ms</span>
        </div>
    `;
}

/** Render main_api /answer response — combined AgenticRAG + DART. */
function showPipelineResult(container, data, elapsed, status) {
    const motion = data.motion;
    const errors = data.errors;
    const fileUrl = joinUrl(SERVICES.dart.baseUrl, motion?.motion_file_url);

    const errBanner = errors
        ? `<div class="result-card-section result-error-banner">
               <div class="result-card-label">⚠️ Service Errors</div>
               <pre class="result-json" style="color:var(--error)">${syntaxHighlight(JSON.stringify(errors, null, 2))}</pre>
           </div>`
        : '';

    const motionBlock = motion ? `
        <div class="result-card-section">
            <div class="result-card-label">🏃 DART Motion (<code>${escapeHtml(motion.text_prompt || '')}</code>)</div>
            <div class="result-card-row" style="gap:12px;margin-top:4px">
                <span class="chip">🎬 ${motion.num_frames} frames</span>
                <span class="chip">⏱ ${motion.duration_seconds}s @ ${motion.fps}fps</span>
                ${fileUrl ? `<a class="chip chip-link" href="${fileUrl}" target="_blank">📥 Download NPZ</a>` : ''}
            </div>
        </div>` : `
        <div class="result-card-section">
            <div class="result-card-label">🏃 DART Motion</div>
            <div class="result-card-value" style="color:var(--text-muted)">Not generated or unavailable</div>
        </div>`;

    const exercisesBlock = (data.exercises && data.exercises.length > 0) ? `
        <div class="result-card-section">
            <div class="result-card-label">🏋️ Exercises</div>
            <div class="result-card-row" style="gap:8px; margin-top:4px;">
                ${data.exercises.map(ex => `
                    <button class="chip chip-link" onclick="visualizeExercise('${escapeHtml(ex.name)}')" 
                            style="border:1px solid var(--accent); background:transparent; cursor:pointer;"
                            title="Generate motion for this exercise">
                        ▶ Visualize: ${escapeHtml(ex.name)}
                    </button>
                `).join('')}
            </div>
        </div>` : '';

    container.className = 'result-container result-success';
    container.innerHTML = `
        <div class="result-card">
            ${errBanner}
            <div class="result-card-section">
                <div class="result-card-label">💬 AgenticRAG Answer</div>
                <div class="result-card-value result-text">${escapeHtml(data.text_answer || '')}</div>
            </div>
            ${exercisesBlock}
            ${motionBlock}
            <details>
                <summary style="cursor:pointer;color:var(--text-muted);font-size:0.8rem;margin-top:8px">Full JSON</summary>
                <pre class="result-json">${syntaxHighlight(JSON.stringify(data, null, 2))}</pre>
            </details>
        </div>
        <div class="result-meta">
            <span>✅ Status: ${status}</span>
            <span>⏱️ ${elapsed}ms wall-clock</span>
            <span>🔧 Pipeline: ${Math.round(data.generation_time_ms || elapsed)}ms</span>
            ${errors ? `<span style="color:var(--error)">⚠️ ${Object.keys(errors).length} error(s)</span>` : ''}
        </div>
    `;
}

/** Render SpeechLLm /synthesize response with audio player and download link. */
function showSpeechLLmResult(container, data, elapsed, status) {
    const audioFilename = data.audio_file ? data.audio_file.split('\\').pop().split('/').pop() : null;
    const audioUrl = audioFilename ? `${SERVICES.speechllm.baseUrl}/audio/${encodeURIComponent(audioFilename)}` : null;

    container.className = 'result-container result-success';
    container.innerHTML = `
        <div class="result-card">
            <div class="result-card-section">
                <div class="result-card-label">🎵 Generated Audio</div>
                <div class="result-card-value">
                    ${audioUrl ? `
                        <audio controls style="width:100%;margin-bottom:8px;">
                            <source src="${audioUrl}" type="audio/wav">
                            Your browser does not support the audio element.
                        </audio>
                        <a href="${audioUrl}" target="_blank" style="color:var(--accent);text-decoration:none;">
                            📥 Download Audio File
                        </a>
                    ` : 'Audio file not available'}
                </div>
            </div>
            <div class="result-card-section">
                <div class="result-card-label">💬 Synthesized Text</div>
                <div class="result-card-value result-text">${escapeHtml(data.text || '')}</div>
            </div>
            <div class="result-card-row">
                <div class="result-card-section">
                    <div class="result-card-label">🎭 Emotion</div>
                    <div class="result-card-value">${data.emotion || 'N/A'}</div>
                </div>
                <div class="result-card-section">
                    <div class="result-card-label">⏱ Duration</div>
                    <div class="result-card-value">${data.duration_seconds ? `${data.duration_seconds.toFixed(2)}s` : 'N/A'}</div>
                </div>
                <div class="result-card-section">
                    <div class="result-card-label">🆔 Request ID</div>
                    <div class="result-card-value"><code>${data.request_id || 'N/A'}</code></div>
                </div>
            </div>
            <details>
                <summary style="cursor:pointer;color:var(--text-muted);font-size:0.8rem;margin-top:8px">Full JSON</summary>
                <pre class="result-json">${syntaxHighlight(JSON.stringify(data, null, 2))}</pre>
            </details>
        </div>
        <div class="result-meta">
            <span>✅ Status: ${status}</span>
            <span>⏱️ ${elapsed}ms</span>
        </div>
    `;
}

// ==============================
// NPZ Runner
// ==============================

let npzPlaybackTimer = null;
let npzSummary = null;

function getNpzBaseUrl() {
    return document.getElementById('npz-base-url')?.value?.trim() || 'http://127.0.0.1:8090';
}

async function loadNpzSummary() {
    const btn = document.getElementById('btn-npz-load');
    const container = document.getElementById('result-npz');
    const timer = document.getElementById('timer-npz');
    const slider = document.getElementById('npz-frame-slider');
    const frameInput = document.getElementById('npz-frame-input');
    const baseUrl = getNpzBaseUrl();

    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Loading...';
    container.innerHTML = '<div class="result-placeholder">Loading NPZ summary...</div>';
    const start = performance.now();
    updateTimer(timer, start);

    try {
        const response = await fetch(`${baseUrl}/api/npz/summary`);
        if (!response.ok) throw new Error(`HTTP ${response.status}: ${await response.text()}`);

        npzSummary = await response.json();
        const maxFrame = Math.max(0, (npzSummary.num_frames || 1) - 1);
        slider.max = String(maxFrame);
        frameInput.max = String(maxFrame);

        const elapsed = Math.round(performance.now() - start);
        clearTimerInterval();
        showResult(container, npzSummary, elapsed, response.status);
        addLog('success', 'NPZ Runner', `Loaded ${npzSummary.num_frames} frames from ${npzSummary.file}`);
        await loadNpzFrame(0);

    } catch (err) {
        clearTimerInterval();
        showError(container, err.message, Math.round(performance.now() - start));
        addLog('error', 'NPZ Runner', err.message);
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<span class="btn-icon">▶</span> Load NPZ';
    }
}

async function loadNpzFrame(frame) {
    const container = document.getElementById('result-npz');
    const slider = document.getElementById('npz-frame-slider');
    const frameInput = document.getElementById('npz-frame-input');
    const baseUrl = getNpzBaseUrl();
    const safeFrame = Math.max(0, parseInt(frame, 10) || 0);

    slider.value = String(safeFrame);
    frameInput.value = String(safeFrame);

    try {
        const response = await fetch(`${baseUrl}/api/npz/frame?i=${safeFrame}`);
        if (!response.ok) throw new Error(`HTTP ${response.status}: ${await response.text()}`);

        const data = await response.json();
        const payload = { summary: npzSummary || null, frame: data };
        const json = syntaxHighlight(JSON.stringify(payload, null, 2));

        container.className = 'result-container result-success';
        container.innerHTML = `
            <div class="result-content">
                <pre class="result-json">${json}</pre>
            </div>
            <div class="result-meta">
                <span>Frame ${data.frame}</span>
                <span>t=${data.time_seconds}s</span>
                <span>transl=(${data.transl.map(v => Number(v).toFixed(3)).join(', ')})</span>
            </div>
        `;
    } catch (err) {
        showError(container, err.message, 0);
        addLog('error', 'NPZ Runner', err.message);
    }
}

function stopNpzPlayback() {
    if (npzPlaybackTimer) { clearInterval(npzPlaybackTimer); npzPlaybackTimer = null; }
    const playBtn = document.getElementById('btn-npz-play');
    if (playBtn) playBtn.innerHTML = '<span class="btn-icon">▶</span> Play';
}

function toggleNpzPlayback() {
    const slider = document.getElementById('npz-frame-slider');
    const timer = document.getElementById('timer-npz');
    const playBtn = document.getElementById('btn-npz-play');

    if (!npzSummary) { loadNpzSummary(); return; }
    if (npzPlaybackTimer) { stopNpzPlayback(); return; }

    const fps = npzSummary.fps || 30;
    const intervalMs = Math.max(1, Math.floor(1000 / fps));
    const maxFrame = parseInt(slider.max, 10) || 0;
    let frame = parseInt(slider.value, 10) || 0;

    playBtn.innerHTML = '<span class="btn-icon">⏸</span> Pause';
    updateTimer(timer, performance.now() - frame * intervalMs);

    npzPlaybackTimer = setInterval(async () => {
        if (frame > maxFrame) { stopNpzPlayback(); clearTimerInterval(); return; }
        await loadNpzFrame(frame++);
    }, intervalMs);
}

function setupNpzControls() {
    const slider = document.getElementById('npz-frame-slider');
    const frameInput = document.getElementById('npz-frame-input');
    if (!slider || !frameInput) return;

    slider.addEventListener('input', () => { stopNpzPlayback(); loadNpzFrame(slider.value); });
    frameInput.addEventListener('change', () => {
        stopNpzPlayback();
        loadNpzFrame(parseInt(frameInput.value, 10) || 0);
    });
}

// ==============================
// Generic Result Rendering
// ==============================

function showResult(container, data, elapsed, status) {
    const json = syntaxHighlight(JSON.stringify(data, null, 2));
    container.className = 'result-container result-success';
    container.innerHTML = `
        <div class="result-content"><pre class="result-json">${json}</pre></div>
        <div class="result-meta">
            <span>✅ Status: ${status}</span>
            <span>⏱️ ${elapsed}ms</span>
            <span>📦 ${JSON.stringify(data).length} bytes</span>
        </div>
    `;
}

function showError(container, message, elapsed) {
    container.className = 'result-container result-error';
    container.innerHTML = `
        <div class="result-content">
            <pre class="result-json" style="color: var(--error);">❌ Error: ${escapeHtml(message)}</pre>
        </div>
        <div class="result-meta">
            <span>❌ Failed</span>
            <span>⏱️ ${elapsed}ms</span>
        </div>
    `;
}

// ==============================
// JSON Syntax Highlighting
// ==============================

function syntaxHighlight(json) {
    json = escapeHtml(json);
    return json.replace(
        /("(\\u[\da-fA-F]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)/g,
        (match) => {
            let cls = 'json-number';
            if (/^"/.test(match)) cls = /:$/.test(match) ? 'json-key' : 'json-string';
            else if (/true|false/.test(match)) cls = 'json-bool';
            else if (/null/.test(match)) cls = 'json-null';
            return `<span class="${cls}">${match}</span>`;
        }
    );
}

function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}



// ==============================
// Timer
// ==============================

let timerInterval = null;

function updateTimer(timerEl, startTime) {
    clearTimerInterval();
    timerInterval = setInterval(() => {
        timerEl.textContent = `${Math.round(performance.now() - startTime)}ms`;
    }, 50);
}

function clearTimerInterval() {
    if (timerInterval) { clearInterval(timerInterval); timerInterval = null; }
}

// ==============================
// Response Log
// ==============================

function addLog(type, service, message) {
    const container = document.getElementById('log-container');
    const emptyMsg = container.querySelector('.log-empty');
    if (emptyMsg) emptyMsg.remove();

    const now = new Date().toLocaleTimeString('en-US', { hour12: false });
    const entry = document.createElement('div');
    entry.className = 'log-entry';
    entry.innerHTML = `
        <span class="log-time">${now}</span>
        <span class="log-badge ${type}">${type}</span>
        <span class="log-message"><strong>${service}</strong> — ${escapeHtml(message)}</span>
    `;
    container.insertBefore(entry, container.firstChild);
    while (container.children.length > 50) container.removeChild(container.lastChild);
}

function clearLog() {
    document.getElementById('log-container').innerHTML =
        '<div class="log-empty">No requests yet. Test an endpoint above.</div>';
}

// ==============================
// Keyboard shortcuts
// ==============================

document.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        const activePanel = document.querySelector('.test-panel.active');
        if (!activePanel) return;
        const id = activePanel.id;
        if (id === 'panel-rag-test') testAgenticRAG();
        else if (id === 'panel-dart-test') testDART();
        else if (id === 'panel-pipeline-test') testPipeline();
        else if (id === 'panel-npz-test') loadNpzSummary();
    }
});

window.addEventListener('DOMContentLoaded', () => { setupNpzControls(); });
