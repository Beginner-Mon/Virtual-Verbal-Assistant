// ==============================
// Configuration
// ==============================

const SERVICES = {
    rag: {
        name: 'AgenticRAG',
        baseUrl: 'http://localhost:8000',
        healthUrl: 'http://localhost:8000/health',
    },
    dart: {
        name: 'DART',
        baseUrl: 'http://localhost:5001',
        healthUrl: 'http://localhost:5001/health',
    },
    orchestrator: {
        name: 'Orchestrator',
        baseUrl: 'http://localhost:8080',
        healthUrl: 'http://localhost:8080/health',
    },
};

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

        if (data.status === 'healthy') {
            dot.className = 'status-dot online';
            label.textContent = `✅ Healthy (${elapsed}ms)`;
            addLog('success', `${service.name}`, `Health check passed in ${elapsed}ms`);
        } else {
            dot.className = 'status-dot offline';
            label.textContent = `⚠️ Unexpected: ${JSON.stringify(data)}`;
            addLog('error', `${service.name}`, `Unexpected response: ${JSON.stringify(data)}`);
        }
    } catch (err) {
        dot.className = 'status-dot offline';
        label.textContent = `❌ Offline — ${err.message}`;
        addLog('error', `${service.name}`, `Health check failed: ${err.message}`);
    }
}

async function checkAllHealth() {
    await Promise.all([
        checkHealth('rag'),
        checkHealth('dart'),
        checkHealth('orchestrator'),
    ]);
}

// Auto-check on load
window.addEventListener('DOMContentLoaded', () => {
    checkAllHealth();
});

// ==============================
// Tab Switching
// ==============================

function switchTab(tabId) {
    // Deactivate all tabs and panels
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.test-panel').forEach(p => p.classList.remove('active'));

    // Activate selected
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
        const response = await fetch('http://localhost:8000/query', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query,
                user_id: userId,
                conversation_history: [],
            }),
        });

        const elapsed = Math.round(performance.now() - start);
        clearTimerInterval();

        if (!response.ok) {
            const error = await response.text();
            throw new Error(`HTTP ${response.status}: ${error}`);
        }

        const data = await response.json();
        showResult(container, data, elapsed, response.status);
        addLog('success', 'AgenticRAG', `Query processed in ${elapsed}ms`);

    } catch (err) {
        clearTimerInterval();
        const elapsed = Math.round(performance.now() - start);
        showError(container, err.message, elapsed);
        addLog('error', 'AgenticRAG', err.message);
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<span class="btn-icon">▶</span> Send Query';
    }
}

async function testDART() {
    const prompt = document.getElementById('dart-prompt').value;
    const primitives = parseInt(document.getElementById('dart-primitives').value) || 20;
    const guidance = parseFloat(document.getElementById('dart-guidance').value) || 5.0;
    const steps = parseInt(document.getElementById('dart-steps').value) || 10;
    const seedInput = document.getElementById('dart-seed').value;
    const seed = seedInput ? parseInt(seedInput) : null;
    const btn = document.getElementById('btn-dart');
    const container = document.getElementById('result-dart');
    const timer = document.getElementById('timer-dart');

    if (!prompt) return;

    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Generating...';
    container.innerHTML = '<div class="result-placeholder">Generating motion...</div>';

    const start = performance.now();
    updateTimer(timer, start);

    try {
        const body = {
            text_prompt: prompt,
            num_primitives: primitives,
            guidance_scale: guidance,
            num_steps: steps,
        };
        if (seed !== null) body.seed = seed;

        const response = await fetch('http://localhost:5001/generate_motion', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });

        const elapsed = Math.round(performance.now() - start);
        clearTimerInterval();

        if (!response.ok) {
            const error = await response.text();
            throw new Error(`HTTP ${response.status}: ${error}`);
        }

        const data = await response.json();
        showResult(container, data, elapsed, response.status);
        addLog('success', 'DART', `Motion generated in ${elapsed}ms — ${data.num_frames} frames, ${data.duration_seconds?.toFixed(1)}s`);

    } catch (err) {
        clearTimerInterval();
        const elapsed = Math.round(performance.now() - start);
        showError(container, err.message, elapsed);
        addLog('error', 'DART', err.message);
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
    container.innerHTML = '<div class="result-placeholder">Running full pipeline (AgenticRAG → DART)...</div>';

    const start = performance.now();
    updateTimer(timer, start);

    try {
        const response = await fetch('http://localhost:8080/answer', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query,
                user_id: userId,
                conversation_history: [],
            }),
        });

        const elapsed = Math.round(performance.now() - start);
        clearTimerInterval();

        if (!response.ok) {
            const error = await response.text();
            throw new Error(`HTTP ${response.status}: ${error}`);
        }

        const data = await response.json();
        showResult(container, data, elapsed, response.status);

        const motionInfo = data.motion ? ` | Motion: ${data.motion.num_frames} frames` : ' | No motion';
        addLog('success', 'Pipeline', `Completed in ${elapsed}ms${motionInfo}`);

    } catch (err) {
        clearTimerInterval();
        const elapsed = Math.round(performance.now() - start);
        showError(container, err.message, elapsed);
        addLog('error', 'Pipeline', err.message);
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<span class="btn-icon">🚀</span> Run Full Pipeline';
    }
}

// ==============================
// NPZ Runner
// ==============================

let npzPlaybackTimer = null;
let npzSummary = null;

function getNpzBaseUrl() {
    const value = document.getElementById('npz-base-url')?.value?.trim();
    return value || 'http://127.0.0.1:8090';
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
        if (!response.ok) {
            const text = await response.text();
            throw new Error(`HTTP ${response.status}: ${text}`);
        }
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
        const elapsed = Math.round(performance.now() - start);
        showError(container, err.message, elapsed);
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
        if (!response.ok) {
            const text = await response.text();
            throw new Error(`HTTP ${response.status}: ${text}`);
        }
        const data = await response.json();
        const payload = {
            summary: npzSummary || null,
            frame: data,
        };
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
    if (npzPlaybackTimer) {
        clearInterval(npzPlaybackTimer);
        npzPlaybackTimer = null;
    }
    const playBtn = document.getElementById('btn-npz-play');
    if (playBtn) {
        playBtn.innerHTML = '<span class="btn-icon">▶</span> Play';
    }
}

function toggleNpzPlayback() {
    const playBtn = document.getElementById('btn-npz-play');
    const slider = document.getElementById('npz-frame-slider');
    const timer = document.getElementById('timer-npz');

    if (!npzSummary) {
        loadNpzSummary();
        return;
    }

    if (npzPlaybackTimer) {
        stopNpzPlayback();
        return;
    }

    const fps = npzSummary.fps || 30;
    const intervalMs = Math.max(1, Math.floor(1000 / fps));
    const maxFrame = parseInt(slider.max, 10) || 0;
    let frame = parseInt(slider.value, 10) || 0;

    playBtn.innerHTML = '<span class="btn-icon">⏸</span> Pause';
    const start = performance.now() - frame * intervalMs;
    updateTimer(timer, start);

    npzPlaybackTimer = setInterval(async () => {
        if (frame > maxFrame) {
            stopNpzPlayback();
            clearTimerInterval();
            return;
        }
        await loadNpzFrame(frame);
        frame += 1;
    }, intervalMs);
}

function setupNpzControls() {
    const slider = document.getElementById('npz-frame-slider');
    const frameInput = document.getElementById('npz-frame-input');
    if (!slider || !frameInput) return;

    slider.addEventListener('input', () => {
        stopNpzPlayback();
        loadNpzFrame(slider.value);
    });

    frameInput.addEventListener('change', () => {
        stopNpzPlayback();
        const frame = parseInt(frameInput.value, 10) || 0;
        loadNpzFrame(frame);
    });
}

// ==============================
// Result Rendering
// ==============================

function showResult(container, data, elapsed, status) {
    const json = syntaxHighlight(JSON.stringify(data, null, 2));
    container.className = 'result-container result-success';
    container.innerHTML = `
        <div class="result-content">
            <pre class="result-json">${json}</pre>
        </div>
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
        function (match) {
            let cls = 'json-number';
            if (/^"/.test(match)) {
                if (/:$/.test(match)) {
                    cls = 'json-key';
                } else {
                    cls = 'json-string';
                }
            } else if (/true|false/.test(match)) {
                cls = 'json-bool';
            } else if (/null/.test(match)) {
                cls = 'json-null';
            }
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
        const elapsed = Math.round(performance.now() - startTime);
        timerEl.textContent = `${elapsed}ms`;
    }, 50);
}

function clearTimerInterval() {
    if (timerInterval) {
        clearInterval(timerInterval);
        timerInterval = null;
    }
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

    // Keep max 50 entries
    while (container.children.length > 50) {
        container.removeChild(container.lastChild);
    }
}

function clearLog() {
    const container = document.getElementById('log-container');
    container.innerHTML = '<div class="log-empty">No requests yet. Test an endpoint above.</div>';
}

// ==============================
// Keyboard shortcuts
// ==============================

document.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        const activePanel = document.querySelector('.test-panel.active');
        if (activePanel) {
            const id = activePanel.id;
            if (id === 'panel-rag-test') testAgenticRAG();
            else if (id === 'panel-dart-test') testDART();
            else if (id === 'panel-pipeline-test') testPipeline();
            else if (id === 'panel-npz-test') loadNpzSummary();
        }
    }
});

window.addEventListener('DOMContentLoaded', () => {
    setupNpzControls();
});
