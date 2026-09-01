// ── Base URL resolution ────────────────────────────────────────────
// Priority: ?api_base= query param → window.TEST_UI_API_BASE_URL → default :8000
// NOTE: 8080 is reserved for the Owner's Spring service — the ECA backend runs on :8000.
function resolveBase(paramName, globalName, fallback) {
    const p = new URLSearchParams(window.location.search).get(paramName);
    return String(p || window[globalName] || fallback).replace(/\/$/, '');
}

const API_BASE    = resolveBase('api_base',    'TEST_UI_API_BASE_URL',    'http://localhost:8000');
const SPEECH_BASE = resolveBase('speech_base', 'TEST_UI_SPEECH_BASE_URL', 'http://localhost:5000');
const SEARXNG_BASE= resolveBase('searxng_base','TEST_UI_SEARXNG_BASE_URL','http://localhost:6666');
const KIMODO_BASE = resolveBase('kimodo_base', 'TEST_UI_KIMODO_BASE_URL', 'http://localhost:5001');

console.log('[test-ui] endpoints', { API_BASE, SPEECH_BASE, SEARXNG_BASE, KIMODO_BASE });

// ── Service registry ───────────────────────────────────────────────
// External services (TTS/SearXNG/Kimodo) don't send CORS headers, so a direct
// browser fetch always fails cross-origin (false negative) regardless of file://
// or http origin. Instead we read their real status from the backend's
// /health/detailed, which probes each dependency server-side (no CORS). The
// `detailedKey` maps a card to the matching `checks.<key>` field.
// speechllm/searxng map to real server-side probes in /health/detailed (accurate).
// kimodo (:5001) has NO server-side probe — backend's `mcp` check counts MCP tools,
// which is decoupled from the :5001 HTTP server — so we probe it directly. That means
// a cross-origin browser fetch shows "down" when :5001 is offline (correct) but also
// when it's up without CORS headers (false negative — a fundamental browser limit).
// VieNeu TTS (:5000) removed — no TTS server exists in this repo (client-only);
// nothing to monitor. Re-add a card + `speechllm` entry if a server is stood up.
const SERVICES = {
    langgraph: { name: 'LangGraph Agent', baseUrl: API_BASE,     healthUrl: `${API_BASE}/health` },
    searxng:   { name: 'SearXNG',         baseUrl: SEARXNG_BASE, detailedKey: 'searxng' },
    kimodo:    { name: 'Kimodo MCP',      baseUrl: KIMODO_BASE,  healthUrl: `${KIMODO_BASE}/health` },
};

// ── Helpers ────────────────────────────────────────────────────────
function escapeHtml(str) {
    const d = document.createElement('div');
    d.textContent = String(str ?? '');
    return d.innerHTML;
}

function syntaxHighlight(json) {
    return escapeHtml(json).replace(
        /("(\\u[\da-fA-F]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)/g,
        m => {
            let c = 'json-number';
            if (/^"/.test(m)) c = /:$/.test(m) ? 'json-key' : 'json-string';
            else if (/true|false/.test(m)) c = 'json-bool';
            else if (/null/.test(m)) c = 'json-null';
            return `<span class="${c}">${m}</span>`;
        }
    );
}

function showResult(container, data, elapsed, status) {
    container.className = 'result-container result-success';
    container.innerHTML = `
        <div class="result-content"><pre class="result-json">${syntaxHighlight(JSON.stringify(data, null, 2))}</pre></div>
        <div class="result-meta">
            <span>✅ ${status}</span><span>⏱️ ${elapsed}ms</span>
            <span>📦 ${JSON.stringify(data).length} bytes</span>
        </div>`;
}

function showError(container, message, elapsed) {
    container.className = 'result-container result-error';
    container.innerHTML = `
        <div class="result-content">
            <pre class="result-json" style="color:var(--error)">❌ ${escapeHtml(message)}</pre>
        </div>
        <div class="result-meta"><span>❌ Failed</span><span>⏱️ ${elapsed}ms</span></div>`;
}

let _timerInterval = null;
function startTimer(el, t0) {
    clearInterval(_timerInterval);
    _timerInterval = setInterval(() => { el.textContent = `${Math.round(performance.now() - t0)}ms`; }, 50);
}
function stopTimer() { clearInterval(_timerInterval); _timerInterval = null; }

function addLog(type, service, message) {
    const c = document.getElementById('log-container');
    c.querySelector('.log-empty')?.remove();
    const now = new Date().toLocaleTimeString('en-US', { hour12: false });
    const el = document.createElement('div');
    el.className = 'log-entry';
    el.innerHTML = `<span class="log-time">${now}</span>
        <span class="log-badge ${type}">${type}</span>
        <span class="log-message"><strong>${escapeHtml(service)}</strong> — ${escapeHtml(message)}</span>`;
    c.insertBefore(el, c.firstChild);
    while (c.children.length > 80) c.removeChild(c.lastChild);
}

function clearLog() {
    document.getElementById('log-container').innerHTML =
        '<div class="log-empty">No requests yet.</div>';
}

// ── Tab switching ──────────────────────────────────────────────────
function switchTab(id) {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.test-panel').forEach(p => p.classList.remove('active'));
    document.getElementById(`tab-${id}`).classList.add('active');
    document.getElementById(`panel-${id}`).classList.add('active');
}

// ── Health checks ──────────────────────────────────────────────────
function _setDot(key, ok, text) {
    document.getElementById(`dot-${key}`).className =
        `status-dot ${ok ? 'online' : 'offline'}`;
    document.getElementById(`label-${key}`).textContent = text;
}

async function checkHealth(key) {
    const svc = SERVICES[key];
    document.getElementById(`dot-${key}`).className = 'status-dot checking';
    document.getElementById(`label-${key}`).textContent = 'Checking…';

    // External deps have no CORS → read their status from backend /health/detailed.
    if (svc.detailedKey) {
        try {
            const res  = await fetch(`${API_BASE}/health/detailed`, { signal: AbortSignal.timeout(8000) });
            const data = await res.json();
            const chk  = data.checks?.[svc.detailedKey];
            if (!chk) throw new Error(`no "${svc.detailedKey}" in /health/detailed`);
            const ms = chk.latency_ms != null ? Math.round(chk.latency_ms) : '?';
            _setDot(key, chk.ok, chk.ok ? `✅ OK (${ms}ms, via backend)` : `❌ down (${chk.detail ?? 'no response'})`);
            addLog(chk.ok ? 'success' : 'error', svc.name, chk.ok ? `Healthy (server-side, ${ms}ms)` : 'Down (server-side probe)');
        } catch (err) {
            _setDot(key, false, `❔ backend unreachable`);
            addLog('error', svc.name, `Can't reach backend /health/detailed: ${err.message}`);
        }
        return;
    }

    // Direct check (LangGraph backend itself).
    try {
        const t0  = performance.now();
        const res = await fetch(svc.healthUrl, { signal: AbortSignal.timeout(3000) });
        const ms  = Math.round(performance.now() - t0);
        _setDot(key, res.ok, res.ok ? `✅ OK (${ms}ms)` : `⚠️ HTTP ${res.status}`);
        addLog(res.ok ? 'success' : 'error', svc.name, res.ok ? `Healthy in ${ms}ms` : `HTTP ${res.status}`);
    } catch (err) {
        _setDot(key, false, `❌ ${err.message}`);
        addLog('error', svc.name, err.message);
    }
}

async function checkHealthDetailed() {
    const container = document.getElementById('result-health');
    container.innerHTML = '<div class="result-placeholder">Fetching /health/detailed…</div>';
    const t0 = performance.now();
    try {
        const res = await fetch(`${API_BASE}/health/detailed`, { signal: AbortSignal.timeout(8000) });
        const ms  = Math.round(performance.now() - t0);
        const data = await res.json();
        showResult(container, data, ms, res.status);
        addLog(res.ok ? 'success' : 'error', 'LangGraph', `Detailed health in ${ms}ms — ${data.status ?? res.status}`);
    } catch (err) {
        showError(container, err.message, Math.round(performance.now() - t0));
        addLog('error', 'LangGraph', err.message);
    }
}

async function checkAllHealth() {
    const detailedKeys = Object.keys(SERVICES).filter(k => SERVICES[k].detailedKey);
    const directKeys   = Object.keys(SERVICES).filter(k => !SERVICES[k].detailedKey);

    // Direct checks (LangGraph) in parallel.
    const jobs = directKeys.map(k => checkHealth(k));

    // One /health/detailed fetch feeds all external-dep dots.
    detailedKeys.forEach(k => {
        document.getElementById(`dot-${k}`).className = 'status-dot checking';
        document.getElementById(`label-${k}`).textContent = 'Checking…';
    });
    jobs.push((async () => {
        try {
            const res  = await fetch(`${API_BASE}/health/detailed`, { signal: AbortSignal.timeout(8000) });
            const data = await res.json();
            for (const k of detailedKeys) {
                const chk = data.checks?.[SERVICES[k].detailedKey];
                if (!chk) { _setDot(k, false, '❔ not in /health/detailed'); continue; }
                const ms = chk.latency_ms != null ? Math.round(chk.latency_ms) : '?';
                _setDot(k, chk.ok, chk.ok ? `✅ OK (${ms}ms, via backend)` : `❌ down (${chk.detail ?? 'no response'})`);
            }
        } catch (err) {
            detailedKeys.forEach(k => _setDot(k, false, '❔ backend unreachable'));
            addLog('error', 'LangGraph', `/health/detailed: ${err.message}`);
        }
    })());

    await Promise.all(jobs);
}

// ── SSE Chat test ──────────────────────────────────────────────────
let _chatAbort = null;

async function testChat() {
    const query     = document.getElementById('chat-query').value.trim();
    const userId    = document.getElementById('chat-user-id').value.trim() || 'test-user';
    const sessionId = document.getElementById('chat-session-id').value.trim() || crypto.randomUUID();
    const personaId = document.getElementById('chat-persona').value;
    const webSearch = document.getElementById('chat-web-search').checked;
    const tts       = document.getElementById('chat-tts').checked;
    const btn       = document.getElementById('btn-chat');
    const stopBtn   = document.getElementById('btn-chat-stop');
    const output    = document.getElementById('chat-output');
    const events    = document.getElementById('chat-events');
    const timer     = document.getElementById('timer-chat');

    if (!query) return;

    // Abort previous if running
    if (_chatAbort) { _chatAbort.abort(); }
    _chatAbort = new AbortController();

    btn.disabled = true;
    stopBtn.style.display = 'inline-flex';
    output.textContent = '';
    events.innerHTML = '';
    const t0 = performance.now();
    startTimer(timer, t0);

    try {
        const res = await fetch(`${API_BASE}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query, user_id: userId, session_id: sessionId,
                persona_id: personaId,
                output_mode: tts ? 'speech' : 'text',
                web_search: webSearch,
            }),
            signal: _chatAbort.signal,
        });

        if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`);

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buf = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buf += decoder.decode(value, { stream: true });

            const parts = buf.split('\n\n');
            buf = parts.pop();

            for (const chunk of parts) {
                const lines = chunk.split('\n');
                let eventType = 'message', dataLine = '';
                for (const l of lines) {
                    if (l.startsWith('event:')) eventType = l.slice(6).trim();
                    if (l.startsWith('data:'))  dataLine  = l.slice(5).trim();
                }
                if (!dataLine) continue;

                let payload = {};
                try { payload = JSON.parse(dataLine); } catch { continue; }

                if (eventType === 'token') {
                    output.textContent += payload.content || '';
                    output.scrollTop = output.scrollHeight;
                } else {
                    addEventBadge(events, eventType, payload);
                }
            }
        }

        stopTimer();
        const ms = Math.round(performance.now() - t0);
        addLog('success', 'LangGraph Chat', `Done in ${ms}ms`);

    } catch (err) {
        stopTimer();
        if (err.name !== 'AbortError') {
            output.textContent += `\n\n[ERROR: ${err.message}]`;
            addLog('error', 'LangGraph Chat', err.message);
        } else {
            addLog('info', 'LangGraph Chat', 'Stopped by user');
        }
    } finally {
        btn.disabled = false;
        stopBtn.style.display = 'none';
        _chatAbort = null;
    }
}

function stopChat() {
    if (_chatAbort) { _chatAbort.abort(); _chatAbort = null; }
}

function addEventBadge(container, eventType, payload) {
    const colors = {
        stage: '#7c80ff', tool_executing: '#f59e0b', tool_output: '#10b981',
        speech_pending: '#a78bfa', speech_ready: '#34d399', speech_failed: '#ef4444',
        session_persisted: '#6b7280', done: '#22c55e', error: '#ef4444',
    };
    const color = colors[eventType] || '#888';
    const span = document.createElement('span');
    span.style.cssText = `display:inline-block;margin:2px;padding:2px 8px;border-radius:12px;
        font-size:0.72rem;font-weight:600;background:${color}22;color:${color};border:1px solid ${color}44;`;

    let label = eventType;
    if (eventType === 'stage')    label = `${payload.node} ${payload.status}`;
    if (eventType === 'done')     label = `done · ${payload.total_tokens ?? 0} tok · ${(payload.required_outputs ?? []).join(', ')}`;
    if (eventType === 'error')    label = `error: ${payload.message ?? ''}`;
    if (eventType === 'speech_ready') label = `🔊 speech_ready`;

    span.textContent = label;
    container.appendChild(span);
}

// ── Session management ─────────────────────────────────────────────
async function listSessions() {
    const userId    = document.getElementById('sess-user-id').value.trim() || 'test-user';
    const container = document.getElementById('result-sessions');
    const t0 = performance.now();
    container.innerHTML = '<div class="result-placeholder">Loading…</div>';
    try {
        const res = await fetch(`${API_BASE}/sessions?user_id=${encodeURIComponent(userId)}`);
        const ms  = Math.round(performance.now() - t0);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        renderSessions(container, data, ms);
        addLog('success', 'Sessions', `${data.total} session(s) for ${userId} in ${ms}ms`);
    } catch (err) {
        showError(container, err.message, Math.round(performance.now() - t0));
        addLog('error', 'Sessions', err.message);
    }
}

function renderSessions(container, data, ms) {
    if (!data.sessions?.length) {
        container.className = 'result-container result-success';
        container.innerHTML = `<div class="result-placeholder">No sessions found.</div>
            <div class="result-meta"><span>✅ 200</span><span>⏱️ ${ms}ms</span></div>`;
        return;
    }
    const rows = data.sessions.map(s => `
        <tr>
            <td><code style="font-size:0.78rem">${escapeHtml(s.session_id.slice(0,8))}…</code></td>
            <td>${escapeHtml(s.first_user_message_preview ?? '')}</td>
            <td style="white-space:nowrap">${new Date(s.updated_at).toLocaleString()}</td>
            <td>${s.message_count ?? '—'}</td>
            <td>
                <button class="btn btn-sm btn-outline" onclick="resumeSession('${escapeHtml(s.session_id)}')">Resume</button>
                <button class="btn btn-sm btn-outline" style="color:var(--error)" onclick="deleteSession('${escapeHtml(s.session_id)}')">Delete</button>
            </td>
        </tr>`).join('');

    container.className = 'result-container result-success';
    container.innerHTML = `
        <div style="overflow-x:auto">
            <table style="width:100%;border-collapse:collapse;font-size:0.85rem">
                <thead><tr style="color:var(--text-muted);font-size:0.75rem">
                    <th style="text-align:left;padding:6px 8px">Session</th>
                    <th style="text-align:left;padding:6px 8px">First message</th>
                    <th style="text-align:left;padding:6px 8px">Last updated</th>
                    <th style="text-align:left;padding:6px 8px">Msgs</th>
                    <th style="padding:6px 8px">Actions</th>
                </tr></thead>
                <tbody>${rows}</tbody>
            </table>
        </div>
        <div class="result-meta"><span>✅ ${data.total} sessions</span><span>⏱️ ${ms}ms</span></div>`;
}

async function resumeSession(sessionId) {
    const userId    = document.getElementById('sess-user-id').value.trim() || 'test-user';
    const container = document.getElementById('result-sessions');
    const t0 = performance.now();
    try {
        const res = await fetch(
            `${API_BASE}/sessions/${encodeURIComponent(sessionId)}?user_id=${encodeURIComponent(userId)}`
        );
        const ms   = Math.round(performance.now() - t0);
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail ?? `HTTP ${res.status}`);
        showResult(container, data, ms, res.status);
        // Pre-fill chat session id so user can continue the conversation
        document.getElementById('chat-session-id').value = sessionId;
        document.getElementById('chat-user-id').value   = userId;
        addLog('success', 'Sessions', `Resumed ${sessionId.slice(0,8)}… in ${ms}ms`);
    } catch (err) {
        showError(container, err.message, Math.round(performance.now() - t0));
        addLog('error', 'Sessions', err.message);
    }
}

async function deleteSession(sessionId) {
    if (!confirm(`Delete session ${sessionId.slice(0,8)}…?`)) return;
    const userId    = document.getElementById('sess-user-id').value.trim() || 'test-user';
    const container = document.getElementById('result-sessions');
    const t0 = performance.now();
    try {
        const res = await fetch(
            `${API_BASE}/sessions/${encodeURIComponent(userId)}/${encodeURIComponent(sessionId)}`,
            { method: 'DELETE' }
        );
        const ms   = Math.round(performance.now() - t0);
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail ?? `HTTP ${res.status}`);
        addLog('success', 'Sessions', `Deleted ${sessionId.slice(0,8)}… in ${ms}ms`);
        await listSessions();
    } catch (err) {
        showError(container, err.message, Math.round(performance.now() - t0));
        addLog('error', 'Sessions', err.message);
    }
}

// ── Init ───────────────────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', () => {
    // Pre-fill a random session id for chat
    document.getElementById('chat-session-id').value = crypto.randomUUID();
    checkAllHealth();
});

document.addEventListener('keydown', e => {
    if (e.key !== 'Enter' || e.shiftKey) return;
    const panel = document.querySelector('.test-panel.active');
    if (!panel) return;
    if (panel.id === 'panel-chat')     testChat();
    if (panel.id === 'panel-sessions') listSessions();
});
