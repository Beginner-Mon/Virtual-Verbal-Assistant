// API Handler for ECA UI using Axios
// Uses unified AgenticRAG gateway on port 8000.
// Priority for API base URL:
// 1) Query parameter: ?api_base=https://your-api-host
// 2) Global override: window.ECA_API_BASE_URL
// 3) Local default: http://localhost:8000

const API_BASE_STORAGE_KEY = "eca_api_base_url";

function normalizeApiBaseUrl(url) {
  return String(url || "").trim().replace(/\/$/, "");
}

function getStoredApiBaseUrl() {
  try {
    const stored = window.localStorage.getItem(API_BASE_STORAGE_KEY);
    if (!stored) return null;
    return normalizeApiBaseUrl(stored);
  } catch (_err) {
    return null;
  }
}

function setStoredApiBaseUrl(url) {
  try {
    const normalized = normalizeApiBaseUrl(url);
    if (!normalized) return;
    window.localStorage.setItem(API_BASE_STORAGE_KEY, normalized);
  } catch (_err) {
    // Ignore storage issues (privacy mode, disabled storage, etc.)
  }
}

function parseApiBaseFromQuery() {
  const params = new URLSearchParams(window.location.search);
  const fromQuery = params.get("api_base");
  return fromQuery ? normalizeApiBaseUrl(fromQuery) : null;
}

function resolveApiBaseUrl() {
  const fromQuery = parseApiBaseFromQuery();
  const fromGlobal = normalizeApiBaseUrl(window.ECA_API_BASE_URL);
  const fromStorage = getStoredApiBaseUrl();

  const host = String(window.location.hostname || "").toLowerCase();
  const isLocal = host === "localhost" || host === "127.0.0.1";
  const isNgrok = host.endsWith(".ngrok-free.app") || host.endsWith(".ngrok.app");

  if (fromQuery || fromGlobal || fromStorage) {
    const chosen = fromQuery || fromGlobal || fromStorage;
    if (fromQuery) {
      setStoredApiBaseUrl(fromQuery);
    }
    return chosen;
  }

  // Local browser can safely call local API.
  // Backend (LangGraph FastAPI) runs on 8080. Legacy orchestrator 8000 — removed.
  if (isLocal) {
    return "http://localhost:8080";
  }

  // For ngrok-hosted UI, prefer same-origin API so one 8000 tunnel works
  // when UI is served from /eca on the gateway.
  if (isNgrok) {
    return String(window.location.origin).replace(/\/$/, "");
  }

  // Fallback for other hosted scenarios where API may be reverse-proxied on same origin.
  return String(window.location.origin).replace(/\/$/, "");
}

let API_BASE_URL = resolveApiBaseUrl();
console.log("[ECA_UI] API_BASE_URL =", API_BASE_URL);

function promptForApiBaseUrl() {
  const suggested = parseApiBaseFromQuery() || getStoredApiBaseUrl() || "https://<your-api-tunnel>.ngrok-free.app";
  const entered = window.prompt(
    "This URL only serves the UI. Paste your API base URL (example: https://<your-api-tunnel>.ngrok-free.app)",
    suggested
  );

  if (!entered) {
    return null;
  }

  const normalized = normalizeApiBaseUrl(entered);
  if (!/^https?:\/\//i.test(normalized)) {
    throw new Error("Invalid API URL. Please include http:// or https://");
  }

  setStoredApiBaseUrl(normalized);
  return normalized;
}

const POLL_INTERVAL_MS = 1500;
const NOT_FOUND_RETRY_LIMIT = 3;
const POLL_TIMEOUT_MS = 600000;

function joinApiBase(path) {
  const base = normalizeApiBaseUrl(API_BASE_URL);
  if (!path) {
    return base;
  }
  const suffix = String(path).startsWith("/") ? path : `/${path}`;
  return `${base}${suffix}`;
}

function normalizeMotionFileUrl(rawUrl) {
  if (!rawUrl) {
    return null;
  }

  const value = String(rawUrl).trim();
  if (!value) {
    return null;
  }

  // Already a proxy-friendly relative path.
  if (value.startsWith("/download/")) {
    return joinApiBase(value);
  }

  // Non-absolute paths are assumed to be API-relative.
  if (!/^https?:\/\//i.test(value)) {
    return joinApiBase(value);
  }

  try {
    const artifactUrl = new URL(value);
    const apiBase = new URL(joinApiBase("/"));

    // Keep same-origin absolute URLs unchanged.
    if (artifactUrl.origin.toLowerCase() === apiBase.origin.toLowerCase()) {
      return value;
    }

    // For cross-origin DART URLs, route through API proxy when path is /download/*.
    const marker = "/download/";
    const idx = artifactUrl.pathname.toLowerCase().indexOf(marker);
    if (idx >= 0) {
      const proxiedPath = artifactUrl.pathname.substring(idx);
      return joinApiBase(`${proxiedPath}${artifactUrl.search || ""}`);
    }

    return value;
  } catch {
    return value;
  }
}

function normalizeExercises(exercises) {
  if (!Array.isArray(exercises)) {
    return [];
  }

  const normalized = [];
  const seen = new Set();

  for (const item of exercises) {
    let name = "";
    if (item && typeof item === "object") {
      name = String(item.name || item.title || "").trim();
    } else if (typeof item === "string") {
      name = item.trim();
    }

    if (!name) {
      continue;
    }

    const key = name.toLowerCase();
    if (seen.has(key)) {
      continue;
    }

    seen.add(key);
    normalized.push({ name });
  }

  return normalized;
}

function extractExercisesFromText(answer, limit = 8) {
  const text = String(answer || "").trim();
  if (!text) {
    return [];
  }

  const movementKeywords = /\b(stretch|tilt|squat|lunge|bridge|plank|raise|rotation|twist|extension|flexion|cat\s*-?\s*cow|bird\s*dog|dead\s*bug|glute|hamstring|mobility|hip\s*flexor|thoracic|pelvic|back)\b/i;

  const lines = text.split(/\r?\n/);
  const found = [];
  const seen = new Set();

  for (const line of lines) {
    const match = line.match(/^\s*(?:\d+[\.)]|[-*])\s+(.+)$/);
    if (!match) {
      continue;
    }

    let candidate = String(match[1] || "").trim();
    candidate = candidate.replace(/\[([^\]]+)\]\([^\)]+\)/g, "$1");
    candidate = candidate.replace(/https?:\/\/\S+/g, "");
    candidate = candidate.split(/\s*:\s*/, 1)[0].trim();
    candidate = candidate.split(/\s+[\-–—]\s+/, 1)[0].trim();
    candidate = candidate.replace(/\(.*?\)/g, "").trim();
    candidate = candidate.replace(/[^A-Za-z0-9\s/+\-]/g, "").trim();

    if (!candidate || candidate.length > 60) {
      continue;
    }

    const lowered = candidate.toLowerCase();
    if (lowered.startsWith("source") || lowered.includes("http") || lowered.includes("www")) {
      continue;
    }

    // Avoid turning generic step labels into exercise chips.
    if (!movementKeywords.test(candidate)) {
      continue;
    }

    if (seen.has(lowered)) {
      continue;
    }

    seen.add(lowered);
    found.push({ name: candidate });

    if (found.length >= limit) {
      break;
    }
  }

  return found;
}

function flattenTaskPayload(taskPayload) {
  const task = taskPayload || {};
  const isDirectAnswer = "text_answer" in task;
  const result = isDirectAnswer ? task : (task.result || {});
  const selectedStrategy = String(
    task.selected_strategy || result.selected_strategy || result?.orchestrator_decision?.intent || ""
  ).trim().toLowerCase();
  const answerText = result.text_answer || result.clinical_advice || "";
  const explicitExercises = normalizeExercises(result.exercises);
  const inferredExercises = (!explicitExercises.length && selectedStrategy === "exercise_recommendation")
    ? extractExercisesFromText(answerText)
    : [];
  const normalizedExercises = explicitExercises.length > 0 ? explicitExercises : inferredExercises;
  
  const motionPayload = result.motion || {};
  const motionJob = result.motion_job || {};
  // In AnswerResponse, motion is a dict with motion_file_url directly.
  const motionUrlRaw = result.motion_file_url || motionPayload.motion_file_url || motionJob.motion_file_url || null;
  const motionUrl = normalizeMotionFileUrl(motionUrlRaw);
  const motionPrompt =
    result.exercise_motion_prompt ||
    motionPayload.text_prompt ||
    motionJob?.selected_candidate?.rewritten_prompt ||
    motionJob?.selected_candidate?.text_description ||
    null;

  const ttsObj = result?.metadata?.tts || task?.tts || result?.tts || null;
  console.log("[flattenTaskPayload] Extracted TTS object:", ttsObj, "from taskPayload:", taskPayload);

  return {
    ...result,
    task_id: task.task_id || task.request_id,
    status: task.status || "processing",
    progress_stage: task.progress_stage || "queued",
    selected_strategy: selectedStrategy || null,
    intent: selectedStrategy || null,
    error: task.error || (task.errors ? JSON.stringify(task.errors) : null),
    text_answer: result.text_answer || "",
    clinical_advice: result.text_answer || "",
    exercises: normalizedExercises,
    motion_duration_seconds: result.motion_duration_seconds ?? motionPayload.duration_seconds ?? null,
    motion_error: result.motion_error || task.error || null,
    tts: ttsObj,
    motion: motionUrl
      ? {
          motion_file_url: motionUrl,
          prompt: motionPrompt,
          frames: motionPayload.frames || motionPayload.num_frames || motionJob.frames || null,
          fps: motionPayload.fps || motionJob.fps || null,
          duration_seconds: motionPayload.duration_seconds || motionJob.duration_seconds || result.motion_duration_seconds || null,
          stage: motionJob.stage || null,
        }
      : null,
  };
}

async function fetchChatHistory(userId = "guest") {
  const response = await axios.get(`${API_BASE_URL}/history/${encodeURIComponent(userId)}`);
  const data = response.data || {};
  return Array.isArray(data.messages) ? data.messages : [];
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

/**
 * Poll unified task status until terminal state.
 * Retries initial transient 404 responses to avoid premature failure.
 */
async function pollTaskStatus(taskId, onProgress = null, { stopWhenTextReady = false, maxRetries = 150 } = {}) {
  let notFoundCount = 0;
  let lastTaskData = null;

  for (let attempt = 0; attempt < maxRetries; attempt++) {
    try {
      // Changed to poll /answer/status instead of /tasks for Orchestrator routing
      let statusResp = await axios.get(`${API_BASE_URL}/answer/status/${encodeURIComponent(taskId)}`);
      let data = statusResp.data || {};
      lastTaskData = data;
      const flattened = flattenTaskPayload(data);

      if (onProgress) {
        onProgress(flattened);
      }

      const hasText = Boolean(flattened.text_answer && String(flattened.text_answer).trim());
      if (stopWhenTextReady && hasText) {
        return data;
      }

      if (data.status === "completed" || data.status === "failed") {
        return data;
      }
    } catch (error) {
      const status = error?.response?.status;
      if (status === 404 && notFoundCount < NOT_FOUND_RETRY_LIMIT) {
        notFoundCount += 1;
      } else {
        throw error;
      }
    }

    await sleep(POLL_INTERVAL_MS);
  }

  const timeoutError = new Error("Task polling timed out");
  timeoutError.code = "TASK_POLL_TIMEOUT";
  timeoutError.partial = lastTaskData ? flattenTaskPayload(lastTaskData) : null;
  throw timeoutError;
}

/**
 * Sends a query to unified AgenticRAG endpoint and polls task status.
 * 
 * @param {string} query - The user's input text
 * @param {string} userId - The unique identifier for the user session
 * @param {string} sessionId - Optional session ID
 * @param {function} onProgress - Callback triggered when polling updates
 * @returns {Promise<Object>} Flattened final payload for UI consumption
 */
async function askEca(query, userId = "guest", sessionId = null, onProgress = null, allowRecovery = true) {
  try {
    const payload = { query: query, user_id: userId };
    if (sessionId) payload.session_id = sessionId;

    // 1. Submit unified async task via Orchestrator endpoint (includes TTS)
    const response = await axios.post(`${API_BASE_URL}/answer`, payload);

    const submitData = response.data || {};
    // Orchestrator uses request_id instead of task_id
    const taskId = submitData.request_id || submitData.task_id;
    if (!taskId) {
      throw new Error("Missing request_id from /answer response");
    }

    if (onProgress) {
      onProgress(flattenTaskPayload(submitData));
    }

    if (submitData.status === "completed" || submitData.status === "failed") {
      const payload = flattenTaskPayload(submitData);
      if (submitData.status === "failed" && !payload.text_answer) {
        throw new Error(submitData.error || submitData.errors?.agenticrag || "Task failed sync");
      }
      return payload;
    }

    const submitPayload = flattenTaskPayload(submitData);
    let textReadyPayload = submitPayload;

    if (!submitPayload.text_answer || !String(submitPayload.text_answer).trim()) {
      // 2. Poll until text is available, then return early while motion continues.
      const textReadyTask = await pollTaskStatus(taskId, onProgress, { stopWhenTextReady: true });
      textReadyPayload = flattenTaskPayload(textReadyTask);
    }

    const isTerminal = textReadyPayload.status === "completed" || textReadyPayload.status === "failed";
    if (isTerminal) {
      if (textReadyPayload.status === "failed" && !textReadyPayload.text_answer) {
        const terminalError = new Error(textReadyPayload.error || "Task failed");
        terminalError.partial = textReadyPayload;
        throw terminalError;
      }
      return textReadyPayload;
    }

    const finalPromise = pollTaskStatus(taskId, onProgress)
      .then((finalTask) => {
        const finalPayload = flattenTaskPayload(finalTask);
        if (finalTask.status === "failed" && !finalPayload.text_answer) {
          const terminalError = new Error(finalTask.error || "Task failed");
          terminalError.partial = finalPayload;
          throw terminalError;
        }
        return finalPayload;
      })
      .catch((error) => {
        if (!error.partial) {
          error.partial = textReadyPayload;
        }
        throw error;
      });

    return {
      ...textReadyPayload,
      task_id: taskId,
      finalPromise,
    };
  } catch (error) {
    const status = error?.response?.status;
    const sameOrigin = API_BASE_URL === String(window.location.origin).replace(/\/$/, "");
    if (status === 404 && sameOrigin) {
      if (allowRecovery) {
        try {
          const recoveredBase = promptForApiBaseUrl();
          if (recoveredBase) {
            API_BASE_URL = recoveredBase;
            console.log("[ECA_UI] API_BASE_URL updated to", API_BASE_URL);
            return askEca(query, userId, sessionId, onProgress, false);
          }
        } catch (recoverErr) {
          throw recoverErr;
        }
      }
      throw new Error(
        "This ngrok URL serves UI but not /answer. Provide ?api_base=<api-ngrok-url>, or open /eca/ on the single 8000 tunnel."
      );
    }
    console.error("API Error in askEca:", error);
    throw error;
  }
}

// Session APIs
async function createSession(userId) {
  const response = await axios.post(`${API_BASE_URL}/sessions`, { user_id: userId });
  return response.data;
}

async function listSessions(userId) {
  const response = await axios.get(`${API_BASE_URL}/sessions/${encodeURIComponent(userId)}`);
  const data = response.data;
  if (!data) return [];
  if (Array.isArray(data)) return data;
  if (Array.isArray(data.sessions)) return data.sessions;
  return Object.values(data);
}

async function getSession(userId, sessionId) {
  const response = await axios.get(`${API_BASE_URL}/sessions/${encodeURIComponent(userId)}/${encodeURIComponent(sessionId)}`);
  return response.data;
}

async function deleteSession(userId, sessionId) {
  const response = await axios.delete(`${API_BASE_URL}/sessions/${encodeURIComponent(userId)}/${encodeURIComponent(sessionId)}`);
  return response.data;
}

async function uploadDocument(userId, file) {
  const formData = new FormData();
  formData.append("user_id", userId);
  formData.append("file", file);
  
  const response = await axios.post(`${API_BASE_URL}/documents/upload`, formData, {
    headers: {
      "Content-Type": "multipart/form-data"
    }
  });
  return response.data;
}

async function removeDocument(userId, filename) {
  const response = await axios.delete(`${API_BASE_URL}/documents/${encodeURIComponent(filename)}?user_id=${encodeURIComponent(userId)}`);
  return response.data;
}

async function clearMemory(userId) {
  const response = await axios.delete(`${API_BASE_URL}/memory/${encodeURIComponent(userId)}`);
  return response.data;
}

// ── Phase 5: SSE streaming chat ──────────────────────────────────────


// ── SSE parser ───────────────────────────────────────────────────────
// Parse SSE blocks (separated by blank line) and emit each event via callback.
function _parseSSEBlocks(text, emit) {
  for (const block of text.split(/\r?\n\r?\n/)) {
    if (!block.trim()) continue;
    let eventType = null;
    let dataStr = "";
    for (const line of block.split(/\r?\n/)) {
      if (line.startsWith(":")) continue;   // comment / heartbeat
      if (line.startsWith("event:")) eventType = line.slice(6).trim();
      else if (line.startsWith("data:")) dataStr += line.slice(5).trim();
    }
    if (!eventType) continue;
    try {
      emit(eventType, dataStr ? JSON.parse(dataStr) : {});
    } catch (e) {
      console.warn("Failed to parse SSE data:", dataStr, e);
    }
  }
}


async function streamChat({ query, userId, sessionId, personaId, outputMode, webSearch, onEvent }) {
  const url = joinApiBase("/chat");
  const body = JSON.stringify({
    query, user_id: userId, session_id: sessionId,
    persona_id: personaId || "eca_default",
    output_mode: outputMode || "text",
    web_search: webSearch || false,
  });

  const resp = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json", "Accept": "text/event-stream" },
    body,
  });
  if (!resp.ok) throw new Error(`HTTP ${resp.status}: ${await resp.text()}`);

  // Clone so we can fallback to text() if the reader path consumes the body
  // without yielding events (AV/proxy buffer issue — body is in memory but
  // ReadableStream yields done immediately).
  const respClone = resp.clone();

  // Attempt 1: stream via getReader. If it gives nothing (AV/proxy buffered the
  // response — known issue when antivirus MITM-inspects CORS chunked traffic),
  // fall through to text() which works because body was already buffered.
  const reader = resp.body && resp.body.getReader ? resp.body.getReader() : null;
  if (reader) {
    const decoder = new TextDecoder();
    let buffer = "";
    let emittedAny = false;
    const wrappedEmit = (t, d) => { emittedAny = true; console.log("[SSE]", t, d); onEvent(t, d); };

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lastBoundary = buffer.lastIndexOf("\n\n");
        if (lastBoundary === -1) continue;
        const ready = buffer.slice(0, lastBoundary);
        buffer = buffer.slice(lastBoundary + 2);
        _parseSSEBlocks(ready, wrappedEmit);
      }
      if (buffer.trim()) _parseSSEBlocks(buffer, wrappedEmit);
      if (emittedAny) return;   // streaming worked
      console.warn("[SSE] reader returned 0 events — falling back to text() parsing");
    } catch (e) {
      console.warn("[SSE] reader failed, falling back to text():", e);
    }
  }

  // Fallback: response.body has been buffered server/proxy-side. Body is
  // already in memory — parse all events but EMIT WITH DELAY to simulate
  // typing (real streaming UX). Without this, all 100+ token events fire
  // synchronously and the user sees a single chunk pop in.
  const fullBody = await respClone.text();
  console.log("[SSE-FALLBACK] parsing", fullBody.length, "chars at once (will replay with typing delay)");

  // Collect events first
  const events = [];
  _parseSSEBlocks(fullBody, (t, d) => events.push({ type: t, data: d }));

  // Replay with delay only for token events (~50 tokens/sec feel)
  const TYPING_DELAY_MS = 15;
  for (const e of events) {
    console.log("[SSE-FB]", e.type, e.data);
    onEvent(e.type, e.data);
    if (e.type === "token") {
      await new Promise(r => setTimeout(r, TYPING_DELAY_MS));
    }
  }
}


async function listSessionsV2(userId) {
  const resp = await fetch(joinApiBase(`/sessions?user_id=${encodeURIComponent(userId)}`));
  if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
  return await resp.json();
}


async function resumeSession(sessionId, userId) {
  const resp = await fetch(joinApiBase(`/sessions/${sessionId}/resume?user_id=${encodeURIComponent(userId)}`), {
    method: "POST",
  });
  if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
  return await resp.json();
}


// Expose to UI
window.ECA_API = { streamChat, listSessions: listSessionsV2, resumeSession };

// ── Legacy exports (keep existing React working) ─────────────────────

// Make it globally available for Babel/React script
window.askEca = askEca;
window.pollTaskStatus = pollTaskStatus;
window.fetchChatHistory = fetchChatHistory;
window.flattenTaskPayload = flattenTaskPayload;
window.createSession = createSession;
window.listSessions = listSessions;
window.getSession = getSession;
window.deleteSession = deleteSession;
window.uploadDocument = uploadDocument;
window.removeDocument = removeDocument;
window.clearMemory = clearMemory;
