// API Handler for ECA UI using Axios
// Uses unified AgenticRAG gateway on port 8000.
// Priority for API base URL:
// 1) Query parameter: ?api_base=https://your-api-host
// 2) Global override: window.ECA_API_BASE_URL
// 3) Local default: http://localhost:8000

function resolveApiBaseUrl() {
  const params = new URLSearchParams(window.location.search);
  const fromQuery = params.get("api_base");
  const fromGlobal = window.ECA_API_BASE_URL;

  const host = String(window.location.hostname || "").toLowerCase();
  const isLocal = host === "localhost" || host === "127.0.0.1";
  const isNgrok = host.endsWith(".ngrok-free.app") || host.endsWith(".ngrok.app");

  if (fromQuery || fromGlobal) {
    return String(fromQuery || fromGlobal).replace(/\/$/, "");
  }

  // Local browser can safely call local API.
  if (isLocal) {
    return "http://localhost:8000";
  }

  // For ngrok-hosted UI, prefer same-origin API so one 8000 tunnel works
  // when UI is served from /eca on the gateway.
  if (isNgrok) {
    return String(window.location.origin).replace(/\/$/, "");
  }

  // Fallback for other hosted scenarios where API may be reverse-proxied on same origin.
  return String(window.location.origin).replace(/\/$/, "");
}

const API_BASE_URL = resolveApiBaseUrl();
console.log("[ECA_UI] API_BASE_URL =", API_BASE_URL);

const POLL_INTERVAL_MS = 1500;
const NOT_FOUND_RETRY_LIMIT = 3;
const POLL_TIMEOUT_MS = 180000;

function flattenTaskPayload(taskPayload) {
  const task = taskPayload || {};
  const result = task.result || {};
  const motionPayload = result.motion || {};
  const motionJob = result.motion_job || {};
  const motionUrl = result.motion_file_url || motionJob.motion_file_url || null;
  const motionPrompt =
    result.exercise_motion_prompt ||
    motionJob?.selected_candidate?.rewritten_prompt ||
    motionJob?.selected_candidate?.text_description ||
    null;

  return {
    ...result,
    task_id: task.task_id,
    status: task.status || "processing",
    progress_stage: task.progress_stage || "queued",
    error: task.error || null,
    text_answer: result.text_answer || "",
    clinical_advice: result.text_answer || "",
    motion_duration_seconds: result.motion_duration_seconds ?? null,
    motion: motionUrl
      ? {
          motion_file_url: motionUrl,
          prompt: motionPrompt,
          frames: motionPayload.frames || motionJob.frames || null,
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
async function pollTaskStatus(taskId, onProgress = null) {
  const startedAt = Date.now();
  let notFoundCount = 0;

  while (Date.now() - startedAt < POLL_TIMEOUT_MS) {
    try {
      const response = await axios.get(`${API_BASE_URL}/tasks/${taskId}`);
      const data = response.data || {};

      if (onProgress) {
        onProgress(flattenTaskPayload(data));
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

  throw new Error("Task polling timed out");
}

/**
 * Sends a query to unified AgenticRAG endpoint and polls task status.
 * 
 * @param {string} query - The user's input text
 * @param {string} userId - The unique identifier for the user session
 * @param {function} onProgress - Callback triggered when polling updates
 * @returns {Promise<Object>} Flattened final payload for UI consumption
 */
async function askEca(query, userId = "guest", onProgress = null) {
  try {
    // 1. Submit unified async task
    const response = await axios.post(`${API_BASE_URL}/process_query`, {
      query: query,
      user_id: userId
    });

    const submitData = response.data || {};
    const taskId = submitData.task_id;
    if (!taskId) {
      throw new Error("Missing task_id from /process_query response");
    }

    if (onProgress) {
      onProgress(flattenTaskPayload({
        task_id: taskId,
        status: submitData.status || "processing",
        progress_stage: submitData.progress_stage || "queued",
        result: submitData.result || null,
        error: submitData.error || null,
      }));
    }

    // 2. Poll task state
    const finalTask = await pollTaskStatus(taskId, onProgress);
    if (finalTask.status === "failed") {
      throw new Error(finalTask.error || "Task failed");
    }

    return flattenTaskPayload(finalTask);
  } catch (error) {
    const status = error?.response?.status;
    if (status === 404 && API_BASE_URL === String(window.location.origin).replace(/\/$/, "")) {
      throw new Error(
        "This ngrok URL serves UI but not /process_query. Use the API tunnel via ?api_base=<api-ngrok-url>, or use a single 8000 tunnel and open /eca/."
      );
    }
    console.error("API Error in askEca:", error);
    throw error;
  }
}

// Make it globally available for Babel/React script
window.askEca = askEca;
window.pollTaskStatus = pollTaskStatus;
window.fetchChatHistory = fetchChatHistory;
