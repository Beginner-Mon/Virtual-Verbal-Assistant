(function (global) {
  const BASE_URL = "http://localhost:8080";

  function normalizeHistory(history) {
    if (!Array.isArray(history)) return [];
    return history
      .filter((item) => item && (item.role === "user" || item.role === "assistant") && item.content)
      .map((item) => ({ role: item.role, content: String(item.content) }));
  }

  async function askAnswer({ query, userId = "eca-demo", history = [], motionFormat = "glb", timeoutMs = 120000 } = {}) {
    const trimmedQuery = String(query || "").trim();
    if (!trimmedQuery) {
      throw new Error("Query is required");
    }

    const payload = {
      query: trimmedQuery,
      user_id: userId,
      motion_format: motionFormat,
    };

    const normalizedHistory = normalizeHistory(history);
    if (normalizedHistory.length > 0) {
      payload.conversation_history = normalizedHistory;
    }

    const controller = new AbortController();
    const timeoutHandle = setTimeout(() => controller.abort(), timeoutMs);

    try {
      const response = await fetch(`${BASE_URL}/answer`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
        signal: controller.signal,
      });

      const raw = await response.text();
      let data = null;
      try {
        data = raw ? JSON.parse(raw) : null;
      } catch {
        data = null;
      }

      if (!response.ok) {
        const message = (data && (data.detail || data.message)) || raw || `Request failed (${response.status})`;
        throw new Error(message);
      }

      return data || {};
    } catch (error) {
      if (error.name === "AbortError") {
        throw new Error("Request timed out. Please try again.");
      }
      throw error;
    } finally {
      clearTimeout(timeoutHandle);
    }
  }

  global.EcaApi = {
    askAnswer,
  };
})(window);
