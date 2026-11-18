(() => {
  "use strict";
  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

  const panels = $$("[data-assistant]");
  let statusTimer = null;

  const SESSION_KEY = "siteguard_assistant_session";
  const getSessionId = () => {
    try {
      let id = window.localStorage.getItem(SESSION_KEY);
      if (!id) {
        id = window.crypto?.randomUUID
          ? window.crypto.randomUUID()
          : `s-${Date.now()}-${Math.random().toString(16).slice(2)}`;
        window.localStorage.setItem(SESSION_KEY, id);
      }
      return id;
    } catch (error) {
      return "";
    }
  };
  const resetSessionId = () => {
    try {
      window.localStorage.removeItem(SESSION_KEY);
    } catch (error) {
      /* storage may be unavailable; the server session is still cleared */
    }
    return getSessionId();
  };

  const BASIS_LABELS = {
    general: "General knowledge",
    documents: "Document-supported",
    analysis: "Analysis facts",
  };

  const addMessage = (history, role, text, sources = [], timings = null, mode = "", basis = "") => {
    const welcome = $("[data-chat-welcome]", history);
    welcome?.remove();
    const item = document.createElement("article");
    item.className = `chat-message chat-${role}`;
    const avatar = document.createElement("span");
    avatar.className = "chat-avatar";
    avatar.textContent = role === "assistant" ? "AI" : "You";
    const body = document.createElement("div");
    body.className = "chat-message-body";
    const bubble = document.createElement("div");
    bubble.className = "chat-bubble";
    bubble.textContent = text;
    body.append(bubble);
    if (mode) item.dataset.mode = mode;

    if (role === "assistant" && basis && BASIS_LABELS[basis]) {
      const badge = document.createElement("span");
      badge.className = "chat-mode";
      badge.textContent = BASIS_LABELS[basis];
      body.append(badge);
    }

    if (sources.length) {
      const sourceBox = document.createElement("div");
      sourceBox.className = "chat-sources";
      const title = document.createElement("strong");
      title.textContent = "Sources";
      sourceBox.append(title);
      sources.forEach((source) => {
        const line = document.createElement("span");
        line.className = "source-citation";
        line.textContent = `[${source.citation}] ${source.label || source.filename || "Safety document"}`;
        sourceBox.append(line);
      });
      body.append(sourceBox);
    }

    if (timings && role === "assistant" && (timings.total_ms || timings.llm_ms)) {
      const timing = document.createElement("small");
      timing.className = "chat-timing";
      const bits = [];
      if (Number.isFinite(Number(timings.retrieval_ms)) && Number(timings.retrieval_ms) > 0) {
        bits.push(`Retrieval ${timings.retrieval_ms} ms`);
      }
      if (Number.isFinite(Number(timings.llm_ms)) && Number(timings.llm_ms) > 0) bits.push(`LLM ${timings.llm_ms} ms`);
      if (Number.isFinite(Number(timings.total_ms))) bits.push(`Total ${timings.total_ms} ms`);
      timing.textContent = bits.join(" · ");
      if (bits.length) body.append(timing);
    }

    item.append(avatar, body);
    history.append(item);
    history.scrollTop = history.scrollHeight;
    return item;
  };

  const formatModel = (name) => {
    const raw = String(name || "").trim();
    if (!raw) return "";
    const [base, tag] = raw.split(":");
    const prettyBase = base.charAt(0).toUpperCase() + base.slice(1);
    return tag ? `${prettyBase} ${tag.toUpperCase()}` : prettyBase;
  };

  const renderStatus = (payload) => {
    const assistant = payload?.assistant || payload;
    const mode = assistant?.mode || "unknown";
    const detailPhrases = {
      ready: `Ollama Online · ${assistant.chat_model || assistant.ollama?.chat_model || "Qwen2.5 3B"}`,
      offline: "Ollama Offline · image analysis unaffected",
      models_missing: "Ollama chat model missing",
      no_documents: "Ollama Online · documents optional",
      dependency_missing: "Ollama Online · documents unavailable",
      index_error: "Ollama Online · documents unavailable",
      reindex_required: "Ollama Online · re-index documents",
    };
    const compactPhrases = {
      ready: "Ollama Online",
      offline: "Ollama Offline",
      models_missing: "Model Missing",
      no_documents: "Ollama Online",
      dependency_missing: "Ollama Online",
      index_error: "Ollama Online",
      reindex_required: "Ollama Online",
    };
    $$("[data-assistant-status]").forEach((node) => {
      node.dataset.mode = mode;
      const phrases = node.hasAttribute("data-compact-status") ? compactPhrases : detailPhrases;
      const text = $("[data-assistant-status-text]", node);
      if (text) text.textContent = phrases[mode] || `Assistant status: ${mode.replaceAll("_", " ")}`;

      const model = assistant?.chat_model || assistant?.ollama?.chat_model;
      const modelNode = $("[data-assistant-model]", node);
      if (modelNode && model) modelNode.textContent = formatModel(model);

      const rag = assistant?.rag;
      if (rag) {
        const ragNode = $("[data-assistant-rag]", node);
        if (ragNode) {
          ragNode.textContent = rag.dependency_error || rag.index_error ? "Optional" : "Enabled";
        }
        const docsNode = $("[data-assistant-docs]", node);
        if (docsNode) {
          const count = Number(rag.documents) || 0;
          docsNode.textContent = `${count} Document${count === 1 ? "" : "s"} (optional)`;
        }
      }
    });
  };

  const refreshStatus = async (force = false) => {
    try {
      const response = await fetch("/api/assistant/status", { headers: { Accept: "application/json" } });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Could not read assistant status.");
      renderStatus(data);
      return data;
    } catch (error) {
      $$("[data-assistant-status]").forEach((node) => {
        node.dataset.mode = "offline";
        const text = $("[data-assistant-status-text]", node);
        if (text) text.textContent = "Assistant status unavailable · image analysis remains online";
      });
      return null;
    }
  };

  const submitChat = async (form) => {
    const panel = form.closest("[data-assistant]");
    const input = $("[data-chat-input]", form);
    const history = $("[data-chat-history]", panel || document);
    const button = $("[data-chat-send]", form);
    const message = input?.value?.trim();
    if (!message || !history) return;
    addMessage(history, "user", message);
    input.value = "";
    input.style.height = "";
    if (button) button.disabled = true;
    const pending = addMessage(history, "assistant", "Thinking…", [], null, "loading");
    pending.classList.add("is-loading");
    try {
      const analysisId = panel?.dataset.analysisId || "";
      const response = await fetch("/api/assistant/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "application/json" },
        body: JSON.stringify({ message, analysis_id: analysisId || null, session_id: getSessionId() }),
      });
      const data = await response.json();
      pending.remove();
      if (!response.ok && !data.answer) throw new Error(data.error || "Assistant request failed.");
      addMessage(history, "assistant", data.answer || data.error || "No answer returned.", data.sources || [], data.timings, data.mode || "", data.basis || "");
      if (data.hint) addMessage(history, "assistant", data.hint, [], null, "hint");
      if (data.status) renderStatus({ assistant: { mode: data.status, chat_model: data.model } });
    } catch (error) {
      pending.remove();
      addMessage(history, "assistant", `The request could not be completed: ${error.message}`, [], null, "error");
    } finally {
      if (button) button.disabled = false;
      input?.focus();
    }
  };

  $$("[data-chat-form]").forEach((form) => {
    form.addEventListener("submit", (event) => {
      event.preventDefault();
      submitChat(form);
    });
    const input = $("[data-chat-input]", form);
    input?.addEventListener("input", () => {
      input.style.height = "auto";
      input.style.height = `${Math.min(input.scrollHeight, 120)}px`;
    });
    input?.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        form.requestSubmit();
      }
    });
  });

  $$("[data-suggested-prompts]").forEach((group) =>
    group.addEventListener("click", (event) => {
      const button = event.target.closest("[data-prompt]");
      if (!button) return;
      const panel = group.closest("[data-assistant]") || document;
      const form = $("[data-chat-form]", panel);
      const input = $("[data-chat-input]", form);
      if (!form || !input) return;
      input.value = button.dataset.prompt || "";
      form.requestSubmit();
    }),
  );

  $$("[data-chat-clear]").forEach((button) => button.addEventListener("click", async () => {
    const panel = button.closest("[data-assistant]");
    const history = $("[data-chat-history]", panel);
    const sessionId = getSessionId();
    try {
      await fetch("/api/assistant/clear", {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "application/json" },
        body: JSON.stringify({ session_id: sessionId }),
      });
    } catch (error) {
      /* clearing the server session is best-effort */
    }
    resetSessionId();
    if (!history) return;
    history.innerHTML = '<div class="chat-welcome" data-chat-welcome><span class="welcome-icon">✦</span><strong>How can I help?</strong><p>I answer from general safety knowledge and can explain the current image analysis. Documents are optional.</p></div>';
  }));

  $$("[data-status-refresh]").forEach((button) => button.addEventListener("click", async () => {
    button.disabled = true;
    await refreshStatus(true);
    button.disabled = false;
  }));

  // Administrator document upload and indexing (optional enhancement).
  const documentForm = $("[data-document-form]");
  documentForm?.addEventListener("submit", async (event) => {
    event.preventDefault();
    const input = $("[data-document-input]", documentForm);
    const feedback = $("[data-document-feedback]", documentForm);
    const submit = $("[data-document-submit]", documentForm);
    const file = input?.files?.[0];
    if (!file) {
      if (feedback) feedback.textContent = "Choose a PDF, TXT or Markdown file first.";
      return;
    }
    if (submit) {
      submit.disabled = true;
      submit.textContent = "Uploading and indexing…";
    }
    if (feedback) feedback.textContent = "Saving document and checking local Ollama embeddings…";
    const body = new FormData();
    body.append("file", file);
    try {
      const response = await fetch("/api/assistant/documents", { method: "POST", body });
      const data = await response.json();
      if (feedback) {
        feedback.textContent = data.message || data.error || (data.ok ? "Document indexed." : "Document saved; indexing needs attention.");
        feedback.dataset.state = data.ok ? "success" : "warning";
      }
      if (data.ok && window.siteGuardToast) window.siteGuardToast(data.message || "Document indexed.", "success");
      else if (data.error && window.siteGuardToast) window.siteGuardToast(data.error, "warning");
      if (data.saved || data.ok) window.setTimeout(() => window.location.reload(), 900);
    } catch (error) {
      if (feedback) feedback.textContent = `Upload failed: ${error.message}`;
    } finally {
      if (submit) {
        submit.disabled = false;
        submit.innerHTML = '<svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M12 16.5V4.8M7.8 8.9 12 4.7l4.2 4.2"></path><path d="M4.5 15.5v2.7a2.3 2.3 0 0 0 2.3 2.3h10.4a2.3 2.3 0 0 0 2.3-2.3v-2.7"></path></svg> Upload &amp; index';
      }
    }
  });

  $$("[data-delete-document]").forEach((button) => button.addEventListener("click", async () => {
    const filename = button.dataset.deleteDocument;
    if (!filename || !window.confirm(`Remove ${filename} from the knowledge base?`)) return;
    button.disabled = true;
    try {
      const response = await fetch(`/api/assistant/documents/${encodeURIComponent(filename)}`, { method: "DELETE" });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Delete failed.");
      button.closest("[data-document]")?.remove();
      window.siteGuardToast?.(data.message || "Document removed.", "success");
      refreshStatus(true);
      window.setTimeout(() => window.location.reload(), 700);
    } catch (error) {
      window.siteGuardToast?.(error.message, "error");
      button.disabled = false;
    }
  }));

  $("[data-reindex]")?.addEventListener("click", async (event) => {
    const button = event.currentTarget;
    button.disabled = true;
    button.dataset.original = button.textContent;
    button.textContent = "Re-indexing documents…";
    try {
      const response = await fetch("/api/assistant/reindex", { method: "POST", headers: { Accept: "application/json" } });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Re-index failed.");
      const failed = data.failed?.length || 0;
      window.siteGuardToast?.(`Re-index complete: ${data.indexed?.length || 0} indexed, ${data.skipped?.length || 0} unchanged, ${failed} failed.`, failed ? "warning" : "success");
      window.setTimeout(() => window.location.reload(), 700);
    } catch (error) {
      window.siteGuardToast?.(error.message, "error");
    } finally {
      button.disabled = false;
      button.textContent = "Re-index all documents";
    }
  });

  if (panels.length || $("[data-status-refresh]")) refreshStatus();
  statusTimer = window.setInterval(() => {
    if (document.visibilityState === "visible" && (panels.length || $("[data-status-refresh]"))) refreshStatus();
  }, 60000);
  window.addEventListener("pagehide", () => {
    if (statusTimer) window.clearInterval(statusTimer);
  });
})();
