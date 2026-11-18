(() => {
  "use strict";

  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

  const toast = (message, type = "info") => {
    if (typeof window.siteGuardToast === "function") window.siteGuardToast(message, type);
  };

  const errorBanner = $("[data-video-error]");
  const errorText = $("[data-video-error-text]");
  const showError = (message) => {
    if (!errorBanner || !errorText) {
      toast(message, "error");
      return;
    }
    errorText.textContent = message;
    errorBanner.hidden = false;
    errorBanner.scrollIntoView({ behavior: "smooth", block: "center" });
  };
  const clearError = () => {
    if (errorBanner) errorBanner.hidden = true;
  };

  const formatClock = (seconds) => {
    const value = Number(seconds);
    if (!Number.isFinite(value) || value < 0) return "—";
    const minutes = Math.floor(value / 60);
    const rest = value % 60;
    return `${String(minutes).padStart(2, "0")}:${rest.toFixed(1).padStart(4, "0")}`;
  };

  const filesize = (bytes) => {
    const value = Number(bytes) || 0;
    if (value >= 1073741824) return `${(value / 1073741824).toFixed(2)} GB`;
    if (value >= 1048576) return `${(value / 1048576).toFixed(1)} MB`;
    return `${Math.max(1, Math.round(value / 1024))} KB`;
  };

  // ---------------------------------------------------------------------
  // Video workspace: sample selection, upload, settings, analyze
  // ---------------------------------------------------------------------
  const sampleList = $("[data-sample-list]");
  const uploadForm = $("[data-video-upload-form]");
  if (sampleList || uploadForm) {
    const preview = $("[data-video-preview]");
    const previewEmpty = $("[data-video-preview-empty]");
    const previewMeta = $("[data-video-preview-meta]");
    const metaName = $("[data-meta-name]");
    const metaResolution = $("[data-meta-resolution]");
    const metaDuration = $("[data-meta-duration]");
    const metaFps = $("[data-meta-fps]");
    const selectedChip = $("[data-selected-chip]");
    const selectedTitle = $("[data-selected-title]");
    const selectedHint = $("[data-selected-hint]");
    const analyzeButton = $("[data-analyze-video]");
    const jobProgress = $("[data-job-progress]");
    const jobProgressFill = $("[data-job-progress-fill]");
    const jobProgressText = $("[data-job-progress-text]");

    let selected = null;
    let pollTimer = null;
    const analyzeLabel = analyzeButton ? $(".btn-label", analyzeButton) : null;
    const analyzeLabelHtml = analyzeLabel?.innerHTML || "";

    const setSelected = (source) => {
      selected = source;
      clearError();

      if (selectedChip) {
        selectedChip.hidden = false;
        selectedChip.textContent = source.type === "sample" ? "Sample video" : "Uploaded video";
      }
      if (selectedTitle) selectedTitle.textContent = source.name;
      if (selectedHint) {
        selectedHint.textContent =
          source.type === "sample"
            ? "Ready to analyze. The sample streams from the allowlisted library."
            : "Upload validated. Analysis uses the stored copy identified by its upload ID.";
      }

      if (preview) {
        preview.hidden = false;
        preview.src = source.stream;
        preview.load();
      }
      if (previewEmpty) previewEmpty.hidden = true;
      if (previewMeta) previewMeta.hidden = false;
      if (metaName) metaName.textContent = source.name;
      if (metaResolution) metaResolution.textContent = `${source.width} × ${source.height} px`;
      if (metaDuration) metaDuration.textContent = formatClock(source.duration);
      if (metaFps) metaFps.textContent = `${Number(source.fps || 0).toFixed(2)} fps`;

      if (analyzeButton) {
        analyzeButton.disabled = false;
        analyzeButton.classList.remove("is-busy");
        if (analyzeLabel) analyzeLabel.innerHTML = analyzeLabelHtml;
      }
      if (jobProgress) jobProgress.hidden = true;
    };

    $$("[data-sample]", sampleList || document).forEach((button) => {
      button.addEventListener("click", () => {
        $$("[data-sample]", sampleList || document).forEach((item) => {
          item.classList.remove("is-selected");
          item.setAttribute("aria-pressed", "false");
        });
        button.classList.add("is-selected");
        button.setAttribute("aria-pressed", "true");
        setSelected({
          type: "sample",
          id: button.dataset.sampleId,
          name: button.dataset.sampleName,
          stream: button.dataset.sampleStream,
          width: Number(button.dataset.sampleWidth) || 0,
          height: Number(button.dataset.sampleHeight) || 0,
          fps: Number(button.dataset.sampleFps) || 0,
          duration: Number(button.dataset.sampleDuration) || 0,
        });
        $("[data-video-preview-frame]")?.scrollIntoView({ behavior: "smooth", block: "center" });
      });
    });

    // ---------------------------------------------------------------
    // Upload with real progress from XMLHttpRequest
    // ---------------------------------------------------------------
    if (uploadForm) {
      const input = $("#video-file", uploadForm);
      const drop = $("[data-video-dropzone]", uploadForm);
      const fileWrap = $("[data-video-file-preview]", uploadForm);
      const fileName = $("[data-video-file-name]", uploadForm);
      const submit = $("[data-upload-submit]", uploadForm);
      const progress = $("[data-upload-progress]", uploadForm);
      const progressFill = $("[data-upload-progress-fill]", uploadForm);
      const progressText = $("[data-upload-progress-text]", uploadForm);
      const allowed = /\.(mp4|avi|mov|mkv|webm|m4v)$/i;

      const describeFile = (file) => {
        if (!file) return;
        if (fileName) fileName.textContent = `${file.name} · ${filesize(file.size)}`;
        if (fileWrap) fileWrap.hidden = false;
      };

      const rejectFile = (message) => {
        showError(message);
        if (input) input.value = "";
        if (fileWrap) fileWrap.hidden = true;
        drop?.classList.remove("has-file");
      };

      input?.addEventListener("change", () => {
        const file = input.files?.[0];
        if (!file) return;
        if (!allowed.test(file.name)) {
          rejectFile("Unsupported video type. Choose an MP4, AVI, MOV, MKV, or WEBM file.");
          return;
        }
        clearError();
        describeFile(file);
        drop?.classList.add("has-file");
      });

      if (drop && input) {
        ["dragenter", "dragover"].forEach((name) =>
          drop.addEventListener(name, (event) => {
            event.preventDefault();
            drop.classList.add("is-over");
          }),
        );
        ["dragleave", "drop"].forEach((name) =>
          drop.addEventListener(name, (event) => {
            event.preventDefault();
            drop.classList.remove("is-over");
          }),
        );
        drop.addEventListener("drop", (event) => {
          const file = event.dataTransfer?.files?.[0];
          if (!file) return;
          if (!allowed.test(file.name)) {
            rejectFile("Unsupported video type. Choose an MP4, AVI, MOV, MKV, or WEBM file.");
            return;
          }
          try {
            const transfer = new DataTransfer();
            transfer.items.add(file);
            input.files = transfer.files;
          } catch (_) {
            // The label still opens the native picker when assignment is restricted.
          }
          clearError();
          describeFile(file);
          drop.classList.add("has-file");
        });
      }

      uploadForm.addEventListener("submit", (event) => {
        event.preventDefault();
        const file = input?.files?.[0];
        if (!file) {
          showError("Choose a video file before uploading.");
          return;
        }
        if (!allowed.test(file.name)) {
          showError("Unsupported video type. Choose an MP4, AVI, MOV, MKV, or WEBM file.");
          return;
        }

        clearError();
        if (submit) {
          submit.disabled = true;
          submit.classList.add("is-busy");
          const label = $(".btn-label", submit);
          if (label) label.textContent = "Uploading…";
        }
        if (progress) progress.hidden = false;
        if (progressFill) progressFill.style.width = "0%";
        if (progressText) progressText.textContent = "Uploading video…";

        const body = new FormData();
        body.append("video", file);

        const xhr = new XMLHttpRequest();
        xhr.open("POST", "/api/videos/upload");
        xhr.setRequestHeader("Accept", "application/json");
        xhr.upload.addEventListener("progress", (e) => {
          if (!e.lengthComputable) return;
          const percent = Math.round((e.loaded / e.total) * 100);
          if (progressFill) progressFill.style.width = `${percent}%`;
          if (progressText) progressText.textContent = `Uploading video… ${percent}%`;
        });
        xhr.addEventListener("load", () => {
          let payload = null;
          try {
            payload = JSON.parse(xhr.responseText || "{}");
          } catch (_) {
            payload = null;
          }
          if (submit) {
            submit.disabled = false;
            submit.classList.remove("is-busy");
            const label = $(".btn-label", submit);
            if (label) label.textContent = "Upload and validate";
          }
          if (xhr.status >= 200 && xhr.status < 300 && payload?.ok) {
            if (progressFill) progressFill.style.width = "100%";
            if (progressText) progressText.textContent = "Upload validated.";
            setSelected({
              type: "upload",
              id: payload.upload_id,
              name: payload.filename,
              stream: payload.preview_url,
              width: Number(payload.width) || 0,
              height: Number(payload.height) || 0,
              fps: Number(payload.fps) || 0,
              duration: Number(payload.duration_s) || 0,
            });
            toast("Video uploaded and validated.", "success");
          } else {
            if (progress) progress.hidden = true;
            showError(payload?.error || `Upload failed (HTTP ${xhr.status}).`);
          }
        });
        xhr.addEventListener("error", () => {
          if (submit) {
            submit.disabled = false;
            submit.classList.remove("is-busy");
          }
          if (progress) progress.hidden = true;
          showError("The upload could not reach the server. Check that the app is still running.");
        });
        xhr.send(body);
      });
    }

    // ---------------------------------------------------------------
    // Live setting pills
    // ---------------------------------------------------------------
    const pillFor = {
      sample_fps: [$("#sample-fps"), $("#sample-fps-value"), (v) => `${Number(v).toFixed(1)} fps`],
      max_sampled_frames: [$("#max-frames"), $("#max-frames-value"), (v) => String(Math.round(Number(v)))],
      conf: [$("#video-conf"), $("#video-conf-value"), (v) => Number(v).toFixed(2)],
      min_ratio: [$("#min-ratio"), $("#min-ratio-value"), (v) => `${Math.round(Number(v) * 100)}%`],
    };
    Object.values(pillFor).forEach(([field, output, formatter]) => {
      if (!field || !output) return;
      const update = () => {
        output.value = formatter(field.value);
        output.textContent = formatter(field.value);
      };
      field.addEventListener("input", update);
      update();
    });

    const readSettings = () => {
      const settings = {};
      $$("[data-setting]").forEach((field) => {
        settings[field.dataset.setting] = field.value;
      });
      return settings;
    };

    // ---------------------------------------------------------------
    // Analyze + polling
    // ---------------------------------------------------------------
    const stopPolling = () => {
      if (pollTimer) window.clearTimeout(pollTimer);
      pollTimer = null;
    };

    const setProgress = (percent, message) => {
      if (jobProgress) jobProgress.hidden = false;
      if (jobProgressFill) jobProgressFill.style.width = `${Math.max(0, Math.min(100, percent))}%`;
      if (jobProgressText) jobProgressText.textContent = message;
    };

    const poll = async (statusUrl, jobUrl, attempt = 0) => {
      try {
        const response = await fetch(statusUrl, { headers: { Accept: "application/json" } });
        if (!response.ok) throw new Error(`Status request failed (${response.status}).`);
        const payload = await response.json();
        const job = payload?.job;
        if (!job) throw new Error("Status response was incomplete.");

        if (job.status === "completed") {
          setProgress(100, "Analysis complete. Opening the result…");
          toast("Video analysis complete.", "success");
          window.location.assign(jobUrl);
          return;
        }
        if (job.status === "failed") {
          stopPolling();
          if (analyzeButton) {
            analyzeButton.disabled = false;
            analyzeButton.classList.remove("is-busy");
            if (analyzeLabel) analyzeLabel.innerHTML = analyzeLabelHtml;
          }
          setProgress(job.progress || 0, "Analysis failed.");
          showError(job.error || "Video analysis failed. Try a shorter video or a lower sampling rate.");
          return;
        }

        const total = job.sampled_frames ? ` of ${job.sampled_frames}` : "";
        setProgress(
          Number(job.progress) || 0,
          `${job.status === "queued" ? "Queued" : "Processing"} · ${job.processed_frames || 0}${total} sampled frames · ${Math.round(Number(job.progress) || 0)}%`,
        );
        if (attempt < 1200) {
          pollTimer = window.setTimeout(() => poll(statusUrl, jobUrl, attempt + 1), 1500);
        }
      } catch (error) {
        if (attempt < 20) {
          pollTimer = window.setTimeout(() => poll(statusUrl, jobUrl, attempt + 1), 2000);
        } else {
          stopPolling();
          showError("Lost contact with the video job. Refresh the video library to check its status.");
        }
      }
    };

    analyzeButton?.addEventListener("click", async () => {
      if (!selected) {
        showError("Select a sample video or upload a video first.");
        return;
      }
      clearError();
      stopPolling();
      analyzeButton.disabled = true;
      analyzeButton.classList.add("is-busy");
      if (analyzeLabel) analyzeLabel.textContent = "Queueing analysis…";
      setProgress(0, "Queueing analysis…");

      const body = {
        source_type: selected.type,
        ...(selected.type === "sample" ? { video_id: selected.id } : { upload_id: selected.id }),
        ...readSettings(),
      };

      try {
        const response = await fetch("/api/videos/analyze", {
          method: "POST",
          headers: { "Content-Type": "application/json", Accept: "application/json" },
          body: JSON.stringify(body),
        });
        const payload = await response.json().catch(() => null);
        if (!response.ok || !payload?.ok) {
          throw new Error(payload?.error || `Could not queue the analysis (HTTP ${response.status}).`);
        }
        setProgress(0, "Queued. Waiting for the background worker…");
        poll(payload.status_url, payload.job_url, 0);
      } catch (error) {
        analyzeButton.disabled = false;
        analyzeButton.classList.remove("is-busy");
        if (analyzeLabel) analyzeLabel.innerHTML = analyzeLabelHtml;
        if (jobProgress) jobProgress.hidden = true;
        showError(error.message || "Could not queue the analysis.");
      }
    });

  }

  // -----------------------------------------------------------------
  // Result page: live job progress (runs outside the workspace branch,
  // because the result page has neither a sample list nor an upload form)
  // -----------------------------------------------------------------
  const live = $("[data-job-live]");
  if (live) {
    const statusUrl = live.dataset.jobStatusUrl;
    const liveFill = $("[data-job-live-fill]");
    const liveProgress = $("[data-job-live-progress]");
    const liveMessage = $("[data-job-live-message]");
    let attempts = 0;

    const refresh = async () => {
      attempts += 1;
      try {
        const response = await fetch(statusUrl, { headers: { Accept: "application/json" } });
        if (!response.ok) throw new Error("status unavailable");
        const payload = await response.json();
        const job = payload?.job || {};
        const percent = Math.max(0, Math.min(100, Number(job.progress) || 0));
        if (liveFill) liveFill.style.width = `${percent}%`;
        if (liveProgress) {
          liveProgress.textContent = job.sampled_frames
            ? `${percent.toFixed(0)}% · ${job.processed_frames || 0} of ${job.sampled_frames} frames`
            : `${percent.toFixed(0)}% complete`;
        }
        if (liveMessage) liveMessage.textContent = job.message || "Processing…";
        if (job.status === "completed" || job.status === "failed") {
          window.location.reload();
          return;
        }
      } catch (_) {
        // Transient network failures are retried; the page keeps the last known progress.
      }
      if (attempts < 1200) window.setTimeout(refresh, 2000);
    };
    window.setTimeout(refresh, 1200);
  }

  const annotatedPlayer = $("[data-annotated-player]");
  $$("[data-seek-time]").forEach((button) => {
    button.addEventListener("click", () => {
      if (!annotatedPlayer) return;
      const target = Number(button.dataset.seekTime);
      if (!Number.isFinite(target)) return;
      try {
        annotatedPlayer.currentTime = Math.max(0, target);
        annotatedPlayer.scrollIntoView({ behavior: "smooth", block: "center" });
        const playResult = annotatedPlayer.play();
        if (playResult && typeof playResult.catch === "function") playResult.catch(() => {});
      } catch (_) {
        // Seeking is a convenience; the timestamp is still visible in the table.
      }
    });
  });

  // Top-bar upload shortcut on the video workspace.
  $$("[data-open-video-file]").forEach((button) =>
    button.addEventListener("click", () => {
      $("#upload")?.scrollIntoView({ behavior: "smooth", block: "center" });
      $("#video-file")?.click();
    }),
  );
})();
