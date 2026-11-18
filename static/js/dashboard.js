(() => {
  "use strict";

  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

  // Current-section selector and responsive sidebar.
  const pageNav = $("[data-page-nav]");
  pageNav?.addEventListener("change", () => {
    if (pageNav.value) window.location.assign(pageNav.value);
  });

  const sidebar = $("[data-sidebar]");
  const overlay = $("[data-sidebar-overlay]");
  const toggle = $("[data-menu-toggle]");
  const closeMenu = () => {
    sidebar?.classList.remove("is-open");
    overlay?.setAttribute("hidden", "");
    toggle?.setAttribute("aria-expanded", "false");
  };
  toggle?.addEventListener("click", () => {
    const open = !sidebar?.classList.contains("is-open");
    sidebar?.classList.toggle("is-open", open);
    if (overlay) overlay.hidden = !open;
    toggle.setAttribute("aria-expanded", String(open));
  });
  overlay?.addEventListener("click", closeMenu);
  $$(".side-link", sidebar || document).forEach((link) => link.addEventListener("click", closeMenu));

  // The top-bar status is based on a real local health request.
  const systemStatus = $("[data-system-status]");
  const systemLabel = $("[data-system-label]");
  if (systemStatus && systemLabel) {
    fetch("/api/status", { headers: { Accept: "application/json" } })
      .then((response) => {
        if (!response.ok) throw new Error("health request failed");
        return response.json();
      })
      .then((payload) => {
        const appOk = payload?.application?.status === "ok";
        const ppeOk = payload?.models?.ppe?.present === true;
        const personOk = payload?.models?.person?.present === true;
        const state = !appOk ? "offline" : ppeOk && personOk ? "online" : "degraded";
        systemStatus.dataset.state = state;
        systemLabel.textContent =
          state === "online" ? "System Online" : state === "degraded" ? "System Degraded" : "System Offline";
        systemStatus.title = `App: ${appOk ? "healthy" : "unavailable"} · PPE model: ${ppeOk ? "available" : "missing"} · Person model: ${personOk ? "available" : "missing"}`;
      })
      .catch(() => {
        systemStatus.dataset.state = "offline";
        systemLabel.textContent = "System Unavailable";
      });
  }

  // Upload form preview, drop zone and actual file picker.
  const form = $("[data-analyze-form]");
  if (form) {
    const input = $("#image", form);
    const drop = $("[data-dropzone]", form);
    const preview = $("[data-preview]", form);
    const previewWrap = $("[data-preview-wrap]", form);
    const fileName = $("[data-file-name]", form);
    const conf = $("#conf", form);
    const confValue = $("#conf-value", form);
    const submit = $("[data-submit]", form);
    const label = $(".btn-label", form);
    const originalLabel = label?.innerHTML || "";
    const status = $("[data-status]", form);
    let objectUrl = null;

    const setFile = (file) => {
      if (!file || !preview || !previewWrap) return;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
      objectUrl = URL.createObjectURL(file);
      preview.src = objectUrl;
      previewWrap.hidden = false;
      if (fileName) {
        fileName.hidden = false;
        fileName.textContent = `${file.name} · ${(file.size / 1048576).toFixed(1)} MB`;
      }
      drop?.classList.add("has-file");
    };
    input?.addEventListener("change", () => setFile(input.files?.[0]));
    if (drop && input) {
      ["dragenter", "dragover"].forEach((eventName) =>
        drop.addEventListener(eventName, (event) => {
          event.preventDefault();
          drop.classList.add("is-over");
        }),
      );
      ["dragleave", "drop"].forEach((eventName) =>
        drop.addEventListener(eventName, (event) => {
          event.preventDefault();
          drop.classList.remove("is-over");
        }),
      );
      drop.addEventListener("drop", (event) => {
        const files = event.dataTransfer?.files;
        if (!files?.length) return;
        try {
          input.files = files;
          setFile(files[0]);
        } catch (_) {
          // The label still opens the native picker if assignment is restricted.
        }
      });
    }
    conf?.addEventListener("input", () => {
      if (confValue) confValue.value = Number(conf.value).toFixed(2);
    });
    form.addEventListener("submit", () => {
      if (!input?.files?.length) return;
      if (submit) {
        submit.disabled = true;
        submit.classList.add("is-busy");
      }
      if (label) label.textContent = "Analyzing photo…";
      if (status) status.hidden = false;
    });
    window.addEventListener("pageshow", () => {
      if (submit) {
        submit.disabled = false;
        submit.classList.remove("is-busy");
      }
      if (label) label.innerHTML = originalLabel;
      if (status) status.hidden = true;
    });
  }

  $$("[data-open-file]").forEach((button) =>
    button.addEventListener("click", () => {
      $("#image")?.click();
      $("#upload")?.scrollIntoView({ behavior: "smooth", block: "center" });
    }),
  );
  $$("[data-focus-upload]").forEach((button) =>
    button.addEventListener("click", (event) => {
      const image = $("#image");
      if (image) {
        event.preventDefault();
        image.click();
      }
    }),
  );

  // Annotated image lightbox.
  const dialog = $("[data-lightbox]");
  const lightboxImage = $("[data-lightbox-image]", dialog || document);
  const closeLightbox = () => {
    if (dialog?.open) dialog.close();
  };
  $$("[data-preview-image]").forEach((button) =>
    button.addEventListener("click", () => {
      const source = button.dataset.previewImage;
      if (!source || !dialog || !lightboxImage) return;
      lightboxImage.src = source;
      if (typeof dialog.showModal === "function") dialog.showModal();
      else window.open(source, "_blank", "noopener");
    }),
  );
  $("[data-lightbox-close]")?.addEventListener("click", closeLightbox);
  dialog?.addEventListener("click", (event) => {
    if (event.target === dialog) closeLightbox();
  });
  dialog?.addEventListener("close", () => {
    if (lightboxImage) lightboxImage.removeAttribute("src");
  });

  // Settings slider outputs.
  $$("[data-output-for]").forEach((output) => {
    const input = document.getElementById(output.dataset.outputFor);
    const update = () => {
      if (input) output.value = Number(input.value).toFixed(2);
    };
    input?.addEventListener("input", update);
    update();
  });

  // Small non-blocking toast notifications shared by the page scripts.
  window.siteGuardToast = (message, type = "info") => {
    const region = $("[data-toast-region]");
    if (!region) return;
    const toast = document.createElement("div");
    toast.className = `toast toast-${type}`;
    toast.textContent = String(message);
    region.append(toast);
    window.setTimeout(() => toast.remove(), 5000);
  };
})();
