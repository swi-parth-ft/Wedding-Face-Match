const HARDCODED_FUNCTIONS_BASE_URL =
  "https://us-central1-facefinder-260219-2674.cloudfunctions.net";
const HARDCODED_DRIVE_FOLDER_LINK =
  "https://drive.google.com/drive/folders/1EF9ddrU0dWt3H5c43Rt_yS54TZRK_TgB?usp=sharing";
const SEARCH_TOP_K = 0; // 0 means return all matches
const SEARCH_MAX_DISTANCE = 0.35;
const SEARCH_REQUEST_RETRIES = 5;
const SEARCH_REQUEST_TIMEOUT_MS = 120000;

const state = {
  matches: [],
  capturedImageBase64: "",
  capturedDraftDataUrl: "",
  previewObjectUrl: "",
  cameraStream: null,
  overlayRunning: false,
  overlayRafId: 0,
  overlayRockets: [],
  overlayParticles: [],
  overlayLastTs: 0,
  overlaySpawnAccumulatorMs: 0,
};

const els = {
  datasetLink: document.getElementById("datasetLink"),
  queryImageInput: document.getElementById("queryImageInput"),
  openCameraBtn: document.getElementById("openCameraBtn"),
  clearPhotoBtn: document.getElementById("clearPhotoBtn"),
  querySourceMeta: document.getElementById("querySourceMeta"),
  queryPreview: document.getElementById("queryPreview"),
  searchBtn: document.getElementById("searchBtn"),
  searchMeta: document.getElementById("searchMeta"),
  results: document.getElementById("results"),
  selectAllBtn: document.getElementById("selectAllBtn"),
  downloadSelectedBtn: document.getElementById("downloadSelectedBtn"),
  searchOverlay: document.getElementById("searchOverlay"),
  searchOverlayCanvas: document.getElementById("searchOverlayCanvas"),
  searchOverlayMessage: document.getElementById("searchOverlayMessage"),
  searchOverlaySubtext: document.getElementById("searchOverlaySubtext"),
  cameraModal: document.getElementById("cameraModal"),
  closeCameraBtn: document.getElementById("closeCameraBtn"),
  capturePhotoBtn: document.getElementById("capturePhotoBtn"),
  resetPhotoBtn: document.getElementById("resetPhotoBtn"),
  usePhotoBtn: document.getElementById("usePhotoBtn"),
  cameraVideo: document.getElementById("cameraVideo"),
  cameraPreview: document.getElementById("cameraPreview"),
  cameraOverlay: document.getElementById("cameraOverlay"),
  cameraCanvas: document.getElementById("cameraCanvas"),
  cameraStatus: document.getElementById("cameraStatus"),
};

init();

function init() {
  if (els.datasetLink) {
    els.datasetLink.href = HARDCODED_DRIVE_FOLDER_LINK;
  }

  els.queryImageInput.addEventListener("change", onFileSelectionChanged);
  els.openCameraBtn.addEventListener("click", onOpenCameraModal);
  els.clearPhotoBtn.addEventListener("click", clearQuerySource);
  els.searchBtn.addEventListener("click", onSearch);
  els.selectAllBtn.addEventListener("click", onToggleSelectAll);
  els.downloadSelectedBtn.addEventListener("click", onDownloadSelected);
  els.results.addEventListener("change", onResultsSelectionChanged);

  els.closeCameraBtn.addEventListener("click", closeCameraModal);
  els.capturePhotoBtn.addEventListener("click", capturePhoto);
  els.resetPhotoBtn.addEventListener("click", resetCameraCapture);
  els.usePhotoBtn.addEventListener("click", useCapturedPhoto);

  els.cameraModal.addEventListener("click", (ev) => {
    if (ev.target === els.cameraModal) {
      closeCameraModal();
    }
  });

  document.addEventListener("keydown", (ev) => {
    if (ev.key === "Escape" && !els.cameraModal.classList.contains("hidden")) {
      closeCameraModal();
    }
  });

  window.addEventListener("beforeunload", () => {
    stopCameraStream();
    hideSearchOverlay();
  });
}

function fnUrl(name) {
  return `${HARDCODED_FUNCTIONS_BASE_URL}/${name}`;
}

function onFileSelectionChanged() {
  const file = els.queryImageInput.files?.[0];
  state.capturedImageBase64 = "";

  if (!file) {
    clearQueryPreview();
    els.querySourceMeta.textContent = "No query image selected. Capture or upload one.";
    return;
  }

  showPreviewFromFile(file);
  els.querySourceMeta.textContent = `Using uploaded file: ${file.name}`;
}

function showPreviewFromFile(file) {
  clearQueryPreview();
  state.previewObjectUrl = URL.createObjectURL(file);
  els.queryPreview.src = state.previewObjectUrl;
  els.queryPreview.classList.remove("mirrored");
  els.queryPreview.classList.remove("hidden");
}

function showPreviewFromDataUrl(dataUrl, options = {}) {
  const mirrored = Boolean(options.mirrored);
  clearQueryPreview();
  els.queryPreview.src = dataUrl;
  els.queryPreview.classList.toggle("mirrored", mirrored);
  els.queryPreview.classList.remove("hidden");
}

function clearQueryPreview() {
  if (state.previewObjectUrl) {
    URL.revokeObjectURL(state.previewObjectUrl);
    state.previewObjectUrl = "";
  }
  els.queryPreview.src = "";
  els.queryPreview.classList.remove("mirrored");
  els.queryPreview.classList.add("hidden");
}

function clearQuerySource() {
  els.queryImageInput.value = "";
  state.capturedImageBase64 = "";
  state.capturedDraftDataUrl = "";
  clearQueryPreview();
  els.querySourceMeta.textContent = "No query image selected. Capture or upload one.";
  resetCameraCaptureButtons();
}

function onOpenCameraModal() {
  state.capturedDraftDataUrl = "";
  resetCameraCaptureUi();
  els.cameraModal.classList.remove("hidden");
  els.cameraModal.setAttribute("aria-hidden", "false");
  els.cameraStatus.textContent = "Starting camera...";
  startCamera();
}

function closeCameraModal() {
  els.cameraModal.classList.add("hidden");
  els.cameraModal.setAttribute("aria-hidden", "true");
  stopCameraStream();
  state.capturedDraftDataUrl = "";
  resetCameraCaptureUi();
  els.cameraStatus.textContent = "";
}

async function startCamera() {
  if (state.cameraStream) return;

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: "user",
        width: { ideal: 1280 },
        height: { ideal: 720 },
      },
      audio: false,
    });
    state.cameraStream = stream;
    els.cameraVideo.srcObject = stream;
    els.cameraStatus.textContent = "Camera ready. Align face and capture.";
  } catch (err) {
    els.cameraStatus.textContent = `Camera error: ${errorText(err)}`;
  }
}

function stopCameraStream() {
  if (!state.cameraStream) return;
  for (const track of state.cameraStream.getTracks()) {
    track.stop();
  }
  state.cameraStream = null;
  els.cameraVideo.srcObject = null;
}

function capturePhoto() {
  const video = els.cameraVideo;
  if (!video.videoWidth || !video.videoHeight) {
    els.cameraStatus.textContent = "Camera is not ready yet.";
    return;
  }

  const canvas = els.cameraCanvas;
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    els.cameraStatus.textContent = "Failed to capture photo.";
    return;
  }

  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  state.capturedDraftDataUrl = canvas.toDataURL("image/jpeg", 0.92);
  stopCameraStream();
  els.cameraPreview.src = state.capturedDraftDataUrl;
  setCapturedUiState(true);
  els.cameraStatus.textContent = "Captured. Use this photo or reset.";
}

function resetCameraCapture() {
  state.capturedDraftDataUrl = "";
  els.cameraPreview.src = "";
  setCapturedUiState(false);
  els.cameraStatus.textContent = "Restarting camera...";
  startCamera();
}

function useCapturedPhoto() {
  if (!state.capturedDraftDataUrl) {
    els.cameraStatus.textContent = "Capture a photo first.";
    return;
  }

  els.queryImageInput.value = "";
  const split = state.capturedDraftDataUrl.split(",", 2);
  state.capturedImageBase64 = split.length === 2 ? split[1] : "";
  showPreviewFromDataUrl(state.capturedDraftDataUrl, { mirrored: true });
  els.querySourceMeta.textContent = "Using camera capture.";
  closeCameraModal();
}

function resetCameraCaptureButtons() {
  state.capturedDraftDataUrl = "";
  resetCameraCaptureUi();
}

function resetCameraCaptureUi() {
  els.cameraPreview.src = "";
  setCapturedUiState(false);
}

function setCapturedUiState(hasCapture) {
  els.cameraVideo.classList.toggle("hidden", hasCapture);
  els.cameraOverlay.classList.toggle("hidden", hasCapture);
  els.cameraPreview.classList.toggle("hidden", !hasCapture);
  els.capturePhotoBtn.classList.toggle("hidden", hasCapture);
  els.resetPhotoBtn.classList.toggle("hidden", !hasCapture);
  els.usePhotoBtn.classList.toggle("hidden", !hasCapture);
}

function showSearchOverlay() {
  if (!els.searchOverlay) return;

  if (els.searchOverlayMessage) {
    els.searchOverlayMessage.textContent = "Searching your photos...";
  }
  if (els.searchOverlaySubtext) {
    els.searchOverlaySubtext.textContent = "Please wait while we find your matches.";
  }

  els.searchOverlay.classList.remove("hidden");
  els.searchOverlay.setAttribute("aria-hidden", "false");
  document.body.classList.add("overlay-active");
  startOverlayAnimation();
}

function hideSearchOverlay() {
  stopOverlayAnimation();
  if (!els.searchOverlay) return;
  els.searchOverlay.classList.add("hidden");
  els.searchOverlay.setAttribute("aria-hidden", "true");
  document.body.classList.remove("overlay-active");
}

function startOverlayAnimation() {
  if (state.overlayRunning || !els.searchOverlayCanvas) return;
  state.overlayRunning = true;
  state.overlayRockets = [];
  state.overlayParticles = [];
  state.overlayLastTs = 0;
  state.overlaySpawnAccumulatorMs = 0;
  resizeOverlayCanvas();
  window.addEventListener("resize", resizeOverlayCanvas);
  state.overlayRafId = requestAnimationFrame(stepOverlayAnimation);
}

function stopOverlayAnimation() {
  if (!state.overlayRunning) return;
  state.overlayRunning = false;
  window.removeEventListener("resize", resizeOverlayCanvas);
  if (state.overlayRafId) {
    cancelAnimationFrame(state.overlayRafId);
    state.overlayRafId = 0;
  }
  state.overlayRockets = [];
  state.overlayParticles = [];
  state.overlayLastTs = 0;
  state.overlaySpawnAccumulatorMs = 0;

  const canvas = els.searchOverlayCanvas;
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function resizeOverlayCanvas() {
  const canvas = els.searchOverlayCanvas;
  if (!canvas) return;
  const dpr = Math.max(1, Math.min(2, window.devicePixelRatio || 1));
  const w = Math.max(1, Math.floor(window.innerWidth));
  const h = Math.max(1, Math.floor(window.innerHeight));
  canvas.width = Math.floor(w * dpr);
  canvas.height = Math.floor(h * dpr);
  canvas.style.width = `${w}px`;
  canvas.style.height = `${h}px`;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function stepOverlayAnimation(ts) {
  if (!state.overlayRunning || !els.searchOverlayCanvas) return;
  const canvas = els.searchOverlayCanvas;
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    state.overlayRafId = requestAnimationFrame(stepOverlayAnimation);
    return;
  }

  const width = canvas.clientWidth || window.innerWidth;
  const height = canvas.clientHeight || window.innerHeight;
  if (!state.overlayLastTs) {
    state.overlayLastTs = ts;
  }
  const dt = Math.min(42, ts - state.overlayLastTs);
  state.overlayLastTs = ts;

  ctx.fillStyle = "rgba(16, 4, 12, 0.25)";
  ctx.fillRect(0, 0, width, height);

  state.overlaySpawnAccumulatorMs += dt;
  if (state.overlaySpawnAccumulatorMs >= 340) {
    state.overlaySpawnAccumulatorMs = 0;
    spawnFireworkRocket(width, height);
    if (Math.random() < 0.35) {
      spawnFireworkRocket(width, height);
    }
  }

  ctx.save();
  ctx.globalCompositeOperation = "lighter";

  for (let i = state.overlayRockets.length - 1; i >= 0; i -= 1) {
    const rocket = state.overlayRockets[i];
    rocket.vy += 0.035;
    rocket.x += rocket.vx;
    rocket.y += rocket.vy;

    ctx.fillStyle = `hsla(${rocket.hue}, 98%, 66%, 0.9)`;
    ctx.beginPath();
    ctx.arc(rocket.x, rocket.y, 2.2, 0, Math.PI * 2);
    ctx.fill();

    if (rocket.y <= rocket.targetY || rocket.vy >= -0.3) {
      explodeFirework(rocket);
      state.overlayRockets.splice(i, 1);
    }
  }

  for (let i = state.overlayParticles.length - 1; i >= 0; i -= 1) {
    const p = state.overlayParticles[i];
    p.vx *= 0.986;
    p.vy += 0.046;
    p.x += p.vx;
    p.y += p.vy;
    p.life -= dt * 0.06;

    if (p.life <= 0) {
      state.overlayParticles.splice(i, 1);
      continue;
    }

    const alpha = Math.max(0, Math.min(1, p.life / p.maxLife));
    ctx.fillStyle = `hsla(${p.hue}, 98%, 67%, ${alpha})`;
    ctx.beginPath();
    ctx.arc(p.x, p.y, p.size * alpha + 0.2, 0, Math.PI * 2);
    ctx.fill();
  }

  ctx.restore();

  state.overlayRafId = requestAnimationFrame(stepOverlayAnimation);
}

function spawnFireworkRocket(width, height) {
  state.overlayRockets.push({
    x: width * (0.1 + Math.random() * 0.8),
    y: height + 12,
    vx: (Math.random() - 0.5) * 1.1,
    vy: -(6.4 + Math.random() * 2.2),
    targetY: height * (0.18 + Math.random() * 0.45),
    hue: 8 + Math.random() * 58,
  });

  if (state.overlayRockets.length > 22) {
    state.overlayRockets.splice(0, state.overlayRockets.length - 22);
  }
}

function explodeFirework(rocket) {
  const count = 30 + Math.floor(Math.random() * 24);
  for (let i = 0; i < count; i += 1) {
    const angle = (Math.PI * 2 * i) / count + Math.random() * 0.25;
    const speed = 1.1 + Math.random() * 3.8;
    const life = 45 + Math.random() * 42;
    state.overlayParticles.push({
      x: rocket.x,
      y: rocket.y,
      vx: Math.cos(angle) * speed,
      vy: Math.sin(angle) * speed,
      life,
      maxLife: life,
      size: 1.2 + Math.random() * 2.4,
      hue: rocket.hue + (Math.random() * 26 - 13),
    });
  }

  if (state.overlayParticles.length > 1400) {
    state.overlayParticles.splice(0, state.overlayParticles.length - 1400);
  }
}

async function onSearch() {
  const file = els.queryImageInput.files?.[0];
  let imageBase64 = "";

  if (state.capturedImageBase64) {
    imageBase64 = state.capturedImageBase64;
  } else if (file) {
    imageBase64 = await fileToBase64(file);
  }

  if (!imageBase64) {
    els.searchMeta.textContent = "Capture a face photo or upload one first.";
    return;
  }

  els.searchMeta.textContent = "Searching...";
  els.searchBtn.disabled = true;
  els.results.innerHTML = "";
  state.matches = [];
  els.selectAllBtn.textContent = "Select All";
  els.selectAllBtn.disabled = true;
  els.downloadSelectedBtn.disabled = true;
  showSearchOverlay();

  try {
    const payload = { imageBase64, topK: SEARCH_TOP_K, maxDistance: SEARCH_MAX_DISTANCE };
    const data = await postJson(fnUrl("search"), payload, {
      retries: SEARCH_REQUEST_RETRIES,
      timeoutMs: SEARCH_REQUEST_TIMEOUT_MS,
    });
    state.matches = Array.isArray(data.matches) ? data.matches : [];

    els.searchMeta.textContent = `Query faces: ${
      data.queryFaceCount ?? 0
    } | Matches: ${data.matchCount ?? state.matches.length} | Distance <= ${SEARCH_MAX_DISTANCE.toFixed(
      2
    )} | Indexed faces scanned: ${
      data.indexedFaceCountScanned ?? 0
    }`;

    renderResults(state.matches);
  } catch (err) {
    els.searchMeta.textContent = `Search failed: ${errorText(err)}`;
  } finally {
    hideSearchOverlay();
    els.searchBtn.disabled = false;
  }
}

function renderResults(matches) {
  els.results.innerHTML = "";
  if (!matches.length) {
    els.results.innerHTML = `<p class="status-line">No matches found.</p>`;
    updateSelectionControls();
    return;
  }

  const frag = document.createDocumentFragment();
  for (const m of matches) {
    const card = document.createElement("article");
    card.className = "result-card";
    card.innerHTML = `
      <img src="${escapeHtml(m.previewUrl || "")}" alt="Matched face image preview" loading="lazy" />
      <div class="result-body">
        <p class="result-title">${escapeHtml(m.fileName || m.fileId || "Unknown")}</p>
        <p class="result-meta">Distance: ${Number(m.distance || 0).toFixed(4)} | Score: ${Number(
      m.score || 0
    ).toFixed(4)}</p>
        <label class="check-row">
          <input type="checkbox" data-file-id="${escapeHtml(m.fileId || "")}" />
          Select for zip
        </label>
        <div class="result-links">
          <a href="${escapeHtml(m.webViewLink || "#")}" target="_blank" rel="noreferrer">Open</a>
          <a href="${escapeHtml(m.downloadUrl || "#")}" target="_blank" rel="noreferrer">Download</a>
        </div>
      </div>
    `;
    frag.appendChild(card);
  }
  els.results.appendChild(frag);
  updateSelectionControls();
}

function onResultsSelectionChanged(ev) {
  const target = ev.target;
  if (!(target instanceof HTMLInputElement)) return;
  if (!target.matches('input[type="checkbox"][data-file-id]')) return;
  updateSelectionControls();
}

function onToggleSelectAll() {
  const boxes = getResultCheckboxes();
  if (!boxes.length) return;
  const shouldSelect = boxes.some((box) => !box.checked);
  for (const box of boxes) {
    box.checked = shouldSelect;
  }
  updateSelectionControls();
}

async function onDownloadSelected() {
  const checked = Array.from(
    els.results.querySelectorAll('input[type="checkbox"][data-file-id]:checked')
  );
  const fileIds = checked
    .map((el) => el.getAttribute("data-file-id"))
    .filter((v) => typeof v === "string" && v.length > 0);

  if (!fileIds.length) {
    els.searchMeta.textContent = "Select at least one result to download.";
    return;
  }

  els.downloadSelectedBtn.disabled = true;
  els.searchMeta.textContent = `Preparing zip for ${fileIds.length} file(s)...`;

  try {
    const response = await fetch(fnUrl("download_matches_zip"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        fileIds,
        zipName: `face-matches-${Date.now()}`,
      }),
    });

    if (!response.ok) {
      const txt = await response.text();
      throw new Error(txt || `HTTP ${response.status}`);
    }

    const blob = await response.blob();
    const filename =
      getFilenameFromDisposition(response.headers.get("content-disposition")) ||
      `face-matches-${Date.now()}.zip`;
    downloadBlob(blob, filename);

    const errCount = Number(response.headers.get("x-download-error-count") || 0);
    if (errCount > 0) {
      els.searchMeta.textContent = `Downloaded with ${errCount} file error(s).`;
    } else {
      els.searchMeta.textContent = "ZIP downloaded.";
    }
  } catch (err) {
    els.searchMeta.textContent = `ZIP download failed: ${errorText(err)}`;
  } finally {
    els.downloadSelectedBtn.disabled = false;
  }
}

async function postJson(url, payload, options = {}) {
  const retries = clampInt(options.retries ?? 0, 0, 5, 0);
  const timeoutMs = clampInt(options.timeoutMs ?? 45000, 5000, 180000, 45000);
  let attempt = 0;

  while (true) {
    let response;
    try {
      response = await fetchWithTimeout(
        url,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        },
        timeoutMs
      );
    } catch (err) {
      if (attempt < retries && isRetryableFetchError(err)) {
        attempt += 1;
        await sleep(backoffDelayMs(attempt));
        continue;
      }
      throw err;
    }

    const text = await response.text();
    let data = {};
    try {
      data = text ? JSON.parse(text) : {};
    } catch {
      data = { raw: text };
    }

    if (!response.ok) {
      if (attempt < retries && isRetryableHttpStatus(response.status)) {
        attempt += 1;
        await sleep(backoffDelayMs(attempt));
        continue;
      }
      throw new Error(data.error || data.raw || `HTTP ${response.status}`);
    }
    return data;
  }
}

async function fetchWithTimeout(url, init, timeoutMs) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, {
      ...init,
      signal: controller.signal,
    });
  } finally {
    clearTimeout(timer);
  }
}

function backoffDelayMs(attempt) {
  const base = Math.min(6000, 500 * 2 ** (attempt - 1));
  const jitter = Math.floor(Math.random() * 350);
  return base + jitter;
}

function isRetryableHttpStatus(status) {
  return status === 408 || status === 425 || status === 429 || status === 500 || status === 502 || status === 503 || status === 504;
}

function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const result = String(reader.result || "");
      const b64 = result.includes(",") ? result.split(",", 2)[1] : result;
      resolve(b64);
    };
    reader.onerror = () => reject(new Error("Failed to read file"));
    reader.readAsDataURL(file);
  });
}

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function getFilenameFromDisposition(disposition) {
  if (!disposition) return "";
  const m = disposition.match(/filename="([^"]+)"/i);
  return m ? m[1] : "";
}

function clampInt(raw, min, max, fallback) {
  const n = Number.parseInt(raw, 10);
  if (Number.isNaN(n)) return fallback;
  return Math.min(max, Math.max(min, n));
}

function getResultCheckboxes() {
  return Array.from(els.results.querySelectorAll('input[type="checkbox"][data-file-id]'));
}

function updateSelectionControls() {
  const boxes = getResultCheckboxes();
  if (!boxes.length) {
    els.selectAllBtn.disabled = true;
    els.selectAllBtn.textContent = "Select All";
    els.downloadSelectedBtn.disabled = true;
    return;
  }

  const checkedCount = boxes.filter((box) => box.checked).length;
  const allSelected = checkedCount === boxes.length;
  els.selectAllBtn.disabled = false;
  els.selectAllBtn.textContent = allSelected ? "Unselect All" : "Select All";
  els.downloadSelectedBtn.disabled = checkedCount === 0;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function errorText(err) {
  if (!err) return "Unknown error";
  if (typeof err === "string") return err;
  if (err instanceof Error) return err.message;
  return String(err);
}

function isRetryableFetchError(err) {
  if (err && typeof err === "object" && err.name === "AbortError") {
    return true;
  }
  const msg = errorText(err).toLowerCase();
  return (
    msg.includes("load failed") ||
    msg.includes("failed to fetch") ||
    msg.includes("networkerror") ||
    msg.includes("network connection was lost") ||
    msg.includes("network request failed")
  );
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}
