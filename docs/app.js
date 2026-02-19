const HARDCODED_FUNCTIONS_BASE_URL =
  "https://us-central1-facefinder-260219-2674.cloudfunctions.net";
const HARDCODED_DRIVE_FOLDER_LINK =
  "https://drive.google.com/drive/folders/1EF9ddrU0dWt3H5c43Rt_yS54TZRK_TgB?usp=sharing";
const SEARCH_TOP_K = 0; // 0 means return all matches
const SEARCH_MAX_DISTANCE = 0.35;

const state = {
  matches: [],
  capturedImageBase64: "",
  capturedDraftDataUrl: "",
  previewObjectUrl: "",
  cameraStream: null,
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

  window.addEventListener("beforeunload", stopCameraStream);
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
  els.queryPreview.classList.remove("hidden");
}

function showPreviewFromDataUrl(dataUrl) {
  clearQueryPreview();
  els.queryPreview.src = dataUrl;
  els.queryPreview.classList.remove("hidden");
}

function clearQueryPreview() {
  if (state.previewObjectUrl) {
    URL.revokeObjectURL(state.previewObjectUrl);
    state.previewObjectUrl = "";
  }
  els.queryPreview.src = "";
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
  showPreviewFromDataUrl(state.capturedDraftDataUrl);
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

  try {
    const payload = { imageBase64, topK: SEARCH_TOP_K, maxDistance: SEARCH_MAX_DISTANCE };
    const data = await postJson(fnUrl("search"), payload, { retries: 2 });
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
  let attempt = 0;

  while (true) {
    let response;
    try {
      response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
    } catch (err) {
      if (attempt < retries && isRetryableFetchError(err)) {
        attempt += 1;
        await sleep(600 * attempt);
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
      throw new Error(data.error || data.raw || `HTTP ${response.status}`);
    }
    return data;
  }
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
  const msg = errorText(err).toLowerCase();
  return (
    msg.includes("load failed") ||
    msg.includes("failed to fetch") ||
    msg.includes("networkerror")
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
