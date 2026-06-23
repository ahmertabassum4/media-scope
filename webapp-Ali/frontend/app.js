const form = document.getElementById("uploadForm");
const input = document.getElementById("imageInput");
const urlInput = document.getElementById("urlInput");
const dropZone = document.getElementById("dropZone");
const fileMeta = document.getElementById("fileMeta");
const button = document.getElementById("submitButton");
const clearButton = document.getElementById("clearButton");
const clearImageButton = document.getElementById("clearImageButton");
const explainButton = document.getElementById("explainButton");
const reasoningText = document.getElementById("reasoningText");
const resultsPanel = document.querySelector(".results-panel");
const previewWrap = document.querySelector(".preview-wrap");
const previewImage = document.getElementById("previewImage");
const evidenceOverlay = document.getElementById("evidenceOverlay");
const loadingStatus = document.getElementById("loadingStatus");
const errorBox = document.getElementById("errorBox");
const healthBadge = document.getElementById("healthBadge");
const modeButtons = document.querySelectorAll(".mode-button");
const analyzeView = document.getElementById("analyzeView");
const gameView = document.getElementById("gameView");

const gameImage = document.getElementById("gameImage");
const gameEmpty = document.getElementById("gameEmpty");
const nextGameButton = document.getElementById("nextGameButton");
const checkGameButton = document.getElementById("checkGameButton");
const gameResult = document.getElementById("gameResult");
const gameErrorBox = document.getElementById("gameErrorBox");
const biasSlider = document.getElementById("biasSlider");
const factualitySlider = document.getElementById("factualitySlider");
const biasSliderGroup = document.getElementById("biasSliderGroup");
const factualitySliderGroup = document.getElementById("factualitySliderGroup");
const gameBiasValue = document.getElementById("gameBiasValue");
const gameFactualityValue = document.getElementById("gameFactualityValue");

const fields = {
  biasLabel: document.getElementById("biasLabel"),
  factualityLabel: document.getElementById("factualityLabel"),
};

const evidenceLists = {
  bias: document.getElementById("biasEvidence"),
  factuality: document.getElementById("factualityEvidence"),
};

const markers = {
  bias: document.getElementById("biasMarker"),
  factuality: document.getElementById("factualityMarker"),
};

const labels = {
  bias: ["left", "left-center", "least biased", "right-center", "right"],
  factuality: ["very low", "low", "mixed", "high", "very high"],
};

const spectrumPositions = {
  bias: {
    left: 0,
    "left-center": 25,
    "least biased": 50,
    "right-center": 75,
    right: 100,
  },
  factuality: {
    "very low": 0,
    low: 25,
    mixed: 50,
    high: 75,
    "very high": 100,
  },
};

const labelColors = {
  bias: {
    left: "#2563eb",
    "left-center": "#3b82f6",
    "least biased": "#374151",
    "right-center": "#ea580c",
    right: "#dc2626",
  },
  factuality: {
    "very low": "#b91c1c",
    low: "#d97706",
    mixed: "#ca8a04",
    high: "#0f766e",
    "very high": "#15803d",
  },
};

let currentGameItem = null;
let gameSelection = { bias: labels.bias[2], factuality: labels.factuality[2] };
let lastAnalyzeData = null;

function setText(id, value) {
  fields[id].textContent = value || "-";
}

function setSpectrumMarker(task, label) {
  const marker = markers[task];
  const position = spectrumPositions[task][label];
  const color = labelColors[task][label];
  if (position === undefined) {
    marker.classList.remove("visible");
    return;
  }
  marker.style.left = `${position}%`;
  marker.style.backgroundColor = color || "";
  marker.classList.add("visible");
}

function setResultColor(task, label) {
  const field = task === "bias" ? fields.biasLabel : fields.factualityLabel;
  field.style.color = labelColors[task][label] || "";
}

function showError(box, message) {
  box.textContent = message;
  box.classList.add("visible");
}

function clearError(box) {
  box.textContent = "";
  box.classList.remove("visible");
}

const statusSteps = {
  url: [
    "Capturing the page",
    "Reading provenance",
    "Running the models",
    "Scoring bias & factuality",
    "Almost there",
  ],
  image: [
    "Reading the image",
    "Running the models",
    "Finding evidence regions",
    "Scoring bias & factuality",
    "Almost there",
  ],
};

let statusTimer = null;

function startStatusCycle(mode) {
  const steps = statusSteps[mode] || statusSteps.image;
  let index = 0;
  loadingStatus.textContent = steps[0];
  stopStatusCycle();
  statusTimer = window.setInterval(() => {
    index = Math.min(index + 1, steps.length - 1);
    loadingStatus.textContent = steps[index];
    if (index === steps.length - 1) {
      stopStatusCycle();
    }
  }, 2800);
}

function stopStatusCycle() {
  if (statusTimer !== null) {
    window.clearInterval(statusTimer);
    statusTimer = null;
  }
}

function setLoading(isLoading, mode) {
  button.disabled = isLoading;
  clearButton.disabled = isLoading;
  button.classList.toggle("is-loading", isLoading);
  const label = button.querySelector(".button-label");
  if (label) {
    label.textContent = isLoading ? "Analyzing" : "Analyze";
  }
  previewWrap.classList.toggle("loading", isLoading);
  resultsPanel.classList.toggle("loading", isLoading);
  if (isLoading) {
    startStatusCycle(mode);
  } else {
    stopStatusCycle();
  }
}

function resetReasoning() {
  explainButton.disabled = true;
  explainButton.textContent = "Explain";
  reasoningText.textContent =
    "Run an analysis, then click Explain for a combined factuality and bias rationale.";
}

function updatePreview(file) {
  lastAnalyzeData = null;
  clearEvidence();
  resetReasoning();
  if (!file) {
    fileMeta.textContent = "PNG, JPG, WEBP, BMP, TIFF";
    previewImage.removeAttribute("src");
    previewWrap.classList.remove("has-image");
    clearImageButton.hidden = true;
    return;
  }
  if (urlInput) {
    urlInput.value = "";
  }
  fileMeta.textContent = `${file.name} - ${(file.size / 1024 / 1024).toFixed(2)} MB`;
  previewImage.src = URL.createObjectURL(file);
  previewWrap.classList.add("has-image");
  clearImageButton.hidden = false;
}

// Full reset of the analyze view so the user can start a fresh analysis.
function resetAnalyze() {
  lastAnalyzeData = null;
  input.value = "";
  if (urlInput) {
    urlInput.value = "";
  }
  fileMeta.textContent = "PNG, JPG, WEBP, BMP, TIFF";
  previewImage.removeAttribute("src");
  previewWrap.classList.remove("has-image");
  clearImageButton.hidden = true;
  clearEvidence();
  resetReasoning();
  clearError(errorBox);
  setText("biasLabel", "");
  setText("factualityLabel", "");
  setResultColor("bias", "");
  setResultColor("factuality", "");
  markers.bias.classList.remove("visible");
  markers.factuality.classList.remove("visible");
}

function clearEvidence() {
  evidenceOverlay.innerHTML = "";
  evidenceOverlay.classList.remove("visible");
  Object.values(evidenceLists).forEach((node) => {
    node.innerHTML = "";
    node.classList.remove("visible");
  });
}

function renderResult(data) {
  lastAnalyzeData = data;

  // Render the exact image that was analyzed so the evidence overlay aligns. For URL
  // captures there is no local file, so the screenshot returned by the backend is the
  // only source; the previewImage "load" listener re-runs renderEvidence with correct
  // dimensions once it decodes.
  if (data.screenshot_b64) {
    previewImage.src = `data:image/png;base64,${data.screenshot_b64}`;
    previewWrap.classList.add("has-image");
    clearImageButton.hidden = false;
  }

  setText("biasLabel", data.bias.label);
  setText("factualityLabel", data.factuality.label);
  setResultColor("bias", data.bias.label);
  setResultColor("factuality", data.factuality.label);
  setSpectrumMarker("bias", data.bias.label);
  setSpectrumMarker("factuality", data.factuality.label);

  const errors = [data.bias.error, data.factuality.error].filter(Boolean);
  if (errors.length) {
    showError(errorBox, errors.join(" - "));
  }
  renderEvidence(data);

  // Reasoning is on-demand; enable the button now that we have a screenshot + labels.
  resetReasoning();
  explainButton.disabled = !data.screenshot_b64;
}

function displayedImageBox(imageInfo) {
  const frameWidth = previewWrap.clientWidth;
  const frameHeight = previewWrap.clientHeight;
  const imageWidth = imageInfo.original_size.width;
  const imageHeight = imageInfo.original_size.height;
  if (!frameWidth || !frameHeight || !imageWidth || !imageHeight) {
    return null;
  }
  const frameRatio = frameWidth / frameHeight;
  const imageRatio = imageWidth / imageHeight;

  if (frameRatio > imageRatio) {
    const height = frameHeight;
    const width = height * imageRatio;
    return { left: (frameWidth - width) / 2, top: 0, width, height, imageWidth, imageHeight };
  }
  const width = frameWidth;
  const height = width / imageRatio;
  return { left: 0, top: (frameHeight - height) / 2, width, height, imageWidth, imageHeight };
}

function renderEvidence(data) {
  clearEvidence();
  if (!previewWrap.classList.contains("has-image")) return;

  const allItems = [
    ...(data.bias.evidence || []).map((item, index) => ({ ...item, task: "bias", index: index + 1 })),
    ...(data.factuality.evidence || []).map((item, index) => ({ ...item, task: "factuality", index: index + 1 })),
  ];
  if (!allItems.length) return;

  const imageBox = displayedImageBox(data.image);
  if (!imageBox) return;
  for (const item of allItems) {
    const [x1, y1, x2, y2] = item.bbox;
    const left = imageBox.left + (x1 / imageBox.imageWidth) * imageBox.width;
    const top = imageBox.top + (y1 / imageBox.imageHeight) * imageBox.height;
    const width = ((x2 - x1) / imageBox.imageWidth) * imageBox.width;
    const height = ((y2 - y1) / imageBox.imageHeight) * imageBox.height;
    const box = document.createElement("div");
    box.className = `evidence-box ${item.task}`;
    box.style.left = `${left}px`;
    box.style.top = `${top}px`;
    box.style.width = `${Math.max(8, width)}px`;
    box.style.height = `${Math.max(8, height)}px`;
    const alpha = 0.12 + Math.min(1, item.importance || 0.5) * 0.16;
    box.style.backgroundColor =
      item.task === "bias" ? `rgba(37, 99, 235, ${alpha})` : `rgba(15, 118, 110, ${alpha})`;
    box.title = item.reason || item.text || item.id;

    const badge = document.createElement("span");
    badge.textContent = `${item.task === "bias" ? "B" : "F"}${item.index}`;
    box.appendChild(badge);
    evidenceOverlay.appendChild(box);
  }
  evidenceOverlay.classList.add("visible");
  renderEvidenceList("bias", data.bias);
  renderEvidenceList("factuality", data.factuality);
}

function renderEvidenceList(task, result) {
  const container = evidenceLists[task];
  const items = result.evidence || [];
  if (!items.length) return;
  container.classList.add("visible");
  container.innerHTML = "";
  const title = document.createElement("div");
  title.className = "evidence-title";
  title.textContent = "Evidence";
  container.appendChild(title);
  items.slice(0, 5).forEach((item, index) => {
    const row = document.createElement("div");
    row.className = `evidence-row ${task}`;

    const badge = document.createElement("span");
    badge.className = "evidence-id";
    badge.textContent = `${task === "bias" ? "B" : "F"}${index + 1}`;

    const reason = document.createElement("span");
    reason.textContent = item.reason || item.text || item.id;

    row.appendChild(badge);
    row.appendChild(reason);
    container.appendChild(row);
  });
}

async function analyze({ file, url }) {
  clearError(errorBox);
  setLoading(true, url ? "url" : "image");
  const body = new FormData();
  if (url) {
    body.append("url", url);
  } else {
    body.append("file", file);
  }

  try {
    const response = await fetch("/api/analyze", {
      method: "POST",
      body,
      cache: "no-store",
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "Analysis failed");
    }
    renderResult(data);
  } catch (error) {
    showError(errorBox, error.message || "Analysis failed");
  } finally {
    setLoading(false);
  }
}

async function requestReasoning() {
  if (!lastAnalyzeData || !lastAnalyzeData.screenshot_b64) return;
  clearError(errorBox);
  explainButton.disabled = true;
  explainButton.textContent = "Explaining";
  reasoningText.textContent = "Generating reasoning...";

  const factuality = lastAnalyzeData.factuality || {};
  const bias = lastAnalyzeData.bias || {};
  const evidence = [...(factuality.evidence || []), ...(bias.evidence || [])];

  try {
    const response = await fetch("/api/reason", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      cache: "no-store",
      body: JSON.stringify({
        screenshot_b64: lastAnalyzeData.screenshot_b64,
        factuality_label: factuality.label || null,
        bias_label: bias.label || null,
        evidence,
      }),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "Could not generate reasoning");
    }
    reasoningText.textContent = data.reasoning || "No reasoning returned.";
  } catch (error) {
    reasoningText.textContent = "";
    showError(errorBox, error.message || "Could not generate reasoning");
  } finally {
    explainButton.disabled = false;
    explainButton.textContent = "Explain";
  }
}

window.addEventListener("resize", () => {
  if (lastAnalyzeData) {
    renderEvidence(lastAnalyzeData);
  }
});

function switchMode(mode) {
  const isGame = mode === "game";
  analyzeView.classList.toggle("active", !isGame);
  gameView.classList.toggle("active", isGame);
  modeButtons.forEach((modeButton) => {
    const isActive = modeButton.dataset.mode === mode;
    modeButton.classList.toggle("active", isActive);
    modeButton.setAttribute("aria-selected", String(isActive));
  });
  if (isGame && !currentGameItem) {
    loadGameItem();
  }
}

function updateCheckState() {
  checkGameButton.disabled = !currentGameItem;
}

function getSliderLabel(group) {
  const slider = group === "bias" ? biasSlider : factualitySlider;
  return labels[group][Number(slider.value)] || "";
}

function updateSliderReadout(group) {
  const slider = group === "bias" ? biasSlider : factualitySlider;
  const valueNode = group === "bias" ? gameBiasValue : gameFactualityValue;
  const selected = getSliderLabel(group);
  const color = labelColors[group][selected] || "#141922";

  gameSelection[group] = selected;
  valueNode.textContent = selected;
  valueNode.style.color = color;
  slider.style.setProperty("--thumb-color", color);
}

function clearGameJudgement() {
  biasSliderGroup.classList.remove("correct", "wrong");
  factualitySliderGroup.classList.remove("correct", "wrong");
  gameResult.className = "game-result";
  gameResult.textContent = "";
  nextGameButton.hidden = true;
}

function updateGameSliders() {
  updateSliderReadout("bias");
  updateSliderReadout("factuality");
  updateCheckState();
}

function handleGameSliderInput(group) {
  updateSliderReadout(group);
  updateSliderReadout(group === "bias" ? "factuality" : "bias");
  clearGameJudgement();
  updateCheckState();
}

function resetGameSelection() {
  biasSlider.value = "2";
  factualitySlider.value = "2";
  clearGameJudgement();
  updateGameSliders();
}

async function loadGameItem() {
  clearError(gameErrorBox);
  currentGameItem = null;
  resetGameSelection();
  nextGameButton.disabled = true;
  checkGameButton.disabled = true;
  gameEmpty.textContent = "Loading";
  gameImage.removeAttribute("src");
  gameImage.classList.remove("visible");

  try {
    const response = await fetch("/api/game/random", { cache: "no-store" });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "Could not load image");
    }
    currentGameItem = data;
    gameImage.src = `${data.image_url}?t=${Date.now()}`;
    gameImage.onload = () => {
      gameImage.classList.add("visible");
      gameEmpty.textContent = "";
    };
  } catch (error) {
    currentGameItem = null;
    showError(gameErrorBox, error.message || "Could not load image");
    gameEmpty.textContent = "No image";
  } finally {
    nextGameButton.disabled = false;
    updateCheckState();
  }
}

function markSliderResult(group, selected, correct) {
  const container = group === "bias" ? biasSliderGroup : factualitySliderGroup;
  const valueNode = group === "bias" ? gameBiasValue : gameFactualityValue;
  const isCorrect = selected === correct;

  container.classList.toggle("correct", isCorrect);
  container.classList.toggle("wrong", !isCorrect);
  valueNode.textContent = isCorrect ? selected : `${selected} -> ${correct}`;
  valueNode.style.color = isCorrect ? labelColors[group][correct] || "" : "#b42318";
}

async function checkGameAnswer() {
  if (!currentGameItem) return;
  clearError(gameErrorBox);
  checkGameButton.disabled = true;

  try {
    const response = await fetch("/api/game/answer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      cache: "no-store",
      body: JSON.stringify({
        id: currentGameItem.id,
        bias: gameSelection.bias,
        factuality: gameSelection.factuality,
      }),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "Could not check answer");
    }
    markSliderResult("bias", data.bias.selected, data.bias.correct_label);
    markSliderResult("factuality", data.factuality.selected, data.factuality.correct_label);

    const total = Number(data.bias.is_correct) + Number(data.factuality.is_correct);
    gameResult.className = `game-result visible score-${total}`;
    gameResult.textContent = `${total}/2`;
    nextGameButton.hidden = false;
  } catch (error) {
    showError(gameErrorBox, error.message || "Could not check answer");
    updateCheckState();
  }
}

input.addEventListener("change", () => {
  updatePreview(input.files[0]);
});

urlInput.addEventListener("input", () => {
  if (urlInput.value.trim() && input.files.length) {
    input.value = "";
    updatePreview(null);
  }
});

previewImage.addEventListener("load", () => {
  if (lastAnalyzeData) {
    renderEvidence(lastAnalyzeData);
  }
});

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const url = (urlInput.value || "").trim();
  if (url) {
    analyze({ url });
    return;
  }
  const file = input.files[0];
  if (!file) {
    showError(errorBox, "Select an image or enter a URL first");
    return;
  }
  analyze({ file });
});

explainButton.addEventListener("click", requestReasoning);
clearButton.addEventListener("click", resetAnalyze);
clearImageButton.addEventListener("click", resetAnalyze);

for (const eventName of ["dragenter", "dragover"]) {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.add("dragover");
  });
}

for (const eventName of ["dragleave", "drop"]) {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.remove("dragover");
  });
}

dropZone.addEventListener("drop", (event) => {
  const file = event.dataTransfer.files[0];
  if (!file) return;
  input.files = event.dataTransfer.files;
  updatePreview(file);
});

modeButtons.forEach((modeButton) => {
  modeButton.addEventListener("click", () => switchMode(modeButton.dataset.mode));
});

nextGameButton.addEventListener("click", loadGameItem);
checkGameButton.addEventListener("click", checkGameAnswer);
biasSlider.addEventListener("input", () => {
  handleGameSliderInput("bias");
});
factualitySlider.addEventListener("input", () => {
  handleGameSliderInput("factuality");
});

async function checkHealth() {
  try {
    const response = await fetch("/api/health", { cache: "no-store" });
    const data = await response.json();
    if (!response.ok) throw new Error("API unavailable");
    const ready = data.models?.status === "ok" && data.game_items > 0;
    healthBadge.textContent = ready ? "Ready" : "Offline";
    healthBadge.classList.add(ready ? "ok" : "fail");
  } catch {
    healthBadge.textContent = "API offline";
    healthBadge.classList.add("fail");
  }
}

updateGameSliders();
checkHealth();
