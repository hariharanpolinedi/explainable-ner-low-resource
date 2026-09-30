/**
 * Kokborok NER Studio — Human-In-The-Loop Frontend Controller
 * Experiment 11B
 */

// Global Application State
const state = {
  annotatorId: localStorage.getItem("kok_hitl_annotator_id") || "",
  currentIndex: 0,
  totalCandidates: 1367,
  currentCandidate: null,
  activeFilter: "UNREVIEWED",
  selectedSpan: null, // { start: int, end: int } (inclusive start, exclusive end)
  currentHumanTags: [], // list of BIO tags for active sentence
  isSaving: false
};

// DOM Elements
const elements = {
  annotatorChip: document.getElementById("annotatorChip"),
  annotatorDisplay: document.getElementById("annotatorDisplay"),
  annotatorModal: document.getElementById("annotatorModal"),
  annotatorInput: document.getElementById("annotatorInput"),
  btnSaveAnnotator: document.getElementById("btnSaveAnnotator"),
  
  progressBarFill: document.getElementById("progressBarFill"),
  progressText: document.getElementById("progressText"),
  
  btnPrev: document.getElementById("btnPrev"),
  btnNext: document.getElementById("btnNext"),
  currentIndexDisplay: document.getElementById("currentIndexDisplay"),
  totalIndexDisplay: document.getElementById("totalIndexDisplay"),
  statusFilter: document.getElementById("statusFilter"),
  jumpIdInput: document.getElementById("jumpIdInput"),
  btnJump: document.getElementById("btnJump"),
  
  sentenceIdDisplay: document.getElementById("sentenceIdDisplay"),
  sourceDisplay: document.getElementById("sourceDisplay"),
  licenseDisplay: document.getElementById("licenseDisplay"),
  sentenceStatusBadge: document.getElementById("sentenceStatusBadge"),
  rawSentenceText: document.getElementById("rawSentenceText"),
  
  machineTokenContainer: document.getElementById("machineTokenContainer"),
  humanTokenBoard: document.getElementById("humanTokenBoard"),
  selectionInfo: document.getElementById("selectionInfo"),
  bioTagsStream: document.getElementById("bioTagsStream"),
  annotationNotes: document.getElementById("annotationNotes"),
  
  btnAccept: document.getElementById("btnAccept"),
  btnSaveEdit: document.getElementById("btnSaveEdit"),
  btnReject: document.getElementById("btnReject"),
  btnUncertain: document.getElementById("btnUncertain"),
  btnExport: document.getElementById("btnExport"),
  btnShortcuts: document.getElementById("btnShortcuts"),
  
  shortcutsModal: document.getElementById("shortcutsModal"),
  btnCloseShortcuts: document.getElementById("btnCloseShortcuts"),
  toastContainer: document.getElementById("toastContainer"),
  
  // Sidebar Stats
  statVerified: document.getElementById("statVerified"),
  statUncertain: document.getElementById("statUncertain"),
  statRejected: document.getElementById("statRejected"),
  statUnreviewed: document.getElementById("statUnreviewed"),
  countPER: document.getElementById("countPER"),
  countLOC: document.getElementById("countLOC"),
  countORG: document.getElementById("countORG"),
  countMISC: document.getElementById("countMISC"),
  countTotalEnt: document.getElementById("countTotalEnt"),
  countMultiToken: document.getElementById("countMultiToken")
};

// Initialize Application
document.addEventListener("DOMContentLoaded", () => {
  setupAnnotator();
  setupEventListeners();
  loadStats();
  loadCandidateByIndex(0);
});

// Setup Annotator Management
function setupAnnotator() {
  if (!state.annotatorId) {
    elements.annotatorModal.style.display = "flex";
  } else {
    elements.annotatorDisplay.textContent = state.annotatorId;
    elements.annotatorModal.style.display = "none";
  }

  elements.btnSaveAnnotator.addEventListener("click", () => {
    const val = elements.annotatorInput.value.trim();
    if (val) {
      state.annotatorId = val;
      localStorage.setItem("kok_hitl_annotator_id", val);
      elements.annotatorDisplay.textContent = val;
      elements.annotatorModal.style.display = "none";
      showToast(`Annotator ID set to: ${val}`, "success");
    } else {
      showToast("Please enter a valid Annotator ID", "error");
    }
  });

  elements.annotatorChip.addEventListener("click", () => {
    elements.annotatorInput.value = state.annotatorId;
    elements.annotatorModal.style.display = "flex";
  });
}

// Setup Event Listeners
function setupEventListeners() {
  // Navigation
  elements.btnPrev.addEventListener("click", () => navigateRelative(-1));
  elements.btnNext.addEventListener("click", () => navigateRelative(1));

  elements.statusFilter.addEventListener("change", (e) => {
    state.activeFilter = e.target.value;
    state.currentIndex = 0;
    loadCandidateByIndex(0);
  });

  elements.btnJump.addEventListener("click", () => {
    const sentId = elements.jumpIdInput.value.trim();
    if (sentId) {
      loadCandidateById(sentId);
    }
  });

  elements.jumpIdInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      const sentId = elements.jumpIdInput.value.trim();
      if (sentId) loadCandidateById(sentId);
    }
  });

  // Entity Palette Buttons
  document.querySelectorAll(".btn-entity").forEach((btn) => {
    btn.addEventListener("click", () => {
      const entType = btn.getAttribute("data-type");
      applyEntityTypeToSelection(entType);
    });
  });

  // Decision Action Buttons
  elements.btnAccept.addEventListener("click", () => handleDecision("ACCEPT"));
  elements.btnSaveEdit.addEventListener("click", () => handleDecision("EDIT"));
  elements.btnReject.addEventListener("click", () => handleDecision("REJECT"));
  elements.btnUncertain.addEventListener("click", () => handleDecision("UNCERTAIN"));

  // Export
  elements.btnExport.addEventListener("click", handleExport);

  // Shortcuts Modal
  elements.btnShortcuts.addEventListener("click", () => {
    elements.shortcutsModal.style.display = "flex";
  });
  elements.btnCloseShortcuts.addEventListener("click", () => {
    elements.shortcutsModal.style.display = "none";
  });

  // Global Keyboard Shortcuts
  window.addEventListener("keydown", handleKeydown);
}

// Keyboard Shortcut Handler
function handleKeydown(e) {
  // Ignore shortcuts if focusing an input
  if (["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement.tagName)) {
    return;
  }

  const key = e.key.toUpperCase();

  if (key === "1") {
    e.preventDefault();
    applyEntityTypeToSelection("PER");
  } else if (key === "2") {
    e.preventDefault();
    applyEntityTypeToSelection("LOC");
  } else if (key === "3") {
    e.preventDefault();
    applyEntityTypeToSelection("ORG");
  } else if (key === "4") {
    e.preventDefault();
    applyEntityTypeToSelection("MISC");
  } else if (key === "0" || key === "BACKSPACE" || key === "DELETE") {
    e.preventDefault();
    applyEntityTypeToSelection("CLEAR");
  } else if (key === "A") {
    e.preventDefault();
    handleDecision("ACCEPT");
  } else if (key === "S") {
    e.preventDefault();
    handleDecision("EDIT");
  } else if (key === "R") {
    e.preventDefault();
    handleDecision("REJECT");
  } else if (key === "U") {
    e.preventDefault();
    handleDecision("UNCERTAIN");
  } else if (key === "N" || key === "ARROWRIGHT") {
    e.preventDefault();
    navigateRelative(1);
  } else if (key === "P" || key === "ARROWLEFT") {
    e.preventDefault();
    navigateRelative(-1);
  } else if (key === "ESCAPE") {
    e.preventDefault();
    state.selectedSpan = null;
    updateSelectionDisplay();
    renderHumanTokenBoard();
    elements.shortcutsModal.style.display = "none";
    elements.annotatorModal.style.display = "none";
  }
}

// Fetch Candidate by Index
async function loadCandidateByIndex(index) {
  try {
    const res = await fetch(`/api/candidate?index=${index}&filter_status=${state.activeFilter}`);
    if (!res.ok) {
      if (res.status === 404) {
        showToast("No more sentences matching filter.", "info");
        return;
      }
      throw new Error("Failed to load candidate");
    }
    const cand = await res.json();
    state.currentIndex = index;
    displayCandidate(cand);
  } catch (err) {
    showToast(err.message, "error");
  }
}

// Fetch Candidate by Sentence ID
async function loadCandidateById(sentenceId) {
  try {
    const res = await fetch(`/api/candidate?sentence_id=${sentenceId}`);
    if (!res.ok) {
      throw new Error(`Sentence ${sentenceId} not found.`);
    }
    const cand = await res.json();
    displayCandidate(cand);
  } catch (err) {
    showToast(err.message, "error");
  }
}

// Display Candidate in UI
function displayCandidate(cand) {
  state.currentCandidate = cand;
  state.selectedSpan = null;

  // Initialize human tags from existing human_tags or machine_tags
  if (cand.human_tags && cand.human_tags.length === cand.tokens.length) {
    state.currentHumanTags = [...cand.human_tags];
  } else {
    state.currentHumanTags = [...cand.machine_tags];
  }

  // Update Metadata
  elements.sentenceIdDisplay.textContent = cand.sentence_id;
  elements.sourceDisplay.textContent = cand.source || "Public Corpus";
  elements.licenseDisplay.textContent = cand.license || "Open Access";
  elements.rawSentenceText.textContent = cand.raw_text;
  elements.annotationNotes.value = cand.notes || "";

  // Status Badge
  const status = cand.human_status || "UNREVIEWED";
  elements.sentenceStatusBadge.textContent = status.replace("_", " ");
  elements.sentenceStatusBadge.className = `status-badge status-${status}`;

  // Index Display
  elements.currentIndexDisplay.textContent = `#${state.currentIndex + 1}`;

  // Render Machine Tokens
  renderMachineTokens(cand.tokens, cand.machine_tags);

  // Render Human Tokens
  renderHumanTokenBoard();
  updateBioStream();
  updateSelectionDisplay();
}

// Render Machine Suggestion Tokens
function renderMachineTokens(tokens, machineTags) {
  elements.machineTokenContainer.innerHTML = "";
  tokens.forEach((tok, i) => {
    const tag = machineTags[i] || "O";
    const entType = tag.startsWith("B-") || tag.startsWith("I-") ? tag.substring(2) : "O";

    const chip = document.createElement("div");
    chip.className = `tok-chip machine-tok ent-${entType}`;
    
    const textSpan = document.createElement("span");
    textSpan.className = "tok-text";
    textSpan.textContent = tok;

    const tagSpan = document.createElement("span");
    tagSpan.className = `tok-tag tag-${tag}`;
    tagSpan.textContent = tag;

    chip.appendChild(textSpan);
    chip.appendChild(tagSpan);
    elements.machineTokenContainer.appendChild(chip);
  });
}

// Render Human Interactive Token Board
function renderHumanTokenBoard() {
  const tokens = state.currentCandidate ? state.currentCandidate.tokens : [];
  elements.humanTokenBoard.innerHTML = "";

  tokens.forEach((tok, i) => {
    const tag = state.currentHumanTags[i] || "O";
    const entType = tag.startsWith("B-") || tag.startsWith("I-") ? tag.substring(2) : "O";

    const chip = document.createElement("div");
    const isSelected = isTokenSelected(i);
    chip.className = `tok-chip interactive-tok ent-${entType} ${isSelected ? "selected" : ""}`;
    chip.setAttribute("data-index", i);

    const textSpan = document.createElement("span");
    textSpan.className = "tok-text";
    textSpan.textContent = tok;

    const tagSpan = document.createElement("span");
    tagSpan.className = `tok-tag tag-${tag}`;
    tagSpan.textContent = tag;

    chip.appendChild(textSpan);
    chip.appendChild(tagSpan);

    chip.addEventListener("click", () => handleTokenClick(i));
    elements.humanTokenBoard.appendChild(chip);
  });
}

// Token Click & Span Selection Logic
function handleTokenClick(idx) {
  if (state.selectedSpan === null) {
    state.selectedSpan = { start: idx, end: idx + 1 };
  } else if (state.selectedSpan.start === idx && state.selectedSpan.end === idx + 1) {
    // Deselect if clicking the same single token
    state.selectedSpan = null;
  } else if (idx >= state.selectedSpan.start) {
    // Extend end
    state.selectedSpan.end = idx + 1;
  } else {
    // Prepend start
    state.selectedSpan.start = idx;
  }
  updateSelectionDisplay();
  renderHumanTokenBoard();
}

function isTokenSelected(idx) {
  if (!state.selectedSpan) return false;
  return idx >= state.selectedSpan.start && idx < state.selectedSpan.end;
}

function updateSelectionDisplay() {
  if (!state.selectedSpan || !state.currentCandidate) {
    elements.selectionInfo.textContent = "No token selected. Click any token to begin span.";
    return;
  }
  const toks = state.currentCandidate.tokens.slice(state.selectedSpan.start, state.selectedSpan.end);
  elements.selectionInfo.textContent = `Selected [${state.selectedSpan.start}:${state.selectedSpan.end}]: "${toks.join(" ")}" (${toks.length} token${toks.length > 1 ? "s" : ""})`;
}

// Apply Entity Type to Selected Span
function applyEntityTypeToSelection(entType) {
  if (!state.selectedSpan || !state.currentCandidate) {
    showToast("Please select one or more tokens first.", "info");
    return;
  }

  const { start, end } = state.selectedSpan;
  const toks = state.currentCandidate.tokens;

  if (entType === "CLEAR") {
    for (let i = start; i < end; i++) {
      state.currentHumanTags[i] = "O";
    }
  } else {
    state.currentHumanTags[start] = `B-${entType}`;
    for (let i = start + 1; i < end; i++) {
      state.currentHumanTags[i] = `I-${entType}`;
    }
  }

  // Ensure trailing I-tags after span are cleaned if orphan
  cleanOrphanTags();

  // Reset selection
  state.selectedSpan = null;
  updateSelectionDisplay();
  renderHumanTokenBoard();
  updateBioStream();
}

function cleanOrphanTags() {
  const tags = state.currentHumanTags;
  let prevType = null;
  for (let i = 0; i < tags.length; i++) {
    const t = tags[i];
    if (t.startsWith("B-")) {
      prevType = t.substring(2);
    } else if (t.startsWith("I-")) {
      const itype = t.substring(2);
      if (prevType !== itype) {
        tags[i] = `B-${itype}`;
        prevType = itype;
      }
    } else {
      prevType = null;
    }
  }
}

function updateBioStream() {
  elements.bioTagsStream.innerHTML = "";
  state.currentHumanTags.forEach((t) => {
    const span = document.createElement("span");
    span.className = "bio-tag-chip";
    span.textContent = t;
    elements.bioTagsStream.appendChild(span);
  });
}

// Handle Decision (ACCEPT, EDIT, REJECT, UNCERTAIN)
async function handleDecision(decision) {
  if (!state.annotatorId) {
    elements.annotatorModal.style.display = "flex";
    showToast("Please enter your Annotator ID before saving.", "error");
    return;
  }

  if (!state.currentCandidate || state.isSaving) return;
  state.isSaving = true;

  const notes = elements.annotationNotes.value.trim();
  const payload = {
    sentence_id: state.currentCandidate.sentence_id,
    annotator_id: state.annotatorId,
    decision: decision,
    human_tags: state.currentHumanTags,
    notes: notes
  };

  try {
    const res = await fetch("/api/save", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    if (!res.ok) {
      const errData = await res.json();
      throw new Error(errData.detail || "Failed to save annotation");
    }

    const data = await res.json();
    showToast(`Saved as ${data.candidate.human_status}`, "success");
    updateSidebarStats(data.stats);

    // Advance to next sentence
    navigateRelative(1);
  } catch (err) {
    showToast(err.message, "error");
  } finally {
    state.isSaving = false;
  }
}

// Relative Navigation
function navigateRelative(delta) {
  const nextIdx = state.currentIndex + delta;
  if (nextIdx < 0) {
    showToast("At first sentence.", "info");
    return;
  }
  loadCandidateByIndex(nextIdx);
}

// Load Live Stats
async function loadStats() {
  try {
    const res = await fetch("/api/stats");
    if (res.ok) {
      const stats = await res.json();
      updateSidebarStats(stats);
    }
  } catch (err) {
    console.error("Failed to load stats", err);
  }
}

function updateSidebarStats(stats) {
  if (!stats) return;

  const total = stats.total_candidates || 1367;
  const reviewed = stats.reviewed || 0;
  const pct = total > 0 ? ((reviewed / total) * 100).toFixed(1) : 0;

  elements.progressBarFill.style.width = `${pct}%`;
  elements.progressText.textContent = `${reviewed} / ${total} Reviewed (${pct}%)`;

  elements.statVerified.textContent = stats.human_verified || 0;
  elements.statUncertain.textContent = stats.uncertain || 0;
  elements.statRejected.textContent = stats.rejected || 0;
  elements.statUnreviewed.textContent = stats.unreviewed || (total - reviewed);

  const ents = stats.entities || {};
  elements.countPER.textContent = ents.PER || 0;
  elements.countLOC.textContent = ents.LOC || 0;
  elements.countORG.textContent = ents.ORG || 0;
  elements.countMISC.textContent = ents.MISC || 0;
  elements.countTotalEnt.textContent = ents.total || 0;
  elements.countMultiToken.textContent = ents.multi_token || 0;
}

// Handle Export
async function handleExport() {
  try {
    const res = await fetch("/api/export", { method: "POST" });
    if (res.ok) {
      const data = await res.json();
      showToast("Gold CoNLL & Provenance exported successfully!", "success");
      updateSidebarStats(data.stats);
    } else {
      throw new Error("Export failed.");
    }
  } catch (err) {
    showToast(err.message, "error");
  }
}

// Toast Notifications
function showToast(message, type = "info") {
  const toast = document.createElement("div");
  toast.className = `toast ${type}`;
  toast.textContent = message;
  elements.toastContainer.appendChild(toast);

  setTimeout(() => {
    toast.remove();
  }, 3500);
}
