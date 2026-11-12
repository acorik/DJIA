const statusEl = document.getElementById("status");
const tracklistBody = document.querySelector("#tracklistTable tbody");
const transitionMeta = document.getElementById("transitionMeta");
const timelineA = document.getElementById("timelineA");
const timelineB = document.getElementById("timelineB");
const stepsList = document.getElementById("stepsList");
const schematicEl = document.getElementById("numarkSchematic");

renderSchematic(schematicEl);

let currentPlan = null; // respuesta completa de /api/plan

function setStatus(msg, kind) {
  statusEl.textContent = msg;
  statusEl.className = "status" + (kind ? " " + kind : "");
}

function relationClass(relation) {
  if (!relation) return "";
  if (["misma_clave", "relativa_mayor_menor"].includes(relation)) return "relation-good";
  if (relation.startsWith("vecino")) return "relation-good";
  if (relation.startsWith("energy_boost")) return "relation-warn";
  return "relation-bad";
}

async function postJSON(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || "Error de servidor");
  return data;
}

document.getElementById("analyzeBtn").addEventListener("click", async () => {
  const folder_path = document.getElementById("folderPath").value.trim();
  if (!folder_path) return setStatus("Indica una carpeta.", "error");
  setStatus("Analizando señal de audio real (BPM, tonalidad, estructura)...");
  try {
    const data = await postJSON("/api/analyze", { folder_path });
    setStatus(`Analizadas ${data.tracks.filter(t => !t.error).length} pistas. Ya puedes generar el set.`, "ok");
  } catch (e) {
    setStatus(e.message, "error");
  }
});

document.getElementById("planBtn").addEventListener("click", async () => {
  const folder_path = document.getElementById("folderPath").value.trim();
  const style_prompt = document.getElementById("stylePrompt").value.trim();
  if (!folder_path) return setStatus("Indica una carpeta.", "error");
  if (!style_prompt) return setStatus("Describe la vibra que quieres para el set.", "error");

  setStatus("Calculando el orden óptimo y diseñando las transiciones...");
  try {
    currentPlan = await postJSON("/api/plan", { folder_path, style_prompt });
    renderTracklist(currentPlan);
    setStatus(`Set generado: ${currentPlan.tracklist.length} pistas, ${currentPlan.transitions.length} transiciones.`, "ok");
    if (currentPlan.transitions.length) selectTransition(0);
  } catch (e) {
    setStatus(e.message, "error");
  }
});

function titleFor(filePath, tracksByPath) {
  const t = tracksByPath[filePath];
  return t ? t.title_guess : filePath.split("/").pop();
}

function renderTracklist(plan) {
  tracklistBody.innerHTML = "";
  plan.tracklist.forEach((row, i) => {
    const tr = document.createElement("tr");
    const relation = row.camelot_relation_to_prev;
    const quality = row.transition_quality_0_1;

    tr.innerHTML = `
      <td>${i + 1}</td>
      <td>${titleFor(row.file_path, plan.tracks_by_path)}</td>
      <td>${row.bpm.toFixed(1)}</td>
      <td>${row.pitch_adjust_percent >= 0 ? "+" : ""}${row.pitch_adjust_percent.toFixed(1)}%</td>
      <td><span class="camelot-badge">${row.camelot_code}</span></td>
      <td class="${relationClass(relation)}">${relation ? relation.replace(/_/g, " ") : "—"}</td>
      <td>${row.energy_1_10.toFixed(1)}</td>
      <td>${row.target_energy_1_10.toFixed(1)}</td>
      <td>${quality !== null && quality !== undefined ? (quality * 100).toFixed(0) + "%" : "—"}</td>
    `;

    if (i > 0) {
      tr.classList.add("transition-row");
      tr.dataset.transitionIndex = i - 1;
      tr.addEventListener("click", () => selectTransition(i - 1));
    }
    tracklistBody.appendChild(tr);
  });
}

function secToX(sec, totalSec) {
  return Math.max(0, Math.min(100, (sec / totalSec) * 100));
}

function renderTimeline(container, totalSec, markers) {
  container.innerHTML = "";
  markers.forEach(({ sec, label }) => {
    const pct = secToX(sec, totalSec);
    const marker = document.createElement("div");
    marker.className = "marker";
    marker.style.left = pct + "%";
    const labelEl = document.createElement("div");
    labelEl.className = "marker-label";
    labelEl.style.left = pct + "%";
    labelEl.textContent = label;
    marker.appendChild(labelEl);
    container.appendChild(marker);
  });
}

function selectTransition(idx) {
  if (!currentPlan) return;
  document.querySelectorAll("#tracklistTable tbody tr").forEach((tr) => tr.classList.remove("selected"));
  const activeRow = document.querySelector(`tr[data-transition-index="${idx}"]`);
  if (activeRow) activeRow.classList.add("selected");

  const transition = currentPlan.transitions[idx];
  const fromTrack = currentPlan.tracks_by_path[transition.from_track];
  const toTrack = currentPlan.tracks_by_path[transition.to_track];

  transitionMeta.innerHTML = `
    <strong>${transition.transition_type.replace(/_/g, " ")}</strong> —
    ${transition.description}<br/>
    Pitch necesario: ${transition.pitch_adjust_percent >= 0 ? "+" : ""}${transition.pitch_adjust_percent.toFixed(1)}%
    · Relación Camelot: ${transition.camelot_relation.replace(/_/g, " ")}
    · Solape aprox.: ${transition.overlap_bars} compases`;

  renderTimeline(timelineA, fromTrack.duration_sec, [
    { sec: parseMMSS(transition.cue_out_mmss), label: `Cue Out ${transition.cue_out_mmss}` },
  ]);
  renderTimeline(timelineB, toTrack.duration_sec, [
    { sec: parseMMSS(transition.cue_in_mmss), label: `Cue In ${transition.cue_in_mmss}` },
  ]);

  stepsList.innerHTML = "";
  transition.actions.forEach((action) => {
    const li = document.createElement("li");
    li.innerHTML = `<span class="step-time">${action.t_mmss}</span>
      <span class="step-deck">[Deck ${action.deck}]</span>
      ${action.action} — <em>${action.control}</em> ${action.value ? `(${action.value})` : ""}`;
    stepsList.appendChild(li);
  });

  applyActionsToSchematic(schematicEl, transition.actions);
}

function parseMMSS(mmss) {
  const [m, s] = mmss.split(":").map(Number);
  return m * 60 + s;
}
