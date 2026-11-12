/**
 * Esquema interactivo de la Numark Mixtrack Platinum FX.
 * Construye el layout de cada deck (EQ 3 bandas, Filter, 6 palancas FX,
 * 8 Performance Pads, Pitch Fader) + Crossfader central, y permite resaltar
 * qué control se acciona en cada paso de una transición.
 */

const FX_UNITS = ["HPF", "LPF", "Flanger", "Echo", "Reverb", "Phaser"];
const PAD_LABELS = ["HotCue1", "HotCue2", "HotCue3", "HotCue4", "LoopRoll", "AutoLoop", "FaderCut", "Sample"];

function buildDeckHTML(deckName) {
  const eq = ["High", "Mid", "Low"]
    .map((band) => `<div class="control-chip" data-chip="EQ:${band}:${deckName}">EQ ${band}</div>`)
    .join("");

  const fx = FX_UNITS
    .map((unit) => `<div class="control-chip" data-chip="FX:${unit}:${deckName}">${unit}</div>`)
    .join("");

  const pads = PAD_LABELS
    .map((pad) => `<div class="control-chip" data-chip="Pad:${pad}:${deckName}">${pad}</div>`)
    .join("");

  return `
    <div class="numark-deck" data-deck="${deckName}">
      <h4>Deck ${deckName}</h4>
      <div class="eq-row">${eq}
        <div class="control-chip" data-chip="Filter:${deckName}">Filter</div>
      </div>
      <div class="fx-row">${fx}</div>
      <div class="pads-row">${pads}</div>
      <div class="control-chip" data-chip="PitchFader:${deckName}">Pitch Fader</div>
    </div>`;
}

function renderSchematic(containerEl) {
  containerEl.innerHTML =
    buildDeckHTML("A") +
    buildDeckHTML("B") +
    `<div class="crossfader-row">
       <div class="control-chip" data-chip="Crossfader:both" style="width:120px;">Crossfader</div>
       <div class="crossfader-track"><div class="crossfader-knob" id="crossfaderKnob" style="left:50%;"></div></div>
     </div>`;
}

function resetSchematic(containerEl) {
  containerEl.querySelectorAll(".control-chip").forEach((el) => el.classList.remove("active"));
  const knob = containerEl.querySelector("#crossfaderKnob");
  if (knob) knob.style.left = "50%";
}

/**
 * Dado el `control` y `deck` de un ControlAction del backend, resuelve la
 * clave de chip (`data-chip`) a resaltar en el esquema.
 */
function controlActionToChipKey(control, deck) {
  if (control === "Crossfader") return "Crossfader:both";
  if (control === "PitchFader") return `PitchFader:${deck}`;
  if (control === "Filter") return `Filter:${deck}`;
  if (control.startsWith("EQ:")) return `${control}:${deck}`;
  if (control.startsWith("FX:")) return `${control}:${deck}`;
  if (control.startsWith("Pad:")) {
    const name = control.split(":")[1];
    // Mapea nombres lógicos del backend a los pads físicos mostrados
    const alias = { LoopRoll: "LoopRoll", HotCue: "HotCue1" };
    return `Pad:${alias[name] || name}:${deck}`;
  }
  return null;
}

function highlightAction(containerEl, action) {
  const key = controlActionToChipKey(action.control, action.deck);
  if (!key) return;
  const el = containerEl.querySelector(`[data-chip="${key}"]`);
  if (el) {
    el.classList.add("active");
    el.title = `${action.t_mmss} — ${action.action} (${action.value})`;
  }
  if (action.control === "Crossfader") {
    const knob = containerEl.querySelector("#crossfaderKnob");
    if (knob && /B100|->\s*B/.test(action.value)) knob.style.left = "85%";
    else if (knob && /A100|->\s*A/.test(action.value)) knob.style.left = "15%";
  }
}

function applyActionsToSchematic(containerEl, actions) {
  resetSchematic(containerEl);
  actions.forEach((action) => highlightAction(containerEl, action));
}
