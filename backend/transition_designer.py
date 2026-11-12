"""Diseña transiciones creativas y las traduce a controles físicos EXACTOS de la
Numark Mixtrack Platinum FX: las 6 palancas/paddles de FX centrales (HPF, LPF,
Flanger, Echo, Reverb, Phaser) con Paddle Trigger + Dry/Wet + Beat Multiplier,
EQ de 3 bandas, Filter, los 8 Performance Pads por deck, Pitch Faders y Crossfader.
"""
from __future__ import annotations
from dataclasses import dataclass, field

from .llm_brief import SetBrief

FX_UNITS = ("HPF", "LPF", "Flanger", "Echo", "Reverb", "Phaser")
BEAT_MULTIPLIERS = (1 / 4, 1 / 2, 1, 2, 4)


@dataclass
class ControlAction:
    t_sec: float
    deck: str              # "A" (saliente) | "B" (entrante)
    control: str           # "FX:Echo", "EQ:Low", "Filter", "Pad:LoopRoll", "Crossfader", "PitchFader"
    action: str             # descripción corta de lo que hay que hacer
    value: str = ""          # valor concreto (dry/wet %, beat multiplier, posición EQ...)

    @property
    def t_mmss(self) -> str:
        m, s = divmod(max(0.0, self.t_sec), 60)
        return f"{int(m):02d}:{s:05.2f}"


@dataclass
class TransitionPlan:
    from_track: str
    to_track: str
    transition_type: str
    cue_out_mmss: str
    cue_in_mmss: str
    overlap_bars: int
    pitch_adjust_percent: float
    camelot_relation: str
    description: str
    actions: list[ControlAction] = field(default_factory=list)


def _seconds_per_bar(bpm: float) -> float:
    return 60.0 / bpm * 4.0


def _find_section(sections: list[dict], label: str, prefer_last: bool = True) -> dict | None:
    matches = [s for s in sections if s["label"] == label]
    if not matches:
        return None
    return matches[-1] if prefer_last else matches[0]


def _cue_out_point(outgoing: dict) -> tuple[float, dict | None]:
    sections = outgoing["sections"]
    breakdown = _find_section(sections, "breakdown")
    outro = _find_section(sections, "outro")
    if breakdown:
        return breakdown["start_sec"], breakdown
    if outro:
        return outro["start_sec"], outro
    return max(0.0, outgoing["duration_sec"] - 32.0), None


def _cue_in_point(incoming: dict) -> tuple[float, dict | None]:
    sections = incoming["sections"]
    intro = _find_section(sections, "intro", prefer_last=False)
    if intro:
        return intro["end_sec"], intro
    return 0.0, None


def _choose_archetype(outgoing: dict, incoming: dict, relation: str, brief: SetBrief) -> str:
    out_sections = {s["label"] for s in outgoing["sections"]}
    in_sections = {s["label"] for s in incoming["sections"]}
    harmonic_ok = relation in ("misma_clave", "relativa_mayor_menor", "vecino_+1", "vecino_-1")
    both_have_drop = "drop" in out_sections and "drop" in in_sections

    if (
        brief.allow_double_drop and brief.fx_playfulness_0_1 >= 0.6
        and harmonic_ok and both_have_drop and brief.aggressiveness_0_1 >= 0.5
    ):
        return "loop_roll_double_drop"
    if outgoing["energy"]["melodic_1_10"] >= 6.0 and "breakdown" in out_sections and incoming["sections"]:
        return "wordplay_vocal_cross"
    if brief.aggressiveness_0_1 >= 0.65:
        return "bass_swap_eq_cut"
    return "echo_out_filter_sweep"


def _design_echo_out_filter_sweep(t_out: float, t_in: float, bpm_out: float, bpm_in: float) -> list[ControlAction]:
    spb = _seconds_per_bar(bpm_out)
    actions = [
        ControlAction(t_out - 8 * spb, "A", "FX:LPF", "Baja el Filter de LPF progresivamente",
                      "de 12 a 4 (barrido 8 compases)"),
        ControlAction(t_out - 4 * spb, "B", "EQ:Low", "Mid entrante con Low en mínimo (evitar choque de graves", "Low=0"),
        ControlAction(t_out - 2 * spb, "A", "FX:Echo", "Activa paddle Echo (momentáneo) sobre la salida",
                      "Beat Multiplier=1/2, Dry/Wet 0%->70%"),
        ControlAction(t_out, "both", "Crossfader", "Inicia barrido lento hacia B", "A100 -> A50"),
        ControlAction(t_in, "B", "EQ:Low", "Sube Low de B a posición neutra", "Low=12 (neutro)"),
        ControlAction(t_in + 2 * spb, "A", "FX:Echo", "Suelta paddle Echo, deja cola natural (lock OFF)", "Dry/Wet -> 0%"),
        ControlAction(t_in + 2 * spb, "both", "Crossfader", "Completa barrido a B", "A50 -> B100"),
        ControlAction(t_in + 2 * spb, "A", "EQ:Low", "Corta Low de A a 0 antes de retirarlo", "Low=0"),
    ]
    return actions


def _design_bass_swap_eq_cut(t_out: float, t_in: float, bpm_out: float, bpm_in: float) -> list[ControlAction]:
    spb = _seconds_per_bar(bpm_out)
    return [
        ControlAction(t_out - 4 * spb, "B", "EQ:Low", "Low de B en 0 (silencio de graves) mientras entra", "Low=0"),
        ControlAction(t_out - 2 * spb, "A", "FX:HPF", "Sube HPF cortando graves/medios de A", "HPF 2 compases, Dry/Wet 60%"),
        ControlAction(t_out, "A", "EQ:Low", "Bass Swap: corte seco de Low en A", "Low=0 (instantáneo)"),
        ControlAction(t_out, "B", "EQ:Low", "Bass Swap: entra Low de B a tope en el mismo golpe", "Low=12 (instantáneo)"),
        ControlAction(t_out, "both", "Crossfader", "Corte rápido de crossfader (menos de 1 compás)", "A100 -> B100"),
        ControlAction(t_out + 1 * spb, "A", "FX:HPF", "Suelta HPF de A", "Dry/Wet -> 0%"),
        ControlAction(t_in + 1 * spb, "B", "EQ:Mid", "Reintroduce Mid de B a nivel normal", "Mid=12"),
    ]


def _design_wordplay_vocal_cross(t_out: float, t_in: float, bpm_out: float, bpm_in: float) -> list[ControlAction]:
    spb = _seconds_per_bar(bpm_out)
    return [
        ControlAction(t_out, "A", "EQ:Low", "Baja Low de A a 0 en el breakdown (deja solo voz/melodía)", "Low=0"),
        ControlAction(t_out, "A", "FX:Reverb", "Activa Reverb en lock sobre la voz/melodía saliente",
                      "Beat Multiplier=1, Dry/Wet 40%"),
        ControlAction(t_out + 2 * spb, "B", "EQ:Low", "Entra Low de B suave por debajo de la voz de A", "Low=6 (rampa)"),
        ControlAction(t_out + 4 * spb, "both", "Crossfader", "Cruce lento A->B durante el wordplay", "A80 -> B50"),
        ControlAction(t_in, "A", "FX:Reverb", "Suelta Reverb de A, deja que se apague la cola", "Dry/Wet -> 0%"),
        ControlAction(t_in, "B", "EQ:Low", "Low de B a nivel completo, ya dominando la mezcla", "Low=12"),
        ControlAction(t_in + 2 * spb, "both", "Crossfader", "Cierra crossfader a B", "A50 -> B100"),
    ]


def _design_loop_roll_double_drop(t_out: float, t_in: float, bpm_out: float, bpm_in: float) -> list[ControlAction]:
    spb = _seconds_per_bar(bpm_out)
    drop_point = t_out  # el propio cue_out coincide con el breakdown/pre-drop detectado
    actions = [
        ControlAction(drop_point - 8 * spb, "B", "PitchFader", "Ajusta pitch fader de B para igualar BPM de A",
                      "ver pitch_adjust_percent"),
        ControlAction(drop_point - 4 * spb, "A", "Pad:LoopRoll", "Loop Roll 4 compases (cuenta atrás de tensión)", "4 bars"),
        ControlAction(drop_point - 2 * spb, "A", "Pad:LoopRoll", "Reduce Loop Roll", "2 bars"),
        ControlAction(drop_point - 1 * spb, "A", "Pad:LoopRoll", "Reduce Loop Roll", "1 bar"),
        ControlAction(drop_point - 0.5 * spb, "A", "Pad:LoopRoll", "Loop Roll final, máxima tensión", "1/2 bar"),
        ControlAction(drop_point, "B", "Pad:HotCue", "Dispara Hot Cue del Drop de B exactamente en el golpe", "Cue DROP"),
        ControlAction(drop_point, "both", "Crossfader", "Double Drop: sube B a tope en el golpe conservando A",
                      "A100+B100 simultáneo"),
        ControlAction(drop_point, "A", "FX:Flanger", "Flanger breve en lock sobre A para texturizar el choque", "Dry/Wet 50%, 1 beat"),
        ControlAction(drop_point + 1 * spb, "A", "FX:Flanger", "Suelta Flanger", "Dry/Wet -> 0%"),
        ControlAction(drop_point + 2 * spb, "A", "EQ:Low", "Retira Low de A progresivamente, deja que domine B", "Low 12 -> 0 en 2 compases"),
        ControlAction(drop_point + 2 * spb, "both", "Crossfader", "Cierra crossfader completamente a B", "-> B100"),
    ]
    return actions


_ARCHETYPE_BUILDERS = {
    "echo_out_filter_sweep": _design_echo_out_filter_sweep,
    "bass_swap_eq_cut": _design_bass_swap_eq_cut,
    "wordplay_vocal_cross": _design_wordplay_vocal_cross,
    "loop_roll_double_drop": _design_loop_roll_double_drop,
}

_ARCHETYPE_DESCRIPTIONS = {
    "echo_out_filter_sweep": "Salida con Echo Out + barrido de Filter (LPF), transición suave y profesional.",
    "bass_swap_eq_cut": "Bass Swap agresivo con corte seco de EQ y HPF, crossfader rápido.",
    "wordplay_vocal_cross": "Cruce de voz/melodía sobre el breakdown detectado (wordplay), con Reverb de cola.",
    "loop_roll_double_drop": "Loop Roll en cuenta atrás (4->2->1->1/2) hacia un Double Drop sincronizado.",
}


def design_transition(
    outgoing: dict, incoming: dict, camelot_relation: str, pitch_adjust_percent: float, brief: SetBrief,
) -> TransitionPlan:
    t_out, _out_section = _cue_out_point(outgoing)
    t_in, _in_section = _cue_in_point(incoming)

    archetype = _choose_archetype(outgoing, incoming, camelot_relation, brief)
    builder = _ARCHETYPE_BUILDERS[archetype]
    actions = builder(t_out, t_in, outgoing["bpm"], incoming["bpm"])
    actions.sort(key=lambda a: a.t_sec)

    spb_out = _seconds_per_bar(outgoing["bpm"])
    overlap_bars = max(1, round(abs(t_in - t_out) / spb_out)) if t_in and t_out else 8

    def fmt(t: float) -> str:
        m, s = divmod(max(0.0, t), 60)
        return f"{int(m):02d}:{s:05.2f}"

    return TransitionPlan(
        from_track=outgoing["file_path"],
        to_track=incoming["file_path"],
        transition_type=archetype,
        cue_out_mmss=fmt(t_out),
        cue_in_mmss=fmt(t_in),
        overlap_bars=overlap_bars,
        pitch_adjust_percent=pitch_adjust_percent,
        camelot_relation=camelot_relation,
        description=_ARCHETYPE_DESCRIPTIONS[archetype],
        actions=actions,
    )
