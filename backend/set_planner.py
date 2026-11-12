"""Motor de decisión: calcula el mejor orden posible de las pistas combinando
compatibilidad armónica (Rueda de Camelot), viabilidad de BPM (dentro del
rango de pitch físico de la Numark Mixtrack Platinum FX) y la curva de
energía que pide el brief de estilo (del LLM o del fallback heurístico).
"""
from __future__ import annotations
import math
from dataclasses import dataclass

from . import config
from .camelot import compatibility_score, camelot_distance, camelot_key_from_code
from .llm_brief import SetBrief


@dataclass
class PlannedTrack:
    index: int
    file_path: str
    title_guess: str
    bpm: float
    camelot_code: str
    energy_1_10: float
    target_energy_1_10: float
    pitch_adjust_percent: float          # cuánto habría que mover el pitch fader respecto al BPM original
    camelot_relation_to_prev: str | None
    transition_quality_0_1: float | None


def _bpm_compatibility(bpm_a: float, bpm_b: float, max_flex_percent: float) -> tuple[float, float]:
    """Compatibilidad de BPM considerando también relaciones de doble/mitad tempo
    (habitual entre géneros: 174 house <-> 87 breaks). Devuelve (score_0_1, pitch_adjust_percent)."""
    candidates = [bpm_b, bpm_b * 2.0, bpm_b / 2.0]
    best_score, best_adjust = 0.0, 0.0
    for cand in candidates:
        if cand <= 0:
            continue
        pct_diff = (cand - bpm_a) / bpm_a * 100.0
        if abs(pct_diff) <= max_flex_percent:
            score = 1.0 - (abs(pct_diff) / max_flex_percent) * 0.4
        else:
            score = max(0.0, 1.0 - (abs(pct_diff) / max_flex_percent))
        if score > best_score:
            best_score, best_adjust = score, pct_diff
    return best_score, best_adjust


def _target_energy_curve(brief: SetBrief, n: int) -> list[float]:
    if n <= 1:
        return [brief.opening_energy_1_10]
    lo, hi = brief.opening_energy_1_10, brief.closing_energy_1_10
    if brief.narrative == "ascendente":
        return [lo + (hi - lo) * i / (n - 1) for i in range(n)]
    if brief.narrative == "descendente":
        return [hi + (lo - hi) * i / (n - 1) for i in range(n)]
    if brief.narrative == "plano":
        avg = (lo + hi) / 2.0
        return [avg] * n
    # "ola": oscila entre lo y hi con un semiciclo completo, terminando arriba
    mid = (lo + hi) / 2.0
    amp = (hi - lo) / 2.0
    return [mid + amp * math.sin(math.pi * i / (n - 1) - math.pi / 2) for i in range(n)]


def _pairwise_score(a: dict, b: dict, brief: SetBrief) -> tuple[float, str, float]:
    key_a, key_b = camelot_key_from_code(a["camelot_code"]), camelot_key_from_code(b["camelot_code"])
    harm_score = compatibility_score(key_a, key_b)
    _, relation = camelot_distance(key_a, key_b)
    bpm_score, pitch_adjust = _bpm_compatibility(a["bpm"], b["bpm"], brief.bpm_flex_percent)
    combined = 0.6 * harm_score + 0.4 * bpm_score
    return combined, relation, pitch_adjust


def plan_order(tracks: list[dict], brief: SetBrief) -> list[PlannedTrack]:
    """Construye el orden óptimo: heurística greedy + mejora local 2-opt."""
    n = len(tracks)
    if n == 0:
        return []
    if n == 1:
        t = tracks[0]
        return [PlannedTrack(
            index=0, file_path=t["file_path"], title_guess=t.get("title_guess", ""),
            bpm=t["bpm"], camelot_code=t["camelot_code"], energy_1_10=t["energy"]["overall_1_10"],
            target_energy_1_10=brief.opening_energy_1_10, pitch_adjust_percent=0.0,
            camelot_relation_to_prev=None, transition_quality_0_1=None,
        )]

    target_curve = _target_energy_curve(brief, n)

    # Punto de partida: la pista cuya energía esté más cerca de la energía de apertura deseada.
    start_idx = min(range(n), key=lambda i: abs(tracks[i]["energy"]["overall_1_10"] - brief.opening_energy_1_10))

    remaining = set(range(n)) - {start_idx}
    order = [start_idx]

    while remaining:
        last = order[-1]
        pos = len(order)
        target_e = target_curve[pos] if pos < len(target_curve) else target_curve[-1]

        def candidate_score(i: int) -> float:
            pair_score, _, _ = _pairwise_score(tracks[last], tracks[i], brief)
            energy_penalty = abs(tracks[i]["energy"]["overall_1_10"] - target_e) / 9.0
            return pair_score - 0.35 * energy_penalty

        best = max(remaining, key=candidate_score)
        order.append(best)
        remaining.remove(best)

    # Mejora local 2-opt: intercambia pares de posiciones si reduce el coste total de transición.
    def total_cost(seq: list[int]) -> float:
        cost = 0.0
        for pos in range(len(seq) - 1):
            pair_score, _, _ = _pairwise_score(tracks[seq[pos]], tracks[seq[pos + 1]], brief)
            cost += 1.0 - pair_score
            target_e = target_curve[pos + 1] if pos + 1 < len(target_curve) else target_curve[-1]
            cost += 0.2 * abs(tracks[seq[pos + 1]]["energy"]["overall_1_10"] - target_e) / 9.0
        return cost

    improved = True
    passes = 0
    while improved and passes < 6:
        improved = False
        passes += 1
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                new_order = order[:i] + order[i:j + 1][::-1] + order[j + 1:]
                if total_cost(new_order) < total_cost(order) - 1e-6:
                    order = new_order
                    improved = True

    planned: list[PlannedTrack] = []
    for pos, idx in enumerate(order):
        t = tracks[idx]
        target_e = target_curve[pos] if pos < len(target_curve) else target_curve[-1]
        if pos == 0:
            relation, pitch_adjust, quality = None, 0.0, None
        else:
            prev = tracks[order[pos - 1]]
            quality, relation, pitch_adjust = _pairwise_score(prev, t, brief)
        planned.append(PlannedTrack(
            index=idx, file_path=t["file_path"], title_guess=t.get("title_guess", ""),
            bpm=t["bpm"], camelot_code=t["camelot_code"], energy_1_10=t["energy"]["overall_1_10"],
            target_energy_1_10=round(target_e, 2), pitch_adjust_percent=round(pitch_adjust, 2),
            camelot_relation_to_prev=relation, transition_quality_0_1=
            round(quality, 3) if quality is not None else None,
        ))
    return planned
