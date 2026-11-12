"""Segmentación estructural real de la pista: Intro / Verso / Buildup / Drop /
Breakdown / Outro, en mm:ss y en compases, derivada de cambios reales de
energía, contenido espectral y densidad percusiva — nunca de metadatos.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import librosa

from . import config

SectionLabel = str  # "intro" | "verse" | "buildup" | "drop" | "breakdown" | "outro"


@dataclass
class Section:
    label: SectionLabel
    start_sec: float
    end_sec: float
    start_bar: int
    end_bar: int
    energy_1_10: float

    @property
    def start_mmss(self) -> str:
        return _fmt_mmss(self.start_sec)

    @property
    def end_mmss(self) -> str:
        return _fmt_mmss(self.end_sec)


def _fmt_mmss(seconds: float) -> str:
    m, s = divmod(max(0.0, seconds), 60)
    return f"{int(m):02d}:{s:05.2f}"


def _bar_for_time(t: float, beat_times: np.ndarray, beats_per_bar: int = 4) -> int:
    if len(beat_times) == 0:
        return 0
    beat_idx = int(np.searchsorted(beat_times, t))
    return max(0, beat_idx // beats_per_bar)


def _segment_boundaries(y: np.ndarray, sr: int, duration_sec: float) -> np.ndarray:
    """Detecta fronteras estructurales combinando timbre (MFCC) y armonía (chroma)
    con clustering jerárquico agglomerativo sobre una matriz de auto-similitud.
    """
    hop = config.HOP_LENGTH
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
    features = np.vstack([mfcc, chroma])

    # Nº de segmentos esperado: asumimos secciones de ~16-24s de media en música de club.
    n_segments = max(4, min(14, int(duration_sec // 18)))
    boundary_frames = librosa.segment.agglomerative(features, n_segments)
    boundary_times = librosa.frames_to_time(boundary_frames, sr=sr, hop_length=hop)
    boundary_times = np.unique(np.concatenate([[0.0], boundary_times, [duration_sec]]))
    return boundary_times


def _segment_energy_stats(
    energy_curve: np.ndarray, onset_env: np.ndarray, sub_bass_curve: np.ndarray,
    frame_times: np.ndarray, t0: float, t1: float,
) -> dict:
    mask = (frame_times >= t0) & (frame_times < t1)
    if not np.any(mask):
        mask = frame_times >= t0
    seg_energy = energy_curve[mask]
    seg_onset = onset_env[mask]
    seg_bass = sub_bass_curve[mask]

    mean_energy = float(np.mean(seg_energy)) if seg_energy.size else 0.0
    mean_onset = float(np.mean(seg_onset)) if seg_onset.size else 0.0
    mean_bass = float(np.mean(seg_bass)) if seg_bass.size else 0.0

    if seg_energy.size >= 4:
        first_half = np.mean(seg_energy[: len(seg_energy) // 2])
        second_half = np.mean(seg_energy[len(seg_energy) // 2:])
        trend = second_half - first_half
    else:
        trend = 0.0

    return {"energy": mean_energy, "onset": mean_onset, "bass": mean_bass, "trend": trend}


def classify_sections(
    y: np.ndarray, sr: int, duration_sec: float, beat_times: list[float],
) -> list[Section]:
    hop = config.HOP_LENGTH
    boundaries = _segment_boundaries(y, sr, duration_sec)

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    energy_curve = rms / (rms.max() + 1e-9)

    S = np.abs(librosa.stft(y, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=sr)
    sub_mask = (freqs >= 20) & (freqs <= 120)
    sub_bass_curve = S[sub_mask, :].mean(axis=0)
    sub_bass_curve = sub_bass_curve / (sub_bass_curve.max() + 1e-9)

    n = min(len(onset_env), len(energy_curve), len(sub_bass_curve))
    onset_env, energy_curve, sub_bass_curve = onset_env[:n], energy_curve[:n], sub_bass_curve[:n]
    frame_times = librosa.frames_to_time(np.arange(n), sr=sr, hop_length=hop)

    beat_times_arr = np.array(beat_times)
    global_energy_p75 = float(np.percentile(energy_curve, 75))
    global_energy_p40 = float(np.percentile(energy_curve, 40))
    global_onset_p60 = float(np.percentile(onset_env, 60))

    raw_segments = []
    for i in range(len(boundaries) - 1):
        t0, t1 = float(boundaries[i]), float(boundaries[i + 1])
        if t1 - t0 < 2.0:
            continue
        stats = _segment_energy_stats(energy_curve, onset_env, sub_bass_curve, frame_times, t0, t1)
        raw_segments.append((t0, t1, stats))

    sections: list[Section] = []
    for idx, (t0, t1, stats) in enumerate(raw_segments):
        is_first = idx == 0
        is_last = idx == len(raw_segments) - 1
        high_energy = stats["energy"] >= global_energy_p75 and stats["onset"] >= global_onset_p60
        low_energy = stats["energy"] <= global_energy_p40

        if is_first and low_energy:
            label: SectionLabel = "intro"
        elif is_last and low_energy:
            label = "outro"
        elif high_energy:
            label = "drop"
        elif stats["trend"] > 0.08 and stats["energy"] < global_energy_p75:
            label = "buildup"
        elif low_energy and 0 < idx < len(raw_segments) - 1:
            prev_high = raw_segments[idx - 1][2]["energy"] >= global_energy_p75
            label = "breakdown" if prev_high else "verse"
        else:
            label = "verse"

        energy_1_10 = round(1.0 + 9.0 * float(np.clip(stats["energy"], 0.0, 1.0)), 2)
        sections.append(
            Section(
                label=label,
                start_sec=t0,
                end_sec=t1,
                start_bar=_bar_for_time(t0, beat_times_arr),
                end_bar=_bar_for_time(t1, beat_times_arr),
                energy_1_10=energy_1_10,
            )
        )

    return sections
