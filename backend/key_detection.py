"""Detección de tonalidad musical real por análisis cromático (Krumhansl-Schmuckler).

Prohibido usar metadatos: esto trabaja únicamente sobre el croma extraído de la señal
de audio real (CQT de alta resolución), correlando contra los perfiles tonales clásicos
de Krumhansl & Kessler para mayor y menor, en las 12 transposiciones posibles.
"""
from __future__ import annotations
import numpy as np
import librosa

from .camelot import key_to_camelot, CamelotKey

# Perfiles tonales de Krumhansl-Kessler (correlación de estabilidad por nota de la escala).
_MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def detect_key(y: np.ndarray, sr: int) -> tuple[CamelotKey, float]:
    """Devuelve (CamelotKey, confianza 0..1) a partir de la forma de onda real.

    Usa un Constant-Q Chromagram (más preciso tonalmente que un chroma STFT plano)
    y promedia en el tiempo ponderando por energía para que pasajes de silencio o
    ruido percusivo puro no distorsionen la detección.
    """
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=512)
    rms = librosa.feature.rms(y=y, hop_length=512)[0]
    weights = rms / (rms.sum() + 1e-9)
    chroma_mean = (chroma * weights[np.newaxis, :]).sum(axis=1)
    chroma_mean = chroma_mean / (np.linalg.norm(chroma_mean) + 1e-9)

    best_score = -np.inf
    best_pc = 0
    best_mode = "major"
    scores: list[float] = []

    for pc in range(12):
        major_rot = np.roll(_MAJOR_PROFILE, pc)
        minor_rot = np.roll(_MINOR_PROFILE, pc)
        major_rot_n = major_rot / np.linalg.norm(major_rot)
        minor_rot_n = minor_rot / np.linalg.norm(minor_rot)

        score_major = float(np.dot(chroma_mean, major_rot_n))
        score_minor = float(np.dot(chroma_mean, minor_rot_n))
        scores.extend([score_major, score_minor])

        if score_major > best_score:
            best_score, best_pc, best_mode = score_major, pc, "major"
        if score_minor > best_score:
            best_score, best_pc, best_mode = score_minor, pc, "minor"

    scores_arr = np.array(scores)
    score_range = scores_arr.max() - scores_arr.min() + 1e-9
    confidence = float((best_score - scores_arr.mean()) / score_range)
    confidence = max(0.0, min(1.0, confidence * 2.0))

    return key_to_camelot(best_pc, best_mode), confidence
