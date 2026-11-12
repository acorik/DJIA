"""Orquesta el análisis completo de una pista (DSP + estructura) con caché por hash."""
from __future__ import annotations
import dataclasses
import os

import librosa

from . import config, cache_db
from .audio_analysis import analyze_track, hash_audio_file, TrackAnalysis
from .structure import classify_sections, Section


def _section_to_dict(s: Section) -> dict:
    return {
        "label": s.label,
        "start_sec": round(s.start_sec, 3),
        "end_sec": round(s.end_sec, 3),
        "start_mmss": s.start_mmss,
        "end_mmss": s.end_mmss,
        "start_bar": s.start_bar,
        "end_bar": s.end_bar,
        "energy_1_10": s.energy_1_10,
    }


def _dict_to_section(d: dict) -> Section:
    return Section(
        label=d["label"], start_sec=d["start_sec"], end_sec=d["end_sec"],
        start_bar=d["start_bar"], end_bar=d["end_bar"], energy_1_10=d["energy_1_10"],
    )


def analyze_with_cache(file_path: str) -> dict:
    """Devuelve el análisis completo (dict serializable) de un archivo, usando
    caché en disco si ya se analizó este contenido exacto antes."""
    file_hash = hash_audio_file(file_path)
    cached = cache_db.get_cached(file_hash)
    if cached is not None:
        cached["file_path"] = file_path  # la ruta puede cambiar aunque el audio sea el mismo
        return cached

    analysis: TrackAnalysis = analyze_track(file_path)

    y, sr = librosa.load(file_path, sr=config.SAMPLE_RATE, mono=True)
    sections = classify_sections(y, sr, analysis.duration_sec, analysis.beat_times)

    payload = {
        "file_path": file_path,
        "file_hash": analysis.file_hash,
        "title_guess": os.path.splitext(os.path.basename(file_path))[0],
        "duration_sec": analysis.duration_sec,
        "bpm": analysis.bpm,
        "bpm_stability_0_1": analysis.bpm_stability_0_1,
        "camelot_code": analysis.camelot_code,
        "camelot_name": analysis.camelot_name,
        "key_confidence_0_1": analysis.key_confidence_0_1,
        "energy": dataclasses.asdict(analysis.energy),
        "sections": [_section_to_dict(s) for s in sections],
    }
    cache_db.put_cached(file_hash, file_path, payload)
    return payload


def scan_folder(folder_path: str) -> list[dict]:
    results = []
    for root, _dirs, files in os.walk(folder_path):
        for name in sorted(files):
            if name.lower().endswith(config.SUPPORTED_EXTENSIONS):
                full_path = os.path.join(root, name)
                try:
                    results.append(analyze_with_cache(full_path))
                except Exception as exc:  # noqa: BLE001 - un archivo corrupto no debe tirar todo el escaneo
                    results.append({"file_path": full_path, "error": str(exc)})
    return results
