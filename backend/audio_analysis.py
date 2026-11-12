"""Análisis de señal de audio puro: BPM, tonalidad, energía, timbre.

Regla crítica: nada aquí lee nombres de archivo, tags ID3 ni metadatos. Todo se
deriva de las muestras de audio decodificadas. El espectrograma mel pesado se
calcula en GPU (torchaudio/CUDA) cuando hay una RTX disponible, acelerando el
procesado de carpetas completas; el resto (beat tracking, chroma, MFCC) corre
en CPU vía librosa, que ya es suficientemente rápido para estas tareas.
"""
from __future__ import annotations
import hashlib
from dataclasses import dataclass, field

import numpy as np
import librosa

from . import config
from .gpu import get_device, torch_module
from .key_detection import detect_key
from .camelot import CamelotKey


@dataclass
class EnergyProfile:
    overall_1_10: float
    sub_bass_1_10: float      # peso de graves < 100 Hz (oscuro/pesado)
    percussive_1_10: float    # densidad percusiva (contenido transitorio)
    melodic_1_10: float       # contenido armónico/vocal sostenido
    brightness_1_10: float    # centroide espectral normalizado


@dataclass
class TrackAnalysis:
    file_path: str
    file_hash: str
    duration_sec: float
    bpm: float
    bpm_stability_0_1: float          # 1.0 = tempo perfectamente estable
    camelot_code: str
    camelot_name: str
    key_confidence_0_1: float
    energy: EnergyProfile
    energy_curve: list[float] = field(default_factory=list)   # RMS normalizado por frame (para estructura)
    beat_times: list[float] = field(default_factory=list)
    sr: int = config.SAMPLE_RATE


def hash_audio_file(file_path: str, chunk_size: int = 1 << 20) -> str:
    """Hash estable del contenido (no del nombre) para cachear el análisis."""
    h = hashlib.sha1()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def _compute_mel_spectrogram_gpu(y: np.ndarray, sr: int) -> np.ndarray:
    """Mel-espectrograma acelerado por GPU (CUDA) cuando está disponible; cae a CPU si no."""
    device = get_device()
    if device != "cuda":
        return librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=config.HOP_LENGTH, n_mels=128)

    torch = torch_module()
    import torchaudio.transforms as T

    waveform = torch.from_numpy(y).float().unsqueeze(0).to(device)
    mel_transform = T.MelSpectrogram(
        sample_rate=sr, n_fft=2048, hop_length=config.HOP_LENGTH, n_mels=128,
    ).to(device)
    mel = mel_transform(waveform)
    return mel.squeeze(0).cpu().numpy()


def _bpm_and_stability(y: np.ndarray, sr: int) -> tuple[float, float, np.ndarray]:
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=config.HOP_LENGTH)
    tempo, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_env, sr=sr, hop_length=config.HOP_LENGTH, trim=False,
    )
    bpm = float(tempo if np.ndim(tempo) == 0 else tempo[0])
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=config.HOP_LENGTH)

    if len(beat_times) > 2:
        intervals = np.diff(beat_times)
        expected = 60.0 / bpm
        rel_dev = np.abs(intervals - expected) / expected
        stability = float(np.clip(1.0 - rel_dev.mean() * 4.0, 0.0, 1.0))
    else:
        stability = 0.5

    return bpm, stability, beat_times


def _energy_profile(y: np.ndarray, sr: int, mel_spec: np.ndarray) -> tuple[EnergyProfile, np.ndarray]:
    rms = librosa.feature.rms(y=y, hop_length=config.HOP_LENGTH)[0]
    rms_norm = rms / (rms.max() + 1e-9)

    S = np.abs(librosa.stft(y, hop_length=config.HOP_LENGTH))
    freqs = librosa.fft_frequencies(sr=sr)

    sub_band = S[(freqs >= 20) & (freqs <= 100), :]
    sub_energy = float(np.mean(sub_band)) if sub_band.size else 0.0
    total_energy = float(np.mean(S)) + 1e-9
    sub_bass_ratio = sub_energy / total_energy

    y_harm, y_perc = librosa.effects.hpss(y)
    perc_energy = float(np.mean(np.abs(y_perc)))
    harm_energy = float(np.mean(np.abs(y_harm)))
    perc_ratio = perc_energy / (perc_energy + harm_energy + 1e-9)
    melodic_ratio = 1.0 - perc_ratio

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=config.HOP_LENGTH)[0]
    brightness = float(np.clip(np.mean(centroid) / (sr / 2), 0.0, 1.0))

    overall = float(np.clip(np.mean(rms_norm) * 1.8, 0.0, 1.0))

    def to_scale(x: float) -> float:
        return round(1.0 + 9.0 * np.clip(x, 0.0, 1.0), 2)

    profile = EnergyProfile(
        overall_1_10=to_scale(overall),
        sub_bass_1_10=to_scale(sub_bass_ratio * 3.0),
        percussive_1_10=to_scale(perc_ratio),
        melodic_1_10=to_scale(melodic_ratio),
        brightness_1_10=to_scale(brightness),
    )
    return profile, rms_norm


def analyze_track(file_path: str) -> TrackAnalysis:
    """Analiza un archivo de audio desde cero, solo con la señal real."""
    file_hash = hash_audio_file(file_path)
    y, sr = librosa.load(file_path, sr=config.SAMPLE_RATE, mono=True)
    duration_sec = float(len(y) / sr)

    bpm, stability, beat_times = _bpm_and_stability(y, sr)
    camelot_key, key_conf = detect_key(y, sr)
    mel_spec = _compute_mel_spectrogram_gpu(y, sr)
    energy_profile, energy_curve = _energy_profile(y, sr, mel_spec)

    return TrackAnalysis(
        file_path=file_path,
        file_hash=file_hash,
        duration_sec=duration_sec,
        bpm=round(bpm, 2),
        bpm_stability_0_1=round(stability, 3),
        camelot_code=camelot_key.code,
        camelot_name=camelot_key.name,
        key_confidence_0_1=round(key_conf, 3),
        energy=energy_profile,
        energy_curve=[round(float(v), 4) for v in energy_curve],
        beat_times=[round(float(t), 3) for t in beat_times],
        sr=sr,
    )


def camelot_key_from_analysis(analysis: TrackAnalysis) -> CamelotKey:
    from .camelot import camelot_key_from_code
    return camelot_key_from_code(analysis.camelot_code)
