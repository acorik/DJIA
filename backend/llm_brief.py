"""Traduce el prompt de estilo en lenguaje natural a un "brief" estructurado
(JSON) que consume el motor de decisión determinista (set_planner / transition_designer).

Por qué un LLM solo para esto y no para decidir BPM/key/orden: el LLM nunca ve
el audio ni decide hechos verificables (eso es DSP puro y 100% fiable); solo
interpreta intención creativa en lenguaje natural ("empieza profundo, acaba
reventando, cortes agresivos") y la convierte en parámetros numéricos que el
algoritmo determinista usa para puntuar y ordenar. Así nunca hay alucinaciones
de BPM/tonalidad, pero sí se respeta el vibe que pide el usuario.

Corre 100% offline con llama-cpp-python sobre un GGUF local (Llama 3.1 8B
Instruct Q4_K_M u otro instruct de 7-8B cuantizado encaja holgado en 12 GB de
VRAM). Si no hay modelo descargado en LLM_MODEL_PATH, cae a un parser
heurístico por palabras clave en español/inglés, así la app funciona desde el
primer minuto sin depender de que el usuario ya tenga pesos descargados.
"""
from __future__ import annotations
import json
import os
import re
from dataclasses import dataclass, asdict
from functools import lru_cache

from . import config


@dataclass
class SetBrief:
    opening_energy_1_10: float = 4.0
    closing_energy_1_10: float = 9.0
    aggressiveness_0_1: float = 0.5       # cortes secos, bass swaps, double drops
    fx_playfulness_0_1: float = 0.5       # cuánto jugueteo con palancas FX / loop rolls
    bpm_flex_percent: float = 6.0         # cuánto pitch-bend asumible (límite Numark 8%)
    allow_double_drop: bool = True
    narrative: str = "ascendente"         # "ascendente" | "descendente" | "ola" | "plano"


_SYSTEM_PROMPT = """Eres el cerebro creativo de un DJ profesional. Traduces la petición de un \
usuario sobre el estilo de una sesión a un JSON ESTRICTO con estos campos y rangos:
{
  "opening_energy_1_10": float 1-10,
  "closing_energy_1_10": float 1-10,
  "aggressiveness_0_1": float 0-1,
  "fx_playfulness_0_1": float 0-1,
  "bpm_flex_percent": float 2-8,
  "allow_double_drop": bool,
  "narrative": "ascendente" | "descendente" | "ola" | "plano"
}
Responde ÚNICAMENTE con el JSON, sin texto adicional ni explicaciones."""


@lru_cache(maxsize=1)
def _load_llm():
    if not os.path.isfile(config.LLM_MODEL_PATH):
        return None
    from llama_cpp import Llama
    return Llama(
        model_path=config.LLM_MODEL_PATH,
        n_gpu_layers=config.LLM_N_GPU_LAYERS,
        n_ctx=config.LLM_CONTEXT_SIZE,
        verbose=False,
    )


def _heuristic_brief(prompt: str) -> SetBrief:
    """Fallback sin modelo descargado: parser de palabras clave, determinista y rápido."""
    text = prompt.lower()
    brief = SetBrief()

    if re.search(r"profund|deep|oscur|dark|minimal", text):
        brief.opening_energy_1_10 = 3.0
    if re.search(r"revent|banger|explot|peak\s*time|fuerte|duro|hard", text):
        brief.closing_energy_1_10 = 10.0
    if re.search(r"agresiv|cortes? seco|bass swap|aggressive", text):
        brief.aggressiveness_0_1 = 0.85
    if re.search(r"suave|smooth|fluid|melodic|chill|relaj", text):
        brief.aggressiveness_0_1 = min(brief.aggressiveness_0_1, 0.2)
    if re.search(r"jugueteo|fx|efectos?|flanger|phaser|echo|reverb|palancas?", text):
        brief.fx_playfulness_0_1 = 0.85
    if re.search(r"double\s*drop|dobles? drop", text):
        brief.allow_double_drop = True
    if re.search(r"sin double drop|no.*double drop", text):
        brief.allow_double_drop = False
    if re.search(r"ola|sube.*baja|wave|montaña rusa", text):
        brief.narrative = "ola"
    elif re.search(r"descend|bajando|cool ?down|cierre suave", text):
        brief.narrative = "descendente"
    elif re.search(r"plano|constante|steady|flat", text):
        brief.narrative = "plano"
    else:
        brief.narrative = "ascendente"

    return brief


def _clamp_brief(d: dict) -> SetBrief:
    def clamp(v, lo, hi, default):
        try:
            v = float(v)
        except (TypeError, ValueError):
            return default
        return max(lo, min(hi, v))

    return SetBrief(
        opening_energy_1_10=clamp(d.get("opening_energy_1_10"), 1, 10, 4.0),
        closing_energy_1_10=clamp(d.get("closing_energy_1_10"), 1, 10, 9.0),
        aggressiveness_0_1=clamp(d.get("aggressiveness_0_1"), 0, 1, 0.5),
        fx_playfulness_0_1=clamp(d.get("fx_playfulness_0_1"), 0, 1, 0.5),
        bpm_flex_percent=clamp(d.get("bpm_flex_percent"), 2, config.MAX_PITCH_SHIFT_PERCENT, 6.0),
        allow_double_drop=bool(d.get("allow_double_drop", True)),
        narrative=d.get("narrative") if d.get("narrative") in
        ("ascendente", "descendente", "ola", "plano") else "ascendente",
    )


def build_set_brief(style_prompt: str) -> SetBrief:
    llm = _load_llm()
    if llm is None:
        return _heuristic_brief(style_prompt)

    completion = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": style_prompt},
        ],
        temperature=0.2,
        max_tokens=300,
        response_format={"type": "json_object"},
    )
    raw = completion["choices"][0]["message"]["content"]
    try:
        parsed = json.loads(raw)
        return _clamp_brief(parsed)
    except (json.JSONDecodeError, KeyError, TypeError):
        return _heuristic_brief(style_prompt)


def brief_to_dict(brief: SetBrief) -> dict:
    return asdict(brief)
