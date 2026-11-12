"""Rueda de Camelot: mapeo de tonalidades musicales y reglas de compatibilidad armónica."""
from __future__ import annotations
from dataclasses import dataclass

# Orden de quintas para mayores (B = "major" en notación Camelot) y relativas menores (A).
_CAMELOT_MAJOR_ORDER = ["B", "F#", "Db", "Ab", "Eb", "Bb", "F", "C", "G", "D", "A", "E"]
_CAMELOT_MINOR_ORDER = ["Abm", "Ebm", "Bbm", "Fm", "Cm", "Gm", "Dm", "Am", "Em", "Bm", "F#m", "C#m"]

# Nombres de nota -> índice cromático (C=0 ... B=11)
NOTE_TO_PITCH_CLASS = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5,
    "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11,
}

# Construimos tabla pitch_class + modo -> código Camelot (ej: 8A, 9B)
_KEY_TO_CAMELOT: dict[tuple[int, str], str] = {}
for i, note in enumerate(_CAMELOT_MAJOR_ORDER):
    pc = NOTE_TO_PITCH_CLASS[note]
    _KEY_TO_CAMELOT[(pc, "major")] = f"{i + 1}B"
for i, note in enumerate(_CAMELOT_MINOR_ORDER):
    pc = NOTE_TO_PITCH_CLASS[note.replace("m", "")]
    _KEY_TO_CAMELOT[(pc, "minor")] = f"{i + 1}A"

PITCH_CLASS_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


@dataclass(frozen=True)
class CamelotKey:
    pitch_class: int
    mode: str  # "major" | "minor"
    code: str  # ej "8A"
    name: str  # ej "A minor"

    @property
    def number(self) -> int:
        return int(self.code[:-1])

    @property
    def letter(self) -> str:
        return self.code[-1]


def key_to_camelot(pitch_class: int, mode: str) -> CamelotKey:
    code = _KEY_TO_CAMELOT[(pitch_class % 12, mode)]
    note_name = PITCH_CLASS_NAMES[pitch_class % 12]
    label = f"{note_name} {'major' if mode == 'major' else 'minor'}"
    return CamelotKey(pitch_class=pitch_class % 12, mode=mode, code=code, name=label)


def camelot_distance(a: CamelotKey, b: CamelotKey) -> tuple[int, str]:
    """Devuelve (distancia_pasos, etiqueta_relacion) entre dos claves Camelot.

    0  -> misma clave (mezcla perfecta)
    1  -> +1/-1 en la rueda, mismo modo (vecino armónico)
    0m -> mismo número, modo cambiado (relativa mayor/menor, "energy boost" o "mood shift")
    2  -> salto de energía (+2), aceptable para subir intensidad
    N  -> cualquier otro salto, choque armónico
    """
    if a.code == b.code:
        return 0, "misma_clave"
    if a.letter == b.letter:
        diff = (b.number - a.number) % 12
        diff = diff if diff <= 6 else diff - 12
        if abs(diff) == 1:
            return 1, f"vecino_{'+1' if diff == 1 else '-1'}"
        if abs(diff) == 2:
            return 2, f"energy_boost_{'+2' if diff == 2 else '-2'}"
        return abs(diff), "choque_armonico"
    if a.number == b.number:
        return 0, "relativa_mayor_menor"
    return 6, "choque_armonico"


def camelot_key_from_code(camelot_code: str) -> CamelotKey:
    """Reconstruye un CamelotKey a partir de su código (ej. '8A'), sin depender
    de ninguna librería de audio — útil para el motor de planificación, que solo
    trabaja con los resultados ya serializados del análisis DSP."""
    letter = camelot_code[-1]
    mode = "major" if letter == "B" else "minor"
    for pc in range(12):
        candidate = key_to_camelot(pc, mode)
        if candidate.code == camelot_code:
            return candidate
    raise ValueError(f"Código Camelot inválido: {camelot_code}")


def compatibility_score(a: CamelotKey, b: CamelotKey) -> float:
    """Puntuación 0..1 de compatibilidad armónica para ordenar el set (1 = mezcla perfecta)."""
    dist, relation = camelot_distance(a, b)
    table = {
        "misma_clave": 1.0,
        "relativa_mayor_menor": 0.92,
        "vecino_+1": 0.88,
        "vecino_-1": 0.82,
        "energy_boost_+2": 0.7,
        "energy_boost_-2": 0.55,
    }
    return table.get(relation, max(0.0, 0.35 - 0.05 * dist))
