"""Configuración central del AI DJ Set Copilot."""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

CACHE_DB_PATH = os.path.join(DATA_DIR, "analysis_cache.sqlite3")

# Audio
SAMPLE_RATE = 44100
HOP_LENGTH = 512

# GPU: "cuda" si hay NVIDIA + torch compilado con CUDA, si no cae a "cpu" solo.
FORCE_DEVICE = os.environ.get("DJ_COPILOT_DEVICE", "auto")  # "auto" | "cuda" | "cpu"

# Pitch físico de la Numark Mixtrack Platinum FX: +/-8% / +/-16% / +/-25% / +/-50% seleccionable,
# usamos el rango seguro por defecto para no destrozar el groove.
MAX_PITCH_SHIFT_PERCENT = 8.0

# LLM local (GGUF) para interpretar el prompt de estilo. Descarga cualquier modelo
# instruct de 7-8B cuantizado Q4_K_M (cabe holgado en 12 GB VRAM) y coloca la ruta aquí,
# por ejemplo models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
LLM_MODEL_PATH = os.environ.get(
    "DJ_COPILOT_LLM_PATH",
    os.path.join(BASE_DIR, "models", "llm-brief.gguf"),
)
LLM_N_GPU_LAYERS = int(os.environ.get("DJ_COPILOT_LLM_GPU_LAYERS", "-1"))  # -1 = todas a GPU
LLM_CONTEXT_SIZE = 4096

SUPPORTED_EXTENSIONS = (".mp3", ".wav", ".flac", ".aiff", ".aif", ".m4a", ".ogg")
