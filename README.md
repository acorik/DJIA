# AI DJ Set Copilot

Copilot local, **100% offline**, para construir sets de DJ a partir de una carpeta de
música y un prompt en lenguaje natural, con transiciones diseñadas para la
**Numark Mixtrack Platinum FX** (Serato DJ Pro).

## 1. Arquitectura elegida y por qué

| Pieza | Tecnología | Por qué para una RTX 4070 de 12 GB |
|---|---|---|
| BPM, tonalidad (Camelot), estructura, timbre | **librosa + numpy/scipy (CPU)** | El beat-tracking, chroma CQT y HPSS son ligeros; correrlos en CPU es más rápido que mover tensores pequeños a GPU y evita saturar VRAM con el LLM cargado. |
| Espectrogramas mel pesados | **torchaudio sobre CUDA** | Acelera el procesado por lotes de carpetas grandes sin competir por VRAM con el LLM (unos pocos cientos de MB). |
| Interpretación del prompt de estilo | **LLM local GGUF vía llama-cpp-python** (ej. Llama 3.1 8B Instruct Q4_K_M, ~5-6 GB VRAM) | Cabe holgado en 12 GB dejando margen para los espectrogramas. Se usa **solo** para traducir intención creativa a un JSON de parámetros — nunca decide BPM, tonalidad ni orden, así no hay alucinaciones sobre hechos verificables del audio. |
| Orden del set y transiciones | **Algoritmo determinista** (Rueda de Camelot + compatibilidad de BPM + heurística de estructura) | Un DJ necesita previsibilidad matemática (Camelot, pitch %), no "creatividad" de un LLM aquí. |
| API / UI | **FastAPI + HTML/JS vanilla** | Cero dependencias de build, arranca con un solo comando, UI oscura de cabina. |

**Regla de oro cumplida:** nada del análisis lee el nombre del archivo ni tags ID3.
Todo (BPM, tonalidad, estructura, energía) se deriva de la señal de audio decodificada.
El nombre de archivo solo se usa como etiqueta de visualización en la tabla (`title_guess`).

## 2. Estructura del proyecto

```
DJIA/
├── requirements.txt
├── run.py                      # arranca la app (uvicorn)
├── backend/
│   ├── config.py                # rutas, SR, límites de pitch de la Numark, modelo LLM
│   ├── camelot.py                # Rueda de Camelot + compatibilidad armónica
│   ├── key_detection.py          # Krumhansl-Schmuckler sobre chroma CQT real
│   ├── gpu.py                    # detección de CUDA
│   ├── audio_analysis.py         # BPM, estabilidad, energía, timbre (GPU/CPU)
│   ├── structure.py              # Intro/Verso/Buildup/Drop/Breakdown/Outro en mm:ss y compases
│   ├── cache_db.py                # caché SQLite por hash de audio
│   ├── analysis_service.py        # orquesta análisis + caché + escaneo de carpeta
│   ├── llm_brief.py                # LLM local (o heurística) -> brief de estilo JSON
│   ├── set_planner.py              # orden óptimo del set (Camelot + BPM + curva de energía)
│   ├── transition_designer.py       # transiciones -> controles físicos de la Numark
│   └── main.py                       # API FastAPI + sirve el frontend
├── frontend/
│   ├── index.html
│   ├── style.css
│   ├── app.js                 # llamadas a la API, tabla, timeline
│   └── numark_layout.js        # esquema interactivo de la Platinum FX
├── data/                        # caché SQLite (se crea sola)
└── models/                       # aquí va tu .gguf del LLM local (opcional)
```

## 3. Instalación paso a paso (Windows/Linux con RTX 4070)

### 3.1. Python y entorno virtual
```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate   # Linux
```

### 3.2. Dependencias base
```bash
pip install -r requirements.txt
```

### 3.3. PyTorch + torchaudio con CUDA (reemplaza la entrada ya instalada por la build CUDA)
```bash
pip install torch==2.4.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
```
Verifica que ve tu GPU:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### 3.4. llama-cpp-python con aceleración CUDA (para el LLM que interpreta el prompt)
```bash
# Windows (PowerShell, con Visual Studio Build Tools + CUDA Toolkit instalados):
$env:CMAKE_ARGS="-DGGML_CUDA=on"; pip install llama-cpp-python==0.3.2 --no-cache-dir

# Linux:
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python==0.3.2 --no-cache-dir
```

### 3.5. (Opcional pero recomendado) Descarga un LLM local en GGUF
Descarga cualquier modelo instruct de 7-8B cuantizado Q4_K_M (ej. Llama 3.1 8B
Instruct, Mistral 7B Instruct) y guárdalo como:
```
DJIA/models/llm-brief.gguf
```
Si no haces este paso, la app sigue funcionando: usa un parser de palabras
clave (español/inglés) como alternativa para interpretar el prompt de estilo.

### 3.6. Arrancar la aplicación
```bash
python run.py
```
Abre **http://localhost:8000** en el navegador.

### 3.7. Uso
1. Pega la ruta de tu carpeta de música.
2. Escribe el estilo del set que quieres (ej. *"empieza profundo, acaba
   reventando, cortes agresivos, loop rolls tensos y jugueteo con las
   palancas de FX"*).
3. Pulsa **Analizar carpeta** (primera vez: tarda según tu colección; queda
   cacheado por hash de audio para la próxima).
4. Pulsa **Generar Set**: obtienes el tracklist ordenado por Camelot/BPM/energía
   y, al hacer clic en cualquier transición, el esquema físico exacto de la
   Numark Mixtrack Platinum FX con cada movimiento y su timestamp en mm:ss.

## 4. Notas de hardware

- Sin GPU/CUDA disponible, la app **sigue funcionando** (cae a CPU para los
  espectrogramas y usa el parser heurístico para el prompt si tampoco hay
  `llama-cpp-python` compilado con CUDA) — solo será más lenta en carpetas grandes.
- La caché en `data/analysis_cache.sqlite3` evita reanalizar un archivo que no
  ha cambiado de contenido, aunque lo muevas o renombres.
