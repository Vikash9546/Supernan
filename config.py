"""
config.py – Centralized configuration for the Supernan Hindi Dubbing Pipeline.
All tunable parameters live here so dub_video.py stays clean.
"""

import os
import torch

# ─── Segment to process ───────────────────────────────────────────────────────
START_SEC = 15          # Start of the 15-second clip (seconds)
END_SEC   = 30          # End of the 15-second clip (seconds)

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_DIR  = os.path.join(BASE_DIR, "workspace")   # Intermediate files
OUTPUT_DIR     = os.path.join(BASE_DIR, "output")       # Final output
MODELS_DIR     = os.path.join(BASE_DIR, "models")       # Downloaded weights

# VideoReTalking & GFPGAN repos (cloned by setup.sh)
VIDEORETALKING_DIR = os.path.join(BASE_DIR, "VideoReTalking")
GFPGAN_DIR         = os.path.join(BASE_DIR, "GFPGAN")

# ─── Whisper ──────────────────────────────────────────────────────────────────
# Options: "tiny", "base", "small", "medium", "large"
# "medium" is the sweet spot for accuracy on a free Colab T4
WHISPER_MODEL_SIZE = "medium"

# ─── Translation ──────────────────────────────────────────────────────────────
# IndicTrans2 direction: English → Hindi
INDICTRANS_MODEL  = "ai4bharat/indictrans2-en-indic-1B"
SOURCE_LANG       = "eng_Latn"
TARGET_LANG       = "hin_Deva"

# ─── Coqui XTTS v2 ────────────────────────────────────────────────────────────
XTTS_MODEL_NAME   = "tts_models/multilingual/multi-dataset/xtts_v2"
TTS_LANGUAGE      = "hi"          # Hindi
TTS_SAMPLE_RATE   = 24000         # XTTS v2 native sample rate

# ─── Audio processing ─────────────────────────────────────────────────────────
# Silence-based chunking for long audio (used in batching for >30s clips)
SILENCE_THRESH_DB   = -40         # dB below which to consider silence
MIN_SILENCE_MS      = 500         # Minimum silence length to split on (ms)
KEEP_SILENCE_MS     = 250         # Silence padding kept around each chunk (ms)

# atempo filter bounds — ffmpeg atempo must stay in [0.5, 2.0]
# We chain multiple atempo filters outside this range
ATEMPO_MIN = 0.5
ATEMPO_MAX = 2.0

# ─── VideoReTalking ───────────────────────────────────────────────────────────
# Face detection confidence threshold
FACE_DET_THRESH = 0.9
# Use enhancer inside VideoReTalking (CodeFormer) — set False to use GFPGAN after
VRT_USE_ENHANCER = False   # We apply GFPGAN separately for more control

# ─── GFPGAN ───────────────────────────────────────────────────────────────────
GFPGAN_VERSION    = "1.4"
GFPGAN_UPSCALE    = 1      # 1 = no upscale, keeps original resolution

# ─── Device ───────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─── Logging ──────────────────────────────────────────────────────────────────
LOG_LEVEL = "INFO"

# ─── Ensure directories exist ─────────────────────────────────────────────────
for _d in [WORKSPACE_DIR, OUTPUT_DIR, MODELS_DIR]:
    os.makedirs(_d, exist_ok=True)
