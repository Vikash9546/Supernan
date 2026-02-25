# 🎬 Supernan AI Dubbing Pipeline

A **high-fidelity, zero-cost** Python pipeline that converts English training videos into professional-grade Hindi-dubbed versions with voice cloning and crystal-clear audio.

**Built for:** Supernan AI Intern Challenge  
**Output:** 20-second high-quality dubbed clip with perfect lip-sync and studio-level voice clarity.

---

## 💎 Premium Quality Features

Unlike standard dubbing scripts, this pipeline includes a **4-Pillar Quality Enhancement Suite**:

1. **🎙️ Crystal Clear Voice Cloning**: Applies adaptive denoising (`afftdn`) and high-pass filtering to the original reference audio for cleaner voice extraction.
2. **🗣️ Anti-Fumble Smart Splitting**: Uses a conjunction-aware text splitter (handling `और`, `क्योंकि`, `लेकिन`, etc.) to prevent XTTS from fumbling on long Hindi sentences.
3. **✨ Ultimate Clarity Booster**: Professional FFmpeg audio chain (Equalizer, Treble Boost, Compressor, and Loudnorm) for a "studio" feel.
4. **🔄 Natural Precision Sync**: Caps speed adjustment at **1.15x** (natural human limit) and uses **Smart Video Padding** (freezing frames) instead of "chipmunk" speed-up if audio is long.

---

## 🛠️ Pipeline Architecture

```mermaid
graph TD
    A[🎬 Input Video] --> B[🎞️ FFmpeg Clip Extract]
    B --> C[ FFmpeg Audio Extract]
    C --> D[ Whisper Transcription (audio - English Text)]
    D --> E[ IndicTrans2 Translation (English Text - Hindi Text)]
    E --> F[ Coqui XTTS v2 (Hindi Voice Clone)]
    F --> G[ FFmpeg Atempo Filter Speed Adjust Audio — match duration exactly]
    G --> H[ VideoReTalking (Lip Sync)]
    H --> I[✨ GFPGAN (Face Enhancement)]
    I --> J[🎬 Final Output — 20 sec Video]
```

---

## 🚀 Setup & Usage

### ☁️ Google Colab (Recommended)
Open `supernan_dubbing.ipynb` in Colab for free GPU access (T4).
1. Runtime → Change runtime type → **T4 GPU**.
2. Run all cells.
3. **GitHub Ready**: The notebook is optimized to be small in size (outputs cleared) for perfect display on GitHub.

### 💻 Local Setup
```bash
chmod +x setup.sh
./setup.sh
source venv/bin/activate
python dub_video.py --input supernan_training.mp4
```

---

## 📂 Project Structure

- `supernan_dubbing.ipynb`: Main interactive pipeline (GitHub-optimized).
- `dub_video.py`: Orchestrator script for high-scale processing.
- `utils.py`: Smart audio manipulation and sync-checking utilities.
- `setup.sh`: Automated environment and model weights downloader.
- `workspace/`: Temporary storage for intermediate stages (Denoised Ref, Raw TTS, Clean TTS).

---
