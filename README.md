# 🎬 Supernan Hindi Dubbing Pipeline

A **zero-cost, fully modular** Python pipeline that converts an English training video into a Hindi-dubbed version with voice cloning and lip-sync.

**Built for:** Supernan AI Intern Challenge  
**Output:** 15-second clip (0:15–0:30) with near-perfect lip sync, natural Hindi voice, and face enhancement

---

## Pipeline Architecture

```
Input Video
    ↓
[FFmpeg]         Stage 1: Extract 15-second clip (0:15–0:30)
    ↓
[FFmpeg]         Stage 2: Extract audio (16kHz WAV for Whisper + 44kHz for XTTS)
    ↓
[Whisper]        Stage 3: Transcribe English speech → text
    ↓
[IndicTrans2]    Stage 4: Translate English → natural Hindi
    ↓
[Coqui XTTS v2] Stage 5: Clone original speaker voice + synthesize Hindi
    ↓
[FFmpeg atempo]  Stage 6: Time-stretch audio to exact clip duration
    ↓
[VideoReTalking] Stage 7: Lip-sync video to Hindi audio (GPU required)
    ↓
[GFPGAN v1.4]   Stage 8: Face restoration & sharpening
    ↓
Final 15s Hindi Dubbed MP4
```

---

## Tool Stack (Zero-Cost)

| Task | Tool | Why |
|---|---|---|
| Clip/Audio Extract | `ffmpeg` | Fast, lossless, free |
| Transcription | `openai-whisper` (medium) | Best open-source accuracy |
| Translation | `IndicTrans2` by ai4bharat | Context-aware; much better than literal Google Translate for Indian languages |
| Voice Cloning | `Coqui XTTS v2` | Free, multilingual, zero-shot speaker cloning |
| Audio Stretching | `ffmpeg atempo` | Pitch-preserving, handles >2x and <0.5x via chaining |
| Lip Sync | `VideoReTalking` | Preserves face sharpness better than Wav2Lip |
| Face Enhancement | `GFPGAN v1.4` | Restores high-frequency face detail after inpainting |

**Estimated cost: ₹0** using [Google Colab Free (T4 GPU)](https://colab.research.google.com)

---

## Setup

### Local Setup

```bash
# 1. Clone this repo
git clone https://github.com/YOUR_USERNAME/supernan-dubbing.git
cd supernan-dubbing

# 2. Run the one-command setup
chmod +x setup.sh
./setup.sh

# 3. Activate virtual environment
source venv/bin/activate
```

`setup.sh` will:
- Check for system deps (`ffmpeg`, `git`, `wget`)
- Create a Python venv and install all packages
- Clone and set up `VideoReTalking` and `GFPGAN`
- Download all model weights (~3.5 GB total)

### Google Colab Setup (Recommended for GPU)

Open `supernan_dubbing.ipynb` in Google Colab:
1. Runtime → Change runtime type → **T4 GPU**
2. Run all cells in order
3. Upload your video when prompted

---

## Usage

```bash
# Default: process 0:15–0:30 segment
python dub_video.py --input supernan_training.mp4

# Custom timestamp
python dub_video.py --input supernan_training.mp4 --start 15 --end 30

# Fast test (skips lip-sync GPU step, just dubbing audio)
python dub_video.py --input supernan_training.mp4 --skip-lipsync

# Resume from a specific stage (avoids re-running completed stages)
python dub_video.py --input supernan_training.mp4 --from-stage synthesize_voice

# Check audio sync quality of workspace files
python dub_video.py --input supernan_training.mp4 --check-sync-only
```

Output is saved to `output/final_dubbed.mp4`.

---

## Dependencies

### System (install separately)
```bash
# Ubuntu/Debian / Colab
sudo apt install -y ffmpeg git wget unzip

# macOS
brew install ffmpeg
```

### Python
See `requirements.txt`. Key packages:
- `openai-whisper` — transcription
- `TTS` (Coqui) — voice cloning
- `transformers` — IndicTrans2
- `gfpgan`, `basicsr`, `facexlib` — face enhancement
- `pydub`, `librosa` — audio processing

### External Repos (auto-cloned by `setup.sh`)
- [VideoReTalking](https://github.com/vinthony/video-retalking) — lip sync
- [GFPGAN](https://github.com/TencentARC/GFPGAN) — face enhancement

---

## Project Structure

```
supernan-dubbing/
├── dub_video.py          # Main pipeline orchestrator (9 stages, full CLI)
├── config.py             # Centralized config: timestamps, model sizes, paths
├── utils.py              # Helpers: audio splitting, stretching, sync check
├── requirements.txt      # Python dependencies
├── setup.sh              # One-command environment setup
├── supernan_dubbing.ipynb # Google Colab notebook (recommended)
├── workspace/            # Intermediate files (git-ignored)
│   ├── clip.mp4
│   ├── clip_audio_16k.wav
│   ├── transcript_en.txt
│   ├── translation_hi.txt
│   ├── tts_raw.wav
│   ├── tts_adjusted.wav
│   └── lipsync.mp4
├── output/               # Final output (git-ignored)
│   └── final_dubbed.mp4
├── models/               # Downloaded model weights (git-ignored)
│   └── VideoReTalking/
├── VideoReTalking/       # Cloned by setup.sh
└── GFPGAN/               # Cloned by setup.sh
```

---

## Estimated Cost at Scale

**Current cost for 15s clip: ₹0** (Google Colab Free Tier)

| Component | Free Tier | Paid Scale (A100 Colab Pro) |
|---|---|---|
| Whisper (medium) | ~45s / min of audio | ~8s / min |
| IndicTrans2 | ~10s / min of text | ~2s / min |
| Coqui XTTS v2 | ~2–3 min / min of audio (T4) | ~20s / min (A100) |
| VideoReTalking | ~15 min / 15s clip (T4) | ~90s / 15s clip (A100) |
| GFPGAN | ~3 min / 15s clip (T4) | ~20s / 15s clip (A100) |

**Total wall-clock per minute of video (T4):** ~2–3 hours  
**Total wall-clock per minute of video (A100):** ~15–20 minutes  
**API cost if using A100 Colab Pro:** ~₹35/hour → ~₹9–12 per minute of video

---

## Handling Long Videos (Scale Design)

The pipeline is designed to scale to 500+ hours of video:

1. **Silence-based audio chunking** (`utils.split_on_silence`): Long audio is split at natural pauses. Each chunk is processed independently through Whisper and XTTS, then concatenated. This prevents OOM errors and allows partial retries.

2. **Stage resumability** (`--from-stage`): If any stage fails mid-way, restart from that exact stage using cached workspace files — no need to re-run Whisper or IndicTrans2.

3. **Batch parallelism**: For 500 hours of video, the pipeline can be parallelized across videos using a job queue (e.g., Celery + Redis). Each Colab instance processes one video from a shared queue.

4. **GPU scaling**: Replace single Colab T4 with multi-GPU instances (A100 × 8). VideoReTalking supports `--face_det_batch_size` for throughput scaling.

5. **Cloud storage**: For large batches, input/output videos are stored in Google Cloud Storage, with the pipeline reading/writing via `gsutil`. Intermediate workspace files are stored in `/tmp` to minimize storage costs.

---

## Known Limitations

- **VideoReTalking** can fail on faces that are not front-facing or occluded. Best results with clear, forward-facing speakers.
- **XTTS v2** voice cloning quality depends on the reference clip — needs 5–15 seconds of clean speech. Background noise degrades cloning quality.
- **IndicTrans2** translates well for standard Hindi but may miss domain-specific vocabulary (e.g., nanny training jargon). Post-edit review recommended for production.
- **Audio stretch ratio** is applied uniformly. A smarter approach would align at the sentence level (future work).
- **Colab Free Tier** has session time limits (~12 hours). Long videos may need to be split across sessions using `--from-stage`.

---

## What I'd Improve With More Time

1. **Sentence-level audio alignment**: Instead of stretching the entire audio, align each translated sentence to its corresponding timestamp window (from Whisper word-level timestamps). This would produce near-perfect sync without stretching artifacts.

2. **IndicTrans2 fine-tuning**: Fine-tune on domain-specific nanny/childcare vocabulary for more natural translations.

3. **SadTalker instead of VideoReTalking**: For better head motion generation, especially when the original video has head turns.

4. **Real-time streaming**: Implement a streaming version that processes chunks as they arrive, enabling near-real-time dubbing for live content.

5. **Web UI**: A Gradio interface for non-technical users to upload videos and download dubbed outputs.

---

## Author

Vikash Kumar — Supernan AI Intern Challenge
