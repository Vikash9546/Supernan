#!/usr/bin/env python3
"""
dub_video.py – Supernan Hindi Dubbing Pipeline
===============================================
A modular pipeline that takes a video, extracts a 15-second clip,
transcribes speech, translates to Hindi, clones the voice, and produces
a lip-synced, face-enhanced Hindi dubbed video.

Pipeline Stages:
  1. extract_clip        → FFmpeg: cut 0:15–0:30 from the source video
  2. extract_audio       → FFmpeg: export WAV from the clip
  3. transcribe          → Whisper: speech → English text
  4. translate           → IndicTrans2: English → Hindi text
  5. synthesize_voice    → Coqui XTTS v2: Hindi TTS with voice cloning
  6. adjust_audio_speed  → FFmpeg atempo: match TTS duration to clip duration
  7. lipsync             → VideoReTalking: regenerate lip movements
  8. enhance_face        → GFPGAN: restore face sharpness post lip-sync
  9. mux_output          → FFmpeg: merge enhanced video + synced audio

Usage:
  python dub_video.py --input path/to/video.mp4 [options]

  # Process default segment (15–30s):
  python dub_video.py --input supernan_training.mp4

  # Custom segment:
  python dub_video.py --input supernan_training.mp4 --start 15 --end 30

  # Skip lip-sync (fast test of translation/TTS only):
  python dub_video.py --input supernan_training.mp4 --skip-lipsync

  # Skip specific stages (e.g. restart from step 5 if XTTS already ran):
  python dub_video.py --input supernan_training.mp4 --from-stage synthesize_voice

Author: Vikash Kumar | Supernan AI Intern Challenge
"""

import argparse
import logging
import os
import sys
import shutil
import subprocess
from pathlib import Path

# ── Project imports ───────────────────────────────────────────────────────────
from config import (
    START_SEC, END_SEC, WORKSPACE_DIR, OUTPUT_DIR, MODELS_DIR,
    VIDEORETALKING_DIR, GFPGAN_DIR,
    WHISPER_MODEL_SIZE,
    INDICTRANS_MODEL, SOURCE_LANG, TARGET_LANG,
    XTTS_MODEL_NAME, TTS_LANGUAGE, TTS_SAMPLE_RATE,
    SILENCE_THRESH_DB, MIN_SILENCE_MS, KEEP_SILENCE_MS,
    FACE_DET_THRESH, VRT_USE_ENHANCER,
    GFPGAN_VERSION, GFPGAN_UPSCALE,
    DEVICE,
)
from utils import (
    run_cmd, get_duration,
    stretch_audio,
    split_on_silence, concatenate_wavs,
    check_sync_offset,
)

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(WORKSPACE_DIR, "pipeline.log"), mode="a"),
    ],
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 1 – CLIP EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def extract_clip(
    input_path: str,
    output_path: str,
    start: float,
    end: float,
) -> str:
    """
    Extract a precise time segment from the source video using FFmpeg.

    We use -ss BEFORE -i (input seeking) for speed, then -t for precise duration.
    Re-encoding is avoided by using stream copy (-c copy) plus a keyframe-accurate
    trim pass. For very accurate cuts we re-encode with libx264/aac.

    Args:
        input_path:  Source video file
        output_path: Destination for the clipped video
        start:       Start time in seconds
        end:         End time in seconds

    Returns:
        output_path
    """
    duration = end - start
    logger.info(f"[Stage 1] Extracting clip: {start}s – {end}s ({duration}s)")
    run_cmd([
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", input_path,
        "-t", str(duration),
        "-c:v", "libx264",    # Re-encode for frame-accurate cut
        "-c:a", "aac",
        "-preset", "fast",
        "-crf", "18",         # Near-lossless quality
        output_path
    ], desc="Extracting clip segment")
    logger.info(f"  ✓ Clip saved: {output_path}")
    return output_path


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 2 – AUDIO EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def extract_audio(video_path: str, output_wav: str) -> str:
    """
    Extract audio from a video file as a mono 16kHz WAV.

    Whisper works best with 16kHz mono audio. XTTS speaker reference
    benefits from higher quality, so we save a separate 44.1kHz reference.

    Args:
        video_path: Input video file
        output_wav: Output WAV path (16kHz mono for Whisper)

    Returns:
        output_wav
    """
    logger.info("[Stage 2] Extracting audio")
    run_cmd([
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",                     # No video
        "-acodec", "pcm_s16le",    # 16-bit PCM
        "-ar", "16000",            # 16kHz for Whisper
        "-ac", "1",                # Mono
        output_wav
    ], desc="Extracting audio as 16kHz mono WAV")

    # Also save a higher-quality reference for XTTS v2 voice cloning
    ref_wav = output_wav.replace(".wav", "_ref44k.wav")
    run_cmd([
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "44100",
        "-ac", "1",
        ref_wav
    ], desc="Extracting 44kHz reference audio for voice cloning")

    logger.info(f"  ✓ Audio (16kHz): {output_wav}")
    logger.info(f"  ✓ Reference (44kHz): {ref_wav}")
    return output_wav


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 3 – TRANSCRIPTION
# ═══════════════════════════════════════════════════════════════════════════════

def transcribe(audio_path: str, transcript_path: str) -> str:
    """
    Transcribe English speech to text using OpenAI Whisper (local, free).

    Model selection:
      - "small"  : Fast, good for clean audio
      - "medium" : Best accuracy on accented speech (recommended for production)
      - "large"  : Highest accuracy, needs more VRAM

    Word-level timestamps are extracted so we can detect speech boundaries
    for smarter audio sync (future enhancement).

    Args:
        audio_path:      WAV file to transcribe
        transcript_path: Path to write the English transcript (.txt)

    Returns:
        The transcribed English text (also written to transcript_path)
    """
    import whisper

    logger.info(f"[Stage 3] Transcribing with Whisper ({WHISPER_MODEL_SIZE})")
    model = whisper.load_model(WHISPER_MODEL_SIZE, device=DEVICE)

    result = model.transcribe(
        audio_path,
        language="en",
        word_timestamps=True,    # Word-level timing for future sync enhancement
        verbose=False,
    )
    text = result["text"].strip()

    # Write transcript
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(text + "\n")

    logger.info(f"  ✓ Transcript: {text[:100]}{'...' if len(text) > 100 else ''}")
    logger.info(f"  ✓ Saved to: {transcript_path}")
    return text


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 4 – TRANSLATION (English → Hindi)
# ═══════════════════════════════════════════════════════════════════════════════

def translate(english_text: str, translation_path: str) -> str:
    """
    Translate English text to Hindi using IndicTrans2 (ai4bharat).

    IndicTrans2 is significantly better than Google Translate for Indian
    languages — it's context-aware, handles colloquialisms, and produces
    natural Hindi appropriate for everyday speech (e.g. a nanny's training
    video), not stiff literal translations.

    Batching strategy for long text:
      Text is split into sentences, translated in batches of up to 10 sentences,
      then reassembled. This prevents OOM on long documents.

    Falls back to deep-translator (Google Translate) if IndicTrans2 is not
    available — useful for quick local testing without the 2GB model download.

    Args:
        english_text:     Source English text
        translation_path: Path to write Hindi translation (.txt)

    Returns:
        Hindi text string
    """
    hindi_text = _translate_indictrans2(english_text)
    if hindi_text is None:
        logger.warning("  IndicTrans2 unavailable. Falling back to deep-translator...")
        hindi_text = _translate_fallback(english_text)

    with open(translation_path, "w", encoding="utf-8") as f:
        f.write(hindi_text + "\n")

    logger.info(f"  ✓ Hindi: {hindi_text[:100]}{'...' if len(hindi_text) > 100 else ''}")
    logger.info(f"  ✓ Saved to: {translation_path}")
    return hindi_text


def _translate_indictrans2(text: str) -> str | None:
    """Primary translation using IndicTrans2."""
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        from IndicTransTokenizer import IndicProcessor

        logger.info(f"[Stage 4] Loading IndicTrans2 ({INDICTRANS_MODEL})")
        tokenizer = AutoTokenizer.from_pretrained(
            INDICTRANS_MODEL, trust_remote_code=True
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            INDICTRANS_MODEL, trust_remote_code=True
        ).to(DEVICE)
        ip = IndicProcessor(inference=True)

        # Split into sentences for batching
        sentences = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
        BATCH_SIZE = 10
        translations = []

        for i in range(0, len(sentences), BATCH_SIZE):
            batch = sentences[i : i + BATCH_SIZE]
            batch_input = ip.preprocess_batch(
                batch, src_lang=SOURCE_LANG, tgt_lang=TARGET_LANG
            )
            inputs = tokenizer(
                batch_input,
                truncation=True,
                padding="longest",
                return_tensors="pt",
                return_attention_mask=True,
            ).to(DEVICE)
            with __import__("torch").no_grad():
                outputs = model.generate(
                    **inputs,
                    num_beams=5,
                    num_return_sequences=1,
                    max_length=256,
                )
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            postprocessed = ip.postprocess_batch(decoded, lang=TARGET_LANG)
            translations.extend(postprocessed)

        return " ".join(translations)

    except (ImportError, Exception) as e:
        logger.debug(f"IndicTrans2 error: {e}")
        return None


def _translate_fallback(text: str) -> str:
    """Fallback: deep-translator (Google Translate) for quick testing."""
    try:
        from deep_translator import GoogleTranslator
        translated = GoogleTranslator(source="en", target="hi").translate(text)
        return translated
    except ImportError:
        raise ImportError(
            "Neither IndicTrans2 nor deep-translator is installed.\n"
            "Install fallback: pip install deep-translator"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 5 – HINDI VOICE SYNTHESIS (XTTS v2 Voice Cloning)
# ═══════════════════════════════════════════════════════════════════════════════

def synthesize_voice(
    hindi_text: str,
    reference_wav: str,
    output_wav: str,
) -> str:
    """
    Generate Hindi speech that clones the original speaker's voice using
    Coqui XTTS v2 (free, multilingual, zero-shot speaker cloning).

    XTTS v2 accepts a reference audio clip (3–30 seconds) and generates
    speech in the target language while preserving the speaker's tone,
    pace, and vocal character.

    Batching for long text:
      XTTS v2 can struggle with very long texts (>300 chars) in a single
      inference call. We split on silence-detected sentence boundaries,
      synthesize each chunk, then concatenate.

    Args:
        hindi_text:    Hindi text to synthesize
        reference_wav: Reference audio clip (original speaker, 44kHz WAV)
        output_wav:    Output path for synthesized Hindi WAV

    Returns:
        output_wav
    """
    from TTS.api import TTS
    import torch

    logger.info("[Stage 5] Loading Coqui XTTS v2")
    tts = TTS(XTTS_MODEL_NAME).to(DEVICE)

    # Clean text for TTS (remove special chars that confuse TTS)
    clean_text = hindi_text.strip()

    # Split into manageable chunks (XTTS works best on <300 char segments)
    chunks = _chunk_hindi_text(clean_text, max_chars=250)
    logger.info(f"  Synthesizing {len(chunks)} chunk(s)")

    if len(chunks) == 1:
        # Single chunk — direct synthesis
        tts.tts_to_file(
            text=clean_text,
            speaker_wav=reference_wav,
            language=TTS_LANGUAGE,
            file_path=output_wav,
        )
    else:
        # Multi-chunk: synthesize each, then concatenate
        chunk_paths = []
        chunk_dir = os.path.join(WORKSPACE_DIR, "tts_chunks")
        os.makedirs(chunk_dir, exist_ok=True)

        for i, chunk in enumerate(chunks):
            chunk_path = os.path.join(chunk_dir, f"tts_{i:04d}.wav")
            logger.info(f"  Chunk {i+1}/{len(chunks)}: {chunk[:60]}...")
            tts.tts_to_file(
                text=chunk,
                speaker_wav=reference_wav,
                language=TTS_LANGUAGE,
                file_path=chunk_path,
            )
            chunk_paths.append(chunk_path)

        concatenate_wavs(chunk_paths, output_wav)

    logger.info(f"  ✓ Hindi TTS audio: {output_wav}")
    return output_wav


def _chunk_hindi_text(text: str, max_chars: int = 250) -> list[str]:
    """
    Split Hindi text into chunks at sentence boundaries (|, ।, .)
    to keep each XTTS inference call manageable.
    """
    import re
    # Split on Hindi danda, pipe, or period
    sentences = re.split(r'(?<=[।|.!?])\s+', text)
    chunks, current = [], ""
    for sentence in sentences:
        if len(current) + len(sentence) + 1 <= max_chars:
            current = (current + " " + sentence).strip()
        else:
            if current:
                chunks.append(current)
            current = sentence
    if current:
        chunks.append(current)
    return chunks if chunks else [text]


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 6 – AUDIO SPEED ADJUSTMENT
# ═══════════════════════════════════════════════════════════════════════════════

def adjust_audio_speed(
    tts_wav: str,
    clip_path: str,
    output_wav: str,
) -> str:
    """
    Time-stretch the TTS audio to exactly match the video clip duration.

    Hindi speech is often slightly longer or shorter than English for the
    same content. We use FFmpeg's atempo filter (pitch-preserving) to
    stretch/compress the TTS audio so it aligns perfectly with the
    video's timeline.

    Args:
        tts_wav:    Path to raw XTTS output WAV
        clip_path:  Path to the video clip (used to get target duration)
        output_wav: Path to write duration-adjusted WAV

    Returns:
        output_wav
    """
    logger.info("[Stage 6] Adjusting audio speed to match video duration")
    target_duration = get_duration(clip_path)
    return stretch_audio(tts_wav, output_wav, target_duration)


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 7 – LIP SYNC (VideoReTalking)
# ═══════════════════════════════════════════════════════════════════════════════

def lipsync(
    clip_path: str,
    audio_path: str,
    output_path: str,
) -> str:
    """
    Generate lip-synced video using VideoReTalking.

    VideoReTalking is preferred over Wav2Lip because:
      - It uses a face-parsing network to only modify lip/mouth regions
      - The rest of the face remains untouched (no full-face blur)
      - It handles natural head movements better

    The model checkpoints are downloaded automatically to MODELS_DIR
    on the first run (~1.5 GB total).

    Args:
        clip_path:   Input video clip
        audio_path:  Adjusted Hindi audio (WAV)
        output_path: Output lip-synced video path

    Returns:
        output_path

    Raises:
        FileNotFoundError: If VideoReTalking repo is not cloned
    """
    if not os.path.isdir(VIDEORETALKING_DIR):
        raise FileNotFoundError(
            f"VideoReTalking not found at {VIDEORETALKING_DIR}.\n"
            "Run setup.sh to clone it, or:\n"
            "  git clone https://github.com/vinthony/video-retalking VideoReTalking"
        )

    logger.info("[Stage 7] Running VideoReTalking lip-sync")

    # VideoReTalking inference script
    inference_script = os.path.join(VIDEORETALKING_DIR, "inference.py")
    checkpoint_dir   = os.path.join(MODELS_DIR, "VideoReTalking")

    # Download checkpoints if not present
    _download_vrt_checkpoints(checkpoint_dir)

    cmd = [
        sys.executable, inference_script,
        "--face",        clip_path,
        "--audio",       audio_path,
        "--outfile",     output_path,
        "--checkpoint_dir", checkpoint_dir,
    ]

    # Run from VideoReTalking directory so relative imports work
    logger.info(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        cwd=VIDEORETALKING_DIR,
        capture_output=False,   # Show VRT's own progress output
    )
    if result.returncode != 0:
        raise RuntimeError("VideoReTalking inference failed. Check the logs above.")

    logger.info(f"  ✓ Lip-synced video: {output_path}")
    return output_path


def _download_vrt_checkpoints(checkpoint_dir: str):
    """Download VideoReTalking model weights if not already present."""
    from utils import download_file
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoints = {
        "30_net_gen.pth": "https://github.com/vinthony/video-retalking/releases/download/v0.0.1/30_net_gen.pth",
        "BFM.zip":        "https://github.com/vinthony/video-retalking/releases/download/v0.0.1/BFM.zip",
        "DNet.pt":        "https://github.com/vinthony/video-retalking/releases/download/v0.0.1/DNet.pt",
        "ENet.pth":       "https://github.com/vinthony/video-retalking/releases/download/v0.0.1/ENet.pth",
        "GFPGANv1.3.pth": "https://github.com/vinthony/video-retalking/releases/download/v0.0.1/GFPGANv1.3.pth",
        "GPEN-BFR-512.pth": "https://github.com/vinthony/video-retalking/releases/download/v0.0.1/GPEN-BFR-512.pth",
        "ParseNet-latest.pth": "https://github.com/vinthony/video-retalking/releases/download/v0.0.1/ParseNet-latest.pth",
        "RetinaFace-R50.pth":  "https://github.com/vinthony/video-retalking/releases/download/v0.0.1/RetinaFace-R50.pth",
        "shape_predictor_68_face_landmarks.dat": "https://github.com/vinthony/video-retalking/releases/download/v0.0.1/shape_predictor_68_face_landmarks.dat",
    }

    for filename, url in checkpoints.items():
        dest = os.path.join(checkpoint_dir, filename)
        download_file(url, dest, desc=filename)

    # Unzip BFM if needed
    bfm_zip = os.path.join(checkpoint_dir, "BFM.zip")
    bfm_dir = os.path.join(checkpoint_dir, "BFM")
    if os.path.exists(bfm_zip) and not os.path.isdir(bfm_dir):
        run_cmd(["unzip", "-q", bfm_zip, "-d", checkpoint_dir], desc="Unzipping BFM")


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 8 – FACE ENHANCEMENT (GFPGAN)
# ═══════════════════════════════════════════════════════════════════════════════

def enhance_face(
    input_video: str,
    output_video: str,
) -> str:
    """
    Apply GFPGAN face restoration to sharpen the face region after lip-sync.

    VideoReTalking's mouth inpainting can slightly soften the face.
    GFPGAN (Generative Face Prior GAN) restores high-frequency details
    (pores, wrinkles, eye clarity) while leaving the background untouched.

    Process:
      1. Extract frames from the lip-synced video
      2. Run GFPGAN on each frame
      3. Re-encode video from enhanced frames + original audio

    Args:
        input_video:  Lip-synced video from VideoReTalking
        output_video: Path for the face-enhanced final video

    Returns:
        output_video

    Raises:
        FileNotFoundError: If GFPGAN repo is not cloned
    """
    if not os.path.isdir(GFPGAN_DIR):
        raise FileNotFoundError(
            f"GFPGAN not found at {GFPGAN_DIR}.\n"
            "Run setup.sh or:\n"
            "  git clone https://github.com/TencentARC/GFPGAN GFPGAN\n"
            "  pip install basicsr facexlib gfpgan"
        )

    logger.info("[Stage 8] Running GFPGAN face enhancement")

    frames_dir   = os.path.join(WORKSPACE_DIR, "frames_raw")
    enhanced_dir = os.path.join(WORKSPACE_DIR, "frames_enhanced")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(enhanced_dir, exist_ok=True)

    # Step 8a: Extract frames
    logger.info("  Extracting frames...")
    run_cmd([
        "ffmpeg", "-y",
        "-i", input_video,
        "-qscale:v", "1",          # Highest quality
        "-qmin", "1",
        os.path.join(frames_dir, "frame_%06d.png")
    ], desc="Extracting frames for GFPGAN")

    # Step 8b: GFPGAN inference
    gfpgan_script = os.path.join(GFPGAN_DIR, "inference_gfpgan.py")
    gfpgan_model  = os.path.join(GFPGAN_DIR, "experiments", "pretrained_models",
                                 f"GFPGANv{GFPGAN_VERSION}.pth")

    # Download model if not present
    if not os.path.exists(gfpgan_model):
        from utils import download_file
        os.makedirs(os.path.dirname(gfpgan_model), exist_ok=True)
        download_file(
            f"https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv{GFPGAN_VERSION}.pth",
            gfpgan_model,
            desc=f"GFPGANv{GFPGAN_VERSION}.pth"
        )

    logger.info("  Running GFPGAN on all frames...")
    run_cmd([
        sys.executable, gfpgan_script,
        "-i", frames_dir,
        "-o", os.path.join(WORKSPACE_DIR, "gfpgan_out"),
        "-v", GFPGAN_VERSION,
        "-s", str(GFPGAN_UPSCALE),
        "--bg_upsampler", "None",   # Skip background upscaling (keep original bg)
    ], desc="GFPGAN face restoration")

    # GFPGAN outputs to <out_dir>/restored_imgs/
    gfpgan_restored = os.path.join(WORKSPACE_DIR, "gfpgan_out", "restored_imgs")
    if not os.path.isdir(gfpgan_restored):
        logger.warning("GFPGAN restored_imgs not found, using raw frames instead")
        gfpgan_restored = frames_dir

    # Step 8c: Get original video FPS
    fps_result = run_cmd([
        "ffprobe", "-v", "quiet",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "csv=p=0",
        input_video
    ], desc="Getting video FPS")
    fps_str = fps_result.stdout.strip()   # e.g. "25/1" or "30000/1001"

    # Step 8d: Re-encode from enhanced frames + original audio
    logger.info("  Re-encoding video from enhanced frames...")
    run_cmd([
        "ffmpeg", "-y",
        "-framerate", fps_str,
        "-pattern_type", "glob",
        "-i", os.path.join(gfpgan_restored, "*.png"),
        "-i", input_video,          # Use audio from lip-synced video
        "-map", "0:v",
        "-map", "1:a",
        "-c:v", "libx264",
        "-crf", "17",
        "-preset", "slow",
        "-c:a", "aac",
        "-shortest",
        output_video
    ], desc="Re-encoding with GFPGAN-enhanced frames")

    logger.info(f"  ✓ Enhanced video: {output_video}")
    return output_video


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 9 – FINAL MUXING (video + audio, no enhancement fallback)
# ═══════════════════════════════════════════════════════════════════════════════

def mux_final(video_path: str, audio_path: str, output_path: str) -> str:
    """
    Replace the audio track of a video with the Hindi TTS audio.
    Used as a fallback step when lip-sync is skipped.

    Args:
        video_path:  Video (clip or lip-synced)
        audio_path:  Hindi audio (adjusted speed WAV)
        output_path: Final output path

    Returns:
        output_path
    """
    logger.info("[Mux] Merging video + Hindi audio")
    run_cmd([
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_path,
        "-map", "0:v",
        "-map", "1:a",
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        output_path
    ], desc="Muxing final video + audio")
    logger.info(f"  ✓ Final output: {output_path}")
    return output_path


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════

STAGE_ORDER = [
    "extract_clip",
    "extract_audio",
    "transcribe",
    "translate",
    "synthesize_voice",
    "adjust_audio_speed",
    "lipsync",
    "enhance_face",
]


def run_pipeline(
    input_video: str,
    start: float,
    end: float,
    skip_lipsync: bool = False,
    skip_enhance: bool = False,
    from_stage: str = None,
) -> str:
    """
    Run the full dubbing pipeline, returning the path to the final output video.

    Intermediate files are stored in WORKSPACE_DIR with descriptive names
    so any stage can be re-run in isolation by passing --from-stage.

    Args:
        input_video:   Path to source video file
        start:         Clip start time (seconds)
        end:           Clip end time (seconds)
        skip_lipsync:  If True, skip VideoReTalking (fast testing mode)
        skip_enhance:  If True, skip GFPGAN face enhancement
        from_stage:    If set, skip stages before this one (use cached workspace files)

    Returns:
        Path to the final dubbed video file
    """
    logger.info("=" * 60)
    logger.info("  SUPERNAN HINDI DUBBING PIPELINE")
    logger.info("=" * 60)
    logger.info(f"  Input:    {input_video}")
    logger.info(f"  Segment:  {start}s – {end}s ({end - start}s)")
    logger.info(f"  Device:   {DEVICE}")
    logger.info(f"  Lipsync:  {'DISABLED' if skip_lipsync else 'ENABLED'}")
    logger.info(f"  Enhance:  {'DISABLED' if skip_enhance else 'ENABLED'}")
    logger.info("=" * 60)

    # ── Workspace file paths (fixed names for resumability) ───────────────────
    clip_path       = os.path.join(WORKSPACE_DIR, "clip.mp4")
    audio_16k       = os.path.join(WORKSPACE_DIR, "clip_audio_16k.wav")
    audio_ref       = os.path.join(WORKSPACE_DIR, "clip_audio_16k_ref44k.wav")
    transcript_path = os.path.join(WORKSPACE_DIR, "transcript_en.txt")
    translation_path= os.path.join(WORKSPACE_DIR, "translation_hi.txt")
    tts_raw_wav     = os.path.join(WORKSPACE_DIR, "tts_raw.wav")
    tts_adj_wav     = os.path.join(WORKSPACE_DIR, "tts_adjusted.wav")
    lipsync_path    = os.path.join(WORKSPACE_DIR, "lipsync.mp4")
    final_path      = os.path.join(OUTPUT_DIR, "final_dubbed.mp4")

    # ── Skip-to logic ─────────────────────────────────────────────────────────
    skip = bool(from_stage)
    def should_run(stage: str) -> bool:
        nonlocal skip
        if skip and stage == from_stage:
            skip = False
        return not skip

    # ── Stage 1: Extract Clip ─────────────────────────────────────────────────
    if should_run("extract_clip"):
        extract_clip(input_video, clip_path, start, end)

    # ── Stage 2: Extract Audio ────────────────────────────────────────────────
    if should_run("extract_audio"):
        extract_audio(clip_path, audio_16k)

    # ── Stage 3: Transcribe ───────────────────────────────────────────────────
    if should_run("transcribe"):
        english_text = transcribe(audio_16k, transcript_path)
    else:
        with open(transcript_path, encoding="utf-8") as f:
            english_text = f.read().strip()

    # ── Stage 4: Translate ────────────────────────────────────────────────────
    if should_run("translate"):
        hindi_text = translate(english_text, translation_path)
    else:
        with open(translation_path, encoding="utf-8") as f:
            hindi_text = f.read().strip()

    # ── Stage 5: Synthesize Voice ─────────────────────────────────────────────
    if should_run("synthesize_voice"):
        synthesize_voice(hindi_text, audio_ref, tts_raw_wav)

    # ── Stage 6: Adjust Audio Speed ────────────────────────────────────────────
    if should_run("adjust_audio_speed"):
        adjust_audio_speed(tts_raw_wav, clip_path, tts_adj_wav)

    # ── Sync quality diagnostic ───────────────────────────────────────────────
    try:
        offset_ms, confidence = check_sync_offset(audio_16k, tts_adj_wav)
        logger.info(f"  Sync diagnostic: offset={offset_ms*1000:.1f}ms, confidence={confidence:.3f}")
    except Exception as e:
        logger.debug(f"Sync check skipped: {e}")

    # ── Stage 7: Lip Sync ─────────────────────────────────────────────────────
    if skip_lipsync:
        logger.info("[Stage 7] SKIPPED (--skip-lipsync)")
        mux_final(clip_path, tts_adj_wav, lipsync_path)
    else:
        if should_run("lipsync"):
            lipsync(clip_path, tts_adj_wav, lipsync_path)

    # ── Stage 8: Face Enhancement ─────────────────────────────────────────────
    if skip_enhance:
        logger.info("[Stage 8] SKIPPED (--skip-enhance)")
        shutil.copy2(lipsync_path, final_path)
    else:
        if should_run("enhance_face"):
            try:
                enhance_face(lipsync_path, final_path)
            except FileNotFoundError as e:
                logger.warning(f"GFPGAN not found, copying lipsync output directly.\n  {e}")
                shutil.copy2(lipsync_path, final_path)

    # ── Verify output duration ────────────────────────────────────────────────
    final_duration = get_duration(final_path)
    expected = end - start
    logger.info("=" * 60)
    logger.info(f"  ✅ PIPELINE COMPLETE")
    logger.info(f"  Output:   {final_path}")
    logger.info(f"  Duration: {final_duration:.2f}s (expected {expected:.1f}s)")
    if abs(final_duration - expected) > 1.0:
        logger.warning(f"  ⚠️  Duration mismatch! Δ={abs(final_duration - expected):.2f}s")
    logger.info("=" * 60)
    return final_path


# ═══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Supernan Hindi Dubbing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline, default segment (15–30s):
  python dub_video.py --input supernan_training.mp4

  # Custom segment (0–15s):
  python dub_video.py --input supernan_training.mp4 --start 0 --end 15

  # Test translation + TTS without heavy lip-sync GPU step:
  python dub_video.py --input supernan_training.mp4 --skip-lipsync

  # Resume pipeline from voice synthesis (reuse cached transcript/translation):
  python dub_video.py --input supernan_training.mp4 --from-stage synthesize_voice
        """,
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to input video file"
    )
    parser.add_argument(
        "--start", "-s", type=float, default=START_SEC,
        help=f"Start time in seconds (default: {START_SEC})"
    )
    parser.add_argument(
        "--end", "-e", type=float, default=END_SEC,
        help=f"End time in seconds (default: {END_SEC})"
    )
    parser.add_argument(
        "--skip-lipsync", action="store_true",
        help="Skip VideoReTalking lip-sync (fast test mode, just mux audio)"
    )
    parser.add_argument(
        "--skip-enhance", action="store_true",
        help="Skip GFPGAN face enhancement"
    )
    parser.add_argument(
        "--from-stage", metavar="STAGE",
        choices=STAGE_ORDER,
        help=(
            "Skip all stages before STAGE, reading intermediate files from workspace/. "
            f"Choices: {', '.join(STAGE_ORDER)}"
        )
    )
    parser.add_argument(
        "--check-sync-only", action="store_true",
        help="Only run the sync-quality diagnostic on existing workspace files"
    )
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        parser.error(f"Input file not found: {args.input}")

    if args.end <= args.start:
        parser.error(f"--end ({args.end}) must be greater than --start ({args.start})")

    if args.check_sync_only:
        ref  = os.path.join(WORKSPACE_DIR, "clip_audio_16k.wav")
        dub  = os.path.join(WORKSPACE_DIR, "tts_adjusted.wav")
        offset, conf = check_sync_offset(ref, dub)
        print(f"Sync offset: {offset*1000:.1f}ms | Confidence: {conf:.4f}")
        sys.exit(0)

    out = run_pipeline(
        input_video=args.input,
        start=args.start,
        end=args.end,
        skip_lipsync=args.skip_lipsync,
        skip_enhance=args.skip_enhance,
        from_stage=args.from_stage,
    )
    print(f"\n✅ Done! Output → {out}")


if __name__ == "__main__":
    main()
