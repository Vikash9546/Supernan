"""
utils.py – Helper utilities for the Supernan Hindi Dubbing Pipeline.

Covers:
  - Subprocess execution with logging
  - Audio splitting on silence (batching for long files)
  - Audio duration/speed stretching
  - Sync-quality check (cross-correlation)
  - ffprobe duration helper
"""

import os
import logging
import subprocess
import tempfile
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ─── Subprocess Helper ────────────────────────────────────────────────────────

def run_cmd(cmd: List[str], desc: str = "", check: bool = True) -> subprocess.CompletedProcess:
    """
    Run a shell command, logging stdout/stderr.

    Args:
        cmd:   Command list, e.g. ["ffmpeg", "-i", "input.mp4", ...]
        desc:  Human-readable description shown in logs
        check: If True, raises CalledProcessError on non-zero exit code
    """
    if desc:
        logger.info(f"[CMD] {desc}")
    logger.debug(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.stdout:
        logger.debug(f"  stdout: {result.stdout.strip()}")
    if result.stderr:
        logger.debug(f"  stderr: {result.stderr.strip()}")
    if check and result.returncode != 0:
        logger.error(f"Command failed (exit {result.returncode}): {' '.join(cmd)}")
        logger.error(result.stderr)
        raise subprocess.CalledProcessError(result.returncode, cmd, result.stderr)
    return result


# ─── ffprobe Duration ─────────────────────────────────────────────────────────

def get_duration(filepath: str) -> float:
    """Return the duration of a media file in seconds using ffprobe."""
    result = run_cmd([
        "ffprobe", "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "csv=p=0",
        filepath
    ], desc=f"Probing duration of {os.path.basename(filepath)}")
    return float(result.stdout.strip())


# ─── Audio Speed Stretching ───────────────────────────────────────────────────

def _build_atempo_chain(ratio: float) -> str:
    """
    Build an ffmpeg atempo filter string that supports ratios outside [0.5, 2.0].
    ffmpeg's atempo filter only accepts values in [0.5, 2.0].
    For ratios outside this range, we chain multiple atempo filters.

    Example: ratio=0.3 → "atempo=0.5,atempo=0.6"
             ratio=3.0 → "atempo=2.0,atempo=1.5"
    """
    filters = []
    remaining = ratio
    while remaining < 0.5:
        filters.append("atempo=0.5")
        remaining /= 0.5
    while remaining > 2.0:
        filters.append("atempo=2.0")
        remaining /= 2.0
    filters.append(f"atempo={remaining:.6f}")
    return ",".join(filters)


def stretch_audio(
    input_path: str,
    output_path: str,
    target_duration: float,
) -> str:
    """
    Time-stretch audio so it fits exactly within `target_duration` seconds.
    Uses ffmpeg's atempo (audio) filter — pitch-preserving.

    Args:
        input_path:      Path to input WAV/MP3 file
        output_path:     Path to write stretched audio
        target_duration: Desired duration in seconds

    Returns:
        output_path
    """
    src_duration = get_duration(input_path)
    ratio = src_duration / target_duration  # >1 means speed up, <1 means slow down
    atempo_str = _build_atempo_chain(ratio)
    logger.info(
        f"Stretching audio: {src_duration:.2f}s → {target_duration:.2f}s "
        f"(ratio={ratio:.3f}, filter={atempo_str})"
    )
    run_cmd([
        "ffmpeg", "-y",
        "-i", input_path,
        "-filter:a", atempo_str,
        "-ar", "44100",
        output_path
    ], desc="Stretching audio to match video duration")
    return output_path


# ─── Silence-Based Audio Splitting (Batching for Long Files) ──────────────────

def split_on_silence(
    input_wav: str,
    output_dir: str,
    silence_thresh_db: float = -40,
    min_silence_ms: int = 500,
    keep_silence_ms: int = 250,
) -> List[str]:
    """
    Split a WAV file on silent sections and return a list of chunk file paths.

    This is used for batching long audio through the TTS model (XTTS v2 works
    best on utterance-length segments, not multi-minute files).

    Uses pydub for silence detection.

    Args:
        input_wav:       Path to source WAV file
        output_dir:      Directory to write chunk WAV files
        silence_thresh_db: dB level below which audio is considered silent
        min_silence_ms:  Minimum silence duration to trigger a split
        keep_silence_ms: Silence padding kept at start/end of each chunk

    Returns:
        Sorted list of chunk file paths
    """
    try:
        from pydub import AudioSegment
        from pydub.silence import split_on_silence as _split
    except ImportError:
        raise ImportError("pydub is required for silence splitting. Install it: pip install pydub")

    logger.info(f"Splitting audio on silence: {os.path.basename(input_wav)}")
    audio = AudioSegment.from_wav(input_wav)
    chunks = _split(
        audio,
        min_silence_len=min_silence_ms,
        silence_thresh=silence_thresh_db,
        keep_silence=keep_silence_ms,
    )
    logger.info(f"  → {len(chunks)} chunk(s) detected")

    os.makedirs(output_dir, exist_ok=True)
    chunk_paths = []
    for i, chunk in enumerate(chunks):
        path = os.path.join(output_dir, f"chunk_{i:04d}.wav")
        chunk.export(path, format="wav")
        chunk_paths.append(path)
        logger.debug(f"  Wrote chunk {i}: {len(chunk)/1000:.2f}s → {path}")

    return sorted(chunk_paths)


def concatenate_wavs(wav_paths: List[str], output_path: str) -> str:
    """
    Concatenate a list of WAV files into a single WAV.
    Uses ffmpeg concat demuxer for lossless joining.

    Args:
        wav_paths:   Ordered list of WAV file paths
        output_path: Output path for concatenated WAV

    Returns:
        output_path
    """
    if not wav_paths:
        raise ValueError("No WAV files provided to concatenate")

    # Write a temporary concat list file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for p in wav_paths:
            f.write(f"file '{os.path.abspath(p)}'\n")
        concat_file = f.name

    try:
        run_cmd([
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_file,
            "-c", "copy",
            output_path
        ], desc="Concatenating audio chunks")
    finally:
        os.unlink(concat_file)

    return output_path


# ─── Audio Sync Quality Check ─────────────────────────────────────────────────

def check_sync_offset(
    reference_wav: str,
    dubbed_wav: str,
    sample_rate: int = 16000,
) -> Tuple[float, float]:
    """
    Estimate the timing offset between reference and dubbed audio using
    normalized cross-correlation.

    This is a lightweight diagnostic tool — it won't catch phrase-level
    mis-alignment but gives a useful ballpark for sync quality.

    Args:
        reference_wav: Original audio (ground truth timing)
        dubbed_wav:    Hindi TTS-generated audio
        sample_rate:   Sample rate to resample both to

    Returns:
        (offset_seconds, confidence)
        offset_seconds: Positive = dubbed audio is delayed, Negative = early
        confidence:     Max cross-correlation value (higher = more reliable)
    """
    try:
        import librosa
    except ImportError:
        raise ImportError("librosa required for sync check. Install: pip install librosa")

    ref, _ = librosa.load(reference_wav, sr=sample_rate, mono=True)
    dub, _ = librosa.load(dubbed_wav,   sr=sample_rate, mono=True)

    # Normalize
    ref = ref / (np.max(np.abs(ref)) + 1e-8)
    dub = dub / (np.max(np.abs(dub)) + 1e-8)

    # Cross-correlate
    corr = np.correlate(ref, dub, mode="full")
    lag  = np.argmax(np.abs(corr)) - (len(dub) - 1)
    offset_sec  = lag / sample_rate
    confidence  = float(np.max(np.abs(corr)) / len(ref))

    logger.info(f"Sync check → offset={offset_sec*1000:.1f}ms, confidence={confidence:.3f}")
    return offset_sec, confidence


# ─── Model Download Helper ────────────────────────────────────────────────────

def download_file(url: str, dest_path: str, desc: str = "") -> str:
    """Download a file with progress using wget if not already present."""
    if os.path.exists(dest_path):
        logger.info(f"Already exists, skipping download: {os.path.basename(dest_path)}")
        return dest_path
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    logger.info(f"Downloading {desc or url} → {dest_path}")
    run_cmd(["wget", "-q", "--show-progress", "-O", dest_path, url],
            desc=f"Downloading {desc}")
    return dest_path
