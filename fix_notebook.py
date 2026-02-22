import json
import os

notebook_path = '/Users/vikashkumar/Desktop/Supernan/supernan_dubbing.ipynb'

with open(notebook_path, 'r') as f:
    nb = json.load(f)

def fix_cell_14(source):
    new_source = []
    new_source.append('# ── Cell 14: Stage 6 — Adjust Audio Speed ────────────────────────────────────\n')
    new_source.append('import subprocess, os\n')
    new_source.append('\n')
    new_source.append('TTS_ADJ = "workspace/tts_adjusted.wav"\n')
    new_source.append('\n')
    new_source.append('def get_duration(path):\n')
    new_source.append('    if not os.path.exists(path): return 0.0\n')
    new_source.append('    r = subprocess.run(["ffprobe","-v","quiet","-show_entries","format=duration","-of","csv=p=0",path],\n')
    new_source.append('                       capture_output=True, text=True)\n')
    new_source.append('    try: return float(r.stdout.strip())\n')
    new_source.append('    except: return 0.0\n')
    new_source.append('\n')
    new_source.append('def build_atempo(ratio):\n')
    new_source.append('    filters = []\n')
    new_source.append('    r = ratio\n')
    new_source.append('    while r < 0.5: filters.append("atempo=0.5"); r /= 0.5\n')
    new_source.append('    while r > 2.0: filters.append("atempo=2.0"); r /= 2.0\n')
    new_source.append('    filters.append(f"atempo={r:.6f}")\n')
    new_source.append('    return ",".join(filters)\n')
    new_source.append('\n')
    new_source.append('tts_dur = get_duration(TTS_RAW)\n')
    new_source.append('clip_dur = get_duration(CLIP)\n')
    new_source.append('if tts_dur > 0 and clip_dur > 0:\n')
    new_source.append('    ratio = tts_dur / clip_dur\n')
    new_source.append('    atempo = build_atempo(ratio)\n')
    new_source.append('    print(f"TTS duration: {tts_dur:.2f}s | Clip duration: {clip_dur:.2f}s | ratio={ratio:.3f}")\n')
    new_source.append('    # Speed match + Duration Guard (ensure exactly clip_dur seconds)\n')
    new_source.append('    get_ipython().system(f"ffmpeg -y -i {TTS_RAW} -filter:a \'{atempo},apad\' -t {clip_dur} -ar 44100 {TTS_ADJ} -loglevel warning")\n')
    new_source.append('    print(f"✓ Audio adjusted and padded to {clip_dur:.2f}s")\n')
    new_source.append('else:\n')
    new_source.append('    print("⚠️ Skipping speed adjustment (missing audio files)")\n')
    return new_source

def fix_cell_16(source):
    new_source = []
    new_source.append('# ── Cell 16: Stage 8 — GFPGAN Face Enhancement ───────────────────────────────\n')
    new_source.append('SKIP_ENHANCEMENT = False  # Set to True to skip this slow stage and get output immediately\n')
    new_source.append('FINAL = "output/final_dubbed.mp4"\n')
    new_source.append('FRAMES_DIR = "workspace/frames_raw"\n')
    new_source.append('ENHANCED_DIR = "workspace/gfpgan_out"\n')
    new_source.append('\n')
    new_source.append('import os, subprocess, torch\n')
    new_source.append('\n')
    new_source.append('if SKIP_ENHANCEMENT:\n')
    new_source.append('    print("\u23e9 SKIP_ENHANCEMENT is True. Copying lip-sync video to final output...")\n')
    new_source.append('    get_ipython().system(f"cp {LIPSYNC} {FINAL}")\n')
    new_source.append('    print(f"\u2713 Final output (no enhancement): {FINAL}")\n')
    new_source.append('else:\n')
    new_source.append('    os.makedirs(FRAMES_DIR, exist_ok=True)\n')
    new_source.append('    print("Extracting frames...")\n')
    new_source.append('    get_ipython().system(f"ffmpeg -y -i {LIPSYNC} -qscale:v 1 -qmin 1 {FRAMES_DIR}/frame_%06d.png -loglevel warning")\n')
    new_source.append('\n')
    new_source.append('    print("Running GFPGAN face restoration...")\n')
    new_source.append('    if torch.cuda.is_available():\n')
    new_source.append('        PROCESSOR = "cuda"\n')
    new_source.append('    elif torch.backends.mps.is_available():\n')
    new_source.append('        PROCESSOR = "mps"\n')
    new_source.append('    else:\n')
    new_source.append('        PROCESSOR = "cpu"\n')
    new_source.append('    print(f"Device detected: {PROCESSOR}")\n')
    new_source.append('    get_ipython().system(f"python GFPGAN/inference_gfpgan.py --ext png --device {PROCESSOR} -i {FRAMES_DIR} -o {ENHANCED_DIR} -v 1.4 -s 1 --bg_upsampler None")\n')
    new_source.append('\n')
    new_source.append('    fps_r = subprocess.run(["ffprobe","-v","quiet","-select_streams","v:0",\n')
    new_source.append('                            "-show_entries","stream=r_frame_rate","-of","csv=p=0", LIPSYNC],\n')
    new_source.append('                           capture_output=True, text=True)\n')
    new_source.append('    FPS = fps_r.stdout.strip()\n')
    new_source.append('\n')
    new_source.append('    RESTORED = f"{ENHANCED_DIR}/restored_imgs"\n')
    new_source.append('    print("Re-encoding final video...")\n')
    new_source.append('    get_ipython().system(f"ffmpeg -y -framerate {FPS} -pattern_type glob -i \'{RESTORED}/*.png\' -i {LIPSYNC} -map 0:v -map 1:a -c:v libx264 -crf 17 -preset slow -c:a aac -shortest {FINAL} -loglevel warning")\n')
    new_source.append('\n')
    new_source.append('    final_dur = get_duration(FINAL)\n')
    new_source.append('    print(f"✓ Final output: {FINAL} ({final_dur:.2f}s)")\n')
    new_source.append('\n')
    new_source.append('from IPython.display import Video\n')
    new_source.append('Video(FINAL, width=640)\n')
    return new_source

for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
    
    source = cell['source']
    if not source:
        continue
    
    # Check for Cell 5 (VideoReTalking weights)
    if '# ── Cell 5: Download VideoReTalking Weights' in source[0]:
        new_source = []
        for line in source:
            if '!unzip -q' in line:
                new_source.append(line.replace('!unzip -q', '!unzip -qo'))
            else:
                new_source.append(line)
        cell['source'] = new_source
    
    # Check for Cell 13 (XTTS)
    if '# ── Cell 13: Stage 5 — Coqui XTTS v2 Voice Cloning' in source[0]:
        new_source = []
        new_source.append('# ── Cell 13: Stage 5 — Coqui XTTS v2 Voice Cloning ──────────────────────────\n')
        new_source.append('import os, re, torch\n')
        new_source.append('from TTS.api import TTS\n')
        new_source.append('\n')
        new_source.append('# Fix for PyTorch 2.6+ UnpicklingError - Monkeypatch torch.load\n')
        new_source.append('if not hasattr(torch.load, "__supernan_patch__"):\n')
        new_source.append('    orig_load = torch.load\n')
        new_source.append('    def zipped_load(*args, **kwargs):\n')
        new_source.append('        kwargs["weights_only"] = False\n')
        new_source.append('        return orig_load(*args, **kwargs)\n')
        new_source.append('    zipped_load.__supernan_patch__ = True\n')
        new_source.append('    torch.load = zipped_load\n')
        new_source.append('\n')
        new_source.append('# Auto-accept Coqui CPML license (non-commercial use)\n')
        new_source.append('os.environ["COQUI_TOS_AGREED"] = "1"\n')
        new_source.append('\n')
        new_source.append('TTS_RAW = "workspace/tts_raw.wav"\n')
        
        # Keep the rest of the logic
        start_appending = False
        for line in source:
            if "print('Loading Coqui XTTS v2...')" in line:
                start_appending = True
            if start_appending:
                new_source.append(line)
        cell['source'] = new_source

    # Check for Cell 14 (Speed Adjust)
    if '# ── Cell 14: Stage 6 — Adjust Audio Speed' in source[0]:
        cell['source'] = fix_cell_14(source)

    # Check for Cell 15 (VideoReTalking)
    if '# ── Cell 15: Stage 7 — VideoReTalking Lip Sync' in source[0]:
        new_source = []
        new_source.append('# ── Cell 15: Stage 7 — VideoReTalking Lip Sync ───────────────────────────────\n')
        new_source.append('import sys, os, subprocess\n')
        new_source.append('\n')
        new_source.append('# Path fix for VideoReTalking third_part modules\n')
        new_source.append('VRT_PATH = os.path.abspath("VideoReTalking")\n')
        new_source.append('if VRT_PATH not in sys.path: sys.path.append(VRT_PATH)\n')
        new_source.append('TP_PATH = os.path.join(VRT_PATH, "third_part")\n')
        new_source.append('if TP_PATH not in sys.path: sys.path.append(TP_PATH)\n')
        new_source.append('F3D_PATH = os.path.join(TP_PATH, "face3d")\n')
        new_source.append('if F3D_PATH not in sys.path: sys.path.append(F3D_PATH)\n')
        new_source.append('\n')
        new_source.append('LIPSYNC = "workspace/lipsync.mp4"\n')
        new_source.append('CHECKPOINT_DIR = "models/VideoReTalking"\n')
        new_source.append('\n')
        new_source.append('print("Running VideoReTalking (this takes ~10-20 min on free T4)...")\n')
        new_source.append('result = subprocess.run(\n')
        new_source.append('    [sys.executable, "VideoReTalking/inference.py",\n')
        new_source.append('     "--face", CLIP,\n')
        new_source.append('     "--audio", TTS_ADJ,\n')
        new_source.append('     "--outfile", LIPSYNC,\n')
        new_source.append('     "--checkpoint_dir", CHECKPOINT_DIR],\n')
        new_source.append('    cwd="."\n')
        new_source.append(')\n')
        new_source.append('\n')
        new_source.append('if result.returncode != 0:\n')
        new_source.append('    print("⚠️ VideoReTalking failed. Falling back to simple audio mux...")\n')
        new_source.append('    if os.path.exists(TTS_ADJ):\n')
        new_source.append('        get_ipython().system(f"ffmpeg -y -i {CLIP} -i {TTS_ADJ} -map 0:v -map 1:a -c:v copy -c:a aac -shortest {LIPSYNC} -loglevel warning")\n')
        new_source.append('    else:\n')
        new_source.append('        print("⚠️ Missing TTS_ADJ, using original clip.")\n')
        new_source.append('        get_ipython().system(f"cp {CLIP} {LIPSYNC}")\n')
        new_source.append('\n')
        new_source.append('from IPython.display import Video\n')
        new_source.append('Video(LIPSYNC, width=640, embed=True)\n')
        cell['source'] = new_source

    # Check for Cell 16 (GFPGAN)
    if '# ── Cell 16: Stage 8 — GFPGAN Face Enhancement' in source[0]:
        cell['source'] = fix_cell_16(source)

    # Check for Cell 17 (Download)
    if '# ── Cell 17: Download Output' in source[0]:
        new_source = []
        new_source.append('# ── Cell 17: Download Output ──────────────────────────────────────────────────\n')
        new_source.append('import os\n')
        new_source.append('try:\n')
        new_source.append('    from google.colab import files\n')
        new_source.append('    files.download(FINAL)\n')
        new_source.append('    print(f"✓ Download started for {FINAL}")\n')
        new_source.append('except ImportError:\n')
        new_source.append('    print(f"✓ Output saved to: {os.path.abspath(FINAL)}")\n')
        new_source.append('    print("   (Local machine detected, skipping browser download)")\n')
        cell['source'] = new_source

with open(notebook_path, 'w') as f:
    json.dump(nb, f, indent=4)

print('Successfully fixed supernan_dubbing.ipynb')
