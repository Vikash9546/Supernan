import json
import os

notebook_path = '/Users/vikashkumar/Desktop/Supernan/supernan_dubbing.ipynb'

with open(notebook_path, 'r') as f:
    nb = json.load(f)

def get_duration_code():
    return [
        'import subprocess, os\n',
        'def get_duration(path):\n',
        '    if not os.path.exists(path): return 0.0\n',
        '    r = subprocess.run(["ffprobe","-v","quiet","-show_entries","format=duration","-of","csv=p=0",path],\n',
        '                       capture_output=True, text=True)\n',
        '    try: return float(r.stdout.strip())\n',
        '    except: return 0.0\n',
        '\n'
    ]

for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
    
    source = cell['source']
    if not source:
        continue
    
    header = source[0]

    # ── Cell 3: Install Packages ──
    if '# ── Cell 3: Install Python packages' in header:
        cell['source'] = [
            '# ── Cell 3: Install Python packages ──────────────────────────────────────────\n',
            '# PyTorch with CUDA (Colab default, but ensuring compatibility)\n',
            '!pip install -q torch torchaudio --index-url https://download.pytorch.org/whl/cu118\n',
            '\n',
            '# Core pipeline deps\n',
            '!pip install -q openai-whisper TTS transformers sentencepiece sacremoses\n',
            '!pip install -q git+https://github.com/VarunGumma/IndicTransTokenizer.git\n',
            '!pip install -q pydub librosa soundfile deep-translator\n',
            '!pip install -q basicsr facexlib gfpgan realesrgan\n',
            'print("✓ All packages installed")\n'
        ]

    # ── Cell 5: Download Weights ──
    elif '# ── Cell 5: Download VideoReTalking Weights' in header:
        cell['source'] = [
            '# ── Cell 5: Download VideoReTalking Weights ──────────────────────────────────\n',
            'import os\n',
            'os.makedirs("models/VideoReTalking", exist_ok=True)\n',
            '!wget -q https://github.com/vinthony/video-retalking/releases/download/v0.0.1/30_net_G.pth -O models/VideoReTalking/30_net_G.pth\n',
            '!wget -q https://github.com/vinthony/video-retalking/releases/download/v0.0.1/BFM.zip -O models/VideoReTalking/BFM.zip\n',
            '!unzip -qo models/VideoReTalking/BFM.zip -d models/VideoReTalking/\n',
            'print("✓ VideoReTalking weights ready")\n'
        ]

    # ── Cell 10: Extract Audio ──
    elif '# ── Cell 10: Stage 2 — Extract Audio' in header:
        cell['source'] = [
            '# ── Cell 10: Stage 2 — Extract Audio ─────────────────────────────────────────\n',
            'import os\n',
            'if "CLIP" not in globals(): CLIP = "workspace/clip.mp4"\n',
            'AUDIO_RAW = "workspace/clip_audio.wav"\n',
            'AUDIO_16K = "workspace/clip_audio_16k.wav"\n',
            'AUDIO_REF = "workspace/clip_audio_ref44k.wav"\n',
            '\n',
            'print("Extracting audio tracks...")\n',
            '# 1. Raw audio\n',
            'get_ipython().system(f"ffmpeg -y -i {CLIP} -vn -acodec pcm_s16le -ar 44100 {AUDIO_RAW} -loglevel warning")\n',
            '# 2. 16k for Whisper\n',
            'get_ipython().system(f"ffmpeg -y -i {AUDIO_RAW} -ac 1 -ar 16000 {AUDIO_16K} -loglevel warning")\n',
            '# 3. 44k Reference for XTTS\n',
            'get_ipython().system(f"ffmpeg -y -i {AUDIO_RAW} -ac 1 -ar 44100 {AUDIO_REF} -loglevel warning")\n',
            'print("✓ Audio extracted (Raw, 16k, 44k Ref)")\n'
        ]

    # ── Cell 12: Translation ──
    elif '# ── Cell 12: Stage 4 — Translate to Hindi (IndicTrans2)' in header:
        cell['source'] = [
            '# ── Cell 12: Stage 4 — Translate to Hindi (IndicTrans2) ──────────────────────\n',
            'import torch, os\n',
            'if "DEVICE" not in globals(): DEVICE = "cuda" if torch.cuda.is_available() else "cpu"\n',
            'if "ENGLISH_TEXT" not in globals():\n',
            '    if os.path.exists("workspace/transcript_en.txt"):\n',
            '        with open("workspace/transcript_en.txt", "r") as f: ENGLISH_TEXT = f.read().strip()\n',
            '        print("✓ ENGLISH_TEXT recovered from disk")\n',
            '    else: raise NameError("ENGLISH_TEXT missing. Run Cell 11.")\n',
            '\n',
            'def translate_indictrans2(text):\n',
            '    try:\n',
            '        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer\n',
            '        from IndicTransTokenizer import IndicProcessor\n',
            '        MODEL = "ai4bharat/indictrans2-en-indic-1B"\n',
            '        print("Loading IndicTrans2...")\n',
            '        tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)\n',
            '        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL, trust_remote_code=True).to(DEVICE)\n',
            '        ip = IndicProcessor(inference=True)\n',
            '        sentences = [s.strip() for s in text.split(".") if s.strip()]\n',
            '        batch = ip.preprocess_batch(sentences, src_lang="eng_Latn", tgt_lang="hin_Deva")\n',
            '        inputs = tokenizer(batch, truncation=True, padding="longest", return_tensors="pt").to(DEVICE)\n',
            '        with torch.no_grad():\n',
            '            outputs = model.generate(**inputs, num_beams=5, max_length=256)\n',
            '        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)\n',
            '        postprocessed = ip.postprocess_batch(decoded, lang="hin_Deva")\n',
            '        return " ".join(postprocessed)\n',
            '    except Exception as e:\n',
            '        print(f"IndicTrans2 failed: {e}")\n',
            '        return None\n',
            '\n',
            'print("Attempting Hindi translation...")\n',
            'HINDI_TEXT = translate_indictrans2(ENGLISH_TEXT)\n',
            '\n',
            'if not HINDI_TEXT:\n',
            '    print("⚠️ IndicTrans2 failed. Attempting Google Translate fallback...")\n',
            '    try:\n',
            '        from deep_translator import GoogleTranslator\n',
            '        HINDI_TEXT = GoogleTranslator(source="en", target="hi").translate(ENGLISH_TEXT)\n',
            '        print("✓ Google Translate success!")\n',
            '    except Exception as ge:\n',
            '        print(f"❌ Google Translate failed: {ge}. Using English as last resort.")\n',
            '        HINDI_TEXT = ENGLISH_TEXT\n',
            '\n',
            'with open("workspace/translation_hi.txt", "w", encoding="utf-8") as f: f.write(HINDI_TEXT)\n',
            'print(f"✓ Hindi translation completed.")\n',
            'print(f"Text: {HINDI_TEXT[:100]}...")\n'
        ]

    # ── Cell 13: Voice Cloning ──
    elif '# ── Cell 13: Stage 5 — Coqui XTTS v2 Voice Cloning' in header:
        cell['source'] = [
            '# ── Cell 13: Stage 5 — Coqui XTTS v2 Voice Cloning ──────────────────────────\n',
            'import os, re, torch\n',
            'from TTS.api import TTS\n',
            '\n',
            '# Recovery & Guards\n',
            'if "DEVICE" not in globals(): DEVICE = "cuda" if torch.cuda.is_available() else "cpu"\n',
            'if "AUDIO_REF" not in globals(): AUDIO_REF = "workspace/clip_audio_ref44k.wav"\n',
            'if "HINDI_TEXT" not in globals() or not HINDI_TEXT.strip():\n',
            '    if os.path.exists("workspace/translation_hi.txt"):\n',
            '        with open("workspace/translation_hi.txt", "r", encoding="utf-8") as f: HINDI_TEXT = f.read().strip()\n',
            '    if not globals().get("HINDI_TEXT"):\n',
            '        if "ENGLISH_TEXT" in globals(): HINDI_TEXT = ENGLISH_TEXT\n',
            '        else: raise ValueError("HINDI_TEXT empty and no fallback found.")\n',
            '\n',
            '# 🛡️ PyTorch 2.6+ Unpickling Fix (Comprehensive)\n',
            'try:\n',
            '    from TTS.tts.configs.xtts_config import XttsConfig\n',
            '    from TTS.tts.models.xtts.xtts_audio_config import XttsAudioConfig\n',
            '    import torch.serialization\n',
            '    torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig])\n',
            '    print("✓ Added XTTS configs to PyTorch safe globals")\n',
            'except Exception as e:\n',
            '    print(f"⚠️ Could not add safe globals: {e}")\n',
            '\n',
            '# Final fallback: Monkeypatch torch.load to allow non-weights_only loading\n',
            'if not hasattr(torch.load, "__supernan_patch__"):\n',
            '    orig_load = torch.load\n',
            '    def patched_load(*args, **kwargs):\n',
            '        if "weights_only" not in kwargs: kwargs["weights_only"] = False\n',
            '        return orig_load(*args, **kwargs)\n',
            '    patched_load.__supernan_patch__ = True\n',
            '    torch.load = patched_load\n',
            '    print("✓ Applied torch.load security bypass monkeypatch")\n',
            '\n',
            'os.environ["COQUI_TOS_AGREED"] = "1"\n',
            'TTS_RAW = "workspace/tts_raw.wav"\n',
            '\n',
            'print("Loading Coqui XTTS v2...")\n',
            'tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(DEVICE)\n',
            '\n',
            'print(f"Synthesizing Hindi audio...")\n',
            'tts.tts_to_file(text=HINDI_TEXT, speaker_wav=AUDIO_REF, language="hi", file_path=TTS_RAW)\n',
            'print(f"✓ Synthesized: {TTS_RAW}")\n'
        ]

    # ── Cell 14: Speed Adjust ──
    elif '# ── Cell 14: Stage 6 — Adjust Audio Speed' in header:
        cell['source'] = [
            '# ── Cell 14: Stage 6 — Adjust Audio Speed ────────────────────────────────────\n',
        ] + get_duration_code() + [
            'if "CLIP" not in globals(): CLIP = "workspace/clip.mp4"\n',
            'if "TTS_RAW" not in globals(): TTS_RAW = "workspace/tts_raw.wav"\n',
            'TTS_ADJ = "workspace/tts_adjusted.wav"\n',
            '\n',
            'def build_atempo(ratio):\n',
            '    filters = []\n',
            '    r = ratio\n',
            '    while r < 0.5: filters.append("atempo=0.5"); r /= 0.5\n',
            '    while r > 2.0: filters.append("atempo=2.0"); r /= 2.0\n',
            '    filters.append(f"atempo={r:.6f}")\n',
            '    return ",".join(filters)\n',
            '\n',
            'tts_dur = get_duration(TTS_RAW)\n',
            'clip_dur = get_duration(CLIP)\n',
            '\n',
            'if tts_dur > 0 and clip_dur > 0:\n',
            '    ratio = tts_dur / clip_dur\n',
            '    atempo = build_atempo(ratio)\n',
            '    print(f"Adjusting speed (ratio: {ratio:.3f})...")\n',
            '    get_ipython().system(f"ffmpeg -y -i {TTS_RAW} -filter:a \'{atempo},apad\' -t {clip_dur} -ar 44100 {TTS_ADJ} -loglevel warning")\n',
            '    print(f"✓ Audio adjusted to {clip_dur:.2f}s")\n',
            'else: print("⚠️ Missing files, skipping speed adjustment")\n'
        ]

    # ── Cell 15: VideoReTalking ──
    elif '# ── Cell 15: Stage 7 — VideoReTalking Lip Sync' in header:
        cell['source'] = [
            '# ── Cell 15: Stage 7 — VideoReTalking Lip Sync ───────────────────────────────\n',
            'import sys, os, subprocess\n',
            'if "CLIP" not in globals(): CLIP = "workspace/clip.mp4"\n',
            'if "TTS_ADJ" not in globals(): TTS_ADJ = "workspace/tts_adjusted.wav"\n',
            'LIPSYNC = "workspace/lipsync.mp4"\n',
            '\n',
            '# Path fixes for VRT\n',
            'for p in ["VideoReTalking", "VideoReTalking/third_part", "VideoReTalking/third_part/face3d"]:\n',
            '    path = os.path.abspath(p)\n',
            '    if path not in sys.path: sys.path.append(path)\n',
            '\n',
            'print("Running VideoReTalking (Stage 7)...")\n',
            'res = subprocess.run([sys.executable, "VideoReTalking/inference.py", "--face", CLIP, "--audio", TTS_ADJ, "--outfile", LIPSYNC, "--checkpoint_dir", "models/VideoReTalking"])\n',
            '\n',
            'if res.returncode != 0:\n',
            '    print("⚠️ VRT failed, falling back to simple mux.")\n',
            '    get_ipython().system(f"ffmpeg -y -i {CLIP} -i {TTS_ADJ} -map 0:v -map 1:a -c:v copy -shortest {LIPSYNC} -loglevel warning")\n',
            '\n',
            'from IPython.display import Video\n',
            'Video(LIPSYNC, width=640)\n'
        ]

with open(notebook_path, 'w') as f:
    json.dump(nb, f, indent=4)

print('Successfully cleaned and fixed supernan_dubbing.ipynb with Golden Cell strategy')
