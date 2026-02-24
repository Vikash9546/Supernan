import json
import os

def patch():
    with open('supernan_dubbing.ipynb', 'r') as f:
        nb = json.load(f)

    # --- PATCH CELL 10: Enhanced Reference Audio Extraction ---
    new_cell_10 = [
        "# ── Cell 10: Stage 2 — Extract & Clean Reference Audio ──────────────────────\n",
        "import os\n",
        "if 'CLIP' not in globals(): CLIP = 'workspace/clip.mp4'\n",
        "\n",
        "AUDIO_16K = 'workspace/clip_audio_16k.wav'\n",
        "AUDIO_REF = 'workspace/clip_audio_ref22k.wav'\n",
        "\n",
        "print('Extracting and cleaning audio...')\n",
        "# 1. Extract 16k for Whisper\n",
        "!ffmpeg -y -i {CLIP} -vn -acodec pcm_s16le -ar 16000 -ac 1 {AUDIO_16K} -loglevel warning\n",
        "\n",
        "# 2. Extract 22k Reference for XTTS with Denoising\n",
        "filters = 'afftdn,highpass=f=80,loudnorm=I=-14:LRA=7:TP=-1'\n",
        "!ffmpeg -y -i {CLIP} -vn -af '{filters}' -ac 1 -ar 22050 {AUDIO_REF} -loglevel warning\n",
        "\n",
        "print(f'✓ Audio extracted & cleaned.')\n",
        "print(f'  16kHz (Transcription): {AUDIO_16K}')\n",
        "print(f'  22kHz (Voice Clone Ref): {AUDIO_REF}')\n"
    ]

    # --- PATCH CELL 13: Smart Splitter & Clarity Booster ---
    new_cell_13 = [
        "# ── Cell 13: Stage 5 — Smart XTTS v2 Voice Cloning ──────────────────────────\n",
        "import os, re, torch\n",
        "from TTS.api import TTS\n",
        "\n",
        "# Global State Recovery & Guards\n",
        "if 'DEVICE' not in globals(): DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'\n",
        "if 'AUDIO_REF' not in globals(): AUDIO_REF = 'workspace/clip_audio_ref22k.wav'\n",
        "if 'HINDI_TEXT' not in globals() or not HINDI_TEXT.strip():\n",
        "    if os.path.exists('workspace/translation_hi.txt'):\n",
        "        with open('workspace/translation_hi.txt', 'r', encoding='utf-8') as f: HINDI_TEXT = f.read().strip()\n",
        "\n",
        "os.environ['COQUI_TOS_AGREED'] = '1'\n",
        "TTS_RAW = 'workspace/tts_raw.wav'\n",
        "TTS_CLEAN = 'workspace/tts_clean.wav'\n",
        "os.makedirs('workspace/tts_chunks', exist_ok=True)\n",
        "\n",
        "print('Loading Coqui XTTS v2...')\n",
        "if 'tts' not in globals():\n",
        "    tts = TTS('tts_models/multilingual/multi-dataset/xtts_v2')\n",
        "    if DEVICE == 'cuda': tts = tts.cuda()\n",
        "\n",
        "def smart_split_hindi(text, max_len=80):\n",
        "    \"\"\"Splits text at conjunctions and punctuation for stability.\"\"\"\n",
        "    import re\n",
        "    # Split at punctuation\n",
        "    text = re.sub(r'([।,.!?])', r'\\1|', text)\n",
        "    # Split at common Hindi conjunctions\n",
        "    for conj in [' क्योंकि ', ' इसलिए ', ' या ', ' और ', ' लेकिन ', ' तो ', ' कि ']:\n",
        "        text = text.replace(conj, f'|{conj}')\n",
        "    \n",
        "    raw_parts = [p.strip() for p in text.split('|') if p.strip()]\n",
        "    chunks, curr = [], ''\n",
        "    for p in raw_parts:\n",
        "        if len(curr) + len(p) < max_len:\n",
        "            curr += ' ' + p if curr else p\n",
        "        else:\n",
        "            if curr: chunks.append(curr.strip())\n",
        "            curr = p\n",
        "    if curr: chunks.append(curr.strip())\n",
        "    return chunks\n",
        "\n",
        "chunks = smart_split_hindi(HINDI_TEXT)\n",
        "print(f'Synthesizing {len(chunks)} chunk(s)...')\n",
        "\n",
        "chunk_paths = []\n",
        "for i, chunk in enumerate(chunks):\n",
        "    p = f'workspace/tts_chunks/tts_{i:04d}.wav'\n",
        "    print(f'  [{i+1}/{len(chunks)}] {chunk[:50]}...')\n",
        "    tts.tts_to_file(text=chunk, speaker_wav=AUDIO_REF, language='hi', file_path=p, split_sentences=False)\n",
        "    chunk_paths.append(p)\n",
        "\n",
        "# Concat with natural breaths (100ms silence)\n",
        "if len(chunk_paths) > 1:\n",
        "    !ffmpeg -y -f lavfi -i anullsrc=r=22050:cl=mono -t 0.1 workspace/sil.wav -loglevel warning\n",
        "    with open('workspace/concat_list.txt', 'w') as f:\n",
        "        for cp in chunk_paths:\n",
        "            f.write(f'file {os.path.abspath(cp)}\\n')\n",
        "            f.write(f'file {os.path.abspath('workspace/sil.wav')}\\n')\n",
        "    !ffmpeg -y -f concat -safe 0 -i workspace/concat_list.txt -c copy {TTS_RAW} -loglevel warning\n",
        "else:\n",
        "    import shutil\n",
        "    shutil.copy(chunk_paths[0], TTS_RAW)\n",
        "\n",
        "print('Applying Ultimate Clarity Booster...')\n",
        "filters = (\n",
        "    'highpass=f=200,'\n",
        "    'equalizer=f=3000:t=q:w=1:g=5,'\n",
        "    'equalizer=f=6000:t=q:w=1:g=5,'\n",
        "    'treble=g=5:f=8000,'\n",
        "    'acompressor=threshold=0.08:ratio=4:attack=5:release=50:makeup=2,'\n",
        "    'loudnorm=I=-14:LRA=7:TP=-1'\n",
        ")\n",
        "!ffmpeg -y -i {TTS_RAW} -af '{filters}' -ar 44100 {TTS_CLEAN} -loglevel warning\n",
        "\n",
        "print(f'✓ Balanced & Clear Dub: {TTS_CLEAN}')\n",
        "TTS_FINAL_AUDIO = TTS_CLEAN\n"
    ]

    found = 0
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell.get('source', []))
            if '# ── Cell 10:' in source:\n                cell['source'] = new_cell_10\n                found += 1\n            if '# ── Cell 13:' in source:\n                cell['source'] = new_cell_13\n                found += 1

    with open('supernan_dubbing.ipynb', 'w') as f:
        json.dump(nb, f, indent=4, ensure_ascii=False)
    print(f'Patched {found} cells.')

if __name__ == '__main__':
    patch()
