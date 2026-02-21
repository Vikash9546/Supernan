#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# setup.sh – One-command setup for the Supernan Hindi Dubbing Pipeline
#
# Usage:
#   chmod +x setup.sh && ./setup.sh
#
# What it does:
#   1. Checks system dependencies (ffmpeg, python3, git)
#   2. Creates a Python virtual environment
#   3. Installs all Python packages
#   4. Clones VideoReTalking and GFPGAN repos with their weights
# ──────────────────────────────────────────────────────────────────────────────

set -e  # Exit on any error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ─── Colors ───────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

info()    { echo -e "${GREEN}[INFO]${NC} $1"; }
warning() { echo -e "${YELLOW}[WARN]${NC} $1"; }
error()   { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# ─── Step 1: Check System Dependencies ───────────────────────────────────────
info "Checking system dependencies..."

check_cmd() {
  if ! command -v "$1" &>/dev/null; then
    error "$1 not found. Install it: $2"
  fi
  info "  ✓ $1 found"
}

check_cmd ffmpeg  "sudo apt install ffmpeg (Linux) or brew install ffmpeg (Mac)"
check_cmd git     "sudo apt install git"
check_cmd python3 "sudo apt install python3"
check_cmd wget    "sudo apt install wget"
check_cmd unzip   "sudo apt install unzip"

# ─── Step 2: Python Virtual Environment ──────────────────────────────────────
info "Setting up Python virtual environment..."

if [ ! -d "venv" ]; then
  python3 -m venv venv
  info "  ✓ Created venv/"
else
  info "  ✓ venv/ already exists"
fi

# Activate venv
source venv/bin/activate
info "  ✓ Activated venv"

# Upgrade pip
pip install --quiet --upgrade pip

# ─── Step 3: Install Python Dependencies ─────────────────────────────────────
info "Installing Python dependencies..."

# Install PyTorch first (CPU version for local; Colab handles GPU automatically)
if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  info "  GPU detected — keeping existing torch installation"
else
  warning "  No GPU found. Installing CPU-only PyTorch (lip-sync will be slow)."
  warning "  For GPU, run on Google Colab (free T4) or use: pip install torch --index-url https://download.pytorch.org/whl/cu118"
  pip install --quiet torch torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

pip install --quiet -r requirements.txt

# Install IndicTransTokenizer (not on PyPI)
info "  Installing IndicTransTokenizer from GitHub..."
pip install --quiet git+https://github.com/VarunGumma/IndicTransTokenizer.git

info "  ✓ All Python packages installed"

# ─── Step 4: Clone VideoReTalking ────────────────────────────────────────────
info "Setting up VideoReTalking..."

if [ ! -d "VideoReTalking" ]; then
  git clone https://github.com/vinthony/video-retalking VideoReTalking
  info "  ✓ Cloned VideoReTalking"
else
  info "  ✓ VideoReTalking already exists"
fi

# Install VideoReTalking deps
info "  Installing VideoReTalking requirements..."
pip install --quiet -r VideoReTalking/requirements.txt 2>/dev/null || true

# Download VideoReTalking checkpoints
info "  Downloading VideoReTalking model weights (~1.5 GB)..."
mkdir -p models/VideoReTalking

CHECKPOINTS=(
  "30_net_gen.pth|https://github.com/vinthony/video-retalking/releases/download/v0.0.1/30_net_gen.pth"
  "DNet.pt|https://github.com/vinthony/video-retalking/releases/download/v0.0.1/DNet.pt"
  "ENet.pth|https://github.com/vinthony/video-retalking/releases/download/v0.0.1/ENet.pth"
  "GFPGANv1.3.pth|https://github.com/vinthony/video-retalking/releases/download/v0.0.1/GFPGANv1.3.pth"
  "GPEN-BFR-512.pth|https://github.com/vinthony/video-retalking/releases/download/v0.0.1/GPEN-BFR-512.pth"
  "ParseNet-latest.pth|https://github.com/vinthony/video-retalking/releases/download/v0.0.1/ParseNet-latest.pth"
  "RetinaFace-R50.pth|https://github.com/vinthony/video-retalking/releases/download/v0.0.1/RetinaFace-R50.pth"
  "shape_predictor_68_face_landmarks.dat|https://github.com/vinthony/video-retalking/releases/download/v0.0.1/shape_predictor_68_face_landmarks.dat"
  "BFM.zip|https://github.com/vinthony/video-retalking/releases/download/v0.0.1/BFM.zip"
)

for entry in "${CHECKPOINTS[@]}"; do
  filename="${entry%%|*}"
  url="${entry##*|}"
  dest="models/VideoReTalking/${filename}"
  if [ ! -f "$dest" ]; then
    info "  Downloading ${filename}..."
    wget -q --show-progress -O "$dest" "$url"
  else
    info "  ✓ ${filename} already downloaded"
  fi
done

# Unzip BFM
if [ -f "models/VideoReTalking/BFM.zip" ] && [ ! -d "models/VideoReTalking/BFM" ]; then
  unzip -q models/VideoReTalking/BFM.zip -d models/VideoReTalking/
  info "  ✓ Unzipped BFM"
fi

# ─── Step 5: Clone GFPGAN ────────────────────────────────────────────────────
info "Setting up GFPGAN..."

if [ ! -d "GFPGAN" ]; then
  git clone https://github.com/TencentARC/GFPGAN GFPGAN
  info "  ✓ Cloned GFPGAN"
else
  info "  ✓ GFPGAN already exists"
fi

# Install GFPGAN deps
cd GFPGAN
pip install --quiet basicsr facexlib gfpgan
pip install --quiet -r requirements.txt 2>/dev/null || true
python setup.py develop --quiet 2>/dev/null || true
cd ..

# Download GFPGAN weights
GFPGAN_WEIGHT="GFPGAN/experiments/pretrained_models/GFPGANv1.4.pth"
if [ ! -f "$GFPGAN_WEIGHT" ]; then
  info "  Downloading GFPGANv1.4.pth..."
  mkdir -p GFPGAN/experiments/pretrained_models
  wget -q --show-progress \
    -O "$GFPGAN_WEIGHT" \
    "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
  info "  ✓ GFPGANv1.4.pth downloaded"
else
  info "  ✓ GFPGANv1.4.pth already present"
fi

# ─── Step 6: Create workspace and output directories ─────────────────────────
mkdir -p workspace output models
info "  ✓ Created workspace/, output/, models/"

# ─── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}  ✅ Setup complete!${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo "Activate the environment:  source venv/bin/activate"
echo ""
echo "Run the pipeline:"
echo "  python dub_video.py --input your_video.mp4 --start 15 --end 30"
echo ""
echo "Quick test (no GPU needed, skips lip-sync):"
echo "  python dub_video.py --input your_video.mp4 --skip-lipsync"
