#!/bin/bash
# =====================================================================
# team_setup.sh -- one-time VITS setup on Easley for every team member.
#
# Run it ONCE, on the Easley login node (no GPU needed):
#     bash ~/navajo-speech/team_setup.sh
#
# It is safe to run again: every step skips work that is already done.
# Takes ~30 minutes, mostly waiting on downloads. Use tmux so a dropped
# connection doesn't stop it (see README).
# =====================================================================
set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # this repo
VITS_DIR="$HOME/vits"                                         # where VITS goes

step() { echo; echo "=================================================="; echo "  $*"; echo "=================================================="; }
ok()   { echo "  [ok] $*"; }

# ---------------------------------------------------------------------
step "1/7  Loading conda"
module load miniconda3
ok "miniconda3 loaded"

# ---------------------------------------------------------------------
step "2/7  Getting the VITS code (+ our fixes for new PyTorch)"
if [ -d "$VITS_DIR/.git" ]; then
  ok "~/vits already exists, not downloading again"
else
  git clone https://github.com/jaywalnut310/vits.git "$VITS_DIR"
fi
cd "$VITS_DIR"
if git apply --check "$REPO_DIR/patches/vits_patches.diff" 2>/dev/null; then
  git apply "$REPO_DIR/patches/vits_patches.diff"
  ok "fixes applied to mel_processing.py, train.py, utils.py"
else
  ok "fixes already applied, skipping"
fi

# ---------------------------------------------------------------------
step "3/7  Creating the 'vits' Python environment (slowest step, ~10 min)"
if conda env list | grep -qE '^vits[[:space:]]'; then
  ok "environment 'vits' already exists"
else
  conda create -y -n vits python=3.10
fi
source activate vits
pip install --no-cache-dir torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install --no-cache-dir -r "$REPO_DIR/requirements.txt"
python -c "import torch, librosa; print('  [ok] torch', torch.__version__, '| librosa', librosa.__version__)"

# ---------------------------------------------------------------------
step "4/7  Downloading the LJSpeech dataset (2.6 GB, then unpacking)"
if [ -d LJSpeech-1.1/wavs ] && [ "$(ls LJSpeech-1.1/wavs | wc -l)" -eq 13100 ]; then
  ok "LJSpeech already here (13100 clips)"
else
  wget -c https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
  echo "  unpacking... this prints nothing for several minutes. Don't press anything."
  tar -xjf LJSpeech-1.1.tar.bz2
  rm -f LJSpeech-1.1.tar.bz2
  ok "$(ls LJSpeech-1.1/wavs | wc -l) clips unpacked"
fi
[ -e DUMMY1 ] || ln -s "$VITS_DIR/LJSpeech-1.1/wavs" DUMMY1
ok "DUMMY1 -> LJSpeech-1.1/wavs"

# ---------------------------------------------------------------------
step "5/7  Building the alignment code (monotonic_align)"
cd "$VITS_DIR/monotonic_align"
mkdir -p monotonic_align
python setup.py build_ext --inplace > build.log 2>&1 || { echo "  [!!] build failed, see ~/vits/monotonic_align/build.log"; exit 1; }
cd "$VITS_DIR"
ls monotonic_align/monotonic_align/*.so > /dev/null && ok "alignment code built"

# ---------------------------------------------------------------------
step "6/7  Making the small practice set (200 train / 20 test clips)"
mkdir -p filelists/small logs
head -n 200 filelists/ljs_audio_text_train_filelist.txt.cleaned > filelists/small/train_200.txt.cleaned
head -n 20  filelists/ljs_audio_text_val_filelist.txt.cleaned   > filelists/small/val_20.txt.cleaned
python -c "import json; c=json.load(open('configs/ljs_base.json')); c['data']['training_files']='filelists/small/train_200.txt.cleaned'; c['data']['validation_files']='filelists/small/val_20.txt.cleaned'; c['train'].update(batch_size=8, log_interval=10, eval_interval=200, epochs=40); json.dump(c, open('configs/ljs_small.json','w'), indent=2)"
ok "configs/ljs_small.json"

# ---------------------------------------------------------------------
step "7/7  Adding the team helper scripts"
cp "$REPO_DIR"/scripts/explore_one_clip.py "$REPO_DIR"/scripts/speak.py "$REPO_DIR"/scripts/train.sh "$VITS_DIR/"
ok "explore_one_clip.py, speak.py, train.sh copied into ~/vits"

echo
echo "=================================================="
echo "  SETUP DONE. Try your first command:"
echo
echo "    cd ~/vits"
echo "    python explore_one_clip.py --wav_dir DUMMY1"
echo
echo "  Every new login, run these first:"
echo "    module load miniconda3"
echo "    source activate vits"
echo "    cd ~/vits"
echo "=================================================="
