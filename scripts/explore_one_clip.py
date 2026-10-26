"""
explore_one_clip.py  --  See exactly what VITS "sees" for ONE training example.

Run from inside your vits folder (in the vits conda env):

    python explore_one_clip.py --wav_dir /path/to/LJSpeech-1.1/wavs

Optional:  --index 3      (pick a different line from the filelist)
           --config configs/ljs_base.json

It prints the text -> phonemes -> numbers pipeline, and saves
clip_explained.png with the waveform, spectrogram and mel spectrogram.
No GPU needed; this is safe to run on the login node (it's tiny).
"""
import argparse
import importlib.util
import json
import os

import numpy as np
import torch
import librosa
import matplotlib
matplotlib.use("Agg")          # no screen on the cluster -> save to a file instead
import matplotlib.pyplot as plt
from scipy.io.wavfile import read

p = argparse.ArgumentParser()
p.add_argument("--wav_dir", required=True, help="folder that holds the LJ*.wav files")
p.add_argument("--index", type=int, default=0, help="which line of the filelist to look at")
p.add_argument("--config", default="configs/ljs_base.json")
args = p.parse_args()

# ---------------------------------------------------------------------------
# 0. Settings the model uses (same file train.py reads)
# ---------------------------------------------------------------------------
with open(args.config) as f:
    data_cfg = json.load(f)["data"]
sr      = data_cfg["sampling_rate"]   # 22050 samples per second
n_fft   = data_cfg["filter_length"]   # 1024 samples per FFT window
hop     = data_cfg["hop_length"]      # slide the window 256 samples each step
win     = data_cfg["win_length"]
n_mels  = data_cfg["n_mel_channels"]  # 80 mel bands

# ---------------------------------------------------------------------------
# 1. Read the SAME line from the raw-text filelist and the phoneme filelist
#    Each line is:   path/to/audio.wav|text
# ---------------------------------------------------------------------------
raw_list   = "filelists/ljs_audio_text_val_filelist.txt"
clean_list = "filelists/ljs_audio_text_val_filelist.txt.cleaned"
raw_line   = open(raw_list,   encoding="utf-8").read().splitlines()[args.index]
clean_line = open(clean_list, encoding="utf-8").read().splitlines()[args.index]

wav_path_in_list, raw_text = raw_line.split("|", 1)
_, phonemes                = clean_line.split("|", 1)
wav_path = os.path.join(args.wav_dir, os.path.basename(wav_path_in_list))

print("=" * 70)
print("STEP 1: one training example = (audio file, sentence)")
print("=" * 70)
print("audio file :", wav_path)
print("raw text   :", raw_text)
print("phonemes   :", phonemes)
print("  -> phonemes are 'how it sounds' symbols (IPA). The model reads these,")
print("     not English spelling, because spelling is a bad guide to sound.")

# ---------------------------------------------------------------------------
# 2. Phonemes -> numbers. A neural net only eats numbers, so every symbol
#    gets an ID (like a label index in your KNN code).
#    We load text/symbols.py directly so we don't need phonemizer installed.
# ---------------------------------------------------------------------------
spec_ = importlib.util.spec_from_file_location("symbols", "text/symbols.py")
sym_mod = importlib.util.module_from_spec(spec_)
spec_.loader.exec_module(sym_mod)
symbols = sym_mod.symbols
sym_to_id = {s: i for i, s in enumerate(symbols)}

ids = [sym_to_id[ch] for ch in phonemes]
if data_cfg.get("add_blank", False):
    # VITS puts a 0 ("blank") between every symbol: [a, b] -> [0, a, 0, b, 0]
    with_blanks = [0] * (len(ids) * 2 + 1)
    with_blanks[1::2] = ids
else:
    with_blanks = ids

print()
print("=" * 70)
print("STEP 2: phonemes -> integer IDs (the model's actual text input)")
print("=" * 70)
print(f"vocabulary size : {len(symbols)} symbols")
print("first 10 symbols:", [f"{c!r}->{sym_to_id[c]}" for c in phonemes[:10]])
print(f"length          : {len(ids)} IDs  ({len(with_blanks)} after adding blanks)")
print("first 20 IDs    :", with_blanks[:20])

# ---------------------------------------------------------------------------
# 3. Audio -> numbers. A .wav is just a long list of air-pressure values.
# ---------------------------------------------------------------------------
file_sr, audio = read(wav_path)
assert file_sr == sr, f"wav is {file_sr} Hz but config expects {sr} Hz"
audio = torch.FloatTensor(audio.astype(np.float32)) / data_cfg["max_wav_value"]  # scale to -1..1

print()
print("=" * 70)
print("STEP 3: audio -> waveform (one number per 1/22050 of a second)")
print("=" * 70)
print(f"samples  : {len(audio):,}")
print(f"duration : {len(audio) / sr:.2f} s")
print(f"range    : {audio.min():.3f} to {audio.max():.3f}")

# ---------------------------------------------------------------------------
# 4. Waveform -> spectrogram (same math as VITS mel_processing.py).
#    Chop audio into overlapping windows; for each window measure how much
#    energy is at each frequency. Result: a 2-D picture, frequency x time.
# ---------------------------------------------------------------------------
y = audio.unsqueeze(0)
pad = (n_fft - hop) // 2
y = torch.nn.functional.pad(y.unsqueeze(1), (pad, pad), mode="reflect").squeeze(1)
stft = torch.stft(y, n_fft, hop_length=hop, win_length=win,
                  window=torch.hann_window(win), center=False,
                  normalized=False, onesided=True, return_complex=True)
spec = torch.sqrt(stft.real ** 2 + stft.imag ** 2 + 1e-6).squeeze(0)   # [513, frames]

mel_basis = torch.from_numpy(librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels,
                                                 fmin=data_cfg["mel_fmin"],
                                                 fmax=data_cfg["mel_fmax"])).float()
mel = torch.log(torch.clamp(mel_basis @ spec, min=1e-5))                 # [80, frames]

frames = spec.shape[1]
print()
print("=" * 70)
print("STEP 4: waveform -> spectrogram (a picture of the sound)")
print("=" * 70)
print(f"linear spectrogram shape : {tuple(spec.shape)}  (frequency bins x time frames)")
print(f"mel spectrogram shape    : {tuple(mel.shape)}  ({n_mels} 'ear-like' bands x time frames)")
print(f"1 frame = {hop} samples = {1000 * hop / sr:.1f} ms of audio")
print(f"check: {len(audio):,} samples / {hop} hop ~= {len(audio) // hop} frames")

# ---------------------------------------------------------------------------
# 5. The big picture: what training has to learn
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("STEP 5: what VITS has to learn from this pair")
print("=" * 70)
print(f"INPUT : {len(with_blanks)} symbol IDs")
print(f"OUTPUT: {len(audio):,} audio samples ({frames} spectrogram frames)")
print(f"=> about {frames / len(ids):.1f} frames per phoneme. Nobody tells the model")
print("   which frames belong to which phoneme -- it has to discover that")
print("   alignment itself (that's what monotonic_align does during training).")

# ---------------------------------------------------------------------------
# 6. Plot it
# ---------------------------------------------------------------------------
t = np.arange(len(audio)) / sr
fig, ax = plt.subplots(3, 1, figsize=(12, 9))
ax[0].plot(t, audio.numpy(), linewidth=0.5)
ax[0].set_title("1) Waveform: what the .wav file stores")
ax[0].set_xlabel("time (s)"); ax[0].set_ylabel("amplitude")
ax[0].set_xlim(0, t[-1])

ext = [0, len(audio) / sr, 0, sr / 2]
ax[1].imshow(torch.log(spec).numpy(), origin="lower", aspect="auto", extent=ext, cmap="magma")
ax[1].set_title("2) Linear spectrogram: VITS's encoder learns from this")
ax[1].set_xlabel("time (s)"); ax[1].set_ylabel("frequency (Hz)")

ax[2].imshow(mel.numpy(), origin="lower", aspect="auto",
             extent=[0, len(audio) / sr, 0, n_mels], cmap="magma")
ax[2].set_title("3) Mel spectrogram: used to score how close the generated audio is")
ax[2].set_xlabel("time (s)"); ax[2].set_ylabel("mel band")

fig.suptitle(f'"{raw_text[:90]}{"..." if len(raw_text) > 90 else ""}"', fontsize=10)
fig.tight_layout()
fig.savefig("clip_explained.png", dpi=120)
print()
print("saved -> clip_explained.png")
