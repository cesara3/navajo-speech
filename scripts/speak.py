"""
speak.py -- make your trained VITS model say a sentence.

Run from ~/vits (vits env):
    python speak.py                      # says the test sentence from your plot
    python speak.py --text "Hello there" # says your own sentence (needs espeak)

Saves  generated.wav  (the model)  next to  LJ022-0023.wav  (the real recording)
so you can listen to both and compare.
"""
import argparse
import torch
from scipy.io.wavfile import write

import commons
import utils
from models import SynthesizerTrn
from text.symbols import symbols
from text import cleaned_text_to_sequence

p = argparse.ArgumentParser()
p.add_argument("--text", default=None, help="your own English sentence (optional)")
p.add_argument("--ckpt", default="logs/small_test/G_1000.pth")
p.add_argument("--config", default="configs/ljs_small.json")
p.add_argument("--out", default="generated.wav")
args = p.parse_args()

hps = utils.get_hparams_from_file(args.config)
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Build the model (random knobs) and load the knobs training learned
net_g = SynthesizerTrn(len(symbols),
                       hps.data.filter_length // 2 + 1,
                       hps.train.segment_size // hps.data.hop_length,
                       **hps.model).to(device)
net_g.eval()
utils.load_checkpoint(args.ckpt, net_g, None)

# 2. Text -> sounds (phonemes)
if args.text:
    from text import _clean_text            # needs phonemizer + espeak installed
    phonemes = _clean_text(args.text, hps.data.text_cleaners)
    print("your text :", args.text)
else:
    line = open("filelists/ljs_audio_text_val_filelist.txt.cleaned", encoding="utf-8").readline()
    phonemes = line.split("|", 1)[1].strip()
    print("test sentence (never seen in training) -- real version is LJ022-0023.wav")
print("phonemes  :", phonemes)

# 3. Sounds -> ID numbers (+ blanks), same as step 2 of explore_one_clip.py
ids = cleaned_text_to_sequence(phonemes)
if hps.data.add_blank:
    ids = commons.intersperse(ids, 0)
x = torch.LongTensor(ids).unsqueeze(0).to(device)
x_len = torch.LongTensor([x.size(1)]).to(device)
print("IDs       :", len(ids), "numbers")

# 4. Model: IDs -> durations -> audio  (only the text side is used now)
with torch.no_grad():
    audio = net_g.infer(x, x_len, noise_scale=0.667, noise_scale_w=0.8,
                        length_scale=1.0)[0][0, 0].cpu().float().numpy()

write(args.out, hps.data.sampling_rate, audio)
print(f"audio     : {len(audio):,} samples = {len(audio)/hps.data.sampling_rate:.2f} s")
print("saved ->", args.out)
