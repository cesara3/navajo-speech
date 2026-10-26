# Navajo Speech — Team Setup

Everything the team needs to train a text-to-speech (TTS) model on the Easley cluster.
We start by training a small English model (LJSpeech dataset) so everyone learns
the full process. Then we move to Navajo.

---

## Two windows, two computers

You'll use **two terminal windows**. Check the prompt to know where you are.

| Prompt looks like | You're on | Use it for |
|---|---|---|
| `[yourname@easley ~]$` | **Easley** (the cluster) | almost everything |
| your laptop's prompt (`%` on Mac, `PS C:\>` on Windows) | **your laptop** | only `scp` (copy files) and playing audio |

---

## One-time setup (~30 min, mostly waiting)

**1. Log into Easley** (in your laptop terminal):

```bash
ssh <your_username>@easley.alliance.unm.edu
```

**2. Start tmux**, so the setup keeps going if your Wi-Fi drops:

```bash
tmux new -s setup
```

(If you get disconnected, log back in and type `tmux attach -t setup`.)

**3. Run the setup** (one line, replace `<OWNER>` with the GitHub username that owns this repo):

```bash
git clone https://github.com/<OWNER>/navajo-speech.git ~/navajo-speech && bash ~/navajo-speech/team_setup.sh
```

It goes through 7 steps and ends with **SETUP DONE**. If it stops with an error,
copy the last ~20 lines and send them to the team chat.

---

## Every time you log in

```bash
module load miniconda3
source activate vits
cd ~/vits
```

Your prompt should now start with `(vits)`.

---

## The practice run (what to do the first time)

### Step 1: look at one example (no GPU needed)

```bash
python explore_one_clip.py --wav_dir DUMMY1
```

It prints what the model sees: sentence → phonemes (sounds) → ID numbers → audio → spectrogram.
Add `--index 5` to look at a different sentence.

### Step 2: get a GPU and train (~15 min)

```bash
srun -p interactive --gres=gpu:1 --cpus-per-task=4 --mem=32G -t 01:00:00 --pty bash
```

Wait until the prompt changes from `easley` to something like `easley055`. Then:

```bash
module load miniconda3
source activate vits
cd ~/vits
python train.py -c configs/ljs_small.json -m my_first_run
```

Every 10 steps it prints a list of numbers. **The 4th number is the mel loss**:
how far the model's audio is from the real audio. Watch it drop from ~66 to ~35.

### Step 3: make it speak

```bash
python speak.py --ckpt logs/my_first_run/G_1000.pth
```

This makes `generated.wav`: the model saying a sentence it never trained on.
When you're done with the GPU, type `exit`.

### Step 4: listen (in your **laptop** window)

```bash
scp <your_username>@easley.alliance.unm.edu:~/vits/generated.wav .
scp <your_username>@easley.alliance.unm.edu:~/vits/DUMMY1/LJ022-0023.wav .
```

- **Mac:** `afplay generated.wav` then `afplay LJ022-0023.wav`
- **Windows:** `start generated.wav` then `start LJ022-0023.wav`

The first is our model (robotic). The second is the real recording.

---

## Longer runs (overnight): sbatch

`srun` stops if you close your laptop. For long runs, submit a job instead:

```bash
sbatch train.sh ljs_small.json my_long_run
squeue -u $USER                          # is it running?
tail -f logs/my_long_run/train.log       # watch the loss (Ctrl+C stops watching)
```

To train on more clips, see "Changing the amount of data" below.

### Changing the amount of data

```bash
N=1000; EP=40
head -n $N filelists/ljs_audio_text_train_filelist.txt.cleaned > filelists/small/train_$N.txt.cleaned
python -c "import json; c=json.load(open('configs/ljs_small.json')); c['data']['training_files']='filelists/small/train_$N.txt.cleaned'; c['train']['epochs']=$EP; json.dump(c,open('configs/ljs_$N.json','w'),indent=2)"
sbatch train.sh ljs_$N.json run_$N
```

---

## Team rules

- **Run names:** `name_date_what`, e.g. `cesar_1026_1000clips`. Never reuse a name (it continues the old run).
- **Log every run** in the team experiment sheet: who, date, config, steps, final mel loss, notes.
- **Ask before changing** `team_setup.sh` or `requirements.txt`. Everyone depends on them.
- When you're done with a GPU, `exit` so someone else can use it.

---

## Troubleshooting

| Error | Fix |
|---|---|
| `conda: command not found` | `module load miniconda3` |
| `CondaError: Run 'conda init'` | use `source activate vits` instead of `conda activate` |
| `No module named 'numpy'` (or torch) | you forgot `source activate vits` |
| `No such file or directory` with `scp` | you ran `scp` on Easley; run it in your laptop window |
| Prompt shows `>` and nothing happens | press `Ctrl+C` |
| Setup seems frozen at "unpacking" | it's working, wait (~5 min) |

---

## What's in this repo

| File | What it does |
|---|---|
| `team_setup.sh` | one-time setup (VITS code, environment, data, practice config) |
| `requirements.txt` | Python package versions we all use |
| `patches/vits_patches.diff` | fixes so the 2021 VITS code works with today's PyTorch |
| `scripts/explore_one_clip.py` | shows what the model sees for one clip |
| `scripts/speak.py` | makes a trained model say a sentence |
| `scripts/train.sh` | Slurm script for long training runs |

Helper scripts were written with Claude; training uses the original
[VITS](https://github.com/jaywalnut310/vits) code.
