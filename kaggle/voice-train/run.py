"""Makeasy clone – train Junaid's voice model on Kaggle (free T4 GPU) and make comparison samples.

Input  : private Kaggle dataset aiventa/makeasy-clone-private (voice_train.wav, voice_ref.wav)
Output : /kaggle/working/junaid_voice.zip (RVC .pth + .index) and samples/*.wav
Tools  : Applio (RVC, MIT) for the voice model · Indic Parler-TTS (Apache-2.0, AI4Bharat) for an
         Indian-English base voice · Chatterbox (MIT) as a second base for comparison.
"""
import glob, json, os, shutil, subprocess, sys, time

T0 = time.time()
IN = "/kaggle/input/makeasy-clone-private"
WORK = "/kaggle/working"
TMP = "/kaggle/tmp"
EPOCHS = int(os.environ.get("EPOCHS", "220"))
os.makedirs(TMP, exist_ok=True)
os.makedirs(f"{WORK}/samples", exist_ok=True)
status = {"steps": []}


def log(msg):
    line = f"[{time.time() - T0:7.0f}s] {msg}"
    print(line, flush=True)
    status["steps"].append(line)
    json.dump(status, open(f"{WORK}/status.json", "w"), indent=1)


def sh(cmd, cwd=None, check=True):
    log("$ " + cmd[:160])
    r = subprocess.run(cmd, shell=True, cwd=cwd)
    if check and r.returncode != 0:
        log(f"FAILED ({r.returncode}): {cmd[:120]}")
        raise SystemExit(r.returncode)
    return r.returncode


# ---------------- inputs
for f in os.listdir(IN):
    log(f"input {f} {os.path.getsize(os.path.join(IN, f)) // 1024} KB")
sh("pip install -q uv")
sh("apt-get -qq update > /dev/null 2>&1; apt-get -qq install -y ffmpeg > /dev/null 2>&1 || true", check=False)

# ---------------- 1. Applio (RVC) in its own Python 3.11 env
sh(f"git clone -q --depth 1 https://github.com/IAHispano/Applio.git {TMP}/applio")
A = f"{TMP}/applio"
sh(f"uv venv -q -p 3.11 {TMP}/rvc")
PY = f"{TMP}/rvc/bin/python"
sh(f"uv pip install -q -p {PY} torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu126")
sh(f"uv pip install -q -p {PY} -r {A}/requirements-engine.txt", cwd=A)
sh(f"{PY} core.py prerequisites --pretraineds-hifigan --models --no-exe", cwd=A)

ds = f"{TMP}/dataset"
os.makedirs(ds, exist_ok=True)
shutil.copy(f"{IN}/voice_train.wav", f"{ds}/junaid.wav")
sh(f"{PY} core.py preprocess --model-name junaid --dataset-path {ds} --sample-rate 40000 --cpu-cores 4 "
   f"--cut-preprocess Automatic --chunk-len 3.0 --overlap-len 0.3 --normalization-mode post", cwd=A)
sh(f"{PY} core.py extract --model-name junaid --f0-method rmvpe --cpu-cores 4 --gpu 0 --sample-rate 40000 "
   f"--embedder-model contentvec", cwd=A)
log(f"training {EPOCHS} epochs")
sh(f"{PY} core.py train --model-name junaid --save-every-epoch 20 --save-only-latest --total-epoch {EPOCHS} "
   f"--sample-rate 40000 --batch-size 8 --gpu 0 --pretrained --index-algorithm Auto", cwd=A)
sh(f"{PY} core.py index --model-name junaid --index-algorithm Auto", cwd=A, check=False)

pths = sorted(glob.glob(f"{A}/logs/junaid/junaid_*e_*s.pth"), key=os.path.getmtime) or \
       sorted(glob.glob(f"{A}/logs/junaid/*.pth"), key=os.path.getmtime)
idxs = sorted(glob.glob(f"{A}/logs/junaid/*.index"), key=os.path.getmtime)
if not pths:
    log("no model produced")
    raise SystemExit(1)
pth, idx = pths[-1], (idxs[-1] if idxs else "")
log(f"model {os.path.basename(pth)} index {os.path.basename(idx)}")
mdir = f"{TMP}/model"
os.makedirs(mdir, exist_ok=True)
shutil.copy(pth, f"{mdir}/junaid.pth")
if idx:
    shutil.copy(idx, f"{mdir}/junaid.index")
shutil.make_archive(f"{WORK}/junaid_voice", "zip", mdir)

# ---------------- 2. base voices for the comparison samples
LINES = {
    "intro": "Hi, I'm Junaid, and this is the first time I'm showing my face on Makeasy AI. "
             "For the last few days, every post on this page was made by AI agents that I built myself.",
    "news": "Big news this week! Google just made its AI video model thirty percent cheaper. "
            "Honestly, the speed of these updates is unbelievable.",
}
json.dump(LINES, open(f"{TMP}/lines.json", "w"))

# 2a. Indic Parler-TTS – Indian-English male, energetic
sh(f"uv venv -q -p 3.11 {TMP}/ptts")
PP = f"{TMP}/ptts/bin/python"
sh(f"uv pip install -q -p {PP} torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121")
sh(f"uv pip install -q -p {PP} git+https://github.com/huggingface/parler-tts.git soundfile")
open(f"{TMP}/parler.py", "w").write('''
import json, torch, soundfile as sf
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
dev = "cuda"
m = ParlerTTSForConditionalGeneration.from_pretrained("ai4bharat/indic-parler-tts").to(dev)
tok = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts")
dtok = AutoTokenizer.from_pretrained(m.config.text_encoder._name_or_path)
lines = json.load(open("/kaggle/tmp/lines.json"))
voices = {
  "aakash": "Aakash speaks with an Indian English accent in an energetic, upbeat and expressive tone at a slightly fast pace, like a confident young content creator. The recording is very clear with no background noise.",
  "male": "A young Indian man speaks English with a clear Indian accent in a lively, excited and engaging tone at a moderately fast pace. The recording is very clear and close-up with no background noise.",
}
for v, desc in voices.items():
    d = dtok(desc, return_tensors="pt").to(dev)
    for k, text in lines.items():
        p = tok(text, return_tensors="pt").to(dev)
        torch.manual_seed(7)
        g = m.generate(input_ids=d.input_ids, attention_mask=d.attention_mask,
                       prompt_input_ids=p.input_ids, prompt_attention_mask=p.attention_mask)
        sf.write(f"/kaggle/tmp/base_parler_{v}_{k}.wav", g.cpu().numpy().squeeze(), m.config.sampling_rate)
print("parler done")
''')
sh(f"{PP} {TMP}/parler.py", check=False)

# 2b. Chatterbox with Junaid's reference (energetic settings)
sh(f"uv venv -q -p 3.11 {TMP}/cb")
CP = f"{TMP}/cb/bin/python"
sh(f"uv pip install -q -p {CP} torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124")
sh(f"uv pip install -q -p {CP} chatterbox-tts==0.1.7")
open(f"{TMP}/cb.py", "w").write('''
import json, torchaudio
from chatterbox.tts import ChatterboxTTS
m = ChatterboxTTS.from_pretrained(device="cuda")
for k, text in json.load(open("/kaggle/tmp/lines.json")).items():
    w = m.generate(text, audio_prompt_path="/kaggle/input/makeasy-clone-private/voice_ref.wav", exaggeration=0.8, cfg_weight=0.35)
    torchaudio.save(f"/kaggle/tmp/base_chatterbox_{k}.wav", w, m.sr)
print("chatterbox done")
''')
sh(f"{CP} {TMP}/cb.py", check=False)

# ---------------- 3. convert every base to Junaid's voice with the trained model
for base in sorted(glob.glob(f"{TMP}/base_*.wav")):
    name = os.path.basename(base)[5:-4]
    shutil.copy(base, f"{WORK}/samples/raw_{name}.wav")
    sh(f"{PY} core.py infer --input-path {base} --output-path {WORK}/samples/junaid_{name}.wav "
       f"--pth-path {mdir}/junaid.pth --index-path {mdir}/junaid.index --pitch 0 --index-rate 0.6 "
       f"--protect 0.4 --f0-method rmvpe --export-format WAV", cwd=A, check=False)
log("all done: " + ", ".join(sorted(os.listdir(f"{WORK}/samples"))))
