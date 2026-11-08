"""Makeasy clone – voice samples with the trained model (no retraining).

Inputs (attached by the workflow):
  kernel aiventa/makeasy-voice-train  -> junaid_voice.zip (RVC .pth + .index)
  dataset aiventa/makeasy-clone-private -> voice_ref.wav (+ hf_token.txt if the owner added one)
Output: /kaggle/working/samples/*.wav and logs/*.txt
"""
import glob, json, os, shutil, subprocess, time, zipfile

T0 = time.time()
WORK, TMP = "/kaggle/working", "/kaggle/tmp"
os.makedirs(f"{WORK}/samples", exist_ok=True)
os.makedirs(f"{WORK}/logs", exist_ok=True)
os.makedirs(TMP, exist_ok=True)


def log(m):
    print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)


def sh(cmd, name, cwd=None):
    log("$ " + cmd[:150])
    with open(f"{WORK}/logs/{name}.txt", "a") as f:
        r = subprocess.run(cmd, shell=True, cwd=cwd, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode:
        log(f"FAILED {name} ({r.returncode}) – last lines:")
        print("".join(open(f"{WORK}/logs/{name}.txt").readlines()[-25:]), flush=True)
    return r.returncode


def find(fname):
    for root, _, files in os.walk("/kaggle/input"):
        if fname in files:
            return os.path.join(root, fname)
    return None


model_zip, ref = find("junaid_voice.zip"), find("voice_ref.wav")
log(f"model {model_zip} ref {ref}")
mdir = f"{TMP}/model"
zipfile.ZipFile(model_zip).extractall(mdir)
shutil.copy(ref, f"{TMP}/voice_ref.wav")
hf = find("hf_token.txt")
HF = open(hf).read().strip() if hf else ""

LINES = {
    "intro": "Hi, I'm Junaid, and this is the first time I'm showing my face on Makeasy AI. "
             "For the last few days, every post on this page was made by AI agents that I built myself.",
    "news": "Big news this week! Google just made its AI video model thirty percent cheaper. "
            "Honestly, the speed of these updates is unbelievable.",
}
json.dump(LINES, open(f"{TMP}/lines.json", "w"))
sh("pip install -q uv", "setup")

# ---- Applio (for converting into Junaid's voice)
sh(f"git clone -q --depth 1 https://github.com/IAHispano/Applio.git {TMP}/applio", "applio")
A = f"{TMP}/applio"
sh(f"uv venv -q -p 3.11 {TMP}/rvc", "applio")
PY = f"{TMP}/rvc/bin/python"
sh(f"uv pip install -q -p {PY} torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu126", "applio")
sh(f"uv pip install -q -p {PY} -r requirements-engine.txt", "applio", cwd=A)
sh(f"{PY} core.py prerequisites --no-pretraineds-hifigan --models --no-exe", "applio", cwd=A)

# ---- base voice A: Chatterbox with Junaid's reference (three energy levels)
sh(f"uv venv -q -p 3.11 {TMP}/cb", "chatterbox")
CP = f"{TMP}/cb/bin/python"
sh(f"uv pip install -q -p {CP} 'setuptools<70' torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124 --extra-index-url https://pypi.org/simple", "chatterbox")
sh(f"uv pip install -q -p {CP} chatterbox-tts==0.1.7 'setuptools<70'", "chatterbox")
open(f"{TMP}/cb.py", "w").write('''
import json, torchaudio
from chatterbox.tts import ChatterboxTTS
m = ChatterboxTTS.from_pretrained(device="cuda")
lines = json.load(open("/kaggle/tmp/lines.json"))
for tag, ex, cfg in [("cb_lively", 0.7, 0.4), ("cb_energetic", 0.95, 0.3)]:
    for k, text in lines.items():
        w = m.generate(text, audio_prompt_path="/kaggle/tmp/voice_ref.wav", exaggeration=ex, cfg_weight=cfg, temperature=0.8)
        torchaudio.save(f"/kaggle/tmp/base_{tag}_{k}.wav", w, m.sr)
print("chatterbox done")
''')
sh(f"{CP} {TMP}/cb.py", "chatterbox")

# ---- base voice B: Indic Parler-TTS (Indian English) – only if the owner added a Hugging Face token
if HF:
    sh(f"uv venv -q -p 3.11 {TMP}/ptts", "parler")
    PP = f"{TMP}/ptts/bin/python"
    sh(f"uv pip install -q -p {PP} torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121 --extra-index-url https://pypi.org/simple", "parler")
    sh(f"uv pip install -q -p {PP} git+https://github.com/huggingface/parler-tts.git soundfile sentencepiece", "parler")
    open(f"{TMP}/parler.py", "w").write('''
import json, os, torch, soundfile as sf
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
tok_hf = os.environ["HF_TOKEN"]
m = ParlerTTSForConditionalGeneration.from_pretrained("ai4bharat/indic-parler-tts", token=tok_hf).to("cuda")
tok = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts", token=tok_hf)
dtok = AutoTokenizer.from_pretrained(m.config.text_encoder._name_or_path)
desc = {"parler": "A young Indian man speaks English with a clear Indian accent in a lively, excited and engaging tone at a moderately fast pace. The recording is very clear and close-up with no background noise."}
for tag, d in desc.items():
    di = dtok(d, return_tensors="pt").to("cuda")
    for k, text in json.load(open("/kaggle/tmp/lines.json")).items():
        p = tok(text, return_tensors="pt").to("cuda")
        torch.manual_seed(7)
        g = m.generate(input_ids=di.input_ids, attention_mask=di.attention_mask, prompt_input_ids=p.input_ids, prompt_attention_mask=p.attention_mask)
        sf.write(f"/kaggle/tmp/base_{tag}_{k}.wav", g.cpu().numpy().squeeze(), m.config.sampling_rate)
print("parler done")
''')
    sh(f"HF_TOKEN={HF} {PP} {TMP}/parler.py", "parler")
else:
    log("no hf_token.txt – skipping Indic Parler-TTS")

# ---- convert every base into Junaid's voice
for base in sorted(glob.glob(f"{TMP}/base_*.wav")):
    name = os.path.basename(base)[5:-4]
    shutil.copy(base, f"{WORK}/samples/raw_{name}.wav")
    sh(f"{PY} core.py infer --input-path {base} --output-path {WORK}/samples/junaid_{name}.wav "
       f"--pth-path {mdir}/junaid.pth --index-path {mdir}/junaid.index --pitch 0 --index-rate 0.66 "
       f"--protect 0.4 --f0-method rmvpe --export-format WAV", "convert", cwd=A)
log("samples: " + ", ".join(sorted(os.listdir(f"{WORK}/samples"))))
