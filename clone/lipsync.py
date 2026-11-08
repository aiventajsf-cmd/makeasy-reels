"""Makeasy clone – lip-sync a base face video to new speech on CPU (MuseTalk v1.5, MIT).

Fast CPU variant of MuseTalk's inference: MediaPipe face landmarks instead of DWPose/mmpose
(DWPose took ~6 s per frame on the free runner), and only the frames that are needed are encoded.

    python clone/lipsync.py --mt /tmp/mt --face base.mp4 --audio speech.wav --out out.mp4 [--start 5]

--mt     MuseTalk checkout with ./models (musetalkV15, sd-vae, whisper, face-parse-bisent)
--start  second of the base video to start from (skip the closed-mouth intro, vary per reel)
"""
import argparse, os, subprocess, sys, time, json
import numpy as np, cv2, torch

ap = argparse.ArgumentParser()
ap.add_argument("--mt", required=True)
ap.add_argument("--face", required=True)
ap.add_argument("--audio", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--fps", type=int, default=25)
ap.add_argument("--start", type=float, default=0.0)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--max_h", type=int, default=1280, help="downscale base frames to this height first")
args = ap.parse_args()

T0 = time.time()
torch.set_num_threads(os.cpu_count() or 4)
os.chdir(args.mt)
sys.path.insert(0, args.mt)
from transformers import WhisperModel
from musetalk.utils.utils import load_all_model, datagen
from musetalk.utils.audio_processor import AudioProcessor
from musetalk.utils.face_parsing import FaceParsing
from musetalk.utils.blending import get_image
import mediapipe as mp

def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)

# ---------- audio length -> number of frames
dur = float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", args.audio]).strip())
n_need = int(np.ceil(dur * args.fps))

# ---------- base frames (resampled to fps, scaled), only as many as needed (ping-pong if the base is short)
tmp = "/tmp/clone_base.mp4"
subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", str(args.start), "-i", args.face, "-an",
                "-vf", f"fps={args.fps},scale=-2:'min({args.max_h},ih)'", "-c:v", "libx264", "-crf", "12", "-preset", "veryfast", tmp], check=True)
cap = cv2.VideoCapture(tmp)
base = []
while len(base) < n_need:
    ok, fr = cap.read()
    if not ok:
        break
    base.append(fr)
cap.release()
if not base:
    sys.exit("no frames in base video")
frames = []
seq = base + base[::-1]
while len(frames) < n_need:
    frames.extend(seq)
frames = frames[:n_need]
H, W = frames[0].shape[:2]
log(f"audio {dur:.1f}s -> {n_need} frames; base {len(base)} frames {W}x{H}")

# ---------- face boxes with MediaPipe (MuseTalk rule: x from jaw line, y2 = chin, y1 = nose - (chin - nose))
OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149,
        150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
JAW = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454]
NOSE = 195
fm = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=False,
                                     min_detection_confidence=0.4, min_tracking_confidence=0.4)
boxes, last = [], None
for fr in base:
    res = fm.process(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    if res.multi_face_landmarks:
        lm = np.array([(p.x * W, p.y * H) for p in res.multi_face_landmarks[0].landmark])
        jaw = lm[JAW]
        chin = jaw[:, 1].max()
        nose = lm[NOSE, 1]
        x1, x2 = jaw[:, 0].min(), jaw[:, 0].max()
        y1 = max(0, nose - (chin - nose))
        last = np.array([x1, y1, x2, chin], dtype=np.float32)
    boxes.append(last)
if all(b is None for b in boxes):
    sys.exit("no face found in the base video")
first = next(b for b in boxes if b is not None)
boxes = [first if b is None else b for b in boxes]
# smooth over time to avoid jitter
B = np.array(boxes)
k = 5
B = np.array([B[max(0, i - k):i + k + 1].mean(0) for i in range(len(B))])
seqb = list(B) + list(B[::-1])
boxes_all = [(seqb * (n_need // len(seqb) + 1))[i] for i in range(n_need)]
log("landmarks done")

# ---------- models
dev = torch.device("cpu")
vae, unet, pe = load_all_model(unet_model_path="models/musetalkV15/unet.pth", vae_type="sd-vae",
                               unet_config="models/musetalkV15/musetalk.json", device=dev)
ts = torch.tensor([0], device=dev)
ap_ = AudioProcessor(feature_extractor_path="models/whisper")
whisper = WhisperModel.from_pretrained("models/whisper").to(dev).eval()
fp = FaceParsing(left_cheek_width=90, right_cheek_width=90)
feats, alen = ap_.get_audio_feature(args.audio)
chunks = ap_.get_whisper_chunk(feats, dev, unet.model.dtype, whisper, alen, fps=args.fps,
                               audio_padding_length_left=2, audio_padding_length_right=2)
n = min(n_need, len(chunks))
log("models + audio features ready", n, "chunks")

EXTRA = 10
def crop_box(b):
    x1, y1, x2, y2 = [int(round(v)) for v in b]
    y2 = min(y2 + EXTRA, H)
    return max(0, x1), max(0, y1), min(W, x2), y2

latents = []
for i in range(n):
    x1, y1, x2, y2 = crop_box(boxes_all[i])
    crop = cv2.resize(frames[i][y1:y2, x1:x2], (256, 256), interpolation=cv2.INTER_LANCZOS4)
    latents.append(vae.get_latents_for_unet(crop))
log("latents encoded")

# ---------- generate + blend, streamed straight into ffmpeg
silent = "/tmp/clone_silent.mp4"
ff = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
                       "-r", str(args.fps), "-i", "-", "-c:v", "libx264", "-crf", "16", "-preset", "veryfast",
                       "-pix_fmt", "yuv420p", silent], stdin=subprocess.PIPE)
idx = 0
with torch.no_grad():
    for wb, lb in datagen(whisper_chunks=chunks[:n], vae_encode_latents=latents, batch_size=args.batch, delay_frame=0, device=dev):
        af = pe(wb)
        pred = unet.model(lb.to(dtype=unet.model.dtype), ts, encoder_hidden_states=af).sample
        for res in vae.decode_latents(pred):
            x1, y1, x2, y2 = crop_box(boxes_all[idx])
            res = cv2.resize(res.astype(np.uint8), (x2 - x1, y2 - y1))
            out = get_image(frames[idx].copy(), res, [x1, y1, x2, y2], mode="jaw", fp=fp)
            ff.stdin.write(np.ascontiguousarray(out, dtype=np.uint8).tobytes())
            idx += 1
        if idx % 50 < args.batch:
            log(f"frame {idx}/{n}")
ff.stdin.close(); ff.wait()
subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", silent, "-i", args.audio, "-c:v", "copy", "-c:a", "aac",
                "-b:a", "160k", "-shortest", args.out], check=True)
total = time.time() - T0
log(f"done: {args.out} ({n} frames, {total / max(n, 1):.2f} s/frame)")
json.dump({"frames": n, "seconds": round(total, 1), "s_per_frame": round(total / max(n, 1), 3), "size": [W, H]},
          open(args.out + ".json", "w"))
