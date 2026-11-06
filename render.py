"""Makeasy Spark – free Instagram Reel renderer v2 (runs on GitHub Actions, all free / open source).

    python render.py spec.json out/REEL.mp4

v2 (Nov 2026): energetic Chatterbox voice (MIT) · captions timed to the real speech (faster-whisper) ·
live website screenshots in a phone frame with a scroll · punch-in cuts every ~2 s · whoosh/pop SFX.

Spec (JSON):
{
  "run_id": "spark-20261106-1930",
  "voice_ref": "assets/voice_ref.wav",   # male reference voice Chatterbox copies (made once, see workflow)
  "energy": 0.75,                         # Chatterbox exaggeration 0.25–1.0 (higher = more energetic)
  "music": "",                            # repo path or https URL of a royalty-free MP3; "" = none
  "scenes": [
    {"kind": "hook", "tag": "AI NEWS · THIS WEEK", "title": "3 AI updates", "highlight": "you can use today",
     "vo": "Three AI updates from this week that you can actually use. Number three is wild!",
     "broll": "futuristic technology city night"},
    {"kind": "item", "num": 1, "title": "Cursor now builds AND ships your app",
     "shot": "https://cursor.com", "vo": "...", "broll": "developer coding laptop"},
    {"kind": "cta", "title": "Follow for daily AI updates", "question": "Which one will you try first?",
     "vo": "...", "broll": "phone scrolling social media"}
  ]
}
Env: PIXABAY_API_KEY or PEXELS_API_KEY (stock video; without one brand-colour backgrounds are used), MOCK=1 (offline test).
Output: 1080x1920, 30 fps, H.264 high, yuv420p, AAC 48 kHz stereo, faststart, ≤ 4.5 Mbps (Instagram Reel spec).
"""
import json, math, os, re, shutil, subprocess, sys, tempfile, wave
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

W, H, FPS = 1080, 1920, 30
HERE = os.path.dirname(os.path.abspath(__file__))
FONT = {w: os.path.join(HERE, "fonts", f"Inter-{w}.otf") for w in ("Black", "ExtraBold", "SemiBold")}
DEEP, MINT, SUN, WHITE, INK = (8, 34, 31), (79, 209, 191), (255, 214, 10), (250, 250, 247), (10, 14, 13)
MOCK = os.environ.get("MOCK") == "1"
NO_FOOTAGE = MOCK or not (os.environ.get("PEXELS_API_KEY") or os.environ.get("PIXABAY_API_KEY"))
SR = 24000
LEAD, TAIL = 0.12, 0.18          # tight pacing: tiny pause before/after each line
CUT = 2.0                         # new b-roll shot every ~2 s
# cinematic look on every shot: teal shadows / warm highlights, filmic contrast, soft vignette, fine grain, crisp detail
CINE_GRADE = ("colorbalance=rs=-0.05:gs=0.0:bs=0.07:rm=0.02:bm=-0.02:rh=0.07:gh=0.02:bh=-0.06,"
              "eq=contrast=1.14:saturation=1.08:brightness=-0.035:gamma=0.96,"
              "curves=all='0/0.03 0.25/0.21 0.5/0.5 0.75/0.79 1/0.97',"
              "vignette=angle=PI/4.6:mode=forward,unsharp=5:5:0.55:5:5:0.0,noise=alls=4:allf=t")
CAP_Y = 1395                      # caption centre (kept above Instagram's bottom UI zone)


def log(*a):
    print("[reel]", *a, flush=True)


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd[:6])}...\n{r.stderr[-2500:]}")
    return r.stdout


def write_wav(path, audio, sr=SR):
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
        wf.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())


# ───────────────────────── voice: Chatterbox (MIT) + word timing: faster-whisper ─────────────────────────
_tts, _asr = None, None


def tts(text, ref, energy):
    global _tts
    if MOCK:
        dur = max(1.2, len(text.split()) / 2.9)
        t = np.linspace(0, dur, int(SR * dur), endpoint=False)
        return (0.05 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)
    if _tts is None:
        import torch
        torch.set_num_threads(os.cpu_count() or 4)
        from chatterbox.tts import ChatterboxTTS
        _tts = ChatterboxTTS.from_pretrained(device="cpu")
        if ref and os.path.exists(ref):
            _tts.prepare_conditionals(ref, exaggeration=energy)
    # energetic delivery: high exaggeration + low cfg keeps the pace natural (per Chatterbox docs)
    wav = _tts.generate(text, exaggeration=energy, cfg_weight=0.35, temperature=0.8)
    audio = wav.squeeze(0).numpy().astype(np.float32)
    if _tts.sr != SR:
        import librosa
        audio = librosa.resample(audio, orig_sr=_tts.sr, target_sr=SR)
    # trim leading/trailing silence
    nz = np.where(np.abs(audio) > 0.01)[0]
    if len(nz):
        audio = audio[max(0, nz[0] - int(0.03 * SR)): nz[-1] + int(0.08 * SR)]
    peak = float(np.max(np.abs(audio))) or 1.0
    return (audio / peak * 0.9).astype(np.float32)


def word_times(audio, text, tmp):
    """Word-level timestamps of what was actually spoken."""
    global _asr
    words = text.split()
    dur = len(audio) / SR
    if not MOCK:
        try:
            if _asr is None:
                from faster_whisper import WhisperModel
                _asr = WhisperModel("base.en", device="cpu", compute_type="int8")
            p = os.path.join(tmp, "align.wav"); write_wav(p, audio)
            segs, _ = _asr.transcribe(p, word_timestamps=True, initial_prompt=text, language="en")
            out = [(w.word.strip(), w.start, w.end) for s in segs for w in (s.words or []) if w.word.strip()]
            if out:
                return out
        except Exception as e:  # noqa
            log("alignment failed, spreading words evenly:", e)
    step = dur / max(1, len(words))
    return [(w, i * step, (i + 1) * step) for i, w in enumerate(words)]


# ───────────────────────── sound effects (synthesised, no licence needed) ─────────────────────────
def sfx_whoosh(d=0.32):
    n = int(SR * d); t = np.linspace(0, 1, n)
    noise = np.random.default_rng(7).standard_normal(n)
    # sweeping band: simple one-pole low-pass with moving cutoff
    out, y, a0 = np.zeros(n), 0.0, 0.02
    for i in range(n):
        a = a0 + 0.35 * t[i]
        y += a * (noise[i] - y); out[i] = y
    env = np.sin(np.pi * t) ** 1.6
    return (out * env / (np.max(np.abs(out)) or 1) * 0.35).astype(np.float32)


def sfx_pop(d=0.09):
    n = int(SR * d); t = np.arange(n) / SR
    f = 950 - 4000 * t
    return (np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 45) * 0.28).astype(np.float32)


# ───────────────────────── stock footage (Pexels / Pixabay, free) ─────────────────────────
def pexels_clips(query, n, used):
    import requests
    key = os.environ.get("PEXELS_API_KEY", "")
    out = []
    for q in (query, "cinematic technology", "cinematic city night"):
        r = requests.get("https://api.pexels.com/videos/search", headers={"Authorization": key},
                         params={"query": q, "orientation": "portrait", "size": "medium", "per_page": 20}, timeout=30)
        if r.status_code != 200:
            log("pexels", r.status_code, r.text[:200]); continue
        for v in r.json().get("videos", []):
            if v["id"] in used or v.get("duration", 0) < 3:
                continue
            files = [f for f in v["video_files"] if f.get("height") and f.get("width") and f["height"] > f["width"]]
            files = [f for f in files if f["height"] >= 1280] or files
            if not files:
                continue
            f = min(files, key=lambda f: abs(f["height"] - 1920))
            used.add(v["id"])
            out.append({"url": f["link"], "credit": f'{v["user"]["name"]} (Pexels)'})
            if len(out) >= n:
                return out
    return out


def pixabay_clips(query, n, used):
    """Pixabay videos (free, commercial use, no attribution needed). Prefers portrait, else centre-cropped."""
    import requests
    key = os.environ.get("PIXABAY_API_KEY", "")
    out, cands = [], []
    for q in (query, "cinematic technology", "cinematic city night"):
        r = requests.get("https://pixabay.com/api/videos/", params={"key": key, "q": q[:100], "per_page": 30,
                         "safesearch": "true", "order": "popular"}, timeout=30)
        if r.status_code != 200:
            log("pixabay", r.status_code, r.text[:200]); continue
        for v in r.json().get("hits", []):
            if v["id"] in used or v.get("duration", 0) < 3:
                continue
            files = [f for f in v.get("videos", {}).values() if f.get("url") and f.get("height")]
            if not files:
                continue
            portrait = any(f["height"] > f["width"] for f in files)
            if portrait:
                good = [f for f in files if min(f["width"], f["height"]) >= 1080] or files
                f = min(good, key=lambda f: abs(max(f["width"], f["height"]) - 1920))
            else:  # landscape is centre-cropped to 9:16 → take the sharpest (up to 4K) so it stays crisp
                f = max([f for f in files if max(f["width"], f["height"]) <= 3840] or files, key=lambda f: f["height"])
            cands.append((0 if portrait else 1, len(cands), v["id"], f["url"], v.get("user", "")))
        if len([c for c in cands if c[2] not in used]) >= n:
            break
    for _, _, vid, url, user in sorted(cands):
        if vid in used:
            continue
        used.add(vid)
        out.append({"url": url, "credit": f"{user} (Pixabay)"})
        if len(out) >= n:
            break
    return out


def footage_clips(query, n, used):
    q = query if "cinematic" in query.lower() else f"cinematic {query}"
    clips = footage_search(q, n, used)
    if len(clips) < n:
        clips += footage_search(query, n - len(clips), used)
    return clips


def footage_search(query, n, used):
    clips = []
    if os.environ.get("PEXELS_API_KEY"):
        clips = pexels_clips(query, n, used)
    if len(clips) < n and os.environ.get("PIXABAY_API_KEY"):
        clips += pixabay_clips(query, n - len(clips), used)
    return clips


def download(url, path):
    import requests
    for attempt in range(3):
        try:
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(path, "wb") as fh:
                    for chunk in r.iter_content(1 << 20):
                        fh.write(chunk)
            return path
        except Exception as e:  # noqa
            log("download retry", attempt, e)
    raise RuntimeError(f"could not download {url}")


def make_segment(src, dur, out, idx, mock_color=None):
    """One b-roll shot: exactly `dur` s at 1080x1920@30 with a punch-in zoom and slow drift."""
    z = "1.0+0.08*max(0\\,1-t/0.35)+0.05*t/%f" % max(dur, 0.1)       # soft punch-in, then slow cinematic push
    vf = ("setpts=1.2*PTS," if not mock_color else "") + (f"scale=w='trunc({W}*1.06*({z})/2)*2':h='trunc({H}*1.06*({z})/2)*2':eval=frame:"
          "force_original_aspect_ratio=increase,"
          f"crop={W}:{H}:x='(iw-ow)/2':y='(ih-oh)/2',fps={FPS},"
          + CINE_GRADE + ",format=yuv420p")
    if mock_color:
        src_args = ["-f", "lavfi", "-i", f"color=c={mock_color}:s=1200x2134:r={FPS},noise=alls=14:allf=t"]
    else:
        src_args = ["-stream_loop", "-1", "-i", src]
    run(["ffmpeg", "-y", "-loglevel", "error", *src_args, "-t", f"{dur:.3f}", "-an", "-vf", vf,
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "16", out])


# ───────────────────────── website screenshot (Playwright, free) ─────────────────────────
BLOCKED = re.compile(r"verify you are human|security verification|captcha|access denied|just a moment|"
                     r"unusual traffic|are you a robot|enable javascript and cookies", re.I)


def fallback_card(url, title, path):
    """Clean branded card used when a site blocks screenshots."""
    dom = re.sub(r"^https?://(www\.)?", "", url).split("/")[0]
    img = Image.new("RGB", (1030, 1250), (250, 250, 247)); d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 1030, 120], fill=(236, 238, 240))
    d.rounded_rectangle([40, 30, 990, 90], 30, fill=WHITE); d.text((80, 40), dom, font=font("SemiBold", 34), fill=(90, 96, 100))
    f = font("Black", 92); y = 260
    for line in wrap(title, f, 900):
        d.text((515 - f.getlength(line) / 2, y), line, font=f, fill=INK); y += 104
    d.rounded_rectangle([315, y + 60, 715, y + 150], 45, fill=SUN)
    t2 = "NEW THIS WEEK"; f2 = font("ExtraBold", 38); d.text((515 - f2.getlength(t2) / 2, y + 84), t2, font=f2, fill=INK)
    img.save(path); return path


def screenshot(url, path, title=""):
    if MOCK:
        img = Image.new("RGB", (1030, 2600), (245, 246, 248)); d = ImageDraw.Draw(img)
        d.rectangle([0, 0, 1030, 150], fill=(20, 20, 24)); d.text((40, 50), url, font=font("ExtraBold", 44), fill=WHITE)
        for i in range(12):
            d.rounded_rectangle([40, 220 + i * 200, 990, 380 + i * 200], 24, fill=(225, 229, 235))
        img.save(path); return path
    from playwright.sync_api import sync_playwright
    mobile = dict(viewport={"width": 412, "height": 915}, device_scale_factor=2.5, is_mobile=True,
                  user_agent="Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 "
                             "(KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1")
    desktop = dict(viewport={"width": 520, "height": 1100}, device_scale_factor=2,
                   user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                              "Chrome/130.0 Safari/537.36")
    with sync_playwright() as p:
        b = p.chromium.launch()
        ok = False
        for opts in (mobile, desktop):
            try:
                pg = b.new_page(**opts)
                pg.goto(url, wait_until="domcontentloaded", timeout=45000)
                pg.wait_for_timeout(4000)
                try:
                    pg.locator("text=/^(Accept|Accept all|I agree|Got it|Allow all|OK)$/i").first.click(timeout=1500)
                except Exception:  # noqa
                    pass
                pg.wait_for_timeout(800)
                body = pg.inner_text("body")[:4000]
                if BLOCKED.search(body) or len(body.strip()) < 40:
                    log("screenshot blocked/empty:", url); pg.close(); continue
                pg.screenshot(path=path, full_page=True); ok = True; break
            except Exception as e:  # noqa
                log("screenshot error:", url, e)
        b.close()
    if not ok:
        return fallback_card(url, title, path)
    img = Image.open(path).convert("RGB")
    if img.height > 2800:
        img.crop((0, 0, img.width, 2800)).save(path)
    return path


def phone_card(shot_path, w=600, h=720):
    """Phone-style frame (RGBA) + the screenshot scaled to the frame width (scrolled at render time)."""
    shot = Image.open(shot_path).convert("RGB")
    sw = w - 28
    shot = shot.resize((sw, int(shot.height * sw / shot.width)), Image.LANCZOS)
    m = 40
    frame = Image.new("RGBA", (w + 2 * m, h + 2 * m), (0, 0, 0, 0))
    sh = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    ImageDraw.Draw(sh).rounded_rectangle([m, m + 14, m + w, m + h + 14], 56, fill=(0, 0, 0, 170))
    frame.alpha_composite(sh.filter(ImageFilter.GaussianBlur(22)))
    ImageDraw.Draw(frame).rounded_rectangle([m, m, m + w, m + h], 56, fill=(14, 16, 18, 255))
    mask = Image.new("L", (sw, h - 28), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, sw - 1, h - 29], 44, fill=255)
    return {"frame": frame, "shot": shot, "mask": mask, "m": m, "inner": (m + 14, m + 14), "view_h": h - 28}


# ───────────────────────── text ─────────────────────────
@lru_cache(maxsize=64)
def font(weight, size):
    return ImageFont.truetype(FONT[weight], size)


def wrap(text, fnt, max_w):
    lines, cur = [], ""
    for word in text.split():
        test = (cur + " " + word).strip()
        if fnt.getlength(test) <= max_w or not cur:
            cur = test
        else:
            lines.append(cur); cur = word
    if cur:
        lines.append(cur)
    return lines


def text_block(text, size, weight="Black", color=WHITE, max_w=920, line_h=1.04, shadow=True,
               bg=None, pad=(0, 0), radius=0, align="center", marker=None):
    """RGBA wrapped text. marker=(r,g,b) draws a highlighter box behind each line (modern Reel look)."""
    f = font(weight, size)
    lines = wrap(text, f, max_w - 2 * pad[0])
    lh = int(size * line_h)
    tw = int(max(f.getlength(l) for l in lines))
    w, h = tw + 2 * pad[0], lh * len(lines) + 2 * pad[1] + int(size * 0.14)
    m = 34
    img = Image.new("RGBA", (w + 2 * m + 40, h + 2 * m), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    if bg:
        d.rounded_rectangle([m, m, m + w, m + h], radius=radius, fill=bg)

    def xpos(l):
        return m + pad[0] + (0 if align == "left" else (tw - f.getlength(l)) / 2)

    if marker:
        for i, l in enumerate(lines):
            x = xpos(l)
            d.rounded_rectangle([x - 18, m + pad[1] + i * lh + size * 0.06, x + f.getlength(l) + 18,
                                 m + pad[1] + i * lh + size * 1.08], radius=14, fill=marker + (255,))
    if shadow and not bg and not marker:
        sh = Image.new("RGBA", img.size, (0, 0, 0, 0)); sd = ImageDraw.Draw(sh)
        for i, l in enumerate(lines):
            sd.text((xpos(l) + 3, m + pad[1] + i * lh + 6), l, font=f, fill=(0, 0, 0, 190))
        img = Image.alpha_composite(img, sh.filter(ImageFilter.GaussianBlur(10))); d = ImageDraw.Draw(img)
    for i, l in enumerate(lines):
        d.text((xpos(l), m + pad[1] + i * lh), l, font=f, fill=color)
    return img, m


def caption_img(words, active):
    """Bold creator captions: white words with dark outline, the spoken word yellow and slightly bigger."""
    parts, x = [], 0
    for i, w in enumerate(words):
        f = font("Black", 96 if i == active else 88)
        ww = f.getlength(w.upper())
        parts.append((x, w.upper(), i == active, f)); x += ww + 24
    img = Image.new("RGBA", (int(x) + 80, 190), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    for px, w, act, f in parts:
        d.text((40 + px, 50 if act else 56), w, font=f, fill=SUN if act else WHITE,
               stroke_width=10, stroke_fill=(5, 8, 8, 255))
    return img


def chunk_words(words, max_words=3, max_chars=15):
    chunks, cur = [], []
    for w in words:
        if cur and (len(cur) >= max_words or len(" ".join(x[0] for x in cur + [w])) > max_chars
                    or re.search(r"[.!?,:]$", cur[-1][0])):
            chunks.append(cur); cur = []
        cur.append(w)
    if cur:
        chunks.append(cur)
    return chunks


def shade(strength):
    a = np.zeros((H, W, 4), np.uint8)
    y = np.linspace(0, 1, H)[:, None]
    top = np.clip(1 - y / 0.2, 0, 1) * 140
    bottom = np.clip((y - 0.5) / 0.5, 0, 1) ** 1.3 * 200
    a[..., 3] = np.clip(top + bottom + strength, 0, 235).astype(np.uint8)
    a[..., 0], a[..., 1], a[..., 2] = (4, 10, 10)
    return Image.fromarray(a, "RGBA")


# ───────────────────────── scene layout ─────────────────────────
def scene_elements(sc, card):
    """[(img, x, y, appear_t, anim)] – x,y = top-left px."""
    els, k = [], sc["kind"]

    def centred(img, y):
        return (W - img.width) // 2, y

    if k == "hook":
        if sc.get("tag"):
            im, _ = text_block(sc["tag"].upper(), 34, "ExtraBold", INK, bg=MINT + (255,), pad=(28, 14), radius=40, shadow=False)
            els.append((im, *centred(im, 560), 0.0, "pop"))
        im, _ = text_block(sc["title"], 132, "Black", WHITE, max_w=960)
        els.append((im, *centred(im, 650), 0.0, "pop"))
        if sc.get("highlight"):
            im2, _ = text_block(sc["highlight"], 92, "Black", INK, max_w=940, marker=SUN, shadow=False)
            els.append((im2, *centred(im2, 650 + im.height - 40), 0.35, "pop"))
    elif k == "item":
        badge, _ = text_block(f"#{sc.get('num', 1)}", 54, "Black", INK, bg=SUN + (255,), pad=(26, 8), radius=22, shadow=False)
        els.append((badge, *centred(badge, 250), 0.0, "pop"))
        size = 74
        while size > 50 and len(wrap(sc["title"], font("Black", size), 940)) > 2:
            size -= 4
        im, _ = text_block(sc["title"], size, "Black", WHITE, max_w=940)
        els.append((im, *centred(im, 352), 0.08, "slide"))
    else:  # cta
        im, _ = text_block(sc["title"], 104, "Black", WHITE, max_w=940)
        els.append((im, *centred(im, 520), 0.0, "pop"))
        if sc.get("question"):
            q, _ = text_block(sc["question"], 60, "Black", INK, max_w=900, marker=SUN, shadow=False)
            els.append((q, *centred(q, 520 + im.height + 10), 0.4, "pop"))
    return els


def anim(kind, t):
    p = min(1.0, max(0.0, t / 0.3))
    e = 1 - (1 - p) ** 3
    if kind == "slide":
        return e, int((1 - e) * 80), 1.0
    if kind == "pop":
        return min(1, p * 2), 0, 0.7 + 0.3 * e + 0.08 * math.sin(p * math.pi)
    return e, 0, 1.0


def blit(frame, img, x, y, a=1.0, s=1.0, dy=0):
    im = img
    if s != 1.0:
        im = img.resize((max(1, int(img.width * s)), max(1, int(img.height * s))), Image.BILINEAR)
        x, y = x + (img.width - im.width) // 2, y + (img.height - im.height) // 2
    if a < 1:
        im = im.copy(); im.putalpha(im.getchannel("A").point(lambda v, a=a: int(v * a)))
    frame.alpha_composite(im, (int(x), int(y + dy)))


# ───────────────────────── main ─────────────────────────
def main(spec_path, out_path):
    spec = json.load(open(spec_path))
    scenes = spec["scenes"]
    ref = os.path.join(HERE, spec.get("voice_ref") or "assets/voice_ref.wav")
    energy = float(spec.get("energy", 0.75))
    tmp = tempfile.mkdtemp(prefix="reel_")

    # 1 · voice + word timing → timeline
    t, track, timeline = 0.0, [], []
    for sc in scenes:
        audio = tts(sc["vo"], ref, energy)
        words = word_times(audio, sc["vo"], tmp)
        dur = LEAD + len(audio) / SR + TAIL
        track += [np.zeros(int(LEAD * SR), np.float32), audio, np.zeros(int(TAIL * SR), np.float32)]
        timeline.append({"start": t, "dur": dur, "words": [(w, t + LEAD + s, t + LEAD + e) for w, s, e in words]})
        t += dur
        log(f"voice scene {len(timeline)}: {dur:.1f}s")
    total = t
    vo = np.concatenate(track)

    # 2 · b-roll: a new shot every ~2 s (pattern interrupts)
    used, segs, cuts, credits, idx = set(), [], [], [], 0
    palette = ["#0f3d38", "#1d2b4a", "#3a2a1a", "#123a2c", "#2b1d3a", "#14313d"]
    for i, (sc, tl) in enumerate(zip(scenes, timeline)):
        n = max(1, round(tl["dur"] / CUT))
        clips = [] if NO_FOOTAGE else footage_clips(sc.get("broll") or "technology", n, used)
        part = tl["dur"] / n
        for j in range(n):
            src = None
            if clips:
                c = clips[j % len(clips)]
                src = download(c["url"], os.path.join(tmp, f"src_{i}_{j}.mp4")); credits.append(c["credit"])
            seg = os.path.join(tmp, f"seg_{idx:03d}.mp4")
            make_segment(src, part, seg, idx, mock_color=None if clips else palette[idx % len(palette)])
            segs.append(seg); cuts.append(tl["start"] + j * part); idx += 1
    with open(os.path.join(tmp, "list.txt"), "w") as fh:
        fh.writelines(f"file '{p}'\n" for p in segs)
    bg = os.path.join(tmp, "bg.mp4")
    run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", os.path.join(tmp, "list.txt"), "-c", "copy", bg])
    log(f"b-roll: {len(segs)} shots")

    # 3 · SFX on cuts and pops → mix into voice track
    whoosh, pop = sfx_whoosh(), sfx_pop()
    fx = np.zeros(len(vo) + SR, np.float32)
    for k, ct in enumerate(cuts):
        if k == 0:
            continue
        i0 = max(0, int((ct - 0.16) * SR)); fx[i0:i0 + len(whoosh)] += whoosh[: len(fx) - i0]
    for tl in timeline:
        i0 = int(tl["start"] * SR); fx[i0:i0 + len(pop)] += pop[: len(fx) - i0]
    mix = vo + fx[: len(vo)] * 0.8
    write_wav(os.path.join(tmp, "voice.wav"), mix / max(1.0, float(np.max(np.abs(mix)))))

    # 4 · overlays
    cards = []
    for sc in scenes:
        c = None
        if sc.get("shot"):
            try:
                c = phone_card(screenshot(sc["shot"], os.path.join(tmp, f"shot_{len(cards)}.png"), sc.get("title", "")))
            except Exception as e:  # noqa
                log("screenshot failed:", sc["shot"], e)
        cards.append(c)
    els = [scene_elements(sc, c) for sc, c in zip(scenes, cards)]
    shades = {False: shade(40), True: shade(95)}
    caps = []
    for tl in timeline:
        for ch in chunk_words(tl["words"]):
            for k, (w, s, e) in enumerate(ch):
                nxt = ch[k + 1][1] if k + 1 < len(ch) else e + 0.1
                caps.append((s, nxt, tuple(x[0] for x in ch), k))
    cap_cache = {}
    logo = None
    lp = os.path.join(HERE, "assets", "logo-dark.png")
    if os.path.exists(lp):
        logo = Image.open(lp).convert("RGBA"); logo.thumbnail((260, 66))

    music = spec.get("music") or ""
    if music.startswith("http"):
        music = download(music, os.path.join(tmp, "music.mp3"))
    elif music:
        music = os.path.join(HERE, music)
    music = music if music and os.path.exists(music) else ""

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    dec = subprocess.Popen(["ffmpeg", "-loglevel", "quiet", "-i", bg, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                           stdout=subprocess.PIPE)
    enc_cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
               "-r", str(FPS), "-i", "-", "-i", os.path.join(tmp, "voice.wav")]
    if music:
        enc_cmd += ["-stream_loop", "-1", "-i", music]
        af = (f"[1:a]aresample=48000[v];[2:a]aresample=48000,volume=0.12,afade=t=out:st={max(0, total-1.2):.2f}:d=1.2[m];"
              "[v][m]amix=inputs=2:duration=first:normalize=0,alimiter=limit=0.95[a]")
    else:
        af = "[1:a]aresample=48000[a]"
    enc_cmd += ["-filter_complex", af, "-map", "0:v", "-map", "[a]", "-t", f"{total:.3f}",
                "-c:v", "libx264", "-profile:v", "high", "-preset", "medium", "-crf", "19",
                "-maxrate", "4500k", "-bufsize", "9000k", "-pix_fmt", "yuv420p", "-g", "60",
                "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-ac", "2", "-movflags", "+faststart", out_path]
    enc = subprocess.Popen(enc_cmd, stdin=subprocess.PIPE)

    nframes, fsize = int(total * FPS), W * H * 3
    for fi in range(nframes):
        raw = dec.stdout.read(fsize)
        if len(raw) < fsize:
            break
        tt = fi / FPS
        si = next(i for i, tl in enumerate(timeline) if tt < tl["start"] + tl["dur"] or i == len(timeline) - 1)
        local = tt - timeline[si]["start"]
        card = cards[si]
        frame = Image.frombuffer("RGB", (W, H), raw, "raw", "RGB", 0, 1).convert("RGBA")
        frame.alpha_composite(shades[card is not None])
        for img, x, y, appear, kind in els[si]:
            if local >= appear:
                a, dy, s = anim(kind, local - appear)
                blit(frame, img, x, y, a, s, dy)
        if card:   # phone mockup slides up, then the page scrolls like a screen recording
            p = min(1.0, max(0.0, (local - 0.15) / 0.4)); e = 1 - (1 - p) ** 3
            fr = card["frame"]; fx0 = (W - fr.width) // 2; fy0 = 585 - card["m"] + int((1 - e) * 500)
            if p > 0:
                f2 = fr.copy()
                maxs = max(0, card["shot"].height - card["view_h"])
                off = int(min(maxs, max(0.0, local - 0.9) * 190))
                view = card["shot"].crop((0, off, card["shot"].width, off + card["view_h"]))
                f2.paste(view, card["inner"], card["mask"])
                blit(frame, f2, fx0, fy0, a=min(1, p * 2))
        for s0, e0, ws, act in caps:
            if s0 <= tt < e0:
                key = (ws, act)
                if key not in cap_cache:
                    cap_cache[key] = caption_img(ws, act)
                ci = cap_cache[key]
                sc_ = 0.9 + 0.1 * min(1, (tt - s0) / 0.07)
                blit(frame, ci, (W - ci.width) // 2, CAP_Y - ci.height // 2, s=sc_)
                break
        if logo:
            frame.alpha_composite(logo, ((W - logo.width) // 2, 96))
        bar = Image.new("RGBA", (W - 200, 7), (255, 255, 255, 50))
        ImageDraw.Draw(bar).rounded_rectangle([0, 0, max(7, int((W - 200) * tt / total)), 6], radius=3, fill=SUN + (255,))
        frame.alpha_composite(bar, (100, 186))
        enc.stdin.write(frame.convert("RGB").tobytes())
        if fi % 150 == 0:
            log(f"frame {fi}/{nframes}")
    enc.stdin.close(); enc.wait(); dec.stdout.close(); dec.wait()
    if enc.returncode != 0:
        raise RuntimeError("encoder failed")
    info = {"duration": round(total, 2), "frames": nframes, "credits": sorted(set(credits)),
            "size_mb": round(os.path.getsize(out_path) / 1e6, 2), "voice": "chatterbox", "energy": energy}
    json.dump(info, open(out_path + ".json", "w"))
    log("done", info)
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
