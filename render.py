"""Makeasy Spark – free Instagram Reel renderer (runs on GitHub Actions).

    python render.py spec.json out/REEL.mp4

Spec (JSON):
{
  "run_id": "spark-20261103-1900",
  "voice": "am_michael", "speed": 1.0,          # Kokoro voice (male, free, Apache-2.0)
  "music": "music/track1.mp3",                  # path in repo or https URL; "" = no music
  "scenes": [
    {"kind": "hook", "kicker": "Prompt fix", "title": "Better AI answers", "highlight": "in 3 steps",
     "vo": "Stop scrolling. Your ChatGPT answers are boring because of your prompt.", "broll": "frustrated man laptop"},
    {"kind": "step", "title": "Give it a role", "try": "You are an accountant for small shops.",
     "vo": "Step one. Give it a role.", "broll": "accountant office desk"},
    {"kind": "cta", "title": "Save this for your next prompt", "question": "Which step do you skip?",
     "vo": "Save this and tell me which step you skip.", "broll": "phone scrolling social media"}
  ]
}
Env: PEXELS_API_KEY (stock video), MOCK=1 (offline test: fake voice + coloured backgrounds).
Output: 1080x1920, 30 fps, H.264 high, yuv420p, AAC 48 kHz, faststart, <= 5 Mbps (Instagram Reel spec).
"""
import json, math, os, random, re, shutil, subprocess, sys, tempfile, wave
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

W, H, FPS = 1080, 1920, 30
HERE = os.path.dirname(os.path.abspath(__file__))
FONT = {w: os.path.join(HERE, "fonts", f"Inter-{w}.otf") for w in ("Black", "ExtraBold", "SemiBold")}
DEEP, MINT, SUN, WHITE = (8, 34, 31), (79, 209, 191), (255, 203, 87), (242, 247, 246)
MOCK = os.environ.get("MOCK") == "1"
NO_FOOTAGE = MOCK or not os.environ.get("PEXELS_API_KEY")   # no key → brand-colour backgrounds
LEAD, TAIL = 0.30, 0.30          # pause before / after each voice line (s)
SR = 24000                        # Kokoro sample rate


def log(*a):
    print("[reel]", *a, flush=True)


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd[:6])}...\n{r.stderr[-2500:]}")
    return r.stdout


# ───────────────────────── voice (Kokoro, free & open source) ─────────────────────────
_pipe = None


def tts(text, voice, speed):
    """Return (float32 mono audio @24k, [(word, start_s, end_s), ...])."""
    global _pipe
    if MOCK:
        words = text.split()
        dur = max(1.2, len(words) / 2.6)
        t = np.linspace(0, dur, int(SR * dur), endpoint=False)
        audio = (0.05 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)
        step = dur / len(words)
        return audio, [(w, i * step, (i + 1) * step) for i, w in enumerate(words)]
    if _pipe is None:
        from kokoro import KPipeline
        _pipe = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")
    chunks, words, offset = [], [], 0.0
    for res in _pipe(text, voice=voice, speed=speed, split_pattern=r"\n+"):
        audio = res.audio.cpu().numpy() if hasattr(res.audio, "cpu") else np.asarray(res.audio)
        for tok in (getattr(res, "tokens", None) or []):
            st, en = getattr(tok, "start_ts", None), getattr(tok, "end_ts", None)
            word = (tok.text or "").strip()
            if st is None or en is None or not re.search(r"\w", word):
                if words and word and not re.search(r"\w", word):   # glue punctuation to previous word
                    w0, s0, e0 = words[-1]
                    words[-1] = (w0 + word, s0, e0)
                continue
            words.append((word, offset + st, offset + en))
        chunks.append(audio.astype(np.float32))
        offset += len(audio) / SR
    audio = np.concatenate(chunks) if chunks else np.zeros(SR, np.float32)
    if not words:  # no timestamps – spread words evenly
        ws = text.split(); step = (len(audio) / SR) / max(1, len(ws))
        words = [(w, i * step, (i + 1) * step) for i, w in enumerate(ws)]
    peak = float(np.max(np.abs(audio))) or 1.0
    return (audio / peak * 0.89).astype(np.float32), words


# ───────────────────────── stock footage (Pexels, free) ─────────────────────────
def pexels_clips(query, n, used):
    import requests
    key = os.environ.get("PEXELS_API_KEY", "")
    out = []
    for q in (query, "technology office", "laptop work"):
        r = requests.get("https://api.pexels.com/videos/search", headers={"Authorization": key},
                         params={"query": q, "orientation": "portrait", "size": "medium", "per_page": 15}, timeout=30)
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
            out.append({"url": f["link"], "credit": f'{v["user"]["name"]} (Pexels)', "dur": v["duration"]})
            if len(out) >= n:
                return out
    return out


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
    """Normalise one b-roll piece to exactly `dur` s of 1080x1920@30, slow pan + slight grade."""
    pan = "(iw-ow)*t/%f" % max(dur, 0.1) if idx % 2 == 0 else "(iw-ow)*(1-t/%f)" % max(dur, 0.1)
    vf = (f"scale={int(W*1.12)}:{int(H*1.12)}:force_original_aspect_ratio=increase,"
          f"crop={W}:{H}:x='{pan}':y='(ih-oh)/2',fps={FPS},"
          "eq=contrast=1.05:saturation=1.08:brightness=-0.02,format=yuv420p")
    if mock_color:
        src_args = ["-f", "lavfi", "-i", f"color=c={mock_color}:s=1280x2276:r={FPS},noise=alls=12:allf=t"]
    else:
        src_args = ["-stream_loop", "-1", "-i", src]
    run(["ffmpeg", "-y", "-loglevel", "error", *src_args, "-t", f"{dur:.3f}", "-an", "-vf", vf,
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "16", out])


# ───────────────────────── text drawing ─────────────────────────
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


def text_block(text, size, weight="Black", color=WHITE, max_w=920, line_h=1.06, shadow=True,
               bg=None, pad=(0, 0), radius=0, align="left"):
    """RGBA image of wrapped text (optional rounded background, soft shadow)."""
    f = font(weight, size)
    lines = wrap(text, f, max_w - 2 * pad[0])
    lh = int(size * line_h)
    tw = int(max(f.getlength(l) for l in lines))
    w, h = tw + 2 * pad[0], lh * len(lines) + 2 * pad[1] + int(size * 0.12)
    m = 30  # margin for shadow
    img = Image.new("RGBA", (w + 2 * m, h + 2 * m), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    if bg:
        d.rounded_rectangle([m, m, m + w, m + h], radius=radius, fill=bg)
    if shadow and not bg:
        sh = Image.new("RGBA", img.size, (0, 0, 0, 0)); sd = ImageDraw.Draw(sh)
        for i, l in enumerate(lines):
            x = m + pad[0] + (0 if align == "left" else (tw - f.getlength(l)) / 2)
            sd.text((x + 3, m + pad[1] + i * lh + 5), l, font=f, fill=(0, 0, 0, 170))
        img = Image.alpha_composite(img, sh.filter(ImageFilter.GaussianBlur(9))); d = ImageDraw.Draw(img)
    for i, l in enumerate(lines):
        x = m + pad[0] + (0 if align == "left" else (tw - f.getlength(l)) / 2)
        d.text((x, m + pad[1] + i * lh), l, font=f, fill=color)
    return img, m


def caption_img(words, active):
    """Creator-style caption: 1–3 words, bold, white with dark outline, spoken word in mint & bigger."""
    f = font("Black", 84)
    parts, x = [], 0
    for i, w in enumerate(words):
        ww = f.getlength(w.upper())
        parts.append((x, w.upper(), i == active)); x += ww + 26
    tw = int(x - 26) + 40
    img = Image.new("RGBA", (tw + 60, 170), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    for px, w, act in parts:
        d.text((30 + px, 40), w, font=f, fill=MINT if act else WHITE,
               stroke_width=9, stroke_fill=(8, 20, 18, 255))
    return img


def chunk_words(words, max_words=3, max_chars=16):
    chunks, cur = [], []
    for w in words:
        if cur and (len(cur) >= max_words or len(" ".join(x[0] for x in cur + [w])) > max_chars
                    or re.search(r"[.!?,:]$", cur[-1][0])):
            chunks.append(cur); cur = []
        cur.append(w)
    if cur:
        chunks.append(cur)
    return chunks


def gradient_overlay():
    a = np.zeros((H, W, 4), np.uint8)
    y = np.linspace(0, 1, H)[:, None]
    top = np.clip(1 - y / 0.22, 0, 1) * 150
    bottom = np.clip((y - 0.42) / 0.58, 0, 1) ** 1.2 * 225
    a[..., 3] = np.clip(top + bottom + 40, 0, 235).astype(np.uint8)
    a[..., 0], a[..., 1], a[..., 2] = DEEP
    return Image.fromarray(a, "RGBA")


# ───────────────────────── scene layout ─────────────────────────
def scene_elements(sc, n_step):
    """List of (img, x, y, appear_t, anim) for one scene; positions are top-left in px."""
    els = []
    k = sc["kind"]
    if k == "hook":
        if sc.get("kicker"):
            im, m = text_block(sc["kicker"].upper(), 34, "ExtraBold", DEEP, bg=MINT + (255,), pad=(26, 14), radius=40, shadow=False)
            els.append((im, 86 - m, 560 - m, 0.0, "fade"))
        im, m = text_block(sc["title"], 118, "Black", WHITE, max_w=940)
        els.append((im, 80 - m, 640 - m, 0.05, "pop"))
        y2 = 640 + im.size[1] - 2 * m + 6
        if sc.get("highlight"):
            im2, m2 = text_block(sc["highlight"], 118, "Black", MINT, max_w=940)
            els.append((im2, 80 - m2, y2 - m2, 0.3, "pop"))
    elif k == "step":
        num, m = text_block(f"{n_step:02d}", 300, "Black", (0, 0, 0, 0), shadow=False)
        d = ImageDraw.Draw(num)
        d.text((m, m), f"{n_step:02d}", font=font("Black", 300), fill=(0, 0, 0, 0), stroke_width=5, stroke_fill=MINT + (255,))
        els.append((num, 70 - m, 250 - m, 0.0, "fade"))
        im, m = text_block(sc["title"], 104, "Black", WHITE, max_w=940)
        els.append((im, 80 - m, 600 - m, 0.08, "slide"))
        if sc.get("try"):
            y = 600 + im.size[1] - 2 * m + 40
            lab, ml = text_block("TRY THIS", 30, "ExtraBold", MINT, shadow=False)
            els.append((lab, 84 - ml, y - ml, 0.4, "fade"))
            card, mc = text_block(sc["try"], 50, "SemiBold", WHITE, max_w=900, bg=(8, 34, 31, 215), pad=(34, 26), radius=28, line_h=1.25)
            els.append((card, 80 - mc, y + 52 - mc, 0.5, "slide"))
    else:  # cta
        im, m = text_block(sc["title"], 104, "Black", WHITE, max_w=880)
        els.append((im, 80 - m, 420 - m, 0.0, "slide"))
        y = 420 + im.size[1] - 2 * m + 40
        tag, mt = text_block("SAVE  ·  SEND  ·  FOLLOW", 40, "ExtraBold", MINT, shadow=False)
        els.append((tag, 84 - mt, y - mt, 0.35, "fade"))
        if sc.get("question"):
            q, mq = text_block(sc["question"], 58, "ExtraBold", DEEP, max_w=880, bg=SUN + (255,), pad=(34, 26), radius=30, shadow=False)
            els.append((q, 80 - mq, y + 90 - mq, 0.6, "pop"))
    return els


def anim(kind, t):
    """(alpha 0..1, dy px, scale) for an element `t` s after it appears."""
    p = min(1.0, max(0.0, t / 0.35))
    e = 1 - (1 - p) ** 3
    if kind == "slide":
        return e, int((1 - e) * 70), 1.0
    if kind == "pop":
        s = 0.8 + 0.2 * (1 - (1 - p) ** 3) + 0.05 * math.sin(p * math.pi) * (p < 1)
        return e, 0, s
    return e, 0, 1.0


# ───────────────────────── main ─────────────────────────
def main(spec_path, out_path):
    spec = json.load(open(spec_path))
    voice, speed = spec.get("voice", "am_michael"), float(spec.get("speed", 1.0))
    scenes = spec["scenes"]
    tmp = tempfile.mkdtemp(prefix="reel_")
    log("tmp", tmp)

    # 1 voice per scene → timeline
    t, voice_track, timeline, credits = 0.0, [], [], []
    for i, sc in enumerate(scenes):
        audio, words = tts(sc["vo"], voice, speed)
        vdur = len(audio) / SR
        dur = LEAD + vdur + TAIL
        voice_track += [np.zeros(int(LEAD * SR), np.float32), audio, np.zeros(int(TAIL * SR), np.float32)]
        timeline.append({"start": t, "dur": dur, "words": [(w, t + LEAD + s, t + LEAD + e) for w, s, e in words]})
        t += dur
    total = t
    log(f"voice done – {total:.1f}s")
    vo = np.concatenate(voice_track)
    with wave.open(os.path.join(tmp, "voice.wav"), "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(SR)
        wf.writeframes((np.clip(vo, -1, 1) * 32767).astype(np.int16).tobytes())

    # 2 b-roll: one new clip every ~2.5 s
    used, seg_paths, idx = set(), [], 0
    palette = ["#0f3d38", "#1d2b4a", "#3a2a1a", "#123a2c", "#2b1d3a"]
    for i, (sc, tl) in enumerate(zip(scenes, timeline)):
        n = max(1, round(tl["dur"] / 2.6))
        clips = [] if NO_FOOTAGE else pexels_clips(sc.get("broll") or "technology", n, used)
        if not NO_FOOTAGE and not clips:
            raise RuntimeError(f"no stock footage found for scene {i+1}")
        part = tl["dur"] / n
        for j in range(n):
            src = None
            if clips:
                c = clips[j % len(clips)]
                src = download(c["url"], os.path.join(tmp, f"src_{i}_{j}.mp4"))
                credits.append(c["credit"])
            seg = os.path.join(tmp, f"seg_{idx:03d}.mp4")
            make_segment(src, part, seg, idx, mock_color=palette[idx % len(palette)] if NO_FOOTAGE else None)
            seg_paths.append(seg); idx += 1
    with open(os.path.join(tmp, "list.txt"), "w") as fh:
        fh.writelines(f"file '{p}'\n" for p in seg_paths)
    bg = os.path.join(tmp, "bg.mp4")
    run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", os.path.join(tmp, "list.txt"),
         "-c", "copy", bg])
    log(f"b-roll done – {len(seg_paths)} clips")

    # 3 overlays
    shade = gradient_overlay()
    logo = None
    lp = os.path.join(HERE, "assets", "logo-dark.png")
    if os.path.exists(lp):
        logo = Image.open(lp).convert("RGBA")
        logo.thumbnail((300, 76))
    n_step, scene_els = 0, []
    for sc, tl in zip(scenes, timeline):
        if sc["kind"] == "step":
            n_step += 1
        scene_els.append(scene_elements(sc, n_step))
    cap_events = []  # (start, end, chunk_words, active_idx)
    for tl in timeline:
        for ch in chunk_words(tl["words"]):
            for k, (w, s, e) in enumerate(ch):
                nxt = ch[k + 1][1] if k + 1 < len(ch) else e + 0.12
                cap_events.append((s, nxt, tuple(x[0] for x in ch), k))
    cap_cache = {}

    music = spec.get("music") or ""
    if music.startswith("http"):
        music = download(music, os.path.join(tmp, "music.mp3"))
    elif music:
        music = os.path.join(HERE, music)
    music = music if music and os.path.exists(music) else ""

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    dec = subprocess.Popen(["ffmpeg", "-loglevel", "quiet", "-i", bg, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                           stdout=subprocess.PIPE)
    enc_cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
               "-r", str(FPS), "-i", "-", "-i", os.path.join(tmp, "voice.wav")]
    if music:
        enc_cmd += ["-stream_loop", "-1", "-i", music]
        afilter = (f"[1:a]aresample=48000,volume=1.0[v];[2:a]aresample=48000,volume=0.13,"
                   f"afade=t=out:st={max(0, total-1.5):.2f}:d=1.5[m];[v][m]amix=inputs=2:duration=first:normalize=0,"
                   "alimiter=limit=0.95[a]")
    else:
        afilter = "[1:a]aresample=48000[a]"
    enc_cmd += ["-filter_complex", afilter, "-map", "0:v", "-map", "[a]", "-t", f"{total:.3f}",
                "-c:v", "libx264", "-profile:v", "high", "-preset", "medium", "-crf", "20",
                "-maxrate", "4500k", "-bufsize", "9000k", "-pix_fmt", "yuv420p", "-g", "60",
                "-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "2", "-movflags", "+faststart", out_path]
    enc = subprocess.Popen(enc_cmd, stdin=subprocess.PIPE)

    nframes = int(total * FPS)
    fsize = W * H * 3
    for fi in range(nframes):
        raw = dec.stdout.read(fsize)
        if len(raw) < fsize:
            break
        tt = fi / FPS
        frame = Image.frombuffer("RGB", (W, H), raw, "raw", "RGB", 0, 1).convert("RGBA")
        frame.alpha_composite(shade)
        si = next(i for i, tl in enumerate(timeline) if tt < tl["start"] + tl["dur"] or i == len(timeline) - 1)
        local = tt - timeline[si]["start"]
        for img, x, y, appear, kind in scene_els[si]:
            if local < appear:
                continue
            a, dy, s = anim(kind, local - appear)
            im = img
            if s != 1.0:
                im = img.resize((max(1, int(img.width * s)), max(1, int(img.height * s))), Image.BILINEAR)
                x2, y2 = x + (img.width - im.width) // 2, y + (img.height - im.height) // 2
            else:
                x2, y2 = x, y
            if a < 1:
                im = im.copy(); im.putalpha(im.getchannel("A").point(lambda v, a=a: int(v * a)))
            frame.alpha_composite(im, (int(x2), int(y2 + dy)))
        for s0, e0, ws, act in cap_events:
            if s0 <= tt < e0:
                key = (ws, act)
                if key not in cap_cache:
                    cap_cache[key] = caption_img(ws, act)
                ci = cap_cache[key]
                p = min(1, (tt - s0) / 0.08)
                sc_ = 0.92 + 0.08 * p
                ci2 = ci.resize((int(ci.width * sc_), int(ci.height * sc_)), Image.BILINEAR) if sc_ < 1 else ci
                frame.alpha_composite(ci2, ((W - ci2.width) // 2, int(H * 0.70) - ci2.height // 2))
                break
        if logo:
            frame.alpha_composite(logo, (80, 92))
        bar = Image.new("RGBA", (W - 160, 8), (255, 255, 255, 55))
        ImageDraw.Draw(bar).rounded_rectangle([0, 0, max(8, int((W - 160) * tt / total)), 7], radius=4, fill=MINT + (255,))
        frame.alpha_composite(bar, (80, 186))
        enc.stdin.write(frame.convert("RGB").tobytes())
        if fi % 150 == 0:
            log(f"frame {fi}/{nframes}")
    enc.stdin.close(); enc.wait(); dec.stdout.close(); dec.wait()
    if enc.returncode != 0:
        raise RuntimeError("encoder failed")
    info = {"duration": round(total, 2), "frames": nframes, "credits": sorted(set(credits)),
            "size_mb": round(os.path.getsize(out_path) / 1e6, 2), "voice": voice}
    json.dump(info, open(out_path + ".json", "w"))
    log("done", info)
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
