"""Create the male reference voice that Chatterbox copies (run once; cached by the workflow).

Uses Kokoro (Apache-2.0) so the reference is free to use commercially.
    python scripts/make_voice_ref.py assets/voice_ref.wav [voice]
"""
import sys
import numpy as np
import soundfile as sf
from kokoro import KPipeline

out = sys.argv[1]
voice = sys.argv[2] if len(sys.argv) > 2 else "am_michael"
text = ("Hey everyone! Today I'm showing you three AI tools that will completely change how you work. "
        "Trust me, number two is a total game changer. Let's go!")
pipe = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")
audio = np.concatenate([np.asarray(r.audio) for r in pipe(text, voice=voice, speed=1.12)])
sf.write(out, audio / np.max(np.abs(audio)) * 0.9, 24000)
print("voice reference written:", out, round(len(audio) / 24000, 1), "s")
