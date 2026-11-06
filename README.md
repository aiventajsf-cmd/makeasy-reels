# makeasy-reels

Free Instagram Reel renderer for **Makeasy Spark**. A GitHub Actions run turns a short JSON script into a 1080×1920 Reel: real stock footage (Pixabay, or Pexels), an energetic male AI voice (Chatterbox, MIT, copying a reference voice made with Kokoro, Apache-2.0), word-by-word captions, website screenshots in a phone frame, sound effects, optional royalty-free music, the Makeasy AI logo and a progress bar. The finished MP4 is published on the `videos` branch (GitHub Pages) so Instagram can fetch it, and the result is POSTed to the Make webhook given as `callback`.

- Trigger: **Actions → Render Reel → Run workflow** (inputs `spec`, `callback`), or the GitHub API `POST /repos/aiventajsf-cmd/makeasy-reels/actions/workflows/reel.yml/dispatches`.
- Secret needed: `PIXABAY_API_KEY` (or `PEXELS_API_KEY`) in Settings → Secrets and variables → Actions. Without one, plain brand-colour backgrounds are used.
- Spec format: `render.py` header; example: `examples/ai-news-3-updates.json`.
- Offline layout test: `MOCK=1 python render.py examples/ai-news-3-updates.json out/test.mp4`.
- Music: put royalty-free MP3s in `music/` and pass `"music": "music/<file>.mp3"`.

No keys or passwords are stored in this repository.
