import subprocess, json, torch, platform
info = {"cuda": torch.cuda.is_available(), "python": platform.python_version(), "torch": torch.__version__}
if torch.cuda.is_available():
    info["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    info["vram_gb"] = [round(torch.cuda.get_device_properties(i).total_memory / 1e9, 1) for i in range(torch.cuda.device_count())]
try:
    import urllib.request; urllib.request.urlopen("https://huggingface.co", timeout=15); info["internet"] = True
except Exception as e:
    info["internet"] = str(e)[:80]
print(json.dumps(info)); json.dump(info, open("/kaggle/working/gpu.json", "w"))
