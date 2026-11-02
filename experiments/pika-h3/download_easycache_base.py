"""Fetch the original BF16 H3 base for the fresh local EasyCache comparison."""
import hashlib
import json
import os
from pathlib import Path
import time
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'easycache-bf16'
OUT.mkdir(exist_ok=True)
started = time.perf_counter()
path = Path(hf_hub_download('Comfy-Org/MiniMax-H3',
    filename='diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors',
    revision='0bd506d2e895983a9663037febda27aa3948cf48',
    local_dir=OUT / 'models', cache_dir=OUT / 'hf-cache'))
assert path.stat().st_size == 40225724176
digest = hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()
assert digest == 'a32572fb90b5508b201ec7c2eddcc184b13ddfd3c6f6d2cf06a0b46535d541b4'
record = {'path': str(path.relative_to(ROOT)), 'bytes': path.stat().st_size, 'sha256': digest,
          'download_and_verify_seconds': time.perf_counter() - started,
          'repository': 'Comfy-Org/MiniMax-H3', 'revision': '0bd506d2e895983a9663037febda27aa3948cf48'}
(OUT / 'base-provenance.json').write_text(json.dumps(record, indent=2))
print(json.dumps(record), flush=True)
