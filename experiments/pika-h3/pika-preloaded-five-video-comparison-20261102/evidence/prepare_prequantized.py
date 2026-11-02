"""Create persistent quantized Pika artifacts; this runs outside inference timing."""
import hashlib
import json
from pathlib import Path
import time
import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'preloaded'
OUT.mkdir(exist_ok=True)
torch.set_num_threads(8)
source = ROOT / 'models/loras/pika_h3_flex_fl2va_comfy_bf16.safetensors'
source_hash = hashlib.file_digest(source.open('rb'), 'sha256').hexdigest()
weights = load_file(str(source))
assert len(weights) == 416 and all(w.dtype == torch.bfloat16 for w in weights.values())
records = {}
for precision in ['fp8', 'int8']:
    target = OUT / f'pika-{precision}.safetensors'
    started = time.perf_counter()
    state = {}
    for key, weight in weights.items():
        w = weight.float()
        maximum = 448 if precision == 'fp8' else 127
        scale = (w.abs().amax() / maximum).clamp_min(1e-12)
        value = (w / scale).clamp(-maximum, maximum)
        if precision == 'int8':
            value = value.round()
        state[key] = value.to(torch.float8_e4m3fn if precision == 'fp8' else torch.int8).contiguous()
        state[key + '.quant_scale'] = scale.float()
    save_file(state, str(target), metadata={'source_sha256': source_hash, 'precision': precision,
                                          'scaling': 'symmetric per-tensor absmax', 'format_version': '1'})
    elapsed = time.perf_counter() - started
    records[precision] = {'path': str(target.relative_to(ROOT)), 'bytes': target.stat().st_size,
                          'sha256': hashlib.file_digest(target.open('rb'), 'sha256').hexdigest(),
                          'offline_quantize_and_save_seconds': elapsed, 'weight_tensors': 416,
                          'scale_tensors': 416, 'source_sha256': source_hash}
    print(json.dumps(records[precision]), flush=True)
    del state
(OUT / 'artifacts.json').write_text(json.dumps(records, indent=2))
