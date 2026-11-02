"""Preloaded Pika adapters. Quantized weights remain on GPU throughout each request."""
from pathlib import Path
import time
import torch
import torch.nn.functional as F
import comfy_kitchen as ck
from comfy_kitchen.backends import cuda as cuda_kitchen
from safetensors.torch import load_file
from ml_modules.minimax_h3_turbo import comfyui_h3_turbo as turbo


def install(artifact, precision):
    started = time.perf_counter()
    raw = load_file(str(artifact))
    loaded = time.perf_counter()
    weights = {k: v.to('cuda') for k, v in raw.items() if not k.endswith('.quant_scale')}
    scales = {id(v): raw[k + '.quant_scale'].to('cuda') for k, v in weights.items()}
    # Only fused fc2 merges require this alternate layout. The other branches use weight.T.
    columns = {id(v): v.t().contiguous().t() for k, v in weights.items()
               if '.fc2.' in k and ('lora_down' in k or 'lora_A' in k)}
    torch.cuda.synchronize()
    prepared = time.perf_counter()
    dtype = torch.float8_e4m3fn if precision == 'fp8' else torch.int8
    assert len(weights) == 416 and all(w.dtype == dtype and w.is_cuda for w in weights.values())
    stats = {'precision': precision, 'adapter_instances': 0, 'bypass_calls': 0, 'merge_calls': 0,
             'native_gemm_calls': 0, 'fallback_calls': 0, 'weight_quantizations_in_request': 0,
             'weight_transfers_in_request': 0, 'preloaded_weight_tensors': len(weights),
             'prepacked_merge_rhs': len(columns), 'weight_dtype': str(dtype),
             'activation_scaling': 'dynamic per-tensor absmax' if precision == 'fp8' else 'dynamic per-row absmax',
             'backend': 'torch._scaled_mm' if precision == 'fp8' else 'comfy_kitchen CUDA int8_linear / torch._int_mm merge',
             'shapes': {}}
    preparation = {'artifact_cpu_load_seconds': loaded - started,
                   'gpu_load_and_prepack_seconds': prepared - loaded,
                   'total_adapter_preparation_seconds': prepared - started,
                   'gpu_allocated_bytes_after_preload': torch.cuda.memory_allocated(),
                   'gpu_weight_bytes': sum(w.numel() * w.element_size() for w in weights.values()),
                   'gpu_prepacked_extra_bytes': sum(w.numel() * w.element_size() for w in columns.values())}
    original_frugal = turbo._FrugalLoRA
    original_load = turbo.comfy.lora.load_lora
    original_file_load = turbo.comfy.utils.load_torch_file

    def record(a, b):
        assert a.is_cuda and b.is_cuda and a.dtype == dtype and b.dtype == dtype
        stats['native_gemm_calls'] += 1
        shape = f'{a.shape[0]}x{a.shape[1]}x{b.shape[1]}'
        stats['shapes'][shape] = stats['shapes'].get(shape, 0) + 1

    def fp8_gemm(a, b, sa, sb, out_dtype):
        record(a, b)
        return torch._scaled_mm(a, b, scale_a=sa, scale_b=sb,
                                out_dtype=out_dtype, use_fast_accum=False)

    def int8_linear(x, w, scale):
        assert x.is_cuda and w.is_cuda and w.dtype == torch.int8 and w.is_contiguous()
        # This concrete CUDA backend quantizes activations to INT8 and uses INT8
        # CUTLASS/cuBLAS GEMMs. It cannot dispatch to an eager BF16 implementation.
        stats['native_gemm_calls'] += 1
        shape = f'{x.numel() // x.shape[-1]}x{x.shape[-1]}x{w.shape[0]}'
        stats['shapes'][shape] = stats['shapes'].get(shape, 0) + 1
        return cuda_kitchen.int8_linear(x, w, scale, out_dtype=x.dtype, convrot=False)

    class PreloadedLoRA(original_frugal):
        def __init__(self, loaded_keys, values):
            if len(values) == 9:
                super().__init__(loaded_keys, values)
                return
            if any(values[i] is not None for i in [3, 4, 5]):
                raise ValueError('Only ordinary Pika LoRA is supported')
            up, down = values[:2]
            assert up.is_cuda and down.is_cuda and up.dtype == dtype and down.dtype == dtype
            super().__init__(loaded_keys, (*values, scales[id(up)], scales[id(down)], columns.get(id(down))))
            stats['adapter_instances'] += 1

        def bypass_forward(self, org_forward, x, *args, **kwargs):
            base_out = org_forward(x, *args, **kwargs)
            up, down, alpha = self.weights[:3]
            su, sd = self.weights[6:8]
            assert up.device == x.device == down.device == su.device == sd.device
            factor = (alpha / down.shape[0] if alpha is not None else 1.) * getattr(self, 'multiplier', 1.)
            flat = x.reshape(-1, x.shape[-1]).contiguous()
            rows = flat.shape[0]
            if precision == 'fp8':
                if rows % 16:
                    flat = F.pad(flat, (0, 0, 0, 16 - rows % 16))
                sx = (flat.abs().amax().float() / 448).clamp_min(1e-12)
                qx = ck.quantize_per_tensor_fp8(flat, sx, torch.float8_e4m3fn)
                hidden = fp8_gemm(qx, down.t(), sx, sd, x.dtype)
                sh = (hidden.abs().amax().float() / 448).clamp_min(1e-12)
                qh = ck.quantize_per_tensor_fp8(hidden, sh, torch.float8_e4m3fn)
                delta = fp8_gemm(qh, up.t(), sh, su, x.dtype)[:rows]
            else:
                hidden = int8_linear(flat, down, sd)
                delta = int8_linear(hidden, up, su)
            stats['bypass_calls'] += 1
            return base_out.add_(delta.reshape(base_out.shape), alpha=factor)

        def calculate_weight(self, weight, key, strength, strength_model, offset, function,
                             intermediate_dtype=torch.float32, original_weight=None):
            up, down, alpha = self.weights[:3]
            su, sd, down_column = self.weights[6:9]
            assert up.device == down.device == weight.device == su.device == sd.device
            if down_column is None:
                raise RuntimeError('Merge RHS was not prepacked before timing')
            if precision == 'fp8':
                diff = fp8_gemm(up, down_column, su, sd, torch.float32)
            else:
                record(up, down_column)
                diff = torch._int_mm(up, down_column).float().mul_(su * sd)
            factor = strength * (alpha / down.shape[0] if alpha is not None else 1.)
            weight += function((factor * diff.reshape(weight.shape)).to(weight.dtype))
            stats['merge_calls'] += 1
            return weight

    def load(*args, **kwargs):
        return {k: PreloadedLoRA(v.loaded_keys, v.weights)
                if isinstance(v, turbo.comfy.weight_adapter.LoRAAdapter) else v
                for k, v in original_load(*args, **kwargs).items()}

    def file_load(path, *args, **kwargs):
        if Path(path).name == 'pika_h3_flex_fl2va_comfy_bf16.safetensors':
            return weights
        return original_file_load(path, *args, **kwargs)

    from comfy.weight_adapter.bypass import BypassForwardHook
    original_move = BypassForwardHook._move_adapter_weights_to_device

    def move(self, device, dtype=None):
        if isinstance(self.adapter, PreloadedLoRA):
            # Keep persistent buffers on GPU even if a base module is offloaded.
            if torch.device(device).type == 'cuda':
                assert all(v.device == torch.device(device) for v in self.adapter.weights if isinstance(v, torch.Tensor))
            return
        return original_move(self, device, dtype)

    BypassForwardHook._move_adapter_weights_to_device = move
    turbo.comfy.utils.load_torch_file = file_load
    turbo.comfy.lora.load_lora = load
    turbo._FrugalLoRA = PreloadedLoRA
    return stats, preparation
