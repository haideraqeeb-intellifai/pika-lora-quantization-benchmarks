"""FP8 E4M3 Pika adapter weights and GEMMs; base checkpoint/graph are unchanged."""
from collections import Counter
import torch
import torch.nn.functional as F
import comfy_kitchen as ck
from ml_modules.minimax_h3_turbo import comfyui_h3_turbo as turbo

def install():
    stats={'adapter_instances':0,'bypass_calls':0,'merge_calls':0,'fp8_gemm_calls':0,'fallback_calls':0,
           'weight_dtype':'float8_e4m3fn','activation_dtype':'float8_e4m3fn','accumulation_output':'bfloat16 bypass / float32 merge',
           'scaling':'per-tensor absmax weights; dynamic per-tensor absmax activations','shapes':{}}
    original_frugal=turbo._FrugalLoRA
    original_load=turbo.comfy.lora.load_lora
    def quantize_weight(w):
        scale=(w.float().abs().amax()/448).clamp_min(1e-12)
        return (w.float()/scale).clamp(-448,448).to(torch.float8_e4m3fn).contiguous(),scale.float()
    def gemm(a,b,sa,sb,out_dtype):
        if a.dtype!=torch.float8_e4m3fn or b.dtype!=torch.float8_e4m3fn:raise TypeError('FP8 operands required')
        if not a.is_cuda or not b.is_cuda:raise RuntimeError('CUDA FP8 required')
        stats['fp8_gemm_calls']+=1
        shape=f'{a.shape[0]}x{a.shape[1]}x{b.shape[1]}'
        stats['shapes'][shape]=stats['shapes'].get(shape,0)+1
        return torch._scaled_mm(a,b,scale_a=sa.float(),scale_b=sb.float(),out_dtype=out_dtype,use_fast_accum=False)
    class FP8LoRA(original_frugal):
        def __init__(self,loaded_keys,weights):
            if len(weights)==8:
                super().__init__(loaded_keys,weights)
                return
            if any(weights[i] is not None for i in [3,4,5]):raise ValueError('Only ordinary Pika LoRA is supported')
            up,su=quantize_weight(weights[0]);down,sd=quantize_weight(weights[1])
            super().__init__(loaded_keys,(up,down,*weights[2:],su,sd))
            stats['adapter_instances']+=1

        def bypass_forward(self,org_forward,x,*args,**kwargs):
            base_out=org_forward(x,*args,**kwargs)
            up,down,alpha,*_=self.weights
            su,sd=self.weights[6:8]
            up=up.to(device=x.device);down=down.to(device=x.device)
            su=su.to(device=x.device);sd=sd.to(device=x.device)
            scale=(alpha/down.shape[0] if alpha is not None else 1.)*getattr(self,'multiplier',1.)
            flat=x.reshape(-1,x.shape[-1]).contiguous()
            rows=flat.shape[0]
            if rows%16:flat=F.pad(flat,(0,0,0,16-rows%16))
            sx=(flat.abs().amax().float()/448).clamp_min(1e-12)
            qx=ck.quantize_per_tensor_fp8(flat,sx,torch.float8_e4m3fn)
            hidden=gemm(qx,down.t(),sx,sd,x.dtype)
            sh=(hidden.abs().amax().float()/448).clamp_min(1e-12)
            qh=ck.quantize_per_tensor_fp8(hidden,sh,torch.float8_e4m3fn)
            delta=gemm(qh,up.t(),sh,su,x.dtype)[:rows].reshape(base_out.shape)
            stats['bypass_calls']+=1
            return base_out.add_(delta,alpha=scale)

        def calculate_weight(self,weight,key,strength,strength_model,offset,function,intermediate_dtype=torch.float32,original_weight=None):
            up,down,alpha,*_=self.weights
            su,sd=self.weights[6:8]
            a=up.to(device=weight.device)
            # cuBLAS FP8 requires column-major RHS.
            b=down.to(device=weight.device).t().contiguous().t()
            diff=gemm(a,b,su.to(weight.device),sd.to(weight.device),torch.float32).reshape(weight.shape)
            factor=strength*(alpha/down.shape[0] if alpha is not None else 1.)
            weight+=function((factor*diff).to(weight.dtype))
            stats['merge_calls']+=1
            return weight

    def load(*args,**kwargs):
        return {k:FP8LoRA(v.loaded_keys,v.weights) if isinstance(v,turbo.comfy.weight_adapter.LoRAAdapter) else v
                for k,v in original_load(*args,**kwargs).items()}
    from comfy.weight_adapter.bypass import BypassForwardHook
    original_move=BypassForwardHook._move_adapter_weights_to_device
    def move(self,device,dtype=None):
        return original_move(self,device,None if isinstance(self.adapter,FP8LoRA) else dtype)
    BypassForwardHook._move_adapter_weights_to_device=move
    turbo.comfy.lora.load_lora=load
    turbo._FrugalLoRA=FP8LoRA
    return stats
