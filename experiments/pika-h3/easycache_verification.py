"""Observe the native EasyCache implementation without changing its decisions."""
import torch

def install(enabled):
    from comfy.ldm.minimax.model import MiniMaxH3Model
    from comfy_extras import nodes_easycache as ec
    stats = {'enabled': enabled, 'full_model_evaluations': 0, 'cache_resets': [],
             'large_weight_dtype': None, 'video_input_dtype': None, 'audio_input_dtype': None}
    original = MiniMaxH3Model._forward

    def forward(self, x, *args, **kwargs):
        weight = self.blocks[0].attn.qkv_proj.weight
        assert weight.dtype == torch.bfloat16 and not hasattr(weight, '_layout_cls')
        stats['full_model_evaluations'] += 1
        stats['large_weight_dtype'] = str(weight.dtype)
        stats['video_input_dtype'] = str(x[0].dtype)
        stats['audio_input_dtype'] = str(x[1].dtype)
        return original(self, x, *args, **kwargs)
    MiniMaxH3Model._forward = forward

    original_reset = ec.EasyCacheHolder.reset
    def reset(self):
        stats['cache_resets'].append({'steps_skipped': self.total_steps_skipped,
            'threshold': self.reuse_threshold, 'start_percent': self.start_percent,
            'end_percent': self.end_percent, 'approx_output_change_rates': list(self.approx_output_change_rates)})
        return original_reset(self)
    ec.EasyCacheHolder.reset = reset
    return stats
