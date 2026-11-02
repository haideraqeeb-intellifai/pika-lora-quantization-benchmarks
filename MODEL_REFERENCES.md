# Model references — weights are not distributed

No base weights, LoRAs, quantized parameters, checkpoints, optimizer states or weight shards are stored in Git, Git LFS or release assets.

The full retrieval manifest is [model-manifest.json](model-manifest.json). `tools/fetch_models.py` lists selected assets by default. Add `--download` to retrieve and checksum them into the ignored local model directory. These downloads are large; common encoders/VAEs are additional to the base checkpoint.

| Asset | Purpose | Exact source | Bytes | SHA-256 |
|---|---|---|---:|---|
| H3 FL2VA pruned INT8 ConvRot | Base for both completed quantization comparisons | `Comfy-Org/MiniMax-H3@0bd506d2e895983a9663037febda27aa3948cf48`, `diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors` | 20,970,379,616 | `e889202c41dafb67b10d67b97f0d8541508036a6090af23425a5c2615d03c47a` |
| H3 FL2VA pruned BF16 | Base for later EasyCache comparison | Same repository/revision, `diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors` | 40,225,724,176 | `a32572fb90b5508b201ec7c2eddcc184b13ddfd3c6f6d2cf06a0b46535d541b4` |
| Qwen3VL NVFP4 AWQ | Text encoder | Same repository/revision, `text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | 15,687,142,551 | `35a88d51044231fe332301d7a62aa81e3f2cba62febeb446e2c1e3e0ef76f2c6` |
| H3 video VAE FP16 | Video decode/conditioning | Same repository/revision, `vae/minimax_h3_video_vae_fp16.safetensors` | 5,207,808,496 | `7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522` |
| H3 audio VAE FP32 | Audio decode/conditioning | Same repository/revision, `vae/minimax_h3_audio_vae_fp32.safetensors` | 605,254,808 | `8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48` |
| Pika H3 Flex FL2VA BF16 | Original Pika adapter | `Pika-Labs/Pika-H3-Flex@3ff166447d5edb1406107ab8370c42000ce173e4`, `comfy/loras/pika_h3_flex_fl2va_comfy_bf16.safetensors` | 1,956,172,176 | `68783e6447696ec365d13c0554941e070bee4cfea13c37da52498ec4f9a53f89` |
| SiLU time-embedding grid | Required runtime initialization asset | [larryvrh/ComfyUI-MiniMax-H3-Turbo, pinned commit](https://github.com/larryvrh/ComfyUI-MiniMax-H3-Turbo/blob/4274783a23afcfdbea3b4876cb79effd6c510785/h3_silu_temb_grid.safetensors) | 5,510,600 | `30eb3c2cc7fb6b470d9717ff840d359313ac27cd64b705e32da1baa10f72d6a8` |

Authoritative model repositories: [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/0bd506d2e895983a9663037febda27aa3948cf48) and [Pika-Labs/Pika-H3-Flex](https://huggingface.co/Pika-Labs/Pika-H3-Flex/tree/3ff166447d5edb1406107ab8370c42000ce173e4).

Both model cards identify the **MiniMax H3 Community License Agreement** (`license: other`). Read the [MiniMax model license](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE) and [pinned Pika license](https://huggingface.co/Pika-Labs/Pika-H3-Flex/blob/3ff166447d5edb1406107ab8370c42000ce173e4/LICENSE). These repositories were public and ungated when checked; applicable license terms still govern use. The grid follows its upstream repository's applicable terms; no separate grid license is asserted here.

## Derived quantized adapters

Recreate these locally using the exact BF16 source above and `experiments/pika-h3/prepare_prequantized.py`. Their source manifest and conversion timing are retained in `experiments/pika-h3/preloaded/artifacts.json`.

| Derived file | Bytes | SHA-256 |
|---|---:|---|
| `preloaded/pika-fp8.safetensors` | 978,164,048 | `8e135b33334defd7b4b6c9de200dcbc80127d2e4e2eefa92c9acbd6748ab7f09` |
| `preloaded/pika-int8.safetensors` | 978,161,968 | `f287698c2717abccddc158cb26321dcf574ef26bceeb5682b4f36c1d1c46f218` |

The original runtime LoRA conversion and its measured compute path are in `fp8_adapter.py`; persistent conversion and GPU preload are in `prepare_prequantized.py` and `preloaded_adapter.py`.

The local original cache also held unused Ref2VA/FP8 base variants and the LarryVRH Turbo LoRA. Those were not selected by the Pika quantization requests. Their authoritative repositories are the pinned Comfy-Org model repository above and [larryvrh/MiniMax-H3-Turbo-Lora@43a74557ac3f6539db8e0f2a959d03feb7a81480](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora/tree/43a74557ac3f6539db8e0f2a959d03feb7a81480) (model-card license: Apache-2.0).
