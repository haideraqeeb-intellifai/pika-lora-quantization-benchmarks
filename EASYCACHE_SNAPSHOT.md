# EasyCache snapshot

Snapshot cutoff: **2026-11-03T03:14:49.433181+05:30 (Asia/Calcutta)**.

**Partial: 7/10 terminal runs**. The main workspace may continue after this snapshot; no running job was modified or restarted.

This experiment uses the original **BF16 base** and the BF16 Pika adapter, with native EasyCache on/off. It uses EasyCache, not TeaCache. Do not combine its timings with the INT8-base adapter-quantization experiments.

| Case | EasyCache | Status | Full request seconds | Full model evaluations | Verified skipped steps |
|---|---|---|---:|---:|---:|
| african-elephant | off | success | 232.108 | 3 | 0 |
| african-elephant | on | success | 231.976 | 3 | 0 |
| bengal-tiger | off | success | 232.276 | 3 | 0 |
| bengal-tiger | on | success | 231.892 | 3 | 0 |
| red-fox | off | success | 226.925 | 3 | 0 |
| red-fox | on | success | 231.897 | 3 | 0 |
| scarlet-macaw | on | success | 232.055 | 3 | 0 |

Raw workflows, videos, telemetry and logs for those terminal runs are under `experiments/pika-h3/easycache-bf16/runs/`. No missing run was represented by a zero or fabricated result. GPU model weights are excluded.

No complete five-case EasyCache quality/speed conclusion is made in this snapshot. The preserved `analyze_easycache.py` and `build_easycache_report.py` are intended for a complete generation phase.
