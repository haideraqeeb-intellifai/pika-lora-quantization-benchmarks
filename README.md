# Pika LoRA quantization benchmarks

Experiment scripts, native videos, source inputs, full-frame PSNR/SSIM, execution timings and evidence for Pika H3 Flex on an NVIDIA RTX 5090 32 GB.

This is a publication snapshot. It does not modify or control the original experiment workspace. The snapshot cutoff and complete file inventory are in [PUBLICATION_INVENTORY.json](PUBLICATION_INVENTORY.json).

## Experiments

| Experiment | Base | Adapter / optimization | Snapshot coverage |
|---|---|---|---|
| Original FP8 comparison | Pruned INT8 ConvRot H3 | Original BF16 LoRA versus native FP8 LoRA computation, quantized during each request | All five animals; additional Red Fox repeats |
| Prequantized and preloaded comparison | Same INT8 ConvRot base | Fresh BF16 LoRA versus GPU-resident FP8 and INT8 LoRA | All five animals, 15 new generations |
| Native EasyCache comparison | Original pruned BF16 H3 base | BF16 Pika LoRA, EasyCache off versus on | **Partial snapshot: see the recorded terminal runs below** |

The BF16 label in the first two experiments describes the original **adapter path**. Their base checkpoint is INT8 ConvRot, not an all-BF16 H3 model. The later EasyCache experiment uses the original BF16 base and must not be pooled with those timings. EasyCache and TeaCache are distinct; the two completed quantization experiments used neither.

The proposed merge-LoRA-then-quantize experiment was cancelled before any model run. Its incomplete download and scripts were deleted; it supplies no benchmark result.

## Reports and main findings

- [Original FP8 report, published interactive version](https://s3.renderplatform.com/agent-assets/Raqeeb/pika-fp8-five-video-comparison-20261102/comparison.html) · [local report](experiments/pika-h3/pika-fp8-five-video-comparison-20261102/comparison.html)
- [Preloaded FP8/INT8 report, published interactive version](https://s3.renderplatform.com/agent-assets/Raqeeb/pika-preloaded-five-video-comparison-20261102/comparison.html) · [local report](experiments/pika-h3/pika-preloaded-five-video-comparison-20261102/comparison.html)

Preloaded experiment, full request seconds:

| Video | Fresh BF16 adapter | Preloaded FP8 | Preloaded INT8 |
|---|---:|---:|---:|
| Red Fox | 181.867 | 186.964 | 185.352 |
| Bengal Tiger | 186.209 | 186.553 | 185.192 |
| African Elephant | 185.794 | 186.907 | 185.313 |
| Scarlet Macaw | 185.768 | 186.525 | 185.133 |
| Barn Owl | 185.852 | 187.337 | 185.307 |
| Mean | **185.098** | **186.857** | **185.259** |

Aggregate speedup versus fresh BF16 is **0.990586× for FP8** and **0.999129× for INT8**. Neither produced a meaningful latency improvement in this test. One measured request per animal/configuration limits conclusions about small differences. Weight preparation is excluded from the preloaded request timer and reported separately.

The completed reports score all 362 decoded frames per animal, without resizing or alignment search. All five fresh BF16 controls are pixel-identical to the published originals. Four preloaded FP8 outputs match their previous FP8 outputs exactly; Owl differs slightly (48.309 dB PSNR / 0.995027 SSIM versus its earlier FP8 run). This variation is retained and disclosed.

## Read the reports locally

```bash
python3 -m http.server 8000 --directory .
```

Open `http://localhost:8000/`. Reports contain local relative media links and require no application backend or external JavaScript libraries.

## Layout

```text
experiments/pika-h3/
  cases/                       Original images, audio and reference videos
  cases.json                   Exact prompts, seeds, requests and input hashes
  runs/                        Original BF16/FP8 runs, including repeats and preflight failures
  analysis/                    Original full-frame PSNR/SSIM and review frames
  preloaded/                   FP8/INT8 preparation metadata, runs and analysis
  easycache-bf16/              Completed EasyCache run snapshots and provenance
  pika-*-comparison-*/         Complete portable report directories
  *.py, *.mjs                  Measured replay, preparation, analysis and report scripts
MODEL_REFERENCES.md            Weight locations, revisions, licenses, sizes and hashes
model-manifest.json            Machine-readable download manifest
environment.json               Observed GPU package versions, not a complete lockfile
PUBLICATION_INVENTORY.json     Publication cutoff, source mapping and exclusions
```

## Analysis and report rebuilding

Use Python 3.12+, FFmpeg/FFprobe and Pillow. Install the lightweight analysis dependency with `python -m pip install -r requirements-analysis.txt`. From the repository root:

```bash
python experiments/pika-h3/analyze.py
python experiments/pika-h3/analyze_preloaded.py
python experiments/pika-h3/build_report.py
python experiments/pika-h3/build_preloaded_report.py
```

The scripts retain their measured file layout and cached per-frame metrics. Existing command records intentionally preserve the original absolute execution paths. Do not run the EasyCache analysis/report builder as a full benchmark while its snapshot has missing cases.

Browser verification uses Node.js 22 and Chrome with H.264 support. The excluded browser installer is precisely referenced in `experiments/pika-h3/chrome-download.json`. For example:

```bash
CHROME_BINARY=/absolute/path/to/chrome node experiments/pika-h3/verify_preloaded.mjs
```

## GPU replay requirements

GPU inference requires the external `strontiumcorp/script2video` runtime and its additional dependencies. That private application is not redistributed in this public experiment archive. Authorized users must provide it at `experiments/pika-h3/source` and match the critical file revisions/hashes in `source-provenance.json` and `worker-release-provenance.json`. The partial EasyCache experiment additionally records its changed runtime SHA-256 hashes in `easycache-bf16/base-provenance.json`; the original revision alone does not identify that later modification.

The full observed GPU package versions are in `environment.json` and each result record. The original environment used Python 3.12, PyTorch 2.11.0+cu130, SageAttention 2.2.0, comfy-kitchen 0.2.26 and comfy-aimdo 0.4.11. Replaying GPU inference also requires the pinned model assets in [MODEL_REFERENCES.md](MODEL_REFERENCES.md). The downloader lists its intended files by default; `--download` explicitly retrieves them.

With the external runtime, dependencies and model files provided, run individual cases using the desired environment's Python:

```bash
python experiments/pika-h3/run_case.py --case red-fox --precision bf16
python experiments/pika-h3/run_case.py --case red-fox --precision fp8
python experiments/pika-h3/prepare_prequantized.py
python experiments/pika-h3/run_preloaded_case.py --case red-fox --precision fp8
python experiments/pika-h3/run_preloaded_case.py --case red-fox --precision int8
```

These commands rerun and overwrite the corresponding local result directories. Use a separate clone for new measurements. The preserved suite scripts contain the original host's absolute Python path; individual-case commands avoid that host-specific path. The original `prepare.py` is provenance for the internal UID-lookup input-recovery process; use the included, hashed inputs for public analysis instead of requiring production database access.

## Publication scope

All intentional experiment code, input media, canonical output media, report packages, metrics, failure records and provenance available at the cutoff are included. Runtime input/output duplicates were excluded only after byte-hash verification against their included canonical copies. Weights, installed dependencies, browser sessions, credential/host-access files and unrelated bucket listings are excluded with reasons in the inventory. No new project license is asserted; upstream model and dependency terms continue to apply.
