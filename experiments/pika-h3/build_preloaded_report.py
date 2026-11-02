"""Package only benchmark media and reviewable evidence for the approved report upload."""
import hashlib
import html
import json
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'preloaded'
DEST = ROOT / 'pika-preloaded-five-video-comparison-20261102'
rows = json.loads((SOURCE / 'analysis/summary.json').read_text())
assert len(rows) == 5
DEST.mkdir(exist_ok=True)

def copy(source, target):
    target = DEST / target
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)

for name in ['prepare_prequantized.py', 'preloaded_adapter.py', 'run_preloaded_case.py',
             'run_preloaded_suite.py', 'monitor_preloaded.py', 'analyze_preloaded.py', 'analyze.py',
             'build_preloaded_report.py', 'cases.json', 'input-integrity.json',
             'source-provenance.json', 'worker-release-provenance.json']:
    copy(ROOT / name, 'evidence/' + name)
for name in ['artifacts.json', 'gpu-gates.json', 'gpu-ownership.jsonl', 'int8-kernel-verification.json', 'weight-equivalence.json', 'suite.log']:
    copy(SOURCE / name, 'evidence/' + name)
for row in rows:
    slug = row['slug']
    copy(ROOT / 'pika-fp8-five-video-comparison-20261102' / 'evidence' / slug / 'uid-lookup-summary.json',
         f'evidence/original-timings/{slug}.json')
    for precision in ['bf16', 'fp8', 'int8']:
        directory = SOURCE / 'runs' / slug / precision
        copy(directory / 'output-r1.mp4', f'media/{slug}/{precision}.mp4')
        for name in ['result.json', 'workflow-r1.json', 'process.log']:
            copy(directory / name, f'evidence/runs/{slug}/{precision}/{name}')
    for path in (SOURCE / 'analysis' / slug).rglob('*'):
        if path.is_file():
            copy(path, 'evidence/analysis/' + str(path.relative_to(SOURCE / 'analysis')))
copy(SOURCE / 'analysis/summary.json', 'summary.json')

bf16 = sum(r['times_seconds']['bf16'] for r in rows) / 5
means = {p: sum(r['times_seconds'][p] for r in rows) / 5 for p in ['bf16', 'fp8', 'int8']}
stage_classes = ['MiniMaxH3PikaLoRA', 'SamplerCustomAdvanced', 'VAEDecode', 'SaveVideo']
stages = ''.join('<tr><td>' + html.escape(c) + '</td>' + ''.join(
    f'<td>{sum(r["stage_seconds"][p].get(c, 0) for r in rows)/5:.3f} s</td>'
    for p in ['bf16', 'fp8', 'int8']) + '</tr>' for c in stage_classes)
artifacts = json.loads((SOURCE / 'artifacts.json').read_text())
prep = ''.join(f'<tr><td>{p.upper()}</td><td>{artifacts[p]["offline_quantize_and_save_seconds"]:.3f} s</td>'
               f'<td>{sum(r["preparation"][p]["artifact_cpu_load_seconds"] for r in rows)/5:.3f} s</td>'
               f'<td>{sum(r["preparation"][p]["gpu_load_and_prepack_seconds"] for r in rows)/5:.3f} s</td>'
               f'<td>{artifacts[p]["bytes"]/1e9:.3f} GB</td></tr>' for p in ['fp8', 'int8'])
gates = json.loads((SOURCE / 'gpu-gates.json').read_text())
ownership = [json.loads(s) for s in (SOURCE / 'gpu-ownership.jsonl').read_text().splitlines() if s]
assert ownership and not any(r['competing_workload'] for r in ownership)
assert all(r['quality']['bf16_control']['ssim'] == 1 and r['quality']['bf16_control']['psnr_infinite'] for r in rows)
overview = {
    'cases': 5, 'successful_video_generations': 15, 'mean_seconds': means,
    'aggregate_speedup_vs_fresh_bf16': {p: bf16 / means[p] for p in ['fp8', 'int8']},
    'time_reduction_percent_vs_fresh_bf16': {p: 100 * (1 - means[p] / bf16) for p in ['fp8', 'int8']},
    'gpu_ownership_samples': len(ownership), 'competing_gpu_workloads_detected': 0,
    'bf16_controls_pixel_identical_to_published_originals': 5,
    'fp8_preloaded_pixel_identical_to_previous_fp8': sum(r['quality']['fp8_preload_equivalence']['psnr_infinite'] for r in rows),
    'scored_frames_per_five_video_comparison': 1810,
    'caveat': 'One measured request per case and configuration; small timing differences can be normal run-to-run variation.'}
(DEST / 'verification-summary.json').write_text(json.dumps(overview, indent=2))

page = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Pika: preloaded INT8 / FP8 versus fresh BF16</title><style>
:root{color-scheme:dark;font:16px/1.6 system-ui;background:#0b111a;color:#e2eaf5}*{box-sizing:border-box}body{margin:0}main{max-width:1450px;margin:auto;padding:32px 24px 80px}h1{font-size:clamp(28px,4vw,46px);line-height:1.15;margin:12px 0 22px}h2{margin:42px 0 12px;font-size:24px}p{color:#b9c6d8;max-width:1050px}a{color:#82c8ff}code{overflow-wrap:anywhere}header>span,.tag{color:#8dd9d2;font-size:13px;letter-spacing:.05em}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}.card,.case{background:#131e2d;border:1px solid #2b3c52;border-radius:12px;padding:20px}.card strong{display:block;font-size:30px}.card small{color:#b9c6d8}.scroll{overflow:auto;border:1px solid #2b3c52;border-radius:10px}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;white-space:nowrap}th,td{text-align:right;padding:12px 16px;border-bottom:1px solid #28364a}th:first-child,td:first-child{text-align:left}th{background:#162439;color:#9db4cf}tbody tr:last-child td{border:0}.good{color:#88dbc1}.slow{color:#ffb69b}.case{margin:20px 0}.case h3{margin:0}.videos{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}figure{margin:12px 0}figcaption{font-weight:600;margin-bottom:8px}video{width:100%;aspect-ratio:16/9;background:#05090f;object-fit:contain}.portrait video{aspect-ratio:9/16;max-height:530px}button,input{font:inherit;color:inherit;background:#1c3048;border:1px solid #466080;border-radius:6px;padding:8px 12px}button{cursor:pointer;margin:6px 8px 6px 0}input{width:min(100%,320px)}details{margin-top:14px}summary{cursor:pointer;color:#82c8ff}.evidence{overflow-wrap:anywhere;font-size:14px}.note{padding:14px 18px;border-left:3px solid #8dd9d2;background:#111f2e}footer{margin-top:50px;font-size:14px;color:#aebed3}@media(max-width:700px){main{padding:20px 14px 55px}.cards,.videos{grid-template-columns:1fr}.card strong{font-size:27px}.case{padding:14px}.portrait video{max-height:480px}th,td{padding:10px}}
</style></head><body><main><header><span>MEASURED ON RTX 5090 · FIVE FIXED REQUESTS · 15 NEW GENERATIONS</span>
<h1>Does preloading make Pika INT8 or FP8 faster?</h1>
<p>Persistent quantized adapter files, GPU-resident weights, and prepacked merge matrices. Each configuration uses the same production Pika path and the same INT8 ConvRot H3 base. BF16 labels refer to the adapter computation.</p></header>
<div class="cards" id="totals"></div><p class="note">@@RESULT@@</p>
<h2>Full request times versus fresh BF16</h2><p>Measured from immediately before graph execution through completed video encoding, with CUDA synchronization. The BF16 controls below were generated again in this experiment. Speedup = BF16 seconds ÷ candidate seconds; values above 1× are faster.</p>
<div class="scroll"><table><thead><tr><th>Video</th><th>Fresh BF16</th><th>Preloaded FP8</th><th>FP8 speedup</th><th>Preloaded INT8</th><th>INT8 speedup</th></tr></thead><tbody id="rows"></tbody></table></div>
<h2>Where the time goes</h2><p>Mean node time over five requests. The sampler includes the existing base-model loading and transfer behavior.</p><div class="scroll"><table><thead><tr><th>Stage</th><th>BF16</th><th>FP8 preloaded</th><th>INT8 preloaded</th></tr></thead><tbody>@@STAGES@@</tbody></table></div>
<p>Preloading removes adapter weight preparation. Each quantized request still performs 916 dynamic activation quantizations across its bypass calls and computes the separate low-rank branches. Node timings do not isolate the individual kernel costs.</p>
<h2>Preparation costs, outside the request timer</h2><p>Offline conversion is paid once per saved artifact. Loading and GPU preparation are paid once per process and can be amortized by a persistent worker. Base-model loading, encoders, denoising, decoding, and encoding retain their existing behavior inside the graph.</p><div class="scroll"><table><thead><tr><th>Adapter</th><th>Offline quantize + save</th><th>Artifact CPU load (mean)</th><th>GPU load + prepack (mean)</th><th>Artifact size</th></tr></thead><tbody>@@PREP@@</tbody></table></div>
<h2>Quality versus fresh BF16</h2><p>PSNR and SSIM score all 362 decoded frames per video at native dimensions and 24 fps, using FFmpeg YUV420P. No resizing, alignment search, trimming, or sampled-frame scoring. All five new BF16 outputs are pixel-identical to the published originals.</p><div class="scroll"><table><thead><tr><th>Video</th><th>FP8 PSNR</th><th>FP8 SSIM</th><th>INT8 PSNR</th><th>INT8 SSIM</th></tr></thead><tbody id="quality"></tbody></table></div>
<h2>Compare the videos</h2><input id="filter" aria-label="Filter videos" placeholder="Filter animals…"><div id="cases"></div>
<h2>What was actually tested</h2><p>Each process starts after three consecutive idle GPU checks. For FP8 and INT8, 416 quantized weight tensors and their FP32 scales are loaded onto the GPU before timing. Merge RHS matrices are also prepacked. The measured requests perform no adapter weight quantization or CPU-to-GPU adapter weight transfers. Activation quantization remains dynamic.</p>
<p>FP8 uses E4M3 operands and <code>torch._scaled_mm(use_fast_accum=False)</code>, returning BF16 bypass activations and FP32 merge deltas. INT8 uses the concrete comfy_kitchen CUDA backend with per-row INT8 activation quantization and INT8 CUTLASS/cuBLAS multiplication; merge deltas use <code>torch._int_mm</code> with INT32 accumulation and FP32 rescaling. No BF16 compute fallback is implemented in either quantized adapter.</p>
<p>Every quantized request applied 208 adapters, executed 458 bypass calls and 150 fused-fc2 merge calls, and made 1,066 native quantized GEMMs. The existing bypass-versus-merge split, sampler, three steps, seed, prompts, input images and audio, dimensions, and output encoding were preserved. The base is unchanged throughout.</p>
<p class="note">Timing limits: one measured request per animal and configuration. Differences of a few seconds may reflect normal run-to-run variation. These results establish the observed effect of this implementation on this GPU; they do not establish a universal FP8 or INT8 speedup.</p>
<details><summary>Previous experiment and reproducibility evidence</summary><p>The <a href="https://s3.renderplatform.com/agent-assets/Raqeeb/pika-fp8-five-video-comparison-20261102/comparison.html">previous report</a> quantized adapter weights during each request. This report uses new BF16 controls as the primary timing denominator. FP8 output equivalence to that previous run is recorded per animal.</p><div class="scroll"><table><thead><tr><th>Video</th><th>Previous BF16</th><th>Previous FP8</th><th>Current BF16</th><th>Preloaded FP8</th><th>FP8 pixel-identical?</th></tr></thead><tbody id="previous"></tbody></table></div><p class="evidence"><a href="summary.json">Full results and stage timings</a> · <a href="verification-summary.json">Verification summary</a> · <a href="evidence/artifacts.json">Artifact hashes</a> · <a href="evidence/cases.json">Exact requests and source UIDs</a> · <a href="evidence/input-integrity.json">Input hashes</a> · <a href="evidence/source-provenance.json">Source provenance</a> · <a href="evidence/worker-release-provenance.json">Original worker release provenance</a> · <a href="evidence/gpu-gates.json">Idle GPU gates</a> · <a href="evidence/gpu-ownership.jsonl">GPU ownership samples</a> · <a href="evidence/int8-kernel-verification.json">INT8 kernel verification</a> · <a href="evidence/prepare_prequantized.py">Artifact preparation script</a> · <a href="evidence/preloaded_adapter.py">Adapter implementation</a> · <a href="evidence/run_preloaded_case.py">Replay runner</a> · <a href="evidence/run_preloaded_suite.py">Suite runner</a> · <a href="evidence/analyze_preloaded.py">Analysis script</a></p></details>
<footer>Local isolated benchmark on gpu5090-79, NVIDIA GeForce RTX 5090 32 GB. No production deployment. GPU ownership monitor found no competing compute process. Times exclude model-download time and report processing.</footer>
</main><script>
const data=@@DATA@@;
const mean=p=>data.reduce((s,r)=>s+r.times_seconds[p],0)/data.length;
const fmt=x=>x.toFixed(3)+' s';
const speed=x=>`<span class="${x>=1?'good':'slow'}">${x.toFixed(4)}×</span>`;
const psnr=q=>q.psnr_infinite?'∞':q.psnr_db.toFixed(3)+' dB';
document.querySelector('#totals').innerHTML=['bf16','fp8','int8'].map(p=>`<div class="card"><span>${p==='bf16'?'Fresh BF16 adapter':p.toUpperCase()+' adapter · preloaded'}</span><strong>${fmt(mean(p))}</strong><small>${p==='bf16'?'Mean full request time':`${speed(mean('bf16')/mean(p))} vs BF16 · ${(100*(1-mean(p)/mean('bf16'))).toFixed(2)}% time reduction`}</small></div>`).join('');
document.querySelector('#rows').innerHTML=data.map(r=>`<tr><td>${r.name}</td><td>${fmt(r.times_seconds.bf16)}</td><td>${fmt(r.times_seconds.fp8)}</td><td>${speed(r.speedup.fp8)}</td><td>${fmt(r.times_seconds.int8)}</td><td>${speed(r.speedup.int8)}</td></tr>`).join('');
document.querySelector('#quality').innerHTML=data.map(r=>`<tr><td>${r.name}</td><td>${psnr(r.quality.fp8)}</td><td>${r.quality.fp8.ssim.toFixed(6)}</td><td>${psnr(r.quality.int8)}</td><td>${r.quality.int8.ssim.toFixed(6)}</td></tr>`).join('');
document.querySelector('#previous').innerHTML=data.map(r=>`<tr><td>${r.name}</td><td>${fmt(r.previous_experiment_seconds.bf16)}</td><td>${fmt(r.previous_experiment_seconds.fp8)}</td><td>${fmt(r.times_seconds.bf16)}</td><td>${fmt(r.times_seconds.fp8)}</td><td>${r.quality.fp8_preload_equivalence.psnr_infinite?'Yes':'No'}</td></tr>`).join('');
function render(){const filter=document.querySelector('#filter').value.toLowerCase();document.querySelector('#cases').innerHTML=data.filter(r=>r.name.toLowerCase().includes(filter)).map(r=>`<article class="case ${r.slug==='barn-owl'?'portrait':''}"><h3>${r.name}</h3><div class="videos">${['bf16','fp8','int8'].map(p=>`<figure><figcaption>${p==='bf16'?'Fresh BF16':p.toUpperCase()+' preloaded'} · ${fmt(r.times_seconds[p])}</figcaption><video controls playsinline preload="metadata"><source src="media/${r.slug}/${p}.mp4" type="video/mp4"></video></figure>`).join('')}</div><button class="play">Play together</button><button class="pause">Pause together</button><button class="reset">Restart</button><details><summary>Timing and execution evidence</summary><p class="evidence">${['bf16','fp8','int8'].map(p=>`<a href="evidence/runs/${r.slug}/${p}/result.json">${p.toUpperCase()} result and telemetry</a> · <a href="evidence/runs/${r.slug}/${p}/process.log">log</a> · <a href="evidence/runs/${r.slug}/${p}/workflow-r1.json">workflow</a>`).join('<br>')}</p><p class="evidence">${['bf16-vs-fp8','bf16-vs-int8','original-vs-bf16','old-vs-preloaded-fp8'].map(p=>`<a href="evidence/analysis/${r.slug}/${p}/metrics.json">${p} metrics</a>`).join(' · ')}</p></details></article>`).join('');
document.querySelectorAll('.case').forEach(card=>{const videos=[...card.querySelectorAll('video')];card.querySelector('.play').onclick=async()=>{const t=videos[0].currentTime;videos.forEach(v=>v.currentTime=t);await Promise.all(videos.map(v=>v.play()));};card.querySelector('.pause').onclick=()=>videos.forEach(v=>v.pause());card.querySelector('.reset').onclick=()=>videos.forEach(v=>{v.pause();v.currentTime=0});});}
document.querySelector('#filter').addEventListener('input',render);render();
</script></body></html>'''
parts = []
for p in ['fp8', 'int8']:
    change = 100 * (1 - means[p] / bf16)
    parts.append(f'{p.upper()}: {abs(change):.2f}% {"less" if change >= 0 else "more"} time than fresh BF16 ({bf16/means[p]:.4f}×).')
page = page.replace('@@RESULT@@', ' '.join(parts) + ' These are measured full-request differences across the five fixed videos.')
page = page.replace('@@STAGES@@', stages).replace('@@PREP@@', prep)
page = page.replace('BF16 labels refer to the adapter computation.', 'BF16 labels refer to the original BF16 Pika adapter path.')
page = page.replace('<h2>Compare the videos</h2>',
                    '<p>Four preloaded FP8 outputs are pixel-identical to the previous FP8 outputs. '
                    'Owl differs slightly: 48.309 dB PSNR and 0.995027 SSIM versus its earlier FP8 run. '
                    'All 416 saved FP8 weight tensors and scales exactly match the previous weight quantization formula; '
                    'the source of the small Owl output variation was not isolated.</p>'
                    '<p>Matched-frame review at 0, 4.5, 9.5 and 14 seconds found small changes in articulation, '
                    'head position and fur, skin or feather detail. The sampled frames retain the subjects and scene composition. '
                    'This frame review does not certify every frame or lip-sync quality.</p>'
                    '<h2>Compare the videos</h2>')
page = page.replace('@@DATA@@', json.dumps(rows).replace('</', '<\\/'))
page = page.replace('<details><summary>Timing and execution evidence</summary>',
                    '<details><summary>Matched frames at 0, 4.5, 9.5 and 14 seconds</summary>'
                    '<a href="evidence/analysis/${r.slug}/frames.jpg"><img style="max-width:100%;height:auto" '
                    'src="evidence/analysis/${r.slug}/frames.jpg" alt="Matching frames from BF16, preloaded FP8 and preloaded INT8"></a></details>'
                    '<details><summary>Timing and execution evidence</summary>')
(DEST / 'comparison.html').write_text(page)
manifest = {str(p.relative_to(DEST)): hashlib.file_digest(p.open('rb'), 'sha256').hexdigest()
            for p in sorted(DEST.rglob('*')) if p.is_file() and p.name != 'artifact-integrity.json'}
(DEST / 'artifact-integrity.json').write_text(json.dumps(manifest, indent=2))
print(json.dumps({'path': str(DEST), 'files': len(manifest), 'overview': overview}, indent=2))
