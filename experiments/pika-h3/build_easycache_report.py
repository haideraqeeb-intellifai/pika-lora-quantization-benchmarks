"""Package timing and cache-use evidence for the local BF16 EasyCache comparison."""
import hashlib
import html
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'easycache-bf16'
DEST=ROOT/'pika-bf16-base-easycache-five-videos-20261103'
summary=json.loads((OUT/'analysis/summary.json').read_text())
assert len(summary['cases'])==5
DEST.mkdir(exist_ok=True)
def copy(source,name):
    target=DEST/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
for name in ['base-provenance.json','gpu-gates.json','gpu-ownership.jsonl','suite.log']:
    copy(OUT/name,'evidence/'+name)
for name in ['run_easycache_case.py','run_easycache_suite.py','easycache_verification.py','analyze_easycache.py','analyze.py','build_easycache_report.py','cases.json','input-integrity.json']:
    copy(ROOT/name,'evidence/'+name)
copy(ROOT/'source/ml_modules/comfy_ui/comfy_v14/comfy_extras/nodes_easycache.py','evidence/native_nodes_easycache.py')
for row in summary['cases']:
    slug=row['slug']
    for mode in ['off','on']:
        directory=OUT/'runs'/slug/mode
        copy(directory/'output-r1.mp4',f'media/{slug}/{mode}.mp4')
        for name in ['result.json','workflow-r1.json','process.log']:
            copy(directory/name,f'evidence/runs/{slug}/{mode}/{name}')
    for path in (OUT/'analysis'/slug).iterdir():
        if path.is_file():copy(path,f'evidence/analysis/{slug}/{path.name}')
copy(OUT/'analysis/summary.json','summary.json')
ownership=[json.loads(line) for line in (OUT/'gpu-ownership.jsonl').read_text().splitlines() if line]
assert not any(r['competing_workload'] for r in ownership)
data=json.dumps(summary).replace('</','<\\/')
page='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>BF16 H3 base: EasyCache timing comparison</title><style>
:root{color-scheme:dark;font:16px/1.6 system-ui;background:#0b111a;color:#e2eaf5}*{box-sizing:border-box}body{margin:0}main{max-width:1400px;margin:auto;padding:30px 24px 70px}h1{font-size:clamp(28px,4vw,44px);line-height:1.2}h2{margin-top:36px}p{color:#b9c6d8;max-width:1050px}a{color:#82c8ff}code{overflow-wrap:anywhere}.cards,.videos{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}.card,.case{background:#131e2d;border:1px solid #2b3c52;border-radius:12px;padding:20px}.card strong{display:block;font-size:32px}.case{margin:20px 0}.scroll{overflow:auto;border:1px solid #2b3c52;border-radius:10px}table{width:100%;border-collapse:collapse;white-space:nowrap;font-variant-numeric:tabular-nums}th,td{padding:12px 16px;border-bottom:1px solid #28364a;text-align:right}th:first-child,td:first-child{text-align:left}th{background:#162439;color:#9db4cf}figure{margin:8px 0}video{width:100%;aspect-ratio:16/9;background:#05090f;object-fit:contain}.portrait video{aspect-ratio:9/16;max-height:550px}button,input{font:inherit;color:inherit;background:#1c3048;border:1px solid #466080;border-radius:6px;padding:8px 12px}button{cursor:pointer;margin:8px 8px 0 0}details{margin-top:14px}summary{cursor:pointer;color:#82c8ff}.note{padding:14px 18px;border-left:3px solid #8dd9d2;background:#111f2e}.evidence{overflow-wrap:anywhere;font-size:14px}@media(max-width:700px){main{padding:20px 14px}.cards,.videos{grid-template-columns:1fr}.case{padding:14px}}
</style></head><body><main><p>RTX 5090 · FIVE FIXED REQUESTS · TEN FRESH LOCAL GENERATIONS</p><h1>BF16 H3 base with EasyCache</h1><p>The original BF16 FL2VA pruned base and original BF16 Pika LoRA, with the three-step Pika sampler. These are BF16-base tests; the earlier reports used an INT8 base with BF16 or quantized adapters.</p><div class="cards" id="totals"></div><p class="note" id="conclusion"></p>
<h2>Measured request times</h2><p>Full graph execution through the saved video, synchronized with CUDA. Model download and process initialization are excluded. Each request starts in a fresh process after three idle GPU checks. Test order alternates by case. The 40.2 GB base uses the same dynamic VRAM/offload behavior in both configurations on the 32 GB GPU.</p><div class="scroll"><table><thead><tr><th>Video</th><th>Cache off</th><th>EasyCache on</th><th>Speedup</th><th>Uncached sampler</th><th>Cached sampler</th><th>Skipped evaluations</th></tr></thead><tbody id="rows"></tbody></table></div>
<h2>Cache settings and verification</h2><p>Native ComfyUI EasyCache, reuse threshold <strong>0.2</strong>, start <strong>0%</strong>, end <strong>100%</strong>, subsampling factor 8. Starting at the first step gives the three-step sampler a chance to warm the cache. The production Pika worker disables EasyCache; this experiment adds the native node in the isolated local workflow only. The original EasyCache decision algorithm is unchanged.</p><p>Actual full H3 model evaluations and native cache skip counts are recorded. Enabling the node alone is not treated as evidence of accelerated inference. Model weights and input dtypes are checked at runtime; the pinned BF16 checkpoint has zero quantization metadata entries. SageAttention 2.2 and the original Pika LoRA are unchanged.</p><p class="note">One measured request per video and configuration. Small timing differences may reflect run-to-run variation. Results apply to this three-step Pika workload and these cache settings.</p>
<h2>Output comparison</h2><p>PSNR and SSIM compare EasyCache against uncached BF16-base output over every decoded frame, using FFmpeg YUV420P at native delivered dimensions and 24 fps. No alignment search, resizing, trimming or sampled-frame scoring.</p><div class="scroll"><table><thead><tr><th>Video</th><th>PSNR</th><th>SSIM</th><th>Frames</th></tr></thead><tbody id="quality"></tbody></table></div>
<h2>Videos and per-request evidence</h2><input id="filter" aria-label="Filter animals" placeholder="Filter animals…"><div id="cases"></div><h2>Reproducibility</h2><p class="evidence"><a href="summary.json">Results and aggregate means</a> · <a href="evidence/base-provenance.json">BF16 base hash and precision</a> · <a href="evidence/gpu-gates.json">Idle GPU gates</a> · <a href="evidence/gpu-ownership.jsonl">GPU ownership samples</a> · <a href="evidence/cases.json">Exact input requests</a> · <a href="evidence/input-integrity.json">Input integrity</a> · <a href="evidence/run_easycache_case.py">Replay script</a> · <a href="evidence/run_easycache_suite.py">Suite runner</a> · <a href="evidence/easycache_verification.py">Runtime verification</a> · <a href="evidence/native_nodes_easycache.py">Native EasyCache implementation</a> · <a href="evidence/analyze_easycache.py">Analysis script</a></p></main><script>
const result=@@DATA@@,data=result.cases,fmt=x=>x.toFixed(3)+' s';
document.querySelector('#totals').innerHTML=['off','on'].map(p=>`<div class="card">${p==='off'?'Uncached BF16 base':'BF16 base + EasyCache'}<strong>${fmt(result.mean_seconds[p])}</strong>Mean full request time</div>`).join('');
const change=result.time_reduction_percent;
document.querySelector('#conclusion').textContent=`${result.aggregate_speedup.toFixed(4)}× aggregate speedup; ${Math.abs(change).toFixed(2)}% ${change>=0?'less':'more'} time than uncached BF16. Actual evaluations skipped: ${result.total_steps_skipped}/15.`;
document.querySelector('#rows').innerHTML=data.map(r=>`<tr><td>${r.name}</td><td>${fmt(r.times_seconds.off)}</td><td>${fmt(r.times_seconds.on)}</td><td>${r.speedup.toFixed(4)}×</td><td>${fmt(r.stage_seconds.off.SamplerCustomAdvanced)}</td><td>${fmt(r.stage_seconds.on.SamplerCustomAdvanced)}</td><td>${r.steps_skipped}/3</td></tr>`).join('');
document.querySelector('#quality').innerHTML=data.map(r=>`<tr><td>${r.name}</td><td>${r.quality.psnr_infinite?'∞':r.quality.psnr_db.toFixed(3)+' dB'}</td><td>${r.quality.ssim.toFixed(6)}</td><td>${r.quality.frames}</td></tr>`).join('');
function render(){const q=document.querySelector('#filter').value.toLowerCase();document.querySelector('#cases').innerHTML=data.filter(r=>r.name.toLowerCase().includes(q)).map(r=>`<article class="case ${r.slug==='barn-owl'?'portrait':''}"><h3>${r.name}</h3><div class="videos">${['off','on'].map(p=>`<figure><figcaption>${p==='off'?'Uncached BF16':'BF16 + EasyCache'} · ${fmt(r.times_seconds[p])}</figcaption><video controls playsinline preload="metadata" ${p==='on'?'muted':''}><source src="media/${r.slug}/${p}.mp4" type="video/mp4"></video></figure>`).join('')}</div><button class="play">Play together</button><button class="pause">Pause together</button><button class="reset">Restart</button><details><summary>Execution evidence</summary><p class="evidence">${['off','on'].map(p=>`<a href="evidence/runs/${r.slug}/${p}/result.json">${p} result</a> · <a href="evidence/runs/${r.slug}/${p}/process.log">log</a> · <a href="evidence/runs/${r.slug}/${p}/workflow-r1.json">workflow</a>`).join('<br>')}<br><a href="evidence/analysis/${r.slug}/metrics.json">Full-frame quality metrics</a></p></details></article>`).join('');document.querySelectorAll('.case').forEach(card=>{const videos=[...card.querySelectorAll('video')];card.querySelector('.play').onclick=async()=>{const t=videos[0].currentTime;videos.forEach(v=>v.currentTime=t);await Promise.all(videos.map(v=>v.play()))};card.querySelector('.pause').onclick=()=>videos.forEach(v=>v.pause());card.querySelector('.reset').onclick=()=>videos.forEach(v=>{v.pause();v.currentTime=0})})};document.querySelector('#filter').addEventListener('input',render);render();
</script></body></html>'''
(DEST/'comparison.html').write_text(page.replace('@@DATA@@',data))
manifest={str(p.relative_to(DEST)):hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in sorted(DEST.rglob('*')) if p.is_file() and p.name!='artifact-integrity.json'}
(DEST/'artifact-integrity.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps({'path':str(DEST),'files':len(manifest),'mean_seconds':summary['mean_seconds'],'speedup':summary['aggregate_speedup'],'steps_skipped':summary['total_steps_skipped']},indent=2))
