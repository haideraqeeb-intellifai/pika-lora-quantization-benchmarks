"""Package measured evidence and native videos into a portable comparison report."""
import csv
import html
import json
import math
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parent
DEST=ROOT/'pika-fp8-five-video-comparison-20261102'
DEST.mkdir(exist_ok=True)
results=json.loads((ROOT/'analysis/summary.json').read_text())
cases={c['slug']:c for c in json.loads((ROOT/'cases.json').read_text())}
observations=json.loads((ROOT/'visual-observations.json').read_text())
if len(results)!=5:raise RuntimeError('Five completed comparisons required')
if set(cases)-set(observations):raise RuntimeError('Every case requires a frame review before publishing')
public=[]
for r in results:
    slug=r['slug'];case=cases[slug]
    target=DEST/'media'/slug;target.mkdir(parents=True,exist_ok=True)
    evidence=DEST/'evidence'/slug;evidence.mkdir(parents=True,exist_ok=True)
    for precision in ['bf16','fp8']:
        run=r['baseline' if precision=='bf16' else 'fp8']['runs'][0]
        shutil.copyfile(ROOT/run['output'],target/f'{precision}.mp4')
        shutil.copyfile(ROOT/'runs'/slug/precision/'result.json',evidence/f'{precision}-result.json')
        shutil.copyfile(ROOT/'runs'/slug/precision/'process.log',evidence/f'{precision}-process.log')
        shutil.copyfile(ROOT/'runs'/slug/precision/'workflow-r1.json',evidence/f'{precision}-workflow.json')
        for extra in ['first-call-result.json','raw-process-result.json']:
            source=ROOT/'runs'/slug/precision/extra
            if source.exists():shutil.copyfile(source,evidence/f'{precision}-{extra}')
    shutil.copyfile(ROOT/case['files']['original']['path'],target/'original.mp4')
    shutil.copyfile(ROOT/'analysis'/slug/'frames.jpg',target/'frames.jpg')
    shutil.copyfile(ROOT/'cases'/slug/'case.json',evidence/'source-case.json')
    lookup=json.loads((ROOT/f'{slug}_uid_lookup.json').read_text())
    # Publish exact job settings/timings without credentials or unrelated gateway payloads.
    safe_lookup={'target_uid':lookup['target_uid'],'time_window':lookup['time_window'],'summary':lookup['summary'],
                 'source_host':case['source_host'],'source_worker_release':case['source_worker_release'],'source_gateway_revision':case['source_gateway_revision'],'source_timings':case['source_timings'],
                 'server_start_time':case['source_server_start'],'server_end_time':case['source_server_end'],
                 'method':'mlagent-uid-lookup read-only bounded exact UID query; selected source message/response; secrets excluded'}
    (evidence/'uid-lookup-summary.json').write_text(json.dumps(safe_lookup,indent=2))
    for name in r['quality']:
        shutil.copytree(ROOT/'analysis'/slug/name,evidence/name,dirs_exist_ok=True)
    public.append({k:v for k,v in r.items() if k not in {'baseline','fp8'}} | {
        'source_url':case['url'],'source_timings':case['source_timings'],'prompt':case['message']['prompt'],'visual_observation':observations.get(slug,'Frame review pending'),
        'settings':{k:case['message'][k] for k in ['seed','num_inference_steps','width','height','fps','num_frames','input_mode','model']},
        'bf16_uid':r['baseline']['runs'][0]['uid'],'fp8_uid':r['fp8']['runs'][0]['uid'],
        'warm_control':{'bf16_seconds':r['baseline']['runs'][1]['execution_seconds'],'fp8_seconds':r['fp8']['runs'][1]['execution_seconds'],
                        'speedup':r['baseline']['runs'][1]['execution_seconds']/r['fp8']['runs'][1]['execution_seconds']}
                        if len(r['baseline']['runs'])>1 and len(r['fp8']['runs'])>1 else None,
        'gpu_telemetry_bf16':r['baseline']['runs'][0]['telemetry'].get('gpu'),
        'gpu_telemetry_fp8':r['fp8']['runs'][0]['telemetry'].get('gpu'),
        'stages_bf16':r['baseline']['runs'][0]['telemetry'].get('stages'),
        'stages_fp8':r['fp8']['runs'][0]['telemetry'].get('stages')})
for name in ['source-provenance.json','worker-release-provenance.json','gpu-gates.json','gpu-ownership.jsonl','input-integrity.json','quantization-verification.json','visual-observations.json','run_case.py','run_suite.py','fp8_adapter.py','analyze.py','prepare.py']:
    shutil.copyfile(ROOT/name,DEST/name)
(DEST/'results.json').write_text(json.dumps(public,indent=2))
with (DEST/'results.csv').open('w') as f:
    fields=['name','source_uid','source_host','source_seconds','bf16_seconds','fp8_seconds','historical_speedup','matched_host_speedup','matched_host_time_reduction_percent','psnr_db','ssim','matched_psnr_db','matched_ssim']
    w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
    for r in public:w.writerow({k:r.get(k) for k in fields} | {'psnr_db':r['quality']['original_vs_fp8']['psnr_db'],'ssim':r['quality']['original_vs_fp8']['ssim'],
                                                            'matched_psnr_db':r['quality']['bf16_vs_fp8']['psnr_db'],'matched_ssim':r['quality']['bf16_vs_fp8']['ssim']})
bf=sum(r['bf16_seconds'] for r in public);fp=sum(r['fp8_seconds'] for r in public)
source=sum(r['source_seconds'] for r in public)
data=json.dumps(public).replace('</','<\\/')
report=r'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Pika FP8 adapter · Five-video comparison</title>
<style>
:root{color-scheme:dark;--bg:#0a1019;--panel:#111d2c;--ink:#e8eff9;--muted:#a3b4ca;--line:#2a3a50;--green:#71e5be;--amber:#ffce78;--blue:#86b8ff}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.6 system-ui,sans-serif}main{max-width:1500px;margin:auto;padding:40px 24px}h1{font-size:clamp(30px,4vw,58px);line-height:1.12;letter-spacing:-.04em;margin:12px 0 20px}h2{font-size:24px;margin:34px 0 16px}h3{margin:0}p{max-width:1060px}a{color:var(--blue)}a:focus,button:focus,select:focus{outline:3px solid var(--green);outline-offset:3px}.muted,small{color:var(--muted)}.eyebrow{color:var(--green);font-weight:700;letter-spacing:.12em;text-transform:uppercase}.summary,.case,.box{background:var(--panel);border:1px solid var(--line);border-radius:18px;padding:24px}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:24px 0}.stat{background:#142437;border:1px solid var(--line);border-radius:14px;padding:18px}.stat strong{display:block;font-size:30px;line-height:1.25}.stat span{color:var(--muted);font-size:13px}.callout{border-left:3px solid var(--amber);padding:12px 18px;background:#242217;border-radius:0 10px 10px 0}.table-wrap{overflow:auto;border:1px solid var(--line);border-radius:12px}table{width:100%;border-collapse:collapse;white-space:nowrap;font-variant-numeric:tabular-nums}th,td{text-align:right;padding:12px 14px;border-bottom:1px solid var(--line)}th:first-child,td:first-child{text-align:left}th{background:#152337;cursor:pointer}th button{font:inherit;background:transparent;border:0;padding:0;color:inherit;cursor:pointer}tbody tr:hover{background:#16273a}.toolbar{display:flex;flex-wrap:wrap;gap:14px;align-items:center;margin:18px 0}button,select,input{font:inherit;background:#172a3e;color:var(--ink);border:1px solid #405773;border-radius:8px;padding:8px 12px}button{cursor:pointer}input{min-width:220px}.case{margin:24px 0}.case-head{display:flex;flex-wrap:wrap;justify-content:space-between;gap:14px;margin-bottom:18px}.badge{display:inline-block;padding:4px 9px;background:#203e42;border:1px solid #3c6b67;border-radius:999px;color:var(--green);font-size:12px}.media-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}figure{margin:0;min-width:0}figcaption{margin-bottom:8px;color:var(--muted);font-size:13px}video{width:100%;height:auto;max-height:540px;display:block;background:#030609;border:1px solid var(--line);border-radius:10px}video.portrait{max-height:600px;object-fit:contain}details{margin-top:18px;border-top:1px solid var(--line);padding-top:12px}summary{cursor:pointer;color:var(--blue)}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#0b1421;padding:16px;border-radius:10px;font:13px/1.6 ui-monospace,monospace}.metrics{display:flex;flex-wrap:wrap;gap:8px 20px;margin:14px 0;font-variant-numeric:tabular-nums}.metrics b{color:var(--green)}.contact{display:block;width:100%;height:auto;margin-top:12px}.method-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}li{margin:8px 0}.verdict{font-size:18px;color:var(--green)}.footer{margin:32px 0;color:var(--muted);font-size:13px}@media(max-width:900px){.stats{grid-template-columns:1fr 1fr}.media-grid{grid-template-columns:1fr}.method-grid{grid-template-columns:1fr}main{padding:24px 14px}.summary,.case,.box{padding:18px}video{max-height:640px}}@media(max-width:450px){.stats{gap:8px}.stat{padding:12px}.stat strong{font-size:24px}input{min-width:0;width:100%}}
code{overflow-wrap:anywhere}.box,.stat,details,article{min-width:0}</style><main>
<header><div class="eyebrow">Pika Flex · Precision experiment · 2 November 2026</div><h1>What changes when the Pika adapter runs in FP8?</h1>
<p class="muted">Five exact requests from the animal podcast batch. The H3 INT8 ConvRot base, Pika sampler, conditioning, seed, step count and output settings are held fixed. Only Pika adapter weights and adapter GEMMs change from BF16 to FP8 E4M3.</p></header>
<section class="summary"><h2 style="margin-top:0">Measured result</h2><p class="verdict" id="verdict"></p><div class="stats" id="stats"></div>
<p class="callout">PSNR and SSIM measure similarity to a reference output, not whether a generated video is better. Higher is closer. Historical timings came from different RTX 5090 workers; use the fresh, same-host BF16 comparison to assess the precision change.</p>
<p><a href="results.csv">Download comparison CSV</a> · <a href="results.json">Raw results JSON</a> · <a href="source-provenance.json">Source-code verification</a> · <a href="gpu-gates.json">GPU idle checks</a> · <a href="gpu-ownership.jsonl">GPU ownership samples</a></p></section>
<h2>Quality and execution time</h2><p class="muted">All video scores include all 362 decoded frames at the original resolution. Execution time covers the Comfy graph through saved output, including graph model loading; it excludes process initialization, network downloads, uploads and metric computation. One first-call timing per case/configuration; Red Fox also has a warm repeat below.</p>
<div class="toolbar"><label>Quality reference <select id="reference"><option value="original_vs_fp8">Published original → FP8</option><option value="bf16_vs_fp8">Fresh BF16 → FP8</option><option value="original_vs_bf16">Published original → fresh BF16</option></select></label><span class="muted">Click a column header to sort.</span></div>
<div class="table-wrap"><table><thead><tr><th><button data-sort="name">Video</button></th><th><button data-sort="psnr">PSNR dB ↑</button></th><th><button data-sort="ssim">SSIM ↑</button></th><th><button data-sort="source_seconds">Original log s</button></th><th><button data-sort="bf16_seconds">Fresh BF16 s</button></th><th><button data-sort="fp8_seconds">FP8 s</button></th><th><button data-sort="matched_host_speedup">Same-host speedup</button></th><th><button data-sort="historical_speedup">Historical ratio</button></th></tr></thead><tbody id="rows"></tbody></table></div>
<h2>Watch the five comparisons</h2><div class="toolbar"><input id="filter" aria-label="Filter videos" placeholder="Filter by animal"><span class="muted">Synchronized controls play all three versions; audio plays from the original only.</span></div><div id="cases"></div>
<h2>What was tested</h2><div class="method-grid"><section class="box"><h3>Precision and execution</h3><ul>
<li><strong>Base:</strong> unchanged <code>minimax_h3_fl2va_pruned_int8_convrot.safetensors</code>, MiniMax revision <code>0bd506d2</code>.</li>
<li><strong>Pika:</strong> official Comfy BF16 FL2VA adapter, revision <code>3ff16644</code>, SHA-256 verified. Converted on load to FP8 E4M3, per-tensor absmax scales. Activations are dynamically scaled to FP8 before both low-rank projections.</li>
<li><strong>Real FP8 GEMMs:</strong> explicit <code>torch._scaled_mm</code> with FP8 operands. No BF16 fallback path exists. Bypass outputs return to the original BF16 activation type; the fused fc2 merge uses FP8 operands with FP32 output, retaining the existing base merge route.</li>
<li><strong>Everything else:</strong> strength 1, SageAttention 2.2.0, dynamic VRAM, Pika re-noise sampler, 3 steps, shifts 12/3, seed <code>427739230855701667</code>. Same prompt/image/last image/driving audio. No TeaCache.</li>
<li><strong>Hardware:</strong> one RTX 5090 on gpu5090-79. Fresh processes; no competing GPU processes observed at the pre-run gates. Driver-only idle baseline: 2 MiB.</li>
</ul></section><section class="box"><h3>Measurement and interpretation</h3><ul>
<li><strong>Selection:</strong> Red Fox, Bengal Tiger, African Elephant, Scarlet Macaw and Barn Owl were selected before generation to span fur, skin, feathers, gestures and landscape/portrait.</li>
<li><strong>Video:</strong> 15.0833 seconds, 362 frames, 24 FPS; four 1280×720 outputs and one 720×1280 output. Native landscape generation is 1280×736, cropped by the unchanged production workflow.</li>
<li><strong>Metrics:</strong> FFmpeg PSNR and SSIM on decoded YUV420P, corresponding frames with timestamps reset. No resizing, motion alignment, skipped frames or cherry-picked segments. PSNR uses combined MSE; SSIM is the mean over all frames. Per-frame logs are downloadable.</li>
<li><strong>Speedup:</strong> BF16 execution seconds ÷ FP8 execution seconds. Above 1× is faster. The aggregate is the ratio of the five total times. The historical ratio uses only original <code>comfy_execution_seconds</code>, excluding queue, input-download and upload time.</li>
<li><strong>Controls:</strong> every input file was hashed; production request settings and backend times were recovered with <code>mlagent-uid-lookup</code>. Fresh replays are direct local inference runs, so their UIDs and timings are in the linked process/result logs, not inserted into the production database.</li>
<li><strong>Limits:</strong> single timing samples cannot establish statistical significance. Adapter-only FP8 says nothing about quantizing the base or other H3 configurations. These jobs use supplied driving audio, which is muxed unchanged; audio-generation quality is not tested.</li>
</ul></section></div>
<details><summary>Source, reproducibility and implementation evidence</summary><p>Original report: <a href="https://s3.renderplatform.com/agent-assets/internal-ml-sota/pika-flex-animal-batch-report-20261102-corrected/99a5e8df5e42_index.html">Pika Flex animal podcast batch</a>. Source worker releases: <code>v4.9.129</code> for four cases and <code>v4.9.125</code> for Scarlet Macaw. <a href="worker-release-provenance.json">Relevant inference source blobs match both release tags</a>, including Pika application, sampler, workflow, assets and driving-audio model/conditioning code.</p><p><a href="fp8_adapter.py">FP8 implementation</a> · <a href="run_case.py">Replay runner</a> · <a href="analyze.py">Metric implementation</a> · <a href="prepare.py">Input preparation</a> · <a href="run_suite.py">Sequential runner</a></p><p>Preflight found stale driving-audio code in the local checkout. It was replaced with exact source-job code before the baseline/FP8 runs; the failed preflight produced no compared output. Production was not changed.</p></details>
<p class="footer">Report includes original delivered videos, native BF16/FP8 replay outputs, prompts, settings, timings and per-frame scoring evidence. All processing and scoring are complete before this report is packaged.</p>
</main><script>
const DATA=__DATA__;
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const n=(x,d=2)=>x==null?'∞':Number(x).toFixed(d);
const sum=k=>DATA.reduce((a,r)=>a+r[k],0);
const speed=sum('bf16_seconds')/sum('fp8_seconds');const change=100*(1-sum('fp8_seconds')/sum('bf16_seconds'));
document.querySelector('#verdict').textContent=`FP8 adapter execution is ${speed.toFixed(3)}× the BF16 replay speed across these five cases (${Math.abs(change).toFixed(1)}% ${change>=0?'less':'more'} execution time).`;
const scores=DATA.map(r=>r.quality.original_vs_fp8);
document.querySelector('#stats').innerHTML=[['5 / 5','Matched video pairs'],[n(speed,3)+'×','Same-host aggregate speedup'],[n(Math.min(...scores.map(q=>q.psnr_db)))+'–'+n(Math.max(...scores.map(q=>q.psnr_db))),'PSNR range vs original · dB'],[n(scores.reduce((s,q)=>s+q.ssim,0)/5,5),'Mean SSIM vs original']].map(([a,b])=>`<div class="stat"><strong>${a}</strong><span>${b}</span></div>`).join('');
let sort='name',direction=1;
function table(){const ref=document.querySelector('#reference').value;let data=[...DATA];const val=(r,k)=>k==='psnr'?r.quality[ref].psnr_db??Infinity:k==='ssim'?r.quality[ref].ssim:r[k];data.sort((a,b)=>typeof val(a,sort)==='string'?direction*val(a,sort).localeCompare(val(b,sort)):direction*(val(a,sort)-val(b,sort)));document.querySelector('#rows').innerHTML=data.map(r=>`<tr><td><a href="#${r.slug}">${esc(r.name)}</a></td><td>${n(r.quality[ref].psnr_db)}</td><td>${n(r.quality[ref].ssim,6)}</td><td>${n(r.source_seconds)}</td><td>${n(r.bf16_seconds)}</td><td>${n(r.fp8_seconds)}</td><td>${n(r.matched_host_speedup,3)}×</td><td>${n(r.historical_speedup,3)}×</td></tr>`).join('')}
document.querySelector('#reference').addEventListener('change',table);document.querySelectorAll('[data-sort]').forEach(b=>b.addEventListener('click',()=>{direction=sort===b.dataset.sort?-direction:1;sort=b.dataset.sort;table()}));table();
function stageTable(r){const stages=['MiniMaxH3ImageToVideo','MiniMaxH3PikaLoRA','SamplerCustomAdvanced','VAEDecode','SaveVideo'];const value=(side,k)=>(r['stages_'+side]?.nodes||[]).find(s=>s.class_type===k)?.duration_seconds;return '<div class="table-wrap"><table><thead><tr><th>Node stage</th><th>BF16 s</th><th>FP8 s</th></tr></thead><tbody>'+stages.map(k=>'<tr><td>'+k+'</td><td>'+n(value('bf16',k))+'</td><td>'+n(value('fp8',k))+'</td></tr>').join('')+'</tbody></table></div><p class="muted">Node lifecycle wall times include deferred loading and transfers. Sampler timing includes denoising and its model loading/offload work; it is not a pure CUDA-kernel timer.</p>'}
function cards(){const query=document.querySelector('#filter').value.toLowerCase();document.querySelector('#cases').innerHTML=DATA.filter(r=>r.name.toLowerCase().includes(query)).map(r=>{const q=r.quality.original_vs_fp8,control=r.quality.original_vs_bf16;return `<article class="case" id="${r.slug}"><div class="case-head"><div><h3>${esc(r.name)}</h3><small>${r.settings.width}×${r.settings.height} · 362 frames · 3 steps</small></div><span class="badge">${n(r.matched_host_speedup,3)}× same-host speedup</span></div><div class="media-grid">${[['original','Published original · BF16 Pika adapter',r.source_seconds],['bf16','Fresh BF16 replay · same GPU',r.bf16_seconds],['fp8','FP8 weights + FP8 adapter GEMMs',r.fp8_seconds]].map(([key,label,seconds],i)=>`<figure><figcaption>${label}<br>${n(seconds)} s execution</figcaption><video ${i?'muted':''} controls preload="metadata" playsinline ${r.settings.height>r.settings.width?'class="portrait"':''} aria-label="${esc(r.name)} ${key}"><source src="media/${r.slug}/${key}.mp4" type="video/mp4"></video></figure>`).join('')}</div><div class="toolbar"><button class="play">Play all</button><button class="pause">Pause all</button><button class="restart">Restart</button></div><div class="metrics"><span>Original → FP8 PSNR <b>${n(q.psnr_db)} dB</b></span><span>SSIM <b>${n(q.ssim,6)}</b></span><span>BF16 replay → FP8 SSIM <b>${n(r.quality.bf16_vs_fp8.ssim,6)}</b></span><span>Original → BF16 replay SSIM <b>${n(control.ssim,6)}</b></span></div><p class="muted">Frame review: ${esc(r.visual_observation)}</p><details><summary>Frame comparison at 0, 4.5, 9.5 and 14 seconds</summary><a href="media/${r.slug}/frames.jpg"><img loading="lazy" class="contact" alt="${esc(r.name)} original, BF16 and FP8 frame comparison" src="media/${r.slug}/frames.jpg"></a></details><details><summary>Exact prompt, settings, timing provenance and FP8 verification</summary><p>Original UID: <code>${esc(r.source_uid)}</code><br>Original worker: <code>${esc(r.source_host)}</code><br>FP8 run UID: <code>${esc(r.fp8_uid)}</code></p><p>Historical worker end-to-end: ${n(r.source_end_to_end_seconds)} s; Comfy graph: ${n(r.source_seconds)} s. Input preparation and upload are excluded from the ratio.</p>${r.warm_control?`<p>Additional Red Fox repeat: warmed BF16 ${n(r.warm_control.bf16_seconds)} s; warmed FP8 ${n(r.warm_control.fp8_seconds)} s; ${n(r.warm_control.speedup,3)}×. These repeats are excluded from the five-case first-call aggregate.</p>`:''}${stageTable(r)}<pre>${esc(JSON.stringify(r.settings,null,2))}</pre><pre>${esc(r.prompt)}</pre><pre>${esc(JSON.stringify(r.fp8_verification,null,2))}</pre><p><a href="evidence/${r.slug}/source-case.json">Source request</a> · <a href="evidence/${r.slug}/uid-lookup-summary.json">UID lookup timing evidence</a> · <a href="evidence/${r.slug}/bf16-result.json">BF16 telemetry</a> · <a href="evidence/${r.slug}/fp8-result.json">FP8 telemetry</a> · <a href="evidence/${r.slug}/fp8-process.log">FP8 process log</a> · <a href="evidence/${r.slug}/original_vs_fp8/psnr-frames.log">Per-frame PSNR</a> · <a href="evidence/${r.slug}/original_vs_fp8/ssim-frames.log">Per-frame SSIM</a></p></details></article>`}).join('');
document.querySelectorAll('.case').forEach(card=>{const videos=[...card.querySelectorAll('video')];card.querySelector('.play').onclick=()=>{const t=videos[0].currentTime;videos.forEach(v=>{v.currentTime=t;v.play().catch(()=>{})})};card.querySelector('.pause').onclick=()=>videos.forEach(v=>v.pause());card.querySelector('.restart').onclick=()=>videos.forEach(v=>{v.pause();v.currentTime=0});videos[0].addEventListener('seeking',()=>videos.slice(1).forEach(v=>{if(Math.abs(v.currentTime-videos[0].currentTime)>.08)v.currentTime=videos[0].currentTime}));videos[0].addEventListener('timeupdate',()=>{if(!videos[0].paused)videos.slice(1).forEach(v=>{if(Math.abs(v.currentTime-videos[0].currentTime)>.15)v.currentTime=videos[0].currentTime})})})}
document.querySelector('#filter').addEventListener('input',cards);cards();
</script></html>'''
report=report.replace('__DATA__',data)
(DEST/'comparison.html').write_text(report)
print(json.dumps({'report':str(DEST/'comparison.html'),'cases':len(public),'matched_host_aggregate_speedup':bf/fp,'historical_ratio':source/fp},indent=2))
