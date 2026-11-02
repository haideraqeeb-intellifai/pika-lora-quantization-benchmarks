"""Compare fresh local BF16-base runs with EasyCache on and off."""
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'easycache-bf16'
helpers = {'__file__': str(ROOT / 'analyze.py'), '__name__': 'helpers'}
exec((ROOT / 'analyze.py').read_text().split('\nall_results=[]')[0], helpers)
rows = []
for case in json.loads((ROOT / 'cases.json').read_text()):
    slug = case['slug']
    results = {mode: json.loads((OUT / 'runs' / slug / mode / 'result.json').read_text()) for mode in ['off','on']}
    runs = {mode: value['runs'][0] for mode,value in results.items()}
    assert all(r['status']=='success' for r in runs.values())
    graphs = {mode:json.loads((OUT/'runs'/slug/mode/'workflow-r1.json').read_text()) for mode in ['off','on']}
    graphs['on'].pop('easycache')
    graphs['on']['7']['inputs']['model']=['sage',0]
    for graph in graphs.values():graph['17']['inputs']['filename_prefix']='OUTPUT'
    assert graphs['off']==graphs['on'], f'Unexpected workflow change: {slug}'
    assert runs['off']['easycache_verification']['full_model_evaluations']==3
    stats = runs['on']['easycache_verification']
    # The extra-node loader uses a separate module identity from regular imports.
    # Read its own end-of-sampling counter and cross-check actual H3 evaluations.
    log=(OUT/'runs'/slug/'on/process.log').read_text()
    skip_records=re.findall(r'EasyCache - skipped (\d+)/3 steps',log)
    assert len(skip_records)==1
    skips=int(skip_records[0])
    assert stats['full_model_evaluations']+skips==3
    paths = {mode:ROOT/run['output'] for mode,run in runs.items()}
    quality = helpers['metrics'](paths['off'],paths['on'],OUT/'analysis'/slug)
    for mode,path in paths.items():
        decoded = subprocess.run(['ffmpeg','-nostdin','-v','error','-threads','2','-i',str(path),'-f','null','-'],capture_output=True,text=True,timeout=180)
        decoded.check_returncode()
        (OUT/'analysis'/slug/f'{mode}-decode.log').write_text(decoded.stderr)
    times = {mode:run['execution_seconds'] for mode,run in runs.items()}
    stages = {mode:{node['class_type']:node['duration_seconds'] for node in run['telemetry']['stages']['nodes']} for mode,run in runs.items()}
    row = {'name':case['name'],'slug':slug,'times_seconds':times,'speedup':times['off']/times['on'],
           'time_reduction_percent':100*(1-times['on']/times['off']),'steps_skipped':skips,
           'full_evaluations':stats['full_model_evaluations'],'skip_counter_source':'native EasyCache sampler log',
           'verification':stats,'quality':quality,'stage_seconds':stages}
    rows.append(row)
    print(json.dumps({key:row[key] for key in ['name','times_seconds','speedup','steps_skipped']}),flush=True)
means = {mode:sum(r['times_seconds'][mode] for r in rows)/5 for mode in ['off','on']}
summary = {'cases':rows,'mean_seconds':means,'aggregate_speedup':means['off']/means['on'],
           'time_reduction_percent':100*(1-means['on']/means['off']),
           'total_steps_skipped':sum(r['steps_skipped'] for r in rows),
           'easycache_parameters':{'reuse_threshold':0.2,'start_percent':0.0,'end_percent':1.0},
           'base':'original BF16 H3 FL2VA pruned; no quantization metadata',
           'steps':3,'valid_generations':10,'caveat':'One measured request per case and configuration.'}
(OUT/'analysis/summary.json').write_text(json.dumps(summary,indent=2))
