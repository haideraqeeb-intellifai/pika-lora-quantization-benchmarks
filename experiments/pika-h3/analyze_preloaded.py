"""Full-frame quality and paired graph timings for the preloaded experiment."""
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'preloaded'
# Reuse the exact full-frame scoring method from the first experiment.
namespace = {'__file__': str(ROOT / 'analyze.py'), '__name__': 'metric_helpers'}
helpers = (ROOT / 'analyze.py').read_text().split('\nall_results=[]')[0]
helpers = helpers.replace("['Published original / BF16 adapter','Matched-host BF16 replay','FP8 adapter / FP8 GEMMs']",
                          "['Fresh BF16 adapter','Preloaded FP8 adapter','Preloaded INT8 adapter']")
exec(helpers, namespace)
metrics = namespace['metrics']

rows = []
for case in json.loads((ROOT / 'cases.json').read_text()):
    slug = case['slug']
    paths = {p: OUT / 'runs' / slug / p / 'result.json' for p in ['bf16', 'fp8', 'int8']}
    if not all(p.exists() for p in paths.values()):
        continue
    results = {p: json.loads(path.read_text()) for p, path in paths.items()}
    if not all(r['runs'][0]['status'] == 'success' for r in results.values()):
        continue
    runs = {p: r['runs'][0] for p, r in results.items()}
    reference_graph = json.loads((ROOT / 'runs' / slug / 'bf16/workflow-r1.json').read_text())
    for graph in [reference_graph]:
        for node in graph.values():
            if node['class_type'] == 'SaveVideo':
                node['inputs']['filename_prefix'] = 'OUTPUT'
    for precision in ['bf16', 'fp8', 'int8']:
        graph = json.loads((OUT / 'runs' / slug / precision / 'workflow-r1.json').read_text())
        for node in graph.values():
            if node['class_type'] == 'SaveVideo':
                node['inputs']['filename_prefix'] = 'OUTPUT'
        assert graph == reference_graph, f'Workflow changed: {slug}/{precision}'
    for precision in ['fp8', 'int8']:
        v = runs[precision]['quantized_verification']
        assert v['adapter_instances'] == 208 and v['bypass_calls'] == 458 and v['merge_calls'] == 150
        assert v['native_gemm_calls'] == 1066 and sum(v['shapes'].values()) == 1066
        assert v['fallback_calls'] == v['weight_quantizations_in_request'] == v['weight_transfers_in_request'] == 0
    outputs = {p: ROOT / r['output'] for p, r in runs.items()}
    analysis = OUT / 'analysis' / slug
    quality = {p: metrics(outputs['bf16'], outputs[p], analysis / f'bf16-vs-{p}') for p in ['fp8', 'int8']}
    quality['bf16_control'] = metrics(ROOT / case['files']['original']['path'], outputs['bf16'], analysis / 'original-vs-bf16')
    quality['fp8_preload_equivalence'] = metrics(ROOT / 'runs' / slug / 'fp8/output-r1.mp4', outputs['fp8'], analysis / 'old-vs-preloaded-fp8')
    namespace['contact'](case, outputs['bf16'], outputs['fp8'], outputs['int8'], analysis / 'frames.jpg')
    for p, path in outputs.items():
        decoded = subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-threads', '2', '-i', str(path), '-f', 'null', '-'],
                                 capture_output=True, text=True, timeout=180)
        decoded.check_returncode()
        (analysis / f'{p}-decode.log').write_text(decoded.stderr)
    stages = {}
    for precision, run in runs.items():
        values = {}
        for node in run['telemetry']['stages']['nodes']:
            key = node.get('class_type', node.get('class', 'unknown'))
            values[key] = values.get(key, 0) + node['duration_seconds']
        stages[precision] = values
    times = {p: r['execution_seconds'] for p, r in runs.items()}
    old = {p: json.loads((ROOT / 'runs' / slug / p / 'result.json').read_text())['runs'][0]['execution_seconds']
           for p in ['bf16', 'fp8']}
    row = {'slug': slug, 'name': case['name'], 'source_uid': case['uid'], 'times_seconds': times,
           'graph_matches_previous_bf16_except_output_prefix': True,
           'speedup': {p: times['bf16'] / times[p] for p in ['fp8', 'int8']},
           'time_reduction_percent': {p: 100 * (1 - times[p] / times['bf16']) for p in ['fp8', 'int8']},
           'previous_experiment_seconds': old, 'quality': quality, 'stage_seconds': stages,
           'preparation': {p: results[p]['adapter_preparation'] for p in ['fp8', 'int8']},
           'verification': {p: runs[p]['quantized_verification'] for p in ['fp8', 'int8']}}
    rows.append(row)
    print(json.dumps({k: row[k] for k in ['name', 'times_seconds', 'speedup']}), flush=True)
(OUT / 'analysis').mkdir(exist_ok=True)
(OUT / 'analysis/summary.json').write_text(json.dumps(rows, indent=2))
