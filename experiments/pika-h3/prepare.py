import concurrent.futures
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import requests

ROOT = Path(__file__).resolve().parent
items = json.loads((ROOT / 'selected.json').read_text())
base = Path('/home/raqeeb/code/script2video/cache/minimax_h3')
for category in ('diffusion_models', 'text_encoders', 'vae'):
    (ROOT / 'models' / category).mkdir(exist_ok=True, parents=True)
    for path in (base / category).glob('*.safetensors'):
        target = ROOT / 'models' / category / path.name
        if not target.exists():
            target.symlink_to(path)
for name, path in {
    'h3_silu_temb_grid.safetensors': Path('/home/raqeeb/code/h3-optimization/larryvrh-h3-turbo-lora/upstream/h3_silu_temb_grid.safetensors'),
    'minimax_h3_turbo_v4_step600_ema.safetensors': Path('/home/raqeeb/code/h3-optimization/larryvrh-h3-turbo-lora/models/loras/minimax_h3_turbo_v4_step600_ema.safetensors'),
}.items():
    target = ROOT / 'models/loras' / name
    if not target.exists():
        target.symlink_to(path)

def fetch(url, path):
    if not path.exists():
        with requests.get(url, stream=True, timeout=90) as response:
            response.raise_for_status()
            with path.open('wb') as output:
                shutil.copyfileobj(response.raw, output)
    return {'url': url, 'path': str(path.relative_to(ROOT)), 'bytes': path.stat().st_size,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}

def prepare(item):
    slug = item['name'].lower().replace(' ', '-')
    case_dir = ROOT / 'cases' / slug
    case_dir.mkdir(parents=True, exist_ok=True)
    lookup = json.loads((ROOT / f'{slug}_uid_lookup.json').read_text())
    row = [r for r in lookup['backend_messages'] if str(r['code']) == '200'][-1]
    message = {k:v for k,v in row['message'].items() if k not in {'api_key','webhook_url','request_time'}}
    message['webhook_url'] = ''
    files = {}
    for key, suffix in [('image_path', '.jpg'), ('last_image_path', '.jpg'), ('audio_path', '.wav')]:
        if message.get(key):
            files[key] = fetch(message[key], case_dir / (key + suffix))
    files['original'] = fetch(item['url'], case_dir / 'original.mp4')
    probe = json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(case_dir/'audio_path.wav')]))
    duration = float(probe['streams'][0]['duration'])
    message['num_frames'] = max(5, math.ceil(duration * 24))
    case = {**item,'slug':slug,'message':message,'files':files,'source_host':row['host_name'],
            'source_timings':row['response']['timings'],'source_effective':row['response']['effective'],
            'source_server_start':row['server_start_time'],'source_server_end':row['server_end_time'],
            'audio_duration':duration}
    (case_dir/'case.json').write_text(json.dumps(case,indent=2))
    print(json.dumps({'case':slug,'frames':message['num_frames'],'size':[message['width'],message['height']],'duration':duration}),flush=True)
    return case

with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
    cases=list(pool.map(prepare,items))
(ROOT/'cases.json').write_text(json.dumps(cases,indent=2))
