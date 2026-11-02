"""Isolated replay through the production Pika graph, with optional FP8 adapter GEMMs."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--case', required=True)
parser.add_argument('--precision', choices=['bf16','fp8'],required=True)
parser.add_argument('--repeats',type=int,default=1)
args = parser.parse_args()
sys.argv = [sys.argv[0]]
sys.path.insert(0, str(ROOT/'source'))
os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
os.environ['DO_NOT_TRACK']='1'
import torch
from ml_modules.minimax_h3_turbo import infer as engine
from ml_modules.minimax_h3_turbo.assets import ADAPTER_HASHES, ADAPTER_SIZES, PIKA_FL2VA, GRID
from ml_modules.minimax_h3_turbo.workflow import build_workflow

case = next(c for c in json.loads((ROOT/'cases.json').read_text()) if c['slug']==args.case)
run_dir=ROOT/'runs'/args.case/args.precision
run_dir.mkdir(parents=True,exist_ok=True)
runtime_root=run_dir/'runtime'
model_root=ROOT/'models'

def validate_selected(root):
    # Only FL2VA + Pika is exercised; do not require unrelated Ref2VA or character-swap adapters.
    for name in [f'loras/{PIKA_FL2VA}',f'loras/{GRID}']:
        p=Path(root)/name
        if p.stat().st_size != ADAPTER_SIZES[name]: raise ValueError(name+' size')
        with p.open('rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
        if digest != ADAPTER_HASHES[name]: raise ValueError(name+' hash')
    for category,name in [('diffusion_models',case['source_effective']['diffusion_checkpoint']),
                           ('text_encoders','qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors'),
                           ('vae','minimax_h3_video_vae_fp16.safetensors'),('vae','minimax_h3_audio_vae_fp32.safetensors')]:
        if not (Path(root)/category/name).is_file():raise FileNotFoundError(name)
engine.validate_assets=validate_selected

result={'case':args.case,'precision':args.precision,'source_uid':case['uid'],
        'configuration':case['source_effective'],'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version','--format=csv,noheader'],text=True).strip(),
        'packages':{n:importlib.metadata.version(n) for n in ['torch','sageattention','comfy-kitchen','comfy-aimdo','av','transformers']},'runs':[]}
started=time.perf_counter()
models=engine.load_models(model_root,runtime_root)
result['runtime_initialization_seconds']=time.perf_counter()-started
if args.precision=='fp8':
    from fp8_adapter import install
    fp8_stats=install()
else:fp8_stats=None

names={}
for key,entry in case['files'].items():
    if key=='original':continue
    source=ROOT/entry['path']
    name=key+source.suffix
    shutil.copyfile(source,runtime_root/'input'/name)
    names[key]=name

for repeat in range(args.repeats):
    uid=f'pika-fp8-ab-{args.case}-{args.precision}-20261102-r{repeat+1}'
    prefix=uid+'/output'
    graph,effective=build_workflow(case['message'],first_frame_name=names.get('image_path'),last_frame_name=names.get('last_image_path'),
                                  audio_name=names.get('audio_path'),output_prefix=prefix,seed=case['message']['seed'])
    (run_dir/f'workflow-r{repeat+1}.json').write_text(json.dumps(graph,indent=2))
    torch.cuda.synchronize()
    start=time.perf_counter()
    record={'uid':uid,'start_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'effective':effective}
    try:
        observation=engine.infer(models,graph,prompt_id=uid,timeout_seconds=9000)
        torch.cuda.synchronize()
        record['execution_seconds']=time.perf_counter()-start
        source=next((runtime_root/'output'/uid).glob('output_*.mp4'))
        target=run_dir/f'output-r{repeat+1}.mp4'
        shutil.copyfile(source,target)
        record.update(status='success',output=str(target.relative_to(ROOT)),telemetry=observation['telemetry'])
        if fp8_stats is not None:record['fp8_verification']=json.loads(json.dumps(fp8_stats))
    except Exception:
        record.update(status='failed',execution_seconds=time.perf_counter()-start,traceback=traceback.format_exc())
        if fp8_stats is not None:record['fp8_verification']=json.loads(json.dumps(fp8_stats))
        result['runs'].append(record)
        (run_dir/'result.json').write_text(json.dumps(result,indent=2))
        raise
    result['runs'].append(record)
    (run_dir/'result.json').write_text(json.dumps(result,indent=2))
    print('BENCHMARK_RESULT '+json.dumps({k:v for k,v in record.items() if k not in {'effective','telemetry'}}),flush=True)
