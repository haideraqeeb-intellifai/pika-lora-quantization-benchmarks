"""Sequential benchmark launcher; refuses competing GPU workloads."""
import argparse
import json
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parent
PYTHON='/home/raqeeb/uv/envs/h3_5090/bin/python'
parser=argparse.ArgumentParser()
parser.add_argument('--skip-fox-bf16',action='store_true')
parser.add_argument('--wait-for-pid',type=int)
args=parser.parse_args()
if args.wait_for_pid:
    import os
    while True:
        try:os.kill(args.wait_for_pid,0)
        except ProcessLookupError:break
        time.sleep(5)
gate=json.loads((ROOT/'easycache-bf16/gpu-gates.json').read_text()) if (ROOT/'easycache-bf16/gpu-gates.json').exists() else []
cases=json.loads((ROOT/'cases.json').read_text())
for index,case in enumerate(cases):
    for precision in (['off','on'] if index%2==0 else ['on','off']):
        if args.skip_fox_bf16 and case['slug']=='red-fox' and precision=='bf16':continue
        directory=ROOT/'easycache-bf16/runs'/case['slug']/precision
        directory.mkdir(parents=True,exist_ok=True)
        result_path=directory/'result.json'
        if result_path.exists():
            result=json.loads(result_path.read_text())
            if result.get('runs') and all(x['status']=='success' for x in result['runs']):continue
            stamp=str(int(time.time()))
            result_path.rename(directory/f'failed-result-{stamp}.json')
            if (directory/'process.log').exists():(directory/'process.log').rename(directory/f'failed-process-{stamp}.log')
        consecutive_idle=0
        for _ in range(60):
            gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True).strip()
            processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],text=True).strip()
            entry={'time_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'before':case['slug']+'/'+precision,'gpu':gpu,'processes':processes}
            gate.append(entry)
            (ROOT/'easycache-bf16/gpu-gates.json').write_text(json.dumps(gate,indent=2))
            if processes or int(gpu.split(',')[1])!=0 or int(gpu.split(',')[2])>2:
                consecutive_idle=0
            else:consecutive_idle+=1
            if consecutive_idle==3:break
            time.sleep(1)
        else:raise RuntimeError('GPU not at established 2 MiB driver-only idle baseline: '+gpu+' '+processes)
        command=[PYTHON,'-u',str(ROOT/'run_easycache_case.py'),'--case',case['slug'],'--cache',precision,'--repeats','1']
        print('START '+case['slug']+' '+precision,flush=True)
        with (directory/'process.log').open('w') as log:
            code=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=14400).returncode
        print('DONE '+case['slug']+' '+precision+' '+str(code),flush=True)
        if code:raise RuntimeError('run failed: '+case['slug']+' '+precision)
print('SUITE_COMPLETE',flush=True)
