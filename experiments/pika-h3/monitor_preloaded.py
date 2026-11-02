"""Host-level ownership telemetry. Records only PIDs, GPU use and owned/not-owned."""
import json
from pathlib import Path
import subprocess
import time
ROOT=Path(__file__).resolve().parent
start=time.monotonic()
with (ROOT/'preloaded/gpu-ownership.jsonl').open('a',buffering=1) as output:
    while time.monotonic()-start<7200:
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits'],text=True).strip()
        raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True).strip()
        processes=[]
        for row in raw.splitlines():
            if not row.strip():continue
            pid,mem=row.split(',')
            try:owned=str(ROOT/'run_preloaded_case.py').encode() in Path(f'/proc/{int(pid)}/cmdline').read_bytes()
            except FileNotFoundError:owned=True
            processes.append({'pid':int(pid),'memory_mib':mem.strip(),'benchmark_owned':owned})
        output.write(json.dumps({'time_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'gpu':gpu,'processes':processes,'competing_workload':any(not p['benchmark_owned'] for p in processes)})+'\n')
        if (ROOT/'preloaded/generation-complete').exists():break
        time.sleep(5)
