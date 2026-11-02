"""Full-frame, frame-aligned PSNR/SSIM on delivered video streams."""
import argparse
import json
import math
from pathlib import Path
import re
import subprocess
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('--available',action='store_true')
args=parser.parse_args()

def probe(path):
    return json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(path)]))

def metrics(reference,candidate,out):
    out.mkdir(parents=True,exist_ok=True)
    saved=out/'metrics.json'
    if saved.exists():return json.loads(saved.read_text())
    rp=probe(reference);cp=probe(candidate)
    rv=next(s for s in rp['streams'] if s['codec_type']=='video')
    cv=next(s for s in cp['streams'] if s['codec_type']=='video')
    for key in ['width','height','nb_frames','r_frame_rate']:
        if rv[key]!=cv[key]:raise ValueError(f'Metadata mismatch {key}: {rv[key]} vs {cv[key]}')
    graph=(f'[0:v]settb=AVTB,setpts=PTS-STARTPTS,format=yuv420p,split=2[rp][rs];'
           f'[1:v]settb=AVTB,setpts=PTS-STARTPTS,format=yuv420p,split=2[cp][cs];'
           f'[rp][cp]psnr=stats_file={out/"psnr-frames.log"}[p];'
           f'[rs][cs]ssim=stats_file={out/"ssim-frames.log"}[s]')
    command=['ffmpeg','-nostdin','-hide_banner','-threads','2','-i',str(reference),'-threads','2','-i',str(candidate),
             '-filter_complex_threads','1','-filter_complex',graph,'-map','[p]','-map','[s]','-f','null','-']
    process=subprocess.run(command,capture_output=True,text=True,timeout=300)
    (out/'ffmpeg.log').write_text(process.stderr)
    process.check_returncode()
    psnr=re.findall(r'PSNR y:(\S+) u:(\S+) v:(\S+) average:(\S+)',process.stderr)[-1]
    ssim=re.findall(r'SSIM Y:(\S+) \([^)]*\) U:(\S+) \([^)]*\) V:(\S+) \([^)]*\) All:(\S+)',process.stderr)[-1]
    pn=sum(1 for _ in (out/'psnr-frames.log').open())
    sn=sum(1 for _ in (out/'ssim-frames.log').open())
    if pn!=int(rv['nb_frames']) or sn!=pn:raise ValueError('Not every frame scored')
    result={'psnr_db':float(psnr[3]) if psnr[3]!='inf' else None,'psnr_infinite':psnr[3]=='inf',
            'ssim':float(ssim[3]),'psnr_y_db':float(psnr[0]) if psnr[0]!='inf' else None,'ssim_y':float(ssim[0]),
            'frames':pn,'width':rv['width'],'height':rv['height'],'fps':rv['r_frame_rate'],
            'reference':str(reference.relative_to(ROOT)),'candidate':str(candidate.relative_to(ROOT)),
            'method':'FFmpeg full-frame YUV420P; identical dimensions/FPS/frame count; PTS reset; no scaling, alignment search, trimming or frame sampling',
            'command':command}
    saved.write_text(json.dumps(result,indent=2))
    return result

def contact(case,original,baseline,fp8,out):
    if out.exists():return
    frames=[]
    width=480
    height=round(width*case['message']['height']/case['message']['width'])
    if height%2:height+=1
    for t in [0,4.5,9.5,14]:
        row=[]
        for label,path in [('Original',original),('BF16 replay',baseline),('FP8',fp8)]:
            raw=subprocess.check_output(['ffmpeg','-nostdin','-v','error','-ss',str(t),'-i',str(path),'-frames:v','1','-vf',f'scale={width}:{height}',
                                         '-f','rawvideo','-pix_fmt','rgb24','-threads','1','-'])
            row.append(Image.frombytes('RGB',(width,height),raw))
        frames.append(row)
    canvas=Image.new('RGB',(width*3,(height+32)*4+42),'#0b121c')
    draw=ImageDraw.Draw(canvas)
    for i,label in enumerate(['Published original / BF16 adapter','Matched-host BF16 replay','FP8 adapter / FP8 GEMMs']):draw.text((i*width+12,12),label,fill='#e5edf8')
    for ri,row in enumerate(frames):
        y=42+ri*(height+32)
        for ci,im in enumerate(row):
            draw.text((ci*width+12,y+6),f't = {[0,4.5,9.5,14][ri]:.1f} s',fill='#aebed3')
            canvas.paste(im,(ci*width,y+32))
    canvas.save(out,quality=92)

all_results=[]
for case in json.loads((ROOT/'cases.json').read_text()):
    slug=case['slug'];bp=ROOT/'runs'/slug/'bf16/result.json';fp=ROOT/'runs'/slug/'fp8/result.json'
    if not bp.exists() or not fp.exists():
        if args.available:continue
        raise FileNotFoundError(slug)
    b=json.loads(bp.read_text());f=json.loads(fp.read_text())
    br=next((r for r in b['runs'] if r['status']=='success'),None)
    fr=next((r for r in f['runs'] if r['status']=='success'),None)
    if not br or not fr:
        if args.available:continue
        raise RuntimeError('Failed generation '+slug)
    verification=fr.get('fp8_verification',{})
    if (verification.get('adapter_instances')!=208 or verification.get('fp8_gemm_calls',0)<=0
            or verification.get('fallback_calls')!=0
            or sum(verification.get('shapes',{}).values())!=verification.get('fp8_gemm_calls')):
        raise RuntimeError('FP8 execution verification failed: '+slug)
    original=ROOT/case['files']['original']['path'];baseline=ROOT/br['output'];fp8=ROOT/fr['output']
    output=ROOT/'analysis'/slug
    # Keep generation and score computation separate when running the full report.
    quality={name:metrics(r,c,output/name) for name,r,c in [('original_vs_fp8',original,fp8),('bf16_vs_fp8',baseline,fp8),('original_vs_bf16',original,baseline)]}
    if len(b['runs'])>1:
        quality['bf16_repeatability']=metrics(baseline,ROOT/b['runs'][1]['output'],output/'bf16_repeatability')
    if len(f['runs'])>1:
        quality['fp8_repeatability']=metrics(fp8,ROOT/f['runs'][1]['output'],output/'fp8_repeatability')
    contact(case,original,baseline,fp8,output/'frames.jpg')
    # Decode complete delivered files; do not rely on metadata alone.
    for label,path in [('original',original),('bf16',baseline),('fp8',fp8)]:
        completed=subprocess.run(['ffmpeg','-nostdin','-v','error','-threads','2','-i',str(path),'-f','null','-'],capture_output=True,text=True,timeout=180)
        completed.check_returncode()
        (output/f'{label}-decode.log').write_text(completed.stderr)
    row={'name':case['name'],'slug':slug,'source_uid':case['uid'],'source_host':case['source_host'],
         'source_seconds':case['source_timings']['comfy_execution_seconds'],'source_end_to_end_seconds':case['source_timings']['end_to_end_seconds'],
         'bf16_seconds':br['execution_seconds'],'fp8_seconds':fr['execution_seconds'],
         'historical_speedup':case['source_timings']['comfy_execution_seconds']/fr['execution_seconds'],
         'matched_host_speedup':br['execution_seconds']/fr['execution_seconds'],
         'matched_host_time_reduction_percent':100*(br['execution_seconds']-fr['execution_seconds'])/br['execution_seconds'],
         'quality':quality,'fp8_verification':fr['fp8_verification'],'baseline':b,'fp8':f}
    all_results.append(row)
    print(json.dumps({k:v for k,v in row.items() if k not in {'quality','baseline','fp8','fp8_verification'}}),flush=True)
(ROOT/'analysis/summary.json').write_text(json.dumps(all_results,indent=2))
