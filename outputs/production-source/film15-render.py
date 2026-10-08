"""Resumable local Wan2.2 FastWan production queue. GPU jobs run serially."""
import os, sys, json, time, threading, subprocess, shutil, argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
REPO=ROOT/'work/Wan2GP'
OUT=ROOT/'outputs/film15/native'
WORK=ROOT/'work/film15'
OUT.mkdir(parents=True,exist_ok=True);WORK.mkdir(parents=True,exist_ok=True)
parser=argparse.ArgumentParser()
parser.add_argument('queue',type=Path)
parser.add_argument('--limit',type=int,default=1000)
parser.add_argument('--vae-tile',type=int,default=512)
parser.add_argument('--sampler',choices=['unipc','dmd'],default='unipc')
args=parser.parse_args()
queue=json.loads(args.queue.resolve().read_text(encoding='utf-8-sig'))
os.environ.update(HF_HOME=str(ROOT/'work/hf-cache'),HF_HUB_DISABLE_TELEMETRY='1',HF_HUB_ETAG_TIMEOUT='60',HF_HUB_DOWNLOAD_TIMEOUT='120',TRITON_CACHE_DIR=str(ROOT/'work/triton-cache'),PYTHONUTF8='1')
sys.path.insert(0,str(REPO));os.chdir(REPO)
from shared.api import init
session=init(root=REPO,output_dir=OUT,cli_args=['--attention','sage2','--profile','3','--perc-reserved-mem-max','0.25','--vram-allocator','vmm'],console_isatty=False)
from models.wan.modules.vae2_2 import Wan2_2_VAE
Wan2_2_VAE.get_VAE_tile_size=staticmethod(lambda *a,**k:args.vae_tile)
if args.sampler=='dmd':
    sys.path.insert(0,str(ROOT/'outputs/production-source'))
    import film15_dmd_scheduler
    import models.wan.any2video as wan_module
    film15_dmd_scheduler.install(wan_module)
done=0
for shot in queue:
    reportfile=WORK/(shot['id']+'-report.json')
    if reportfile.exists():
        previous=json.loads(reportfile.read_text())
        if previous.get('success') and previous.get('sampler','unipc')==args.sampler:
            print('SKIP',shot['id'],flush=True);continue
        archive=WORK/'old-reports';archive.mkdir(exist_ok=True)
        shutil.copy2(reportfile,archive/(reportfile.stem+'-'+str(int(time.time()))+'.json'))
    if (WORK/'pause-video').exists():
        print('PAUSED BETWEEN SHOTS',flush=True);break
    if done>=args.limit:break
    if not Path(shot['image']).exists():
        print('MISSING KEYFRAME',shot['id'],flush=True);continue
    settings={
      'model_type':'ti2v_2_2_fastwan','prompt':shot['prompt'],
      'negative_prompt':'frozen still image, static slideshow, photorealism, 3D, camera spinning, fast chaotic movement, distorted face, extra limbs, extra fingers, changing character identity, flicker, blurry, captions, text, logo, watermark',
      'image_prompt_type':'S','image_start':shot['image'],'image_mode':0,
      'resolution':'1280x704','video_length':shot.get('frames',241),'force_fps':'24',
      'num_inference_steps':3,'guidance_scale':1,'flow_shift':3,'seed':shot.get('seed',41071),
      'sample_solver':'euler' if args.sampler=='dmd' else 'unipc',
      'prompt_enhancer':'','repeat_generation':1,'output_filename':shot['id']
    }
    start=time.time();telemetry=[];stop=threading.Event()
    def monitor():
        while not stop.is_set():
            try:
                r=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu,power.draw,temperature.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,creationflags=0x08000000)
                telemetry.append({'elapsed':round(time.time()-start,2),'gpu':r.stdout.strip()})
            except Exception:pass
            stop.wait(5)
    threading.Thread(target=monitor,daemon=True).start()
    print('SHOT_START',shot['id'],flush=True)
    job=session.submit_task(settings)
    last=0
    for event in job.events.iter(timeout=.5):
        if event.kind=='progress' and time.time()-last>8:
            p=event.data;print('PROGRESS',shot['id'],p.phase,p.current_step,p.total_steps,flush=True);last=time.time()
    result=job.result();stop.set()
    report={'id':shot['id'],'success':result.success,'seconds':round(time.time()-start,2),'vae_tile':args.vae_tile,'sampler':args.sampler,'files':[str(p) for p in result.generated_files],'errors':[str(e.message) for e in result.errors],'settings':settings,'telemetry':telemetry}
    reportfile.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('SHOT_RESULT',json.dumps({k:v for k,v in report.items() if k not in ['settings','telemetry']}),flush=True)
    if not result.success:sys.exit(2)
    done+=1
print('BATCH_COMPLETE',done,flush=True)
