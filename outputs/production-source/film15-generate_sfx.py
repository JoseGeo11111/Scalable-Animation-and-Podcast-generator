from pathlib import Path
import os, sys, time, json, argparse
ROOT=Path(__file__).resolve().parents[2]
os.environ['HF_HOME']=str(ROOT/'work/hf-cache')
os.environ['TORCHDYNAMO_DISABLE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
sys.path.insert(0,str(ROOT/'work/film15-sound/MOSS-TTS'))
import torch
import soundfile as sf
import numpy as np
from moss_soundeffect_v2 import MossSoundEffectPipeline
ap=argparse.ArgumentParser(); ap.add_argument('--only'); ap.add_argument('--steps',type=int,default=100); ap.add_argument('--force',action='store_true'); ap.add_argument('--seed-offset',type=int,default=0); args=ap.parse_args()
OUT=ROOT/'outputs/film15/audio/sfx'; OUT.mkdir(parents=True,exist_ok=True)
prompt_file=ROOT/'outputs/production-source/film15-sfx_prompts.json'
if not prompt_file.exists():prompt_file=Path(__file__).parent/'sfx_prompts.json'
jobs=json.loads(prompt_file.read_text())
pipe=MossSoundEffectPipeline.from_pretrained(str(ROOT/'work/film15-sound/models/moss-sfx'),torch_dtype=torch.bfloat16,device='cuda')
records=[]
for i,j in enumerate(jobs):
 if args.only and j['id']!=args.only: continue
 dest=OUT/(j['id']+'.wav')
 if dest.exists() and not args.force: print('SKIP',dest,flush=True); continue
 t=time.perf_counter(); torch.cuda.reset_peak_memory_stats()
 audio=pipe(prompt=j['prompt'],seconds=j['seconds'],num_inference_steps=args.steps,cfg_scale=4.,sigma_shift=5.,seed=61700+i+args.seed_offset)
 # Native WAV writing avoids torchaudio's optional FFmpeg/TorchCodec DLL chain on Windows.
 wav=audio[0].float().cpu().numpy().T
 sf.write(dest,wav,pipe.sample_rate,subtype='PCM_24')
 y,sr=sf.read(dest)
 record={**j,'path':str(dest),'model':'OpenMOSS-Team/MOSS-SoundEffect-v2.0','license':'Apache-2.0','steps':args.steps,'seed':61700+i+args.seed_offset,'generation_seconds':time.perf_counter()-t,'sample_rate':sr,'duration':len(y)/sr,'peak_dbfs':float(20*np.log10(max(1e-12,np.max(np.abs(y))))),'rms_dbfs':float(20*np.log10(max(1e-12,np.sqrt(np.mean(y*y))))),'max_allocated_vram_bytes':torch.cuda.max_memory_allocated()}
 (OUT/(j['id']+'.json')).write_text(json.dumps(record,indent=2))
 print(json.dumps(record),flush=True); records.append(record)
print('COMPLETE',len(records),flush=True)
