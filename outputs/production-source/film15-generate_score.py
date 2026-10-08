from pathlib import Path
import os,sys,time,json,argparse,shutil
ROOT=Path(__file__).resolve().parents[2]
REPO=ROOT/'work/film15-sound/ACE-Step-1.5'
os.environ['HF_HOME']=str(ROOT/'work/hf-cache')
os.environ['ACESTEP_CHECKPOINTS_DIR']=str(REPO/'checkpoints')
os.environ['TOKENIZERS_PARALLELISM']='false'
sys.path.insert(0,str(REPO)); os.chdir(REPO)
import torch, soundfile as sf, numpy as np
import acestep.model_downloader as downloader
# DiT-only inference needs these three components. No language-model generation is used.
downloader.MAIN_MODEL_COMPONENTS=['acestep-v15-turbo','vae','Qwen3-Embedding-0.6B']
from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import GenerationParams,GenerationConfig,generate_music
ap=argparse.ArgumentParser();ap.add_argument('--only');args=ap.parse_args()
OUT=ROOT/'outputs/film15/audio/score';OUT.mkdir(parents=True,exist_ok=True)
dit=AceStepHandler(); llm=LLMHandler()
status,ready=dit.initialize_service(project_root=str(REPO),config_path='acestep-v15-turbo',device='cuda',use_flash_attention=False,compile_model=False,offload_to_cpu=True,offload_dit_to_cpu=False,quantization=None)
print(status,flush=True)
if not ready: raise RuntimeError(status)
prompt_file=ROOT/'outputs/production-source/film15-score_prompts.json'
if not prompt_file.exists():prompt_file=ROOT/'work/film15-sound/score_prompts.json'
jobs=json.loads(prompt_file.read_text())
for i,j in enumerate(jobs):
 if args.only and j['id']!=args.only:continue
 dest=OUT/(j['id']+'.wav')
 if dest.exists():print('SKIP',dest,flush=True);continue
 params=GenerationParams(caption=j['caption'],lyrics='[Instrumental]',instrumental=True,bpm=j['bpm'],duration=j['duration'],keyscale=j['keyscale'],timesignature='4',thinking=False,use_cot_metas=False,use_cot_caption=False,use_cot_language=False,inference_steps=8,seed=62000+i)
 config=GenerationConfig(batch_size=1,allow_lm_batch=False,use_random_seed=False,audio_format='wav32')
 t=time.perf_counter();torch.cuda.reset_peak_memory_stats()
 result=generate_music(dit,llm,params,config,save_dir=str(OUT/'raw'))
 if not result.success:raise RuntimeError(result.error)
 shutil.copyfile(result.audios[0]['path'],dest)
 y,sr=sf.read(dest)
 record={**j,'model':'ACE-Step/Ace-Step1.5','license':'MIT','mode':'DiT-only, no LM','path':str(dest),'seed':62000+i,'generation_seconds':time.perf_counter()-t,'sample_rate':sr,'actual_duration':len(y)/sr,'peak_dbfs':float(20*np.log10(max(1e-12,np.max(np.abs(y))))),'max_allocated_vram_bytes':torch.cuda.max_memory_allocated()}
 (OUT/(j['id']+'.json')).write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)
print('COMPLETE',flush=True)
