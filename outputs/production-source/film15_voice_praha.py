"""Malayalam-specific 350M local CPU/GPU narration comparison."""
import os,sys,json,time,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; WORK=ROOT/'work/film15-voice'
os.environ['HF_HOME']=str(WORK/'hf-cache'); os.environ['HF_HUB_CACHE']=str(WORK/'hf-cache')
import huggingface_hub.file_download as hf_files
hf_files.are_symlinks_supported=lambda cache_dir=None:False
import torch,numpy as np,soundfile as sf
from transformers import AutoTokenizer,AutoModelForCausalLM
from snac import SNAC

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--device',default='cpu'); ap.add_argument('--script'); args=ap.parse_args()
    torch.set_num_threads(6); torch.manual_seed(4070)
    mid='Praha-Labs/LFM-MALAYALAM-TTS-v0.1'; cache=WORK/'hf-cache'
    tok=AutoTokenizer.from_pretrained(mid,cache_dir=cache)
    model=AutoModelForCausalLM.from_pretrained(mid,cache_dir=cache,torch_dtype=torch.float32 if args.device=='cpu' else torch.bfloat16,attn_implementation='sdpa').to(args.device).eval()
    codec=SNAC.from_pretrained('hubertsiuzdak/snac_24khz',cache_dir=cache).to(args.device).eval()
    scenes=json.loads(Path(args.script).read_text(encoding='utf-8'))['scenes'] if args.script else [{'id':'test-praha','narration_ml':'കടൽ കടന്നുവന്നത് കുരുമുളകിന്റെ കഥ മാത്രമല്ല. മനുഷ്യരും വിശ്വാസങ്ങളും ഇവിടെ കണ്ടുമുട്ടി. കേരളത്തിലെ ക്രിസ്തീയ പാരമ്പര്യം നമ്മുടെ നാടിന്റെ ചരിത്രത്തിന്റെ ഭാഗമാണ്.'}]
    for s in scenes:
        t=time.perf_counter(); ids=[64403]+tok(s['narration_ml']).input_ids+[7,64404]; x=torch.tensor([ids],device=args.device)
        with torch.inference_mode(): y=model.generate(x,attention_mask=torch.ones_like(x),max_new_tokens=2000,do_sample=True,temperature=.6,top_p=.95,repetition_penalty=1.1,eos_token_id=64402,pad_token_id=64407)[0,len(ids):].tolist()
        audio=[]
        for n in y:
            if n==64402: break
            if 64410<=n<93082: audio.append(n)
        audio=audio[:len(audio)//7*7]
        a=np.asarray(audio,dtype=np.int64).reshape(-1,7)-np.asarray([64410+i*4096 for i in range(7)])
        if np.any((a<0)|(a>4095)): raise RuntimeError('Invalid codec tokens')
        a=torch.tensor(a,device=args.device)
        with torch.inference_mode(): wav=codec.decode([a[:,0].reshape(1,-1),a[:,[1,4]].reshape(1,-1),a[:,[2,3,5,6]].reshape(1,-1)]).cpu().float().numpy().reshape(-1)
        p=ROOT/'outputs/film15/audio/voice'/(s['id']+'.wav'); sf.write(p,wav,24000,subtype='PCM_24')
        meta={'id':s['id'],'text':s['narration_ml'],'duration':len(wav)/24000,'file':str(p),'model':mid,'device':args.device,'render_seconds':time.perf_counter()-t,'sample_rate':24000}
        p.with_suffix('.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps({k:meta[k] for k in ['id','duration','render_seconds']}),flush=True)
if __name__=='__main__': main()
