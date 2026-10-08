"""Project-local Svara narration. Preset voice only; no voice cloning.
Uses official Svara token framing and SNAC hierarchical code ordering.
"""
import os, sys, json, time, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT/'work/film15-voice'
os.environ['HF_HOME'] = str(WORK/'hf-cache')
os.environ['HF_HUB_CACHE'] = str(WORK/'hf-cache')
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
import numpy as np
import torch
import soundfile as sf
import huggingface_hub.file_download as hf_files
hf_files.are_symlinks_supported = lambda cache_dir=None: False
from transformers import AutoTokenizer, AutoModelForCausalLM
from snac import SNAC

def prompt_ids(tok, text, voice):
    # The publisher's v1 Space uses a different framing from its newer API repo.
    # Follow the v1 checkpoint's demonstrated framing exactly.
    ids = tok(f'{voice}: {text}', add_special_tokens=True).input_ids
    return [128259] + ids + [128009,128260]

def decode(ids, codec):
    seq=[]
    for t in ids:
        if t in (128258,128262,128009): break
        if 128266 <= t < 156938: seq.append(t)
    seq=seq[:len(seq)//7*7]
    if len(seq)<7: raise RuntimeError('No complete audio frame')
    a=np.asarray(seq,dtype=np.int64).reshape(-1,7)-np.asarray([128266+i*4096 for i in range(7)])
    invalid=int(np.sum((a<0)|(a>4095)))
    if invalid: raise RuntimeError(f'Invalid audio token sequence: {invalid}')
    a=torch.tensor(a,device='cuda')
    codes=[a[:,0].reshape(1,-1),a[:,[1,4]].reshape(1,-1),a[:,[2,3,5,6]].reshape(1,-1)]
    with torch.inference_mode(): x=codec.decode(codes).float().cpu().numpy().reshape(-1)
    # Conservative trim only below -50dB; preserve 100ms breathing room.
    active=np.flatnonzero(np.abs(x)>0.0032)
    if active.size: x=x[max(0,active[0]-2400):min(len(x),active[-1]+2400)]
    x=np.nan_to_num(x)
    return x, len(seq)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--script'); ap.add_argument('--test',action='store_true'); ap.add_argument('--voice',default='Malayalam (Male)'); ap.add_argument('--batch',type=int,default=3); ap.add_argument('--limit',type=int); ap.add_argument('--seed',type=int,default=4070); args=ap.parse_args()
    torch.set_num_threads(6); torch.manual_seed(args.seed)
    (WORK/'narration.pid').write_text(str(os.getpid()))
    out=ROOT/'outputs/film15/audio/voice'; out.mkdir(parents=True,exist_ok=True)
    if args.test:
        scenes=[{'id':'test-male-v1' if 'Male' in args.voice else 'test-female-v1','narration_ml':'കടൽ കടന്നുവന്നത് കുരുമുളകിന്റെ കഥ മാത്രമല്ല. മനുഷ്യരും വിശ്വാസങ്ങളും ഇവിടെ കണ്ടുമുട്ടി. കേരളത്തിലെ ക്രിസ്തീയ പാരമ്പര്യം നമ്മുടെ നാടിന്റെ ചരിത്രത്തിന്റെ ഭാഗമാണ്.'}]
    else: scenes=json.loads(Path(args.script).read_text(encoding='utf-8'))['scenes']
    if args.limit: scenes=scenes[:args.limit]
    scenes=[s for s in scenes if not (out/(str(s['id'])+'.wav')).exists()]
    if not scenes: print('No pending narration scenes'); return
    t0=time.perf_counter(); print('Loading Svara BF16 on CUDA',flush=True)
    tok=AutoTokenizer.from_pretrained('kenpath/svara-tts-v1',cache_dir=WORK/'hf-cache')
    model=AutoModelForCausalLM.from_pretrained('kenpath/svara-tts-v1',cache_dir=WORK/'hf-cache',torch_dtype=torch.bfloat16,attn_implementation='sdpa',device_map='cuda').eval()
    codec=SNAC.from_pretrained('hubertsiuzdak/snac_24khz',cache_dir=WORK/'hf-cache').eval().to('cuda')
    print('Models ready',round(time.perf_counter()-t0,2),flush=True)
    manifest=[]
    for offset in range(0,len(scenes),args.batch):
        if (WORK/'stop-after-batch').exists():
            print('Stopped at requested batch boundary',flush=True); break
        group=scenes[offset:offset+args.batch]; ids=[prompt_ids(tok,s['narration_ml'],args.voice) for s in group]; width=max(map(len,ids))
        batch=torch.tensor([[128263]*(width-len(i))+i for i in ids],device='cuda'); mask=batch.ne(128263)
        t=time.perf_counter()
        with torch.inference_mode():
            outputs=model.generate(input_ids=batch,attention_mask=mask,max_new_tokens=2600,do_sample=True,temperature=.65,top_p=.9,top_k=50,repetition_penalty=1.1,eos_token_id=[128258,128262],pad_token_id=128263,use_cache=True)
        elapsed=time.perf_counter()-t
        for s,row in zip(group,outputs):
            x,n=decode(row[width:].tolist(),codec); fn=out/(str(s['id'])+'.wav'); sf.write(fn,x,24000,subtype='PCM_24')
            generated=row[width:].tolist()
            data={'id':s['id'],'file':str(fn),'duration':len(x)/24000,'sample_rate':24000,'tokens':n,'model':'kenpath/svara-tts-v1','voice':args.voice,'seed':args.seed,'batch_render_seconds':elapsed,'text':s['narration_ml'],'peak':float(np.max(np.abs(x))),'rms':float(np.sqrt(np.mean(x*x))),'capped':not any(z in (128258,128262) for z in generated),'local':True,'v1_prompt_framing':True}
            fn.with_suffix('.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8'); manifest.append(data)
            print(json.dumps({k:data[k] for k in ['id','duration','tokens','batch_render_seconds','capped']}),flush=True)
        (out/'latest-batch.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Complete',round(time.perf_counter()-t0,2),flush=True)
    if not args.test:
        requested=json.loads(Path(args.script).read_text(encoding='utf-8'))['scenes']
        records=[]
        for s in requested:
            p=out/(str(s['id'])+'.json')
            if p.exists():
                record=json.loads(p.read_text(encoding='utf-8'))
                if record.get('voice')==args.voice and record.get('v1_prompt_framing') and record.get('text')==s['narration_ml']: records.append(record)
        (out/'durations.json').write_text(json.dumps({'scenes':records,'total_duration':sum(r['duration'] for r in records),'complete':len(records)==len(requested),'model':'kenpath/svara-tts-v1','voice':args.voice},ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__': main()
