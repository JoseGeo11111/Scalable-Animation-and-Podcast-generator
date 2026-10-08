"""Independent Malayalam-finetuned ASR check on local narration, CPU or CUDA."""
import os,json,sys,unicodedata,time,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; CACHE=ROOT/'work/film15-voice/hf-cache'
os.environ['HF_HUB_CACHE']=str(CACHE)
import torch,soundfile as sf,numpy as np,jiwer
from scipy.signal import resample_poly
from transformers import WhisperProcessor,WhisperForConditionalGeneration
import huggingface_hub.file_download as hf_files
hf_files.are_symlinks_supported=lambda cache_dir=None:False
torch.set_num_threads(6)
ap=argparse.ArgumentParser(); ap.add_argument('--watch',action='store_true'); ap.add_argument('--device',default='cpu'); ap.add_argument('--resume',action='store_true'); ap.add_argument('files',nargs='*'); args=ap.parse_args()
p=WhisperProcessor.from_pretrained('openai/whisper-small',cache_dir=CACHE)
m=WhisperForConditionalGeneration.from_pretrained('adalat-ai/whisper-small-ml-rmft',cache_dir=CACHE,torch_dtype=torch.float16 if args.device=='cuda' else torch.float32).to(args.device).eval()
def norm(x):return ''.join(c for c in unicodedata.normalize('NFC',x) if unicodedata.category(c)[0] in 'LMN' or c.isspace())
def files_to_check():
    if not args.watch:
        yield from args.files; return
    scenes=json.loads((ROOT/'outputs/production-source/film15-script.json').read_text(encoding='utf-8'))['scenes']
    for scene in scenes:
        f=ROOT/'outputs/film15/audio/voice'/(scene['id']+'.wav')
        while not f.exists() or not f.with_suffix('.json').exists(): time.sleep(5)
        yield str(f)
results=[]
for fn in files_to_check():
    if args.resume and Path(fn).with_suffix('.ml-asr.json').exists():
        results.append(json.loads(Path(fn).with_suffix('.ml-asr.json').read_text(encoding='utf-8'))); continue
    f=Path(fn); x,sr=sf.read(f,dtype='float32'); x=resample_poly(x,16000,sr); texts=[]
    metadata=json.loads(f.with_suffix('.json').read_text(encoding='utf-8'))
    # Retakes supply actual sentence boundaries; avoid cutting a historical name
    # between arbitrary windows when those independently generated boundaries exist.
    spans=[(round(u['start']*16000),round((u['start']+u['duration'])*16000)) for u in metadata.get('utterances',[])]
    if not spans: spans=[(start,min(len(x),start+6*16000)) for start in range(0,len(x),6*16000)]
    for start,end in spans:
        features=p(x[start:end],sampling_rate=16000,return_tensors='pt').input_features.to(device=args.device,dtype=m.dtype)
        with torch.inference_mode(): ids=m.generate(features,language='ml',task='transcribe',max_new_tokens=400)
        texts.append(p.batch_decode(ids,skip_special_tokens=True)[0])
    got=' '.join(texts); expected=metadata['text']
    data={'file':str(f),'model':'adalat-ai/whisper-small-ml-rmft','expected':expected,'recognized':got,'cer':jiwer.cer(norm(expected),norm(got)),'note':'Automatic recognition only; no native listening certification.'}
    f.with_suffix('.ml-asr.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps({'file':str(f),'cer':data['cer']}),flush=True)
    results.append(data)
    (ROOT/'outputs/film15/audio/voice/malayalam-asr-audit.json').write_text(json.dumps({'count':len(results),'scenes':results,'flagged':[r['file'] for r in results if r['cer']>.15]},ensure_ascii=False,indent=2),encoding='utf-8')
