"""CPU-only independent ASR of final mixed narration excerpts."""
import os,json,unicodedata,time,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; CACHE=ROOT/'work/film15-voice/hf-cache'
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['HF_HUB_OFFLINE']='1'; os.environ['HF_HUB_CACHE']=str(CACHE)
os.environ['OMP_NUM_THREADS']='3'; os.environ['MKL_NUM_THREADS']='3'
import torch,numpy as np,soundfile as sf,jiwer
from scipy.signal import resample_poly
from transformers import WhisperProcessor,WhisperForConditionalGeneration
torch.set_num_threads(3); torch.set_num_interop_threads(1)
timeline=json.loads((ROOT/'outputs/production-source/film15-timeline.json').read_text(encoding='utf-8'))
OUT=ROOT/'outputs/film15/audio'; VOICE=OUT/'voice'
ap=argparse.ArgumentParser();ap.add_argument('--source',default='mix.wav');ap.add_argument('--report',default='mix-asr-spotcheck.json');ap.add_argument('--ids',default='03,11,30,45,60');ap.add_argument('--whole-scene',action='store_true');args=ap.parse_args()
ids=set(args.ids.split(','))
p=WhisperProcessor.from_pretrained('openai/whisper-small',cache_dir=CACHE,local_files_only=True)
m=WhisperForConditionalGeneration.from_pretrained('adalat-ai/whisper-small-ml-rmft',cache_dir=CACHE,local_files_only=True,torch_dtype=torch.float32).to('cpu').eval()
def norm(x):return ''.join(c for c in unicodedata.normalize('NFC',x) if unicodedata.category(c)[0] in 'LMN' or c.isspace())
report={'device':'cpu','torch_threads':3,'inter_op_threads':1,'model':'adalat-ai/whisper-small-ml-rmft','source':'outputs/film15/audio/mix.wav','tempo':timeline['voice_tempo'],'independent_without_script_prompt':True,'scenes':[],'note':'ASR is not native listening certification. Compare meaningful words, dates and negations; CER also reflects spacing, names, tempo and segment boundary changes.'}
report['source']=str(OUT/args.source)
with sf.SoundFile(OUT/args.source) as mix:
    sr=mix.samplerate
    for s in timeline['scenes']:
        if s['id'] not in ids:continue
        started=time.perf_counter()
        meta=json.loads((VOICE/(s['id']+'.json')).read_text(encoding='utf-8'))
        raw=json.loads((VOICE/(s['id']+'.ml-asr.json')).read_text(encoding='utf-8'))
        spans=[(u['start'],u['start']+u['duration']) for u in meta.get('utterances',[])]
        if not spans:spans=[(st,min(meta['duration'],st+6)) for st in np.arange(0,meta['duration'],6)]
        if args.whole_scene:spans=[(0,meta['duration'])]
        pieces=[]
        for start,end in spans:
            a=s['voice_start']+start/s['tempo']; b=min(s['voice_end'],s['voice_start']+end/s['tempo'])
            mix.seek(round(a*sr)); samples=mix.read(round((b-a)*sr),dtype='float32',always_2d=True).mean(axis=1)
            samples=resample_poly(samples,16000,sr)
            features=p(samples,sampling_rate=16000,return_tensors='pt').input_features
            with torch.inference_mode():out=m.generate(features,language='ml',task='transcribe',max_new_tokens=400)
            text=p.batch_decode(out,skip_special_tokens=True)[0]
            pieces.append({'mix_start':a,'mix_end':b,'recognized':text})
        recognized=' '.join(x['recognized'] for x in pieces)
        record={'id':s['id'],'voice_start':s['voice_start'],'voice_end':s['voice_end'],'expected':s['narration_ml'],'mixed_recognized':recognized,'mixed_cer':jiwer.cer(norm(s['narration_ml']),norm(recognized)),'raw_recognized':raw['recognized'],'raw_cer':raw['cer'],'segments':pieces,'cpu_wall_seconds':time.perf_counter()-started}
        report['scenes'].append(record)
        (OUT/args.report).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps({'id':s['id'],'raw_cer':raw['cer'],'mixed_cer':record['mixed_cer'],'cpu_wall_seconds':record['cpu_wall_seconds']}),flush=True)
print('Complete',flush=True)
