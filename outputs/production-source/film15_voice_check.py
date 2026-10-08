"""Local ASR audit. Recognition disagreement flags review; it is not human QA."""
import os, json, sys, argparse, re, unicodedata
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
WORK=ROOT/'work/film15-voice'
os.environ['HF_HUB_CACHE']=str(WORK/'hf-cache')
import torch
os.add_dll_directory(str(Path(torch.__file__).parent/'lib'))
import huggingface_hub.file_download as hf_files
hf_files.are_symlinks_supported=lambda cache_dir=None:False
from faster_whisper import WhisperModel
from huggingface_hub import snapshot_download
import jiwer

def norm(x): return ''.join(c for c in unicodedata.normalize('NFC',x.lower()) if unicodedata.category(c)[0] in 'LMN' or c.isspace()).strip()
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('files',nargs='+'); ap.add_argument('--device',default='cpu'); args=ap.parse_args()
    d=snapshot_download('mobiuslabsgmbh/faster-whisper-large-v3-turbo',cache_dir=WORK/'hf-cache',local_files_only=True)
    m=WhisperModel(d,device=args.device,compute_type='int8' if args.device=='cpu' else 'float16',cpu_threads=6)
    results=[]
    for fn in args.files:
        p=Path(fn); meta=json.loads(p.with_suffix('.json').read_text(encoding='utf-8'))
        segments,info=m.transcribe(str(p),language='ml',beam_size=5,vad_filter=True,condition_on_previous_text=False,chunk_length=6,max_new_tokens=400)
        seg=list(segments); got=' '.join(x.text.strip() for x in seg)
        data={'file':str(p),'expected':meta['text'],'recognized':got,'cer':jiwer.cer(norm(meta['text']),norm(got)),'wer':jiwer.wer(norm(meta['text']),norm(got)),'segments':[{'start':x.start,'end':x.end,'text':x.text,'avg_logprob':x.avg_logprob} for x in seg],'note':'ASR output is fallible. Not native-speaker listening approval.'}
        p.with_suffix('.asr.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8'); results.append(data)
        print(json.dumps(data,ensure_ascii=True),flush=True)
    (ROOT/'outputs/film15/audio/voice/asr-review.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
if __name__=='__main__': main()
