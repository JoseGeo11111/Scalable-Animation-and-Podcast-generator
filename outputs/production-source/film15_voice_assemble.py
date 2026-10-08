"""Prepare short narration utterances, or join only genuine synthesized speech."""
import json,re,argparse
from pathlib import Path
import numpy as np,soundfile as sf
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/film15/audio/voice'; SRC=ROOT/'outputs/production-source'

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--prepare',action='store_true'); args=ap.parse_args()
    film=json.loads((SRC/'film15-script.json').read_text(encoding='utf-8'))
    chunks=[]
    for s in film['scenes']:
        sentences=[x.strip() for x in re.split(r'(?<=[.!?।])\s+',s['narration_ml']) if x.strip()]
        for n,text in enumerate(sentences): chunks.append({'id':f"{s['id']}_u{n+1:02}",'scene_id':s['id'],'narration_ml':text})
    if args.prepare:
        (SRC/'film15-voice-utterances.json').write_text(json.dumps({'scenes':chunks},ensure_ascii=False,indent=2),encoding='utf-8'); print('Utterances',len(chunks)); return
    manifest=[]
    for s in film['scenes']:
        parts=[c for c in chunks if c['scene_id']==s['id']]; waves=[]; timing=[]; t=0.
        for i,c in enumerate(parts):
            p=OUT/(c['id']+'.wav'); x,sr=sf.read(p,dtype='float32'); assert sr==24000 and x.ndim==1
            meta=json.loads(p.with_suffix('.json').read_text(encoding='utf-8'))
            timing.append({'id':c['id'],'start':t,'end':t+len(x)/sr,'text':c['narration_ml']}); waves.append(x); t+=len(x)/sr
            if i<len(parts)-1: waves.append(np.zeros(3600,dtype=np.float32)); t+=.15
        y=np.concatenate(waves); p=OUT/(s['id']+'.wav'); sf.write(p,y,24000,subtype='PCM_24')
        d={'id':s['id'],'file':str(p),'duration':len(y)/24000,'sample_rate':24000,'model':meta['model'],'voice':meta.get('voice','Default model voice'),'text':s['narration_ml'],'utterances':timing,'added_sentence_pause_seconds':.15,'local':True,'peak':float(np.max(np.abs(y)))}
        p.with_suffix('.json').write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8'); manifest.append(d)
    result={'total_speech_seconds':sum(x['duration'] for x in manifest),'scenes':manifest}
    (OUT/'narration-manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8'); print('Scenes',len(manifest),'Seconds',result['total_speech_seconds'])
if __name__=='__main__': main()
