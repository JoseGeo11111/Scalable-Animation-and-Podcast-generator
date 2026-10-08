from pathlib import Path
import json,shutil,numpy as np,soundfile as sf,subprocess
ROOT=Path(__file__).resolve().parents[2]; WORK=ROOT/'work/film15-sound'; OUT=ROOT/'outputs/film15/audio'; SRC=ROOT/'outputs/production-source'
records={}
for kind in ['sfx','score']:
 rows=[]
 for p in sorted((OUT/kind).glob('*.json')):
  row=json.loads(p.read_text()); y,sr=sf.read(row['path'],dtype='float32',always_2d=True)
  row.update(channels=y.shape[1],finite=bool(np.isfinite(y).all()),rms_dbfs=float(20*np.log10(max(1e-12,np.sqrt(np.mean(y*y))))),samples=len(y))
  assert row['finite'] and len(y)>0 and sr==48000
  rows.append(row)
 records[kind]=rows
preview=[]; index=[];t=0
for row in records['score']:
 y,sr=sf.read(row['path'],dtype='float32',always_2d=True);x=y[20*sr:30*sr].copy()
 gain=min(10**(-21/20)/max(1e-9,np.sqrt(np.mean(x*x))),.85/max(1e-9,np.max(np.abs(x))));x*=gain
 f=round(.25*sr);x[:f]*=np.linspace(0,1,f)[:,None];x[-f:]*=np.linspace(1,0,f)[:,None]
 preview.extend([x,np.zeros((sr//2,2),np.float32)]);index.append({'id':row['id'],'start':t,'source_start':20});t+=10.5
sf.write(OUT/'score-audition.wav',np.concatenate(preview),48000,subtype='PCM_24')
subprocess.run(['ffmpeg','-y','-hide_banner','-loglevel','error','-i',str(OUT/'score-audition.wav'),'-c:a','libmp3lame','-b:a','256k',str(OUT/'score-audition.mp3')],check=True)
(OUT/'score-audition-index.json').write_text(json.dumps(index,indent=2))
summary={k:{'count':len(v),'audio_seconds':sum(x.get('actual_duration',x.get('duration',0)) for x in v),'generation_seconds':sum(x['generation_seconds'] for x in v),'max_allocated_vram_gib':max(x['max_allocated_vram_bytes'] for x in v)/2**30} for k,v in records.items()}
report={'summary':summary,'assets':records,'benchmark_scope':'Generation timers exclude process startup, model loading, installation and download. Selected-writing retake replaces near-silent first candidate. No cloud sound or music generation. Mono effects, stereo score. Technical signal checks passed; no claim of human listening or native instrument authentication.','rejected':{'writing_quill_initial':'RMS -63.98dBFS; preserved in work/film15-sound/quiet-writing-first.*'}}
(OUT/'generation-report.json').write_text(json.dumps(report,indent=2))
for name in ['generate_sfx.py','generate_score.py','sfx_prompts.json','score_prompts.json','mix_film_audio.py','preview_sfx.py','package_audio.py']:
 shutil.copyfile(WORK/name,SRC/('film15-'+name))
# Copies of generation scripts resolve ROOT identically from production-source.
# Their prompt paths still reference their work install, documented and included as separate files.
print(json.dumps(summary,indent=2))
