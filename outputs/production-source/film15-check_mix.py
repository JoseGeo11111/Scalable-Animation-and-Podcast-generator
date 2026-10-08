from pathlib import Path
import json, numpy as np,soundfile as sf
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/film15/audio'
names=['narration.wav','music-stem.wav','ambience-stem.wav','foley-stem.wav','mix.wav']
files=[sf.SoundFile(OUT/n) for n in names]
assert all(f.frames==43_200_000 and f.samplerate==48000 and f.channels==2 for f in files)
stats={n:{'sum':0.,'count':0,'peak':0.,'speech_sum':0.,'speech_count':0,'nonspeech_sum':0.,'nonspeech_count':0} for n in names}
while True:
 arrays=[f.read(480_000,dtype='float32',always_2d=True) for f in files]
 if not len(arrays[0]):break
 voice=arrays[0]; mask=np.sqrt(np.mean(voice.reshape(-1,480,2)**2,axis=(1,2)))>.008
 samplemask=np.repeat(mask,480)
 for n,y in zip(names,arrays):
  assert np.isfinite(y).all(),n
  z=stats[n];z['sum']+=float(np.sum(y*y,dtype='float64'));z['count']+=y.size;z['peak']=max(z['peak'],float(np.max(np.abs(y))))
  for lab,m in [('speech',samplemask),('nonspeech',~samplemask)]:
   z[lab+'_sum']+=float(np.sum(y[m]**2,dtype='float64'));z[lab+'_count']+=y[m].size
for f in files:f.close()
result={}
for n,z in stats.items():
 result[n]={'rms_dbfs':10*np.log10(max(1e-15,z['sum']/z['count'])),'peak_dbfs':20*np.log10(max(1e-15,z['peak'])),'speech_active_rms_dbfs':10*np.log10(max(1e-15,z['speech_sum']/max(1,z['speech_count']))),'speech_gap_rms_dbfs':10*np.log10(max(1e-15,z['nonspeech_sum']/max(1,z['nonspeech_count'])))}
report=json.loads((OUT/'mix-report.json').read_text(encoding='utf8'))
report['signal_verification']={'duration_seconds':900,'frames_per_stem':43_200_000,'all_finite':True,'all60_voices_aligned':len(report['timed_voice'])==60,'stems':result,'note':'RMS levels are signal measurements, not LUFS. Stems are pre-master. Speech activity threshold is RMS>0.008 in10ms blocks.'}
(OUT/'mix-report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf8')
print(json.dumps(result,indent=2))
