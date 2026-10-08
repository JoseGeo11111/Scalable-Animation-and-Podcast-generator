from pathlib import Path
import json,numpy as np,soundfile as sf,subprocess
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'outputs/film15/audio';SR=48000
clips=[];index=[];t=0
for p in sorted((OUT/'sfx').glob('*.wav')):
 y,sr=sf.read(p,dtype='float32',always_2d=True);assert sr==SR
 n=min(len(y),3*SR)
 candidates=list(range(0,max(1,len(y)-n),SR//2)) or [0]
 start=max(candidates,key=lambda i:float(np.mean(y[i:i+n]**2)))
 x=y[start:start+n].copy();gain=min(10**(-22/20)/max(1e-9,np.sqrt(np.mean(x*x))),.79/max(1e-9,np.max(np.abs(x))))
 x*=gain;f=min(2400,len(x)//4);x[:f]*=np.linspace(0,1,f)[:,None];x[-f:]*=np.linspace(1,0,f)[:,None]
 clips += [x,np.zeros((round(.4*SR),y.shape[1]),np.float32)]
 index.append({'asset':p.name,'preview_start':t,'source_start':start/SR,'duration':n/SR});t+=n/SR+.4
sf.write(OUT/'sfx-audition.wav',np.concatenate(clips),SR,subtype='PCM_24')
(OUT/'sfx-audition-index.json').write_text(json.dumps(index,indent=2))
subprocess.run(['ffmpeg','-y','-hide_banner','-loglevel','error','-i',str(OUT/'sfx-audition.wav'),'-c:a','libmp3lame','-b:a','192k',str(OUT/'sfx-audition.mp3')],check=True)
print('Audition',t,'seconds')
