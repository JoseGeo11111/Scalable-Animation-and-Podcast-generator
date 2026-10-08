"""Audition stronger speech ducking while retaining full ambience between phrases."""
from pathlib import Path
import json,subprocess,re,shutil
import numpy as np
import soundfile as sf
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'outputs/film15/audio';OUT=BASE/'clarity-candidate';OUT.mkdir(exist_ok=True)
INITIAL=BASE/'mixes/initial' if (BASE/'mixes/initial/mix-report.json').exists() else BASE
SR=48000;N=900*SR;step=480
timeline=json.loads((ROOT/'outputs/production-source/film15-timeline.json').read_text(encoding='utf8'))
closing=timeline['scenes'][-1];close_a=round(closing['start']*SR);close_b=round(closing['end']*SR)
voice,sr=sf.read(BASE/'narration.wav',dtype='float32',always_2d=True);assert sr==SR and len(voice)==N
activity=np.sqrt(np.mean(voice.reshape(-1,step,2)**2,axis=(1,2)))
target=np.clip((activity-.008)/.025,0,1);env=np.zeros_like(target)
for i in range(1,len(env)):
    rate=.18 if target[i]>env[i-1] else .018
    env[i]=env[i-1]+rate*(target[i]-env[i-1])
names=['music-stem','ambience-stem','foley-stem'];extra=[2,2.5,4.5]
inputs=[sf.SoundFile(INITIAL/(name+'.wav')) for name in names]
outputs=[sf.SoundFile(OUT/(name+'.wav'),'w',samplerate=SR,channels=2,subtype='PCM_24') for name in names]
with sf.SoundFile(OUT/'premaster.wav','w',samplerate=SR,channels=2,subtype='FLOAT') as master:
    for i in range(0,N,SR*10):
        j=min(N,i+SR*10);e=np.repeat(env[i//step:(j+step-1)//step],step)[:j-i,None]
        total=voice[i:j].copy()
        for k,(source,dest,db) in enumerate(zip(inputs,outputs,extra)):
            y=source.read(j-i,dtype='float32',always_2d=True)*(10**(-db*e/20))
            a=max(i,close_a)-i;b=min(j,close_b)-i
            if b>a:
                if k==2:y[a:b]=0  # The horizon shot has no visible pepper-pouring action.
                else:y[a:b]*=10**(-3*e[a:b]/20)
            dest.write(y);total+=y
        master.write(total)
for f in inputs+outputs:f.close()
def run(cmd):return subprocess.run(cmd,capture_output=True,text=True,check=True)
def meter(path):
    p=run(['ffmpeg','-hide_banner','-i',str(path),'-af','loudnorm=I=-16:TP=-1.5:LRA=9:print_format=json','-f','null','-'])
    return json.loads(re.findall(r'\{[^{}]*"input_i"[^{}]*\}',p.stderr,re.S)[-1])
m=meter(OUT/'premaster.wav')
flt=f"loudnorm=I=-16:TP=-1.5:LRA=9:measured_I={m['input_i']}:measured_TP={m['input_tp']}:measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}:offset={m['target_offset']}:linear=true"
run(['ffmpeg','-y','-hide_banner','-i',str(OUT/'premaster.wav'),'-af',flt,'-ar','48000','-c:a','pcm_s24le',str(OUT/'mix.wav')])
m=meter(OUT/'mix.wav')
run(['ffmpeg','-y','-hide_banner','-i',str(OUT/'mix.wav'),'-c:a','libmp3lame','-b:a','256k',str(OUT/'mix.mp3')])
run(['ffmpeg','-y','-hide_banner','-i',str(OUT/'mix.wav'),'-t','45','-c:a','libmp3lame','-b:a','256k',str(OUT/'opening-sound-preview.mp3')])
r=json.loads((INITIAL/'mix-report.json').read_text(encoding='utf8'))
r.update(integrated_lufs=float(m['input_i']),true_peak_dbtp=float(m['input_tp']),loudness_range_lu=float(m['input_lra']),revision='speech-clarity-v2',clarity_revision='Additional ducking during speech only; original bed level retained between phrases. Source narration timing is unchanged.')
r['mix_choices'].update(score_speech_duck_db=8.5,ambience_speech_duck_db=5.5,foley_speech_duck_db=7)
r['closing_scene_mix']='Additional3dB score/ambience speech duck; removed unrelated pepper-pouring Foley from the horizon scene.'
r['cues']=[c for c in r['cues'] if not (c.get('scene')=='60' and c.get('type')=='foley')]
if 'stem_checks' in r:r.pop('stem_checks')
(OUT/'mix-report.json').write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps({'integrated_lufs':r['integrated_lufs'],'true_peak_dbtp':r['true_peak_dbtp'],'candidate':str(OUT/'mix.wav')}),flush=True)
