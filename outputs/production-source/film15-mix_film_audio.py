"""Assemble a scene-timed 900 second film mix from locally generated assets.

Timeline JSON accepts a scenes list with id, start, duration and optional voice_start.
voice_start is absolute film time; default is scene start + 0.6 seconds.
All mix targets are production choices, not claimed measured loudness.
"""
from pathlib import Path
import json, argparse, subprocess, re, math
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly, butter, sosfilt
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/film15/audio'; WORK=ROOT/'work/film15-sound'
SR=48000
ap=argparse.ArgumentParser();ap.add_argument('timeline');ap.add_argument('--duration',type=float,default=900);args=ap.parse_args()
timeline_doc=json.loads(Path(args.timeline).read_text(encoding='utf8'))
global_tempo=float(timeline_doc.get('voice_tempo',1)) if isinstance(timeline_doc,dict) else 1
timeline=timeline_doc.get('scenes',timeline_doc) if isinstance(timeline_doc,dict) else timeline_doc
script=json.loads((ROOT/'outputs/production-source/film15-script.json').read_text(encoding='utf8'))
scenes={s['id']:s for s in script['scenes']}; N=round(args.duration*SR)
def read(path):
 y,sr=sf.read(path,dtype='float32',always_2d=True)
 if sr!=SR:
  g=math.gcd(sr,SR);y=resample_poly(y,SR//g,sr//g).astype('float32')
 y=sosfilt(butter(2,35,fs=SR,btype='highpass',output='sos'),y,axis=0).astype('float32')
 if y.shape[1]==1:y=np.repeat(y,2,axis=1)
 return y[:,:2]
def level(y,db):
 rms=float(np.sqrt(np.mean(y*y))); peak=float(np.max(np.abs(y)))
 gain=min(10**(db/20)/max(rms,1e-8),.90/max(peak,1e-8))
 return y*gain
def fade(y,seconds=.25):
 y=y.copy();n=min(int(seconds*SR),len(y)//2)
 if n:y[:n]*=np.linspace(0,1,n)[:,None];y[-n:]*=np.linspace(1,0,n)[:,None]
 return y
def add(dst,y,start):
 i=round(start*SR);j=min(len(dst),i+len(y))
 if i<0:y=y[-i:];i=0
 if j>i:dst[i:j]+=y[:j-i]
def loop(y,seconds):
 n=round(seconds*SR)
 if len(y)>=n:return fade(y[:n],.3)
 overlap=min(SR*2,len(y)//5);hop=len(y)-overlap
 out=np.zeros((n,2),np.float32)
 for k in range(0,n,hop):
  part=y.copy();f=min(overlap,len(part)//2)
  part[:f]*=np.sin(np.linspace(0,np.pi/2,f))[:,None]
  part[-f:]*=np.cos(np.linspace(0,np.pi/2,f))[:,None]
  end=min(n,k+len(part));out[k:end]+=part[:end-k]
 return fade(out,.5)
voice=np.memmap(WORK/'voice.f32',dtype='float32',mode='w+',shape=(N,2));voice[:]=0
music=np.memmap(WORK/'music.f32',dtype='float32',mode='w+',shape=(N,2));music[:]=0
amb=np.memmap(WORK/'ambience.f32',dtype='float32',mode='w+',shape=(N,2));amb[:]=0
fx=np.memmap(WORK/'foley.f32',dtype='float32',mode='w+',shape=(N,2));fx[:]=0
cache={p.stem:read(p) for p in (OUT/'sfx').glob('*.wav')}
# A brief generated wood impact, edited from the press Foley, punctuates the stage step.
wood=cache['printing_press']; block=2400
energy=np.asarray([np.mean(wood[i:i+block]**2) for i in range(0,len(wood)-block,block)])
hit=max(0,int(np.argmax(energy))*block-1200)
cache['wood_stage_impact']=fade(wood[hit:hit+round(.75*SR)],.015)
scorefiles=sorted((OUT/'score').glob('0*.wav'));assert len(scorefiles)>=6,'Six generated themes required'
records=[]
timed_voice=[]
amb_for={'pepper':'palm_breeze','coast':'shore_gentle','home':'palm_breeze','map':'sea_voyage','sailing':'sea_voyage','port':'harbor_market','harvest':'palm_breeze','evidence':'palm_breeze','tradition':'palm_breeze','manuscript':'palm_breeze','copper':'palm_breeze','community':'village_morning','church':'palm_breeze','dance':'village_morning','fishing':'sea_voyage','council':'palm_breeze','oath':'palm_breeze','paths':'palm_breeze','school':'village_morning','girl_school':'village_morning','care':'palm_breeze','travel':'sea_voyage','rain':'rain_roof','timeline':'shore_gentle'}
fx_for={'pepper':'pepper_pour','sailing':'cloth_sail','port':'wooden_boat','harvest':'cloth_sail','evidence':'paper_pages','manuscript':'writing_quill','copper':'metal_plate','church':'church_bell','fishing':'wooden_boat','council':'paper_pages','oath':'rope_strain','paths':'footsteps_earth','school':'paper_pages','girl_school':'paper_pages','care':'cloth_sail','travel':'footsteps_earth','rain':'rain_roof'}
amb_for.update(printing='palm_breeze',reader='palm_breeze')
fx_for.update(printing='printing_press',reader='paper_pages',dance='footsteps_earth')
fx_for.pop('rain',None)  # Rain is the continuous bed; avoid a duplicate thunder/rain hit.
# Low midrange reduction gives narration space without making the beds disappear.
hp=butter(2,100,fs=SR,btype='highpass',output='sos')
# Broad -4dB presence dip leaves Malayalam speech space while retaining audible sea/air.
peq_A=10**(-4/40);peq_w=2*np.pi*1400/SR;peq_alpha=np.sin(peq_w)/(2*.7)
peq_b=np.array([1+peq_alpha*peq_A,-2*np.cos(peq_w),1-peq_alpha*peq_A])
peq_a=np.array([1+peq_alpha/peq_A,-2*np.cos(peq_w),1-peq_alpha/peq_A])
amb_eq=np.concatenate([peq_b/peq_a[0],peq_a/peq_a[0]])[None,:]
for row in timeline:
 sid=str(row['id']).zfill(2);start=float(row['start']);duration=float(row['duration']);s=scenes[sid]
 vstart=float(row.get('voice_start',start+.6));tempo=float(row.get('tempo',global_tempo))
 source=Path(row.get('voice_file',OUT/'voice'/f'{sid}.wav'))
 if not source.is_absolute():source=ROOT/source
 target_samples=round((float(row['voice_end'])-vstart)*SR) if 'voice_end' in row else None
 if tempo!=1 or target_samples is not None:
  target=WORK/f'voice-timed-{sid}.wav'
  filters=[f'atempo={tempo:.10f}','aresample=48000']
  if target_samples is not None:filters.extend([f'apad=whole_len={target_samples}',f'atrim=end_sample={target_samples}'])
  subprocess.run(['ffmpeg','-y','-hide_banner','-loglevel','error','-i',str(source),'-af',','.join(filters),'-c:a','pcm_f32le',str(target)],check=True)
  v=read(target)
 else:v=read(source)
 if target_samples is not None:assert abs(len(v)-target_samples)<=1,f'Timed voice sample mismatch {sid}'
 timed_voice.append({'id':sid,'source':str(source),'tempo':tempo,'start':vstart,'end':vstart+len(v)/SR,'samples':len(v),'pitch_preserving':'FFmpeg atempo'})
 assert vstart+len(v)/SR<=start+duration+.15,f'Voice overruns scene {sid}'
 add(voice,fade(level(v,-19),.012),vstart)
 name=amb_for.get(s['setting'],'palm_breeze'); bed=loop(cache[name],duration)
 bed=level(sosfilt(amb_eq,bed,axis=0).astype('float32'),-24.5)
 # A different stereo delay for the mono-generated beds suggests space without phase reversal.
 bed[:,1]=np.roll(bed[:,1],137); add(amb,bed,start)
 hit=fx_for.get(s['setting'])
 hit={'13':'paper_pages','22':'wood_stage_impact','40':'footsteps_earth','53':'paper_pages','60':'pepper_pour'}.get(sid,hit)
 if hit:
  effect=cache[hit];maxlen=min(duration*.58,8 if hit!='church_bell' else 13)
  effect=fade(level(effect[:int(maxlen*SR)],-22.5),.12)
  cue=start+(.25 if sid=='01' else .1)
  add(fx,effect,cue)
  records.append({'scene':sid,'type':'foley','asset':hit,'start':cue,'duration':len(effect)/SR})
 if sid=='40':
  effect=fade(level(cache['paper_pages'][:SR*3],-22.5),.1);cue=start+duration*.58
  add(fx,effect,cue);records.append({'scene':sid,'type':'foley','asset':'paper_pages','start':cue,'duration':len(effect)/SR})
 records.append({'scene':sid,'type':'ambience','asset':name,'start':start,'duration':duration})
intro_end=float(timeline[0]['start'])
if intro_end>0:
 add(amb,level(loop(cache['shore_gentle'],intro_end),-23),0)
 records.append({'type':'ambience','asset':'shore_gentle','start':0,'duration':intro_end,'purpose':'Sound-only opening before the first sentence'})
film_content_end=float(timeline[-1]['start'])+float(timeline[-1]['duration'])
if film_content_end<args.duration:
 credits_duration=args.duration-film_content_end
 add(amb,fade(level(loop(cache['shore_gentle'],credits_duration),-25),min(4,credits_duration/2)),film_content_end)
 records.append({'type':'ambience','asset':'shore_gentle','start':film_content_end,'duration':credits_duration,'purpose':'Coastline returns beneath credits'})
for act in range(1,7):
 rows=[r for r in timeline if scenes[str(r['id']).zfill(2)]['act']['number']==act]
 start=float(rows[0]['start']);end=float(rows[-1]['start'])+float(rows[-1]['duration'])
 if act==6:end=args.duration
 y=read(scorefiles[act-1]);y=sosfilt(hp,y,axis=0).astype('float32')
 if act==6 and len(y)>round((end-start)*SR):y=y[-round((end-start)*SR):]
 y=level(loop(y,end-start),-23.5);y=fade(y,2.5)
 add(music,y,start)
 records.append({'act':act,'type':'score','asset':scorefiles[act-1].name,'start':start,'duration':end-start})
# Speech activity at 100 Hz. The release preserves silence breaths between phrases.
step=480;activity=np.sqrt(np.mean(np.asarray(voice).reshape(-1,step,2)**2,axis=(1,2)))
target=np.clip((activity-.008)/.025,0,1);env=np.zeros_like(target)
for i in range(1,len(env)):
 rate=.18 if target[i]>env[i-1] else .018
 env[i]=env[i-1]+rate*(target[i]-env[i-1])
sf.write(OUT/'narration.wav',voice,SR,subtype='PCM_24')
# Block output keeps memory bounded while video uses system RAM.
with sf.SoundFile(OUT/'premaster.wav','w',samplerate=SR,channels=2,subtype='FLOAT') as f, sf.SoundFile(OUT/'music-stem.wav','w',samplerate=SR,channels=2,subtype='PCM_24') as m, sf.SoundFile(OUT/'ambience-stem.wav','w',samplerate=SR,channels=2,subtype='PCM_24') as a, sf.SoundFile(OUT/'foley-stem.wav','w',samplerate=SR,channels=2,subtype='PCM_24') as x:
 for i in range(0,N,SR*10):
  j=min(N,i+SR*10); e=np.repeat(env[i//step:(j+step-1)//step],step)[:j-i,None]
  mm=music[i:j]*(10**(-6.5*e/20));aa=amb[i:j]*(10**(-3*e/20));xx=fx[i:j]*(10**(-2.5*e/20))
  f.write(voice[i:j]+mm+aa+xx);m.write(mm);a.write(aa);x.write(xx)
def run(cmd):return subprocess.run(cmd,capture_output=True,text=True,check=True)
p=run(['ffmpeg','-hide_banner','-i',str(OUT/'premaster.wav'),'-af','loudnorm=I=-16:TP=-1.5:LRA=9:print_format=json','-f','null','-'])
meter=json.loads(re.findall(r'\{[^{}]*"input_i"[^{}]*\}',p.stderr,re.S)[-1])
flt=f"loudnorm=I=-16:TP=-1.5:LRA=9:measured_I={meter['input_i']}:measured_TP={meter['input_tp']}:measured_LRA={meter['input_lra']}:measured_thresh={meter['input_thresh']}:offset={meter['target_offset']}:linear=true:print_format=json"
run(['ffmpeg','-y','-hide_banner','-i',str(OUT/'premaster.wav'),'-af',flt,'-ar','48000','-c:a','pcm_s24le',str(OUT/'mix.wav')])
p=run(['ffmpeg','-hide_banner','-i',str(OUT/'mix.wav'),'-af','loudnorm=I=-16:TP=-1.5:LRA=9:print_format=json','-f','null','-'])
finalmeter=json.loads(re.findall(r'\{[^{}]*"input_i"[^{}]*\}',p.stderr,re.S)[-1])
run(['ffmpeg','-y','-hide_banner','-i',str(OUT/'mix.wav'),'-c:a','libmp3lame','-b:a','256k',str(OUT/'mix.mp3')])
run(['ffmpeg','-y','-hide_banner','-i',str(OUT/'mix.wav'),'-t','45','-c:a','libmp3lame','-b:a','256k',str(OUT/'opening-sound-preview.mp3')])
report={'duration':N/SR,'sample_rate':SR,'channels':2,'integrated_lufs':float(finalmeter['input_i']),'true_peak_dbtp':float(finalmeter['input_tp']),'loudness_range_lu':float(finalmeter['input_lra']),'mix_choices':{'voice_rms_dbfs':-19,'score_nonvoice_rms_dbfs':-23.5,'score_speech_duck_db':6.5,'ambience_nonvoice_rms_dbfs':-24.5,'ambience_speech_duck_db':3,'foley_base_rms_dbfs':-22.5,'foley_speech_duck_db':2.5},'cues':records,'source_models':['OpenMOSS-Team/MOSS-SoundEffect-v2.0','ACE-Step/Ace-Step1.5'],'speech_source':'See voice scene metadata','notes':'Model synthesis is illustrative. These are not location recordings or verified historical sounds. Stem WAVs are before final master gain; mix.wav is the mastered deliverable.'}
report['edited_effects']={'wood_stage_impact':'0.75 second strongest wood transient edited from locally generated printing_press.wav; illustrative impact, not a location recording.'}
report['timed_voice']=timed_voice
report['equalization']='35Hz high-pass on all inputs for DC/rumble; additional100Hz high-pass on score; broad -4dB presence dip at1400Hz,Q0.7 on scene ambience to leave voice space.'
(OUT/'mix-report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf8')
print(json.dumps({k:v for k,v in report.items() if k!='cues'},indent=2))
