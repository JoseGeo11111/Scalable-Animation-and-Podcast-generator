"""Render brief animated endcards, assemble the measured 900-second film and QA it."""
import argparse,json,math,subprocess,shutil
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import film15_graphics as g
from finish_local_test import subtitle_tracks_off,probe,run,sha256
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs';WORK=ROOT/'work/film15'
BASE='kerala-roots-and-sea-15min';MASTER=OUT/(BASE+'.mp4')
ap=argparse.ArgumentParser();ap.add_argument('--cards-only',action='store_true');ap.add_argument('--validate-existing',action='store_true');args=ap.parse_args()
timeline=({'intro_frames':60,'credits_frames':300} if args.cards_only else json.loads((OUT/'production-source/film15-timeline.json').read_text(encoding='utf8')))
def card_frame(t,credits):
    im=Image.new('RGB',(1920,1080),g.INK);d=ImageDraw.Draw(im)
    for layer in range(4):
        points=[(x,790+layer*58+25*math.sin(x/230+t*.8+layer)) for x in range(0,1921,12)]
        color=[(25,70,75),(33,85,85),(46,105,98),(79,130,112)][layer]
        d.polygon(points+[(1920,1080),(0,1080)],fill=color)
    d.ellipse((1390,90,1660,360),fill=(196,151,84))
    if not credits:
        g.label(im,'വേരുകളും കടലും',(150,345),96,g.CREAM)
        g.label(im,'കേരളത്തിലെ ക്രൈസ്തവരുടെ കഥ',(155,495),40,g.CREAM)
    elif t<5:
        g.label(im,'വേരുകളും കടലും',(150,175),70,g.CREAM)
        g.label(im,'ഓർമ്മ. തെളിവ്. ഒരുമ.',(155,290),40,g.CREAM)
        for j,text in enumerate(['HISTORY: Sources and scene notes accompany this film.',
          'ILLUSTRATIONS: Interpretive composites, not historical portraits.',
          'MAPS: Natural Earth, geoBoundaries and GeoNames.',
          'PLATES: Historical reproduction, Quilon copper plates.']):
            g.latin(im,text,(155,420+j*64),31,g.CREAM)
    else:
        g.label(im,'കേരളം നമ്മുടേത്. കഥയും.',(150,190),66,g.CREAM)
        lines=['AI-assisted pilot • Malayalam narration: local Svara-TTS',
          'Cloud-generated artwork • Local Wan animation • RIFE 30 fps',
          'Local sound: MOSS-SoundEffect 2.0 • Score: ACE-Step 1.5',
          'Music and effects are illustrative, not location recordings.',
          'Source ledger, subtitles and production notes included.']
        for j,text in enumerate(lines):g.latin(im,text.replace('•','/'),(155,340+j*68),29,g.CREAM)
    return im
def make_card(name,n,credits=False):
    dest=WORK/'shot30'/name;dest.parent.mkdir(parents=True,exist_ok=True)
    cmd=['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','1920x1080','-r','30','-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','18','-threads','3','-pix_fmt','yuv420p','-color_primaries','bt709','-color_trc','bt709','-colorspace','bt709',str(dest)]
    p=subprocess.Popen(cmd,stdin=subprocess.PIPE)
    for i in range(n):p.stdin.write(card_frame(i/30,credits).tobytes())
    p.stdin.close();assert p.wait()==0
    card_frame(2 if credits else 1,credits).save(dest.with_suffix('.png'))
    return dest
intro=(WORK/'shot30/intro.mp4' if args.validate_existing else make_card('intro.mp4',timeline['intro_frames']));credits=(WORK/'shot30/credits.mp4' if args.validate_existing else make_card('credits.mp4',timeline['credits_frames'],True))
if args.cards_only:raise SystemExit(0)
files=[intro]
for shot in timeline['shots']:
    path=WORK/'shot30'/(shot['id']+'.mp4')
    if not path.exists():raise FileNotFoundError(path)
    v=next(s for s in probe(path)['streams'] if s['codec_type']=='video')
    assert int(v['nb_frames'])==shot['frames'],(shot['id'],v['nb_frames'],shot['frames'])
    files.append(path)
files.append(credits)
concat=WORK/'concat.txt';concat.write_text('\n'.join("file '"+str(p).replace('\\','/').replace("'","'\\''")+"'" for p in files),encoding='utf8')
chapters=WORK/'chapters.txt';lines=[';FFMETADATA1','title=Roots and the Sea']
for act in range(1,7):
    rows=[s for s in timeline['scenes'] if s['act']['number']==act]
    lines+=['[CHAPTER]','TIMEBASE=1/1000','START='+str(0 if act==1 else round(rows[0]['start']*1000)),'END='+str(900000 if act==6 else round(rows[-1]['end']*1000)),'title='+rows[0]['act']['title_ml']]
chapters.write_text('\n'.join(lines),encoding='utf8')
chapter_text=[]
for act in range(1,7):
    row=next(s for s in timeline['scenes'] if s['act']['number']==act)
    seconds=0 if act==1 else int(row['start'])
    chapter_text.append(f'{seconds//60:02}:{seconds%60:02} '+row['act']['title_ml'])
(OUT/'kerala-roots-and-sea-chapters.txt').write_text('\n'.join(chapter_text),encoding='utf8')
cmd=['ffmpeg','-hide_banner','-y','-f','concat','-safe','0','-i',concat,'-i',OUT/'film15/audio/mix.wav','-i',OUT/(BASE+'.ml.srt'),'-i',OUT/(BASE+'.en.srt'),'-i',chapters,'-map','0:v','-map','1:a','-map','2:s','-map','3:s','-map_metadata','4','-map_chapters','4','-t','900','-c:v','libx264','-preset','medium','-crf','18','-threads','8','-pix_fmt','yuv420p','-r','30','-color_primaries','bt709','-color_trc','bt709','-colorspace','bt709','-bsf:v','h264_metadata=colour_primaries=1:transfer_characteristics=1:matrix_coefficients=1','-c:a','aac','-b:a','320k','-ar','48000','-ac','2','-c:s','mov_text','-metadata:s:s:0','language=mal','-metadata:s:s:0','title=Malayalam','-metadata:s:s:1','language=eng','-metadata:s:s:1','title=English','-disposition:s:0','0','-disposition:s:1','0','-metadata','comment=AI-assisted documentary pilot. Interpretive artwork; sourced modern geography. Locally generated animation, Malayalam voice, music and effects. Native24fps interpolated to30fps;1280x704 source conventionally resized1080p. Subtitles use editorial timing.','-movflags','+faststart',MASTER]
if not args.validate_existing:
    print('ENCODING MASTER',flush=True);run(cmd)
subtitle_tracks_off(MASTER)
print('VALIDATING FULL DECODE',flush=True)
run(['ffmpeg','-v','error','-xerror','-i',MASTER,'-map','0:v','-map','0:a','-f','null','-'])
p=probe(MASTER,True);v=next(s for s in p['streams'] if s['codec_type']=='video');a=next(s for s in p['streams'] if s['codec_type']=='audio');subs=[s for s in p['streams'] if s['codec_type']=='subtitle']
checks={'900_seconds':abs(float(p['format']['duration'])-900)<.04,'27000_frames':int(v['nb_read_frames'])==27000,'30_fps':v['avg_frame_rate']=='30/1','1080p':(v['width'],v['height'])==(1920,1080),'stereo_48k':a['channels']==2 and a['sample_rate']=='48000','two_optional_subtitles':len(subs)==2 and all(s['disposition']['default']==0 for s in subs),'full_decode':True}
assert all(checks.values()),checks
shutil.copy2(OUT/'film15/audio/mix.mp3',OUT/'kerala-roots-and-sea-podcast.mp3')
run(['ffmpeg','-hide_banner','-loglevel','error','-y','-ss','5','-i',MASTER,'-frames:v','1','-q:v','2',OUT/(BASE+'-poster.jpg')])
report={'checks':checks,'file':str(MASTER),'sha256':sha256(MASTER),'bytes':MASTER.stat().st_size,'video':v,'audio':a,'duration':p['format']['duration'],'voice_tempo':timeline['voice_tempo'],'source_voice_seconds':timeline['source_voice_seconds'],'retimed_voice_seconds':timeline['retimed_voice_seconds'],'mix':json.loads((OUT/'film15/audio/mix-report.json').read_text(encoding='utf8'))}
(OUT/'film15/final-qa.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(checks),flush=True)
