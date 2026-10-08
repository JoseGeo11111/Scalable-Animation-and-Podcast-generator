"""Allocate the final 900-second film from measured Malayalam recordings.
Voice retiming is pitch-preserving and documented; no repeated voice or frozen
frames are used to manufacture the running time.
"""
import json,math,html
from pathlib import Path
from PIL import Image,ImageDraw
import film15_graphics as g
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs';WORK=ROOT/'work/film15'
script=json.loads((OUT/'production-source/film15-script.json').read_text(encoding='utf8'))
dur=json.loads((OUT/'film15/audio/voice/durations.json').read_text(encoding='utf8'))
meta={str(s['id']).zfill(2):s for s in dur['scenes']}
queue=json.loads((OUT/'production-source/film15-video-queue.json').read_text())
byid={s['scene_id']:s for s in queue}
override_path=OUT/'production-source/film15-edit-overrides.json'
overrides=json.loads(override_path.read_text(encoding='utf8')) if override_path.exists() else {}
graphics={'04','08','14','16','17','35','38','48','52','55','56'}
fps=30;totalframes=27000;intro=60;endframes=300
pre=10;post=8;chapterextra=24
reveals={'13','16','33','40','59'}
gaps=60*(pre+post)+6*chapterextra+len(reveals)*30
voiceframes=totalframes-intro-endframes-gaps
raw=sum(meta[s['id']]['duration'] for s in script['scenes'])
tempo=raw/(voiceframes/fps)
if not .85<=tempo<=1.35:raise ValueError(f'Voice requires excessive retiming: {tempo:.3f}')
floats=[meta[s['id']]['duration']/tempo*fps for s in script['scenes']]
frames=[int(x) for x in floats]
for i in sorted(range(60),key=lambda i:floats[i]-frames[i],reverse=True)[:voiceframes-sum(frames)]:frames[i]+=1
overlays=OUT/'film15/overlays';overlays.mkdir(exist_ok=True,parents=True)
def overlay(name,main,sub='',position='top'):
    im=Image.new('RGBA',(1920,1080),(0,0,0,0));d=ImageDraw.Draw(im)
    y=90 if position=='top' else 820
    one=g.text_img(main,52,g.CREAM);two=g.text_img(sub,29,g.CREAM) if sub else None
    width=max(one.width,two.width if two else 0)+64
    d.rounded_rectangle((70,y-25,70+width,y+one.height+(two.height+30 if two else 20)),radius=12,fill=(12,44,48,220))
    im.alpha_composite(one,(100,y))
    if two:im.alpha_composite(two,(100,y+one.height+12))
    dest=overlays/(name+'.png');im.save(dest);return str(dest)
labels={
'02':('വേരുകളും കടലും','കേരളത്തിലെ ക്രൈസ്തവരുടെ കഥ'),
'11':('ക്രി.വ. 52','സഭാപാരമ്പര്യം'),
'13':('ആറാം നൂറ്റാണ്ട്','കോസ്മാസിന്റെ രേഖ'),
'21':('ചവിട്ടുനാടകം','ജീവിക്കുന്ന കലാപാരമ്പര്യം'),
'23':('മാർഗംകളി','പാട്ടും ചുവടും ഓർമ്മയും'),
'27':('1498','വാസ്കോ ഡ ഗാമയുടെ വരവ്'),
'30':('1599','ഉദയംപേരൂർ സൂനഹദോസ്'),
'33':('1653','കൂനൻ കുരിശുസത്യം'),
'40':('1819','ബേക്കർ വിദ്യാലയം'),
'41':('1821','കോട്ടയം സി എം എസ് അച്ചടിശാല'),
'45':('1866','സ്ത്രീകളുടെ കൂട്ടായ പ്രവർത്തനം'),
'47':('1956','ലിസി ആശുപത്രി'),
'49':('1964','പുഷ്പഗിരിയിലെ നഴ്സിങ് പരിശീലനം'),
'53':('1887','നസ്രാണി ദീപിക'),
'54':('2005','ഗോതുരുത്തിലെ കലാപരിശീലനം')
}
scenes=[];shots=[];cursor=intro
for index,(s,nvoice) in enumerate(zip(script['scenes'],frames)):
    sid=s['id'];endgap=post+(chapterextra if (index+1)%10==0 else 0)
    scene_pre=pre+(30 if sid in reveals else 0)
    n=scene_pre+nvoice+endgap
    row={**s,'start_frame':cursor,'end_frame':cursor+n,'start':cursor/fps,'end':(cursor+n)/fps,'duration':n/fps,'voice_start':(cursor+scene_pre)/fps,'voice_end':(cursor+scene_pre+nvoice)/fps,'voice_frames':nvoice,'voice_file':meta[sid]['file'],'voice_source_duration':meta[sid]['duration'],'tempo':tempo}
    scenes.append(row)
    if sid in graphics:
        shots.append({'id':sid+'g','scene_id':sid,'kind':'graphic','graphic_id':sid,'frames':n,'start_frame':cursor,'purpose':s['purpose']})
    else:
        q=byid[sid]
        reportpath=WORK/(q['id']+'-report.json')
        source=None
        if reportpath.exists():
            report=json.loads(reportpath.read_text());source=next((p for p in reversed(report['files']) if p.endswith('.mp4')),None)
        source=source or str(OUT/'film15/native'/('PENDING_'+q['id']+'.mp4'))
        label=overlay(sid,*labels[sid]) if sid in labels else None
        if sid in overrides:
            edits=overrides[sid]['shots'];used=0
            for j,edit in enumerate(edits):
                nf=n-used if j==len(edits)-1 else round(n*edit['weight'])
                shots.append({'id':sid+chr(97+j),'scene_id':sid,'kind':'video','frames':nf,'start_frame':cursor+used,'source':source,'queue_id':q['id'],'purpose':s['purpose'],'edit_reason':overrides[sid]['reason'],'overlay':label if j==0 else None,'overlay_seconds':min(4.5,nf/fps),**{k:v for k,v in edit.items() if k!='weight'}})
                used+=nf
            cursor+=n
            continue
        a=round(n*.52);b=n-a
        shots.append({'id':sid+'a','scene_id':sid,'kind':'video','frames':a,'start_frame':cursor,'source':source,'queue_id':q['id'],'source_start':0,'source_end':6.2,'crop_zoom':1,'overlay':label,'overlay_seconds':min(4.5,a/fps),'purpose':s['shots'][0]['purpose']})
        shots.append({'id':sid+'b','scene_id':sid,'kind':'video','frames':b,'start_frame':cursor+a,'source':source,'queue_id':q['id'],'source_start':6.2,'source_end':12,'crop_zoom':1.13 if index%3 else 1,'focus_x':.48,'focus_y':.44,'purpose':s['shots'][1]['purpose']})
    cursor+=n
assert cursor+endframes==totalframes,(cursor,endframes)
timeline={'title':script['title_ml'],'fps':fps,'width':1920,'height':1080,'total_frames':totalframes,'duration':900,'intro_frames':intro,'credits_frames':endframes,'voice_tempo':tempo,'source_voice_seconds':raw,'retimed_voice_seconds':voiceframes/fps,'scenes':scenes,'shots':shots}
(OUT/'production-source/film15-timeline.json').write_text(json.dumps(timeline,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps({k:timeline[k] for k in ['duration','voice_tempo','source_voice_seconds','retimed_voice_seconds']},ensure_ascii=False))
def stamp(sec):
    ms=round(sec*1000);return f'{ms//3600000:02}:{ms//60000%60:02}:{ms//1000%60:02},{ms%1000:03}'
for lang in ['ml','en']:
    cues=[];i=0
    for s in scenes:
        text=s['narration_'+lang]
        # Clause-sized captions; weighted timing is editorial, not a forced aligner.
        import re
        units=[u.strip() for u in re.split(r'(?<=[.!?])\s+',text) if u.strip()]
        weights=[len(u) for u in units];start=s['voice_start'];span=s['voice_end']-start;acc=0
        for u,w in zip(units,weights):
            i+=1;end=start+span*w/sum(weights)
            cues.extend([str(i),stamp(start)+' --> '+stamp(end),u,'']);start=end
    (OUT/f'kerala-roots-and-sea-15min.{lang}.srt').write_text('\n'.join(cues),encoding='utf8')
cards=[]
for s in scenes:
    sid=s['id'];img=f'film15/graphics/previews/{sid}.png' if sid in graphics else 'film15/keyframes/'+byid[sid]['asset']+'.png'
    cards.append(f'<article><img src="{img}"><div><small>{sid} · {s["start"]:.1f}–{s["end"]:.1f}s · {html.escape(s["emotion"])}</small><h2>{html.escape(s["title_ml"])}</h2><p lang="ml">{html.escape(s["narration_ml"])}</p><p>{html.escape(s["narration_en"])}</p><details><summary>Shot purpose and sources</summary><p>{html.escape(s["visual_brief_en"])}</p><p>Sources: {html.escape(", ".join(s["source_ids"]))}</p></details></div></article>')
sources=''.join(f'<li id="{s["id"]}">{html.escape(s["id"])}: '+(f'<a href="{html.escape(s["url"])}">{html.escape(s["title"])}</a>' if s.get('url') else html.escape(s['title']))+'</li>' for s in script['sources'])
page='<!doctype html><meta charset="utf-8"><title>Roots and the Sea · Storyboard</title><style>@font-face{font-family:NotoMalayalam;src:url(assets/fonts/NotoSansMalayalam-Regular.ttf)}body{margin:0;background:#f5eddc;color:#17383d;font:18px/1.65 system-ui,NotoMalayalam,sans-serif}header,main,footer{max-width:1300px;margin:auto;padding:40px}h1{font-size:52px;line-height:1.2}small{color:#a75737}article{display:grid;grid-template-columns:1fr 1fr;gap:30px;background:#fff9ed;margin:28px 0;padding:24px;border-radius:16px}img{width:100%;border-radius:8px}h2{font-size:26px}p[lang=ml]{font-size:21px}a{color:#a75737}@media(max-width:800px){article{display:block}}</style><header><small>15 MINUTES · MALAYALAM · 30 FPS</small><h1>വേരുകളും കടലും</h1><p>Roots and the Sea — a source-led animated documentary about Kerala Christian history and shared belonging.</p><p>Story structure, narration, a visual purpose for each scene, and source notes. Illustrations are interpretive composites; maps and the copper-plate reproduction use identified sources.</p></header><main>'+''.join(cards)+'</main><footer><h2>Sources</h2><ol>'+sources+'</ol></footer>'
(OUT/'kerala-roots-and-sea-storyboard.html').write_text(page,encoding='utf8')
