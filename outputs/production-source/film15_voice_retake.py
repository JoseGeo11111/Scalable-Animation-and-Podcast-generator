"""Prepare purposeful sentence retakes and assemble them without stretching."""
import json,re,sys,shutil
from pathlib import Path
import numpy as np,soundfile as sf
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'outputs/film15/audio/voice'; WORK=ROOT/'work/film15-voice'
film=json.loads((ROOT/'outputs/production-source/film15-script.json').read_text(encoding='utf-8'))
ids=sys.argv[2:]; mode=sys.argv[1]
if mode=='prepare':
    scenes=[]
    for s in film['scenes']:
        if s['id'] not in ids: continue
        spoken=s['narration_ml'].replace('തോമാശ്ലീഹാ','തോമാ ശ്ലീഹാ').replace('ക്രിസ്തുവർഷം','ക്രിസ്തു വർഷം')
        if s['id']=='11': spoken=spoken.replace('തോമാ ശ്ലീഹാ ക്രിസ്തു വർഷം അമ്പത്തിരണ്ടിൽ കേരളത്തിലെത്തിയെന്നാണ്','തോമാ ശ്ലീഹാ, ക്രിസ്തു വർഷം അമ്പത്തിരണ്ടിൽ, കേരളത്തിൽ എത്തിയെന്നാണ്')
        if s['id']=='12': spoken=spoken.replace('അമ്പത്തിരണ്ടെന്ന','അമ്പത്തി രണ്ട് എന്ന')
        if s['id']=='20': spoken=spoken.replace('സുറിയാനി ക്രിസ്ത്യാനി എന്ന പേരുകേട്ട് എല്ലാവരും ഇന്നത്തെ സിറിയയിൽനിന്നു വന്നവരാണെന്ന് കരുതരുത്.','സുറിയാനി ക്രിസ്ത്യാനി എന്ന പേര് കേട്ട്, എല്ലാവരും ഇന്നത്തെ സിറിയയിൽ നിന്ന് വന്നവരാണ് എന്ന് കരുതരുത്.')
        if s['id']=='30': spoken=spoken.replace('ആയിരത്തഞ്ഞൂറ്റിത്തൊണ്ണൂറ്റൊമ്പതിൽ','ആയിരത്തി അഞ്ഞൂറ്റി തൊണ്ണൂറ്റി ഒമ്പതിൽ').replace('ഗോവയിലെ ആർച്ച് ബിഷപ്പ് മെനേസിസിന്റെ നേതൃത്വത്തിൽ ചേർന്ന','ഗോവയിലെ, ആർച്ച് ബിഷപ്പ് മെനേസിസിന്റെ നേതൃത്വത്തിൽ, ചേർന്ന')
        if s['id']=='27': spoken=spoken.replace('ആയിരത്തിനാനൂറ്റിത്തൊണ്ണൂറ്റിയെട്ടിൽ വാസ്കോ ഡ ഗാമയുടെ','ആയിരത്തി നാനൂറ്റി തൊണ്ണൂറ്റി എട്ടിൽ, വാസ്കോ ഡ ഗാമയുടെ')
        if s['id']=='38': spoken=spoken.replace('ആയിരത്തെണ്ണൂറ്റിപ്പതിനേഴിൽ കോട്ടയത്ത് സി എം എസ്','ആയിരത്തി എണ്ണൂറ്റി പതിനേഴിൽ, കോട്ടയത്ത്, സി എം എസ്')
        if s['id']=='39': spoken=spoken.replace('ബെഞ്ചമിൻ ബെയ്‌ലിയും','ബെഞ്ചമിൻ, ബെയ്‌ലിയും')
        if s['id']=='40': spoken=spoken.replace('ആയിരത്തെണ്ണൂറ്റിപ്പത്തൊമ്പതിൽ','ആയിരത്തി എണ്ണൂറ്റി പത്തൊമ്പതിൽ').replace('കോട്ടയത്തെ ബേക്കർ സ്കൂൾ','കോട്ടയത്തെ, ബേക്കർ സ്കൂൾ')
        if s['id']=='41': spoken=spoken.replace('ആയിരത്തെണ്ണൂറ്റിയിരുപത്തൊന്നിൽ ബെയ്‌ലിയുടെ നേതൃത്വത്തിൽ കോട്ടയത്തെ','ആയിരത്തി എണ്ണൂറ്റി ഇരുപത്തൊന്നിൽ, ബെയ്‌ലിയുടെ നേതൃത്വത്തിൽ, കോട്ടയത്തെ')
        if s['id']=='45': spoken=spoken.replace('ആയിരത്തെണ്ണൂറ്റിയറുപത്താറിൽ','ആയിരത്തി എണ്ണൂറ്റി അറുപത്തി ആറിൽ,').replace('ചാവറയും ലെയോപോൾഡ് ബെക്കാറോയും','ചാവറയും, ലെയോപോൾഡ് ബെക്കാറോയും')
        if s['id']=='48': spoken=spoken.replace('തിരുവല്ലയിലെ പുഷ്പഗിരി ആശുപത്രിയുടെ','തിരുവല്ലയിലെ, പുഷ്പഗിരി ആശുപത്രിയുടെ').replace('ആയിരത്തിത്തൊള്ളായിരത്തിയമ്പത്തൊമ്പതിലെ','ആയിരത്തി തൊള്ളായിരത്തി അമ്പത്തി ഒമ്പതിലെ,')
        if s['id']=='52': spoken=spoken.replace('വിദേശകപ്പലിന്റെ വരവിൽ മാത്രം ഒതുങ്ങുന്നില്ലെന്ന്','വിദേശ കപ്പലിന്റെ വരവിൽ മാത്രം, ഒതുങ്ങുന്നില്ല എന്ന്')
        if s['id']=='54': spoken=spoken.replace('ഗോതുരുത്തിൽ രണ്ടായിരത്തിയഞ്ചിൽ','ഗോതുരുത്തിൽ, രണ്ടായിരത്തി അഞ്ചിൽ,')
        for n,t in enumerate(re.split(r'(?<=[.!?।])\s+',spoken)):
            if t.strip(): scenes.append({'id':s['id']+f'_r{n+1:02}','narration_ml':t.strip(),'scene_id':s['id']})
    (WORK/'retake-script.json').write_text(json.dumps({'scenes':scenes},ensure_ascii=False,indent=2),encoding='utf-8'); print('Retake utterances',len(scenes))
elif mode=='assemble':
    entries=json.loads((WORK/'retake-script.json').read_text(encoding='utf-8'))['scenes']; archive=WORK/'retakes'; archive.mkdir(exist_ok=True)
    for s in film['scenes']:
        if ids and s['id'] not in ids: continue
        parts=[e for e in entries if e['scene_id']==s['id']]
        if not parts: continue
        arrays=[]; timing=[]; t=0
        for i,p in enumerate(parts):
            fn=OUT/(p['id']+'.wav'); x,sr=sf.read(fn,dtype='float32'); assert sr==24000
            timing.append({'id':p['id'],'start':t,'duration':len(x)/sr,'spoken_text':p['narration_ml']}); arrays.append(x); t+=len(x)/sr
            if i<len(parts)-1: arrays.append(np.zeros(2880,dtype='float32')); t+=.12
        for ext in ['.wav','.json']:
            old=OUT/(s['id']+ext)
            if old.exists() and not (archive/(s['id']+'.first-pass'+ext)).exists(): shutil.copy2(old,archive/(s['id']+'.first-pass'+ext))
        x=np.concatenate(arrays); fn=OUT/(s['id']+'.wav'); sf.write(fn,x,24000,subtype='PCM_24')
        d={'id':s['id'],'file':str(fn),'duration':len(x)/24000,'sample_rate':24000,'model':'kenpath/svara-tts-v1','voice':'Malayalam (Female)','text':s['narration_ml'],'local':True,'v1_prompt_framing':True,'capped':False,'retake':True,'utterances':timing,'peak':float(np.max(np.abs(x)))}
        fn.with_suffix('.json').write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8'); print(s['id'],d['duration'])
elif mode=='manifest':
    entries=[json.loads((OUT/(s['id']+'.json')).read_text(encoding='utf-8')) for s in film['scenes']]
    for s,e in zip(film['scenes'],entries): assert e['text']==s['narration_ml'] and e['voice']=='Malayalam (Female)' and e.get('v1_prompt_framing') and not e['capped']
    (OUT/'durations.json').write_text(json.dumps({'scenes':entries,'total_duration':sum(e['duration'] for e in entries),'complete':True,'model':'kenpath/svara-tts-v1','voice':'Malayalam (Female)'},ensure_ascii=False,indent=2),encoding='utf-8')
    print('Full voice',sum(e['duration'] for e in entries))
