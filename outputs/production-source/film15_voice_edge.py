"""Cloud Microsoft Edge preset narration fallback. NOT local TTS."""
import asyncio,json,argparse,subprocess,time
from pathlib import Path
import edge_tts
import soundfile as sf
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/film15/audio/voice'
async def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--test',action='store_true'); ap.add_argument('--voice',default='ml-IN-MidhunNeural'); args=ap.parse_args()
    scenes=json.loads((ROOT/'outputs/production-source/film15-script.json').read_text(encoding='utf-8'))['scenes']
    if args.test: scenes=[{'id':'test-edge','narration_ml':'കടൽ കടന്നുവന്നത് കുരുമുളകിന്റെ കഥ മാത്രമല്ല. മനുഷ്യരും വിശ്വാസങ്ങളും ഇവിടെ കണ്ടുമുട്ടി. കേരളത്തിലെ ക്രിസ്തീയ പാരമ്പര്യം നമ്മുടെ നാടിന്റെ ചരിത്രത്തിന്റെ ഭാഗമാണ്.'}]
    for s in scenes:
        p=OUT/(str(s['id'])+'.wav')
        if p.exists(): continue
        mp=p.with_suffix('.mp3'); t=time.perf_counter()
        await edge_tts.Communicate(s['narration_ml'],args.voice,rate='-5%').save(str(mp))
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(mp),'-ar','24000','-ac','1','-c:a','pcm_s24le',str(p)],check=True)
        x,sr=sf.read(p); data={'id':s['id'],'file':str(p),'duration':len(x)/sr,'sample_rate':sr,'model':'Microsoft Edge online Read Aloud','voice':args.voice,'rate':'-5%','text':s['narration_ml'],'peak':float(np.max(np.abs(x))),'rms':float(np.sqrt(np.mean(x*x))),'render_seconds':time.perf_counter()-t,'local':False}
        p.with_suffix('.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps({'id':s['id'],'duration':data['duration']}),flush=True)
if __name__=='__main__': asyncio.run(main())
