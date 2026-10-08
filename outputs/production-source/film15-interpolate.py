"""Exact 30 fps RIFE v4.26 interpolation, applied within each shot only.
Frames are streamed through FFmpeg; source pictures are never blended across cuts.
Usage: gpu-env/python film15-interpolate.py edit.json [--limit N]
"""
import argparse,json,sys,subprocess,time,os,math
from pathlib import Path
import cv2,numpy as np,torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'work/Wan2GP'))
from postprocessing.rife.RIFE_V4 import Model
parser=argparse.ArgumentParser();parser.add_argument('edit',type=Path);parser.add_argument('--limit',type=int,default=999)
args=parser.parse_args();edit=json.loads(args.edit.read_text(encoding='utf8'))
model=Model();model.load_model(str(ROOT/'work/Wan2GP/ckpts/rife4.26.pkl'),-1,device='cuda');model.eval()
torch.set_num_threads(4);torch.backends.cudnn.benchmark=True
outdir=ROOT/'work/film15/shot30';outdir.mkdir(exist_ok=True,parents=True)
count=0
@torch.inference_mode()
def render(shot):
    dest=outdir/(shot['id']+'.mp4')
    temp=dest.with_name(dest.stem+'.partial.mp4')
    if not Path(shot['source']).exists():
        report=json.loads((ROOT/'work/film15'/(shot['queue_id']+'-report.json')).read_text())
        shot['source']=next(p for p in reversed(report['files']) if p.endswith('.mp4'))
    fingerprint={'shot':shot,'source_mtime':Path(shot['source']).stat().st_mtime_ns}
    manifest=dest.with_suffix('.render.json')
    if dest.exists() and manifest.exists() and json.loads(manifest.read_text())==fingerprint:return
    cap=cv2.VideoCapture(shot['source']);fps=cap.get(cv2.CAP_PROP_FPS);total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps<=0 or total<2:raise RuntimeError('Invalid source video: '+shot['source'])
    w=int(cap.get(3));h=int(cap.get(4));source_start=shot.get('source_start',0)
    source_end=shot.get('source_end',(total-1)/fps)
    source_end=min(source_end,(total-1)/fps)
    if source_start>=source_end:raise RuntimeError('Source too short for shot: '+shot['id'])
    n=int(shot['frames']);targetfps=30
    # Frames at exact requested positions; true slow motion where editorial timing requires it.
    positions=np.linspace(source_start*fps,source_end*fps,n)
    zoom=shot.get('crop_zoom',1)
    vf=(f'crop=trunc(iw/{zoom}/2)*2:trunc(ih/{zoom}/2)*2:(iw-ow)*{shot.get("focus_x",.5)}:(ih-oh)*{shot.get("focus_y",.5)},' if zoom>1 else '')+'scale=1920:1080:flags=lanczos:force_original_aspect_ratio=increase:out_color_matrix=bt709,crop=1920:1080,setsar=1'
    command=['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{w}x{h}','-r','30','-i','pipe:0']
    if shot.get('overlay'):
        command+=['-i',shot['overlay'],'-filter_complex',f'[0:v]{vf}[base];[base][1:v]overlay=0:0:enable=\'between(t,0,{shot.get("overlay_seconds",4)})\'[v]','-map','[v]']
    else:command+=['-vf',vf]
    command+=['-an','-c:v','libx264','-preset','fast','-crf','18','-threads','4','-pix_fmt','yuv420p','-color_primaries','bt709','-color_trc','bt709','-colorspace','bt709',str(temp)]
    pipe=subprocess.Popen(command,stdin=subprocess.PIPE)
    cache={};current=-1
    def get(idx):
        nonlocal current
        idx=min(max(idx,0),total-1)
        while current<idx:
            ok,frame=cap.read()
            if not ok:raise RuntimeError('Source decode ended early')
            current+=1
            if current>=idx-1:
                rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                tensor=torch.from_numpy(rgb).permute(2,0,1).unsqueeze(0).to('cuda',dtype=torch.float32)/255
                cache[current]=F.pad(tensor,(0,(-w)%64,0,(-h)%64))
        return cache[idx]
    for j,pos in enumerate(positions):
        lo=int(math.floor(pos));hi=min(lo+1,total-1);t=float(pos-lo)
        first=get(lo);second=get(hi)
        if t<1e-5 or lo==hi:frame=first
        else:frame=model.inference(first,second,t,1)
        data=(frame[0,:,:h,:w].clamp(0,1)*255).round().byte().permute(1,2,0).cpu().numpy()
        pipe.stdin.write(data.tobytes())
        for key in list(cache):
            if key<lo:del cache[key]
    cap.release();pipe.stdin.close()
    if pipe.wait()!=0:raise RuntimeError('FFmpeg failed')
    temp.replace(dest)
    manifest.write_text(json.dumps(fingerprint,ensure_ascii=False,indent=2),encoding='utf8')
    print('INTERPOLATED',shot['id'],n,'frames',flush=True)
for shot in edit['shots']:
    if shot['kind']!='video':continue
    if (ROOT/'work/film15/pause-interpolation').exists():break
    if count>=args.limit:break
    start=time.time();render(shot);count+=1
    print('SHOT_SECONDS',shot['id'],round(time.time()-start,2),flush=True)
