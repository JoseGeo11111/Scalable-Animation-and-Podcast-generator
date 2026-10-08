"""Extract a labelled contact sheet for visual QA; never modifies source media."""
import argparse,json
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
ap=argparse.ArgumentParser();ap.add_argument('video',type=Path);ap.add_argument('--out',type=Path);ap.add_argument('--frames',type=int,default=12);args=ap.parse_args()
cap=cv2.VideoCapture(str(args.video));fps=cap.get(cv2.CAP_PROP_FPS);n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));assert fps and n
cols=4;thumbw=480;thumbh=270;rows=(args.frames+cols-1)//cols
sheet=Image.new('RGB',(cols*thumbw,rows*(thumbh+32)),(17,44,48));d=ImageDraw.Draw(sheet)
for i,t in enumerate(np.linspace(0,(n-1)/fps,args.frames)):
    cap.set(cv2.CAP_PROP_POS_MSEC,float(t)*1000);ok,bgr=cap.read();assert ok,(i,t)
    rgb=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));rgb.thumbnail((thumbw,thumbh))
    x=i%cols*thumbw;y=i//cols*(thumbh+32);sheet.paste(rgb,(x,y));d.text((x+12,y+thumbh+7),f'{t:.2f} seconds',fill='white')
cap.release();dest=args.out or args.video.with_suffix('.contact.jpg');sheet.save(dest,quality=92)
print(json.dumps({'source':str(args.video),'frames':n,'fps':fps,'duration':n/fps,'sheet':str(dest)}))
