"""CPU-only sourced documentary graphics,1920x1080,30fps.

API: render('04',duration_frames,output_path); preview('04',path,progress=.5).
CLI: python film15_graphics.py --previews OR --scene 04 --frames 180 --output path.mp4
Malayalam is explicitly shaped by HarfBuzz and rasterized by FreeType (Pillow
RAQM is unavailable in this environment). No generated typography or geography.
"""
from pathlib import Path
from functools import lru_cache
import argparse,json,math,shutil,subprocess,time
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import uharfbuzz as hb
import freetype

ROOT=Path(__file__).resolve().parents[2]
SRC=Path(__file__).resolve().parent
ASSETS=ROOT/'outputs/assets'
FONT=ASSETS/'fonts/NotoSansMalayalam-Regular.ttf'
LATIN=ASSETS/'fonts/Nunito.ttf'
W,H,FPS=1920,1080,30
TEAL=(17,54,59);SEA=(25,69,74);LAND=(205,212,188);GOLD=(233,179,79)
CREAM=(247,239,220);MUTED=(170,193,188);INK=(26,55,59);RUST=(179,101,70)
SCENES=('04','08','14','16','17','35','38','48','52','55','56','printing1821','anna1959')
POINTS={
 'Kottayam':(76.52132,9.58692,1265911),
 'Tiruvalla':(76.57489,9.38160,1254335),
 'Chendamangalam':(76.23456,10.16322,8181689),
 'Kollam':(76.58469,8.88113,1259091),
}
PROVENANCE={
 'land':'Natural Earth 1:50m, public domain; actual source geometry from land.geojson',
 'kerala':'geoBoundaries IND ADM1 via DataMeet/ECI, CC BY2.5 India; modern orientation only',
 'points':'GeoNames IN.txt, CC BY4.0; named modern town reference areas, not monument footprints',
 'plates':'Quilon plates1and4,1928published reproduction, Commons public-domain scan, uploaded by Challiyan; image not redrawn',
 'point_source':'https://download.geonames.org/export/dump/IN.zip',
 'font':'Noto Sans Malayalam, SIL OFL1.1; HarfBuzz0.56.3 +FreeType explicit glyph shaping',
 'routes':'Indicative connections, not reconstructed vessel tracks',
 'anna_date':'1959 HighCourt appointment, not first1937judicial appointment; optional insert',
 'anna_date_source':'https://www.prd.kerala.gov.in/ml/node/276136 (Kerala official Information/Public Relations,27November2024) corroborates1959HighCourt appointment',
}

def ease(x):
    x=max(0,min(1,x));return x*x*(3-2*x)
def mix(a,b,t):return tuple(int(x+(y-x)*t) for x,y in zip(a,b))

@lru_cache(maxsize=1000)
def text_img(text,size,color=CREAM):
    """Shape full Malayalam syllables first, then draw exact glyph IDs."""
    data=FONT.read_bytes();facehb=hb.Face(data);font=hb.Font(facehb)
    font.scale=(size*64,size*64);hb.ot_font_set_funcs(font)
    buf=hb.Buffer();buf.add_str(text);buf.guess_segment_properties();hb.shape(font,buf)
    ft=freetype.Face(str(FONT));ft.set_pixel_sizes(0,size)
    penx=0;peny=0;glyphs=[];minx=0;miny=0;maxx=1;maxy=1
    for info,pos in zip(buf.glyph_infos,buf.glyph_positions):
        ft.load_glyph(info.codepoint,freetype.FT_LOAD_RENDER)
        bm=ft.glyph.bitmap
        x=round(penx+pos.x_offset/64+ft.glyph.bitmap_left)
        y=round(-peny-pos.y_offset/64-ft.glyph.bitmap_top)
        if bm.width and bm.rows:
            arr=np.array(bm.buffer,dtype=np.uint8).reshape(bm.rows,abs(bm.pitch))[:,:bm.width]
            glyphs.append((x,y,Image.fromarray(arr.copy())))
            minx=min(minx,x);miny=min(miny,y);maxx=max(maxx,x+bm.width);maxy=max(maxy,y+bm.rows)
        penx+=pos.x_advance/64;peny+=pos.y_advance/64
    maxx=max(maxx,round(penx));pad=3
    im=Image.new('RGBA',(maxx-minx+2*pad,maxy-miny+2*pad),(0,0,0,0))
    for x,y,g in glyphs:
        ink=Image.new('RGBA',g.size,(*color,255));ink.putalpha(g)
        im.alpha_composite(ink,(x-minx+pad,y-miny+pad))
    return im

def label(im,text,xy,size=42,color=CREAM,alpha=1,anchor='left'):
    ti=text_img(text,size,color)
    if alpha<.999:
        ti=ti.copy();ti.putalpha(ti.getchannel('A').point(lambda a:int(a*max(0,alpha))))
    x,y=xy
    if anchor=='center':x-=ti.width/2
    elif anchor=='right':x-=ti.width
    im.paste(ti,(int(x),int(y)),ti)

@lru_cache(maxsize=50)
def latin_font(size):return ImageFont.truetype(str(LATIN),size)
def latin(im,text,xy,size=24,color=MUTED):ImageDraw.Draw(im).text(xy,text,font=latin_font(size),fill=color)

@lru_cache(maxsize=4)
def backdrop(dark=True):
    a=np.zeros((H,W,3),dtype=np.uint8)
    for y in range(H):
        t=y/H;a[y,:,:]=mix((20,59,64),(10,35,40),t) if dark else mix((249,241,224),(230,220,198),t)
    return Image.fromarray(a)

def geometry_rings(obj):
    if obj['type']=='FeatureCollection':
        for f in obj['features']:yield from geometry_rings(f)
    elif obj['type']=='Feature':yield from geometry_rings(obj['geometry'])
    elif obj['type']=='Polygon':yield obj['coordinates'][0]
    elif obj['type']=='MultiPolygon':
        for p in obj['coordinates']:yield p[0]

@lru_cache(maxsize=3)
def rings(name):return list(geometry_rings(json.loads((SRC/name).read_text(encoding='utf-8'))))
def merc(lat):return math.degrees(math.log(math.tan(math.pi/4+math.radians(max(-80,min(80,lat)))/2)))

class Projection:
    def __init__(self,bbox,rect):
        lo,la,hi,ha=bbox;x,y,w,h=rect
        dy=merc(ha)-merc(la);dx=hi-lo
        self.scale=min(w/dx,h/dy);self.cx=(lo+hi)/2;self.cy=(merc(la)+merc(ha))/2
        self.px=x+w/2;self.py=y+h/2;self.bbox=bbox;self.rect=rect
    def __call__(self,lon,lat):return(self.px+(lon-self.cx)*self.scale,self.py-(merc(lat)-self.cy)*self.scale)

@lru_cache(maxsize=24)
def map_layer(bbox,rect,kerala=True):
    im=backdrop().copy();d=ImageDraw.Draw(im)
    pr=Projection(bbox,rect)
    # Coarse graticule makes geographic scale legible without political borders.
    for lon in range(-180,181,10):
        x,_=pr(lon,0)
        if 0<x<W:d.line((x,180,x,880),fill=(31,72,76),width=1)
    for lat in range(-60,71,10):
        _,y=pr(0,lat)
        if 180<y<880:d.line((0,y,W,y),fill=(31,72,76),width=1)
    for ring in rings('land.geojson'):
        lons=[p[0] for p in ring];lats=[p[1] for p in ring]
        if max(lons)<bbox[0]-3 or min(lons)>bbox[2]+3 or max(lats)<bbox[1]-3 or min(lats)>bbox[3]+3:continue
        poly=[pr(*p[:2]) for p in ring]
        d.polygon(poly,fill=LAND)
    if kerala:
        for ring in rings('kerala-boundary.geojson'):
            poly=[pr(*p[:2]) for p in ring]
            d.polygon(poly,fill=GOLD);d.line(poly+[poly[0]],fill=(255,221,140),width=2)
    # Keep all maps within a clean image field, leaving stable typography space.
    top=backdrop().crop((0,0,W,170));im.paste(top,(0,0))
    bot=backdrop().crop((0,900,W,H));im.paste(bot,(0,900))
    return im

def top(im,title,kicker='വേരുകളും കടലും',dark=True):
    c=CREAM if dark else INK
    label(im,kicker,(90,44),25,GOLD if dark else RUST)
    label(im,title,(88,99),52,c)

def footer(im,credit,mapnote=True,dark=True):
    c=MUTED if dark else (85,99,89)
    if mapnote:label(im,'ഇന്നത്തെ ഭൂപ്രകൃതി - ചരിത്രകാല അതിർത്തികളല്ല',(90,966),25,c)
    latin(im,credit,(93,1021),20,c)

def dot(im,xy,p,label_text=None,offset=(30,-30)):
    d=ImageDraw.Draw(im);x,y=xy;r=11
    d.ellipse((x-r,y-r,x+r,y+r),fill=GOLD,outline=CREAM,width=3)
    radius=20+18*(.5+.5*math.sin(p*math.pi*4))
    d.ellipse((x-radius,y-radius,x+radius,y+radius),outline=(132,149,104),width=2)
    if label_text:label(im,label_text,(x+offset[0],y+offset[1]),36)

def bezier(a,b,c,n=120):
    return [((1-t)**2*a[0]+2*(1-t)*t*b[0]+t*t*c[0],(1-t)**2*a[1]+2*(1-t)*t*b[1]+t*t*c[1]) for t in np.linspace(0,1,n)]

def route(im,pts,p,color=GOLD,width=6):
    end=max(2,min(len(pts),round(len(pts)*ease(p))))
    if p>0:ImageDraw.Draw(im).line(pts[:end],fill=color,width=width,joint='curve')
    return pts[end-1]

def scene04(p):
    box=(33,-7,96,32);rect=(70,175,1780,700)
    im=map_layer(box,rect).copy();pr=Projection(box,rect)
    top(im,'ലോകത്തേക്കു തുറന്ന തീരം')
    a=pr(76.1,10.3); b=pr(49,18); c=pr(40.5,16)
    route(im,bezier(a,pr(65,4),b),(p-.12)/.7)
    route(im,bezier(a,pr(59,-1),c),(p-.30)/.7,color=(159,201,183),width=4)
    dot(im,a,p)
    # Keep the name above the westward route; a fine leader anchors it to shore.
    ImageDraw.Draw(im).line((a[0]-10,a[1]-15,a[0]-82,a[1]-66,a[0]-150,a[1]-66),fill=CREAM,width=2)
    label(im,'കേരളം',(a[0]-264,a[1]-126),36)
    label(im,'അറേബ്യ',(b[0]-90,b[1]-85),34,INK)
    label(im,'ഇന്ത്യൻ മഹാസമുദ്രം',(980,720),40,MUTED,anchor='center')
    footer(im,'Natural Earth (public domain) · geoBoundaries / DataMeet (CC BY 2.5 IN) · routes: indicative connections')
    return im

def regional(p,title,town=None):
    box=(72.5,7.5,79.5,13.8);rect=(60,185,840,690)
    im=map_layer(box,rect).copy();pr=Projection(box,rect)
    # Right narrative panel masks geography outside the deliberately regional crop.
    im.paste(backdrop().crop((930,170,W,905)),(930,170))
    top(im,title)
    label(im,'കേരളം',(145,285),34,MUTED)
    if town:
        lon,lat,_=POINTS[town]
        names={'Kottayam':'കോട്ടയം','Tiruvalla':'തിരുവല്ല','Chendamangalam':'ചേന്ദമംഗലം'}
        dot(im,pr(lon,lat),p,names.get(town,town),(-220,20))
    return im,pr

def scene14(p):
    im,pr=regional(p,'നൂറ്റാണ്ടുകൾ മുമ്പേ')
    # Broad coastal focus, not an invented exact location for Cosmas's church.
    a=pr(75.15,12.0);b=pr(76.35,9.1)
    ImageDraw.Draw(im).line((a,b),fill=(255,220,140),width=8)
    label(im,'മലബാർ തീരം',(110,770),34)
    label(im,'ആറാം നൂറ്റാണ്ട്',(1040,270),53,GOLD)
    label(im,'കോസ്മാസിന്റെ രേഖ',(1040,360),34)
    d=ImageDraw.Draw(im);d.line((1060,520,1710,520),fill=(82,109,110),width=4)
    x=1060+650*ease((p-.15)/.65);d.line((1060,520,x,520),fill=GOLD,width=7)
    if p>.4:
        label(im,'1498',(1590,584),82,CREAM,alpha=ease((p-.4)*3))
        label(im,'പോർച്ചുഗീസ് വരവ്',(1360,708),34,MUTED,alpha=ease((p-.5)*3))
    footer(im,'Cosmas, Christian Topography III · H1 · broad regional identification, not an exact church site')
    return im

@lru_cache(maxsize=2)
def plate():return Image.open(ASSETS/'quilon-plates-1928-scan.jpg').convert('RGB')

@lru_cache(maxsize=1)
def pattanam_photo():
    return Image.open(ASSETS/'pattanam-pottery-kannanvm-2019.jpg').convert('RGB').resize((2400,1600),Image.Resampling.LANCZOS)

def scene08(p):
    im=backdrop(False).copy();top(im,'മണ്ണിൽ ശേഷിച്ച കടൽബന്ധങ്ങൾ',dark=False)
    src=pattanam_photo();w,h=1170,780
    # Real museum photograph: overview progresses to the lower-left sherd trays.
    # No object reconstruction, removal, relighting or generative alteration.
    z=1+.34*ease(p);cw=src.width/z;ch=src.height/z
    cx=src.width*(.50-.07*ease(p));cy=src.height*(.50+.10*ease(p))
    left=max(0,min(src.width-cw,cx-cw/2));upper=max(0,min(src.height-ch,cy-ch/2))
    detail=src.transform((w,h),Image.Transform.EXTENT,(left,upper,left+cw,upper+ch),resample=Image.Resampling.BICUBIC)
    im.paste(detail,(90,196))
    label(im,'പട്ടണം',(1330,248),73,RUST)
    label(im,'മൺപാത്രശകലങ്ങൾ',(1332,372),36,INK)
    label(im,'മ്യൂസിയം പ്രദർശനം',(1332,427),30,INK)
    ImageDraw.Draw(im).line((1330,556,1330+450*ease(p*1.6),556),fill=RUST,width=3)
    label(im,'മുസിരിസുമായുള്ള ബന്ധം',(1332,640),31,INK,alpha=ease((p-.15)*3))
    label(im,'തിരിച്ചറിയലിൽ',(1332,714),34,INK,alpha=ease((p-.25)*3))
    label(im,'ചർച്ച തുടരുന്നു',(1332,765),34,INK,alpha=ease((p-.25)*3))
    footer(im,'KannanVM / Wikimedia Commons · 5 Jan 2019 · CC BY-SA 4.0 · cropped + animated · Pattanam Museum pottery display',False,False)
    return im
def image_fit_pan(src,size,p,zoom0=1.,zoom1=1.08):
    w,h=size;z=zoom0+(zoom1-zoom0)*ease(p)
    scale=max(w/src.width,h/src.height)*z
    cropw=w/scale;croph=h/scale
    cx=src.width*(.48+.04*p);cy=src.height*(.46+.08*p)
    l=max(0,min(src.width-cropw,cx-cropw/2));t=max(0,min(src.height-croph,cy-croph/2))
    return src.transform((w,h),Image.Transform.EXTENT,(l,t,l+cropw,t+croph),resample=Image.Resampling.BICUBIC)

def scene16(p):
    im=backdrop(False).copy();top(im,'ചെമ്പിൽ നിലനിന്ന വാക്കുകൾ',dark=False)
    label(im,'849',(105,270),190,RUST)
    label(im,'കൊല്ലം',(115,535),54,INK)
    label(im,'തരിസാപ്പള്ളി',(115,634),43,INK)
    label(im,'ചെപ്പേടുകൾ',(115,699),43,INK)
    scan=plate();size=(1050,795)
    # Whole reproduction first, then mild zoom while preserving both plates.
    sf=min(size[0]/scan.width,size[1]/scan.height)*(1+.025*ease(p))
    scaled=scan.resize((round(scan.width*sf),round(scan.height*sf)),Image.Resampling.LANCZOS)
    im.paste(scaled,(780+(1050-scaled.width)//2,190+(795-scaled.height)//2))
    footer(im,'Quilon plates 1 & 4 · 1928 published reproduction · Wikimedia Commons / Challiyan · public domain',False,False)
    return im

def scene17(p):
    im=backdrop(False).copy();top(im,'രേഖപ്പെടുത്തിയ ബന്ധങ്ങൾ',dark=False)
    im.paste(image_fit_pan(plate(),(1720,620),p,1.0,1.18),(100,210))
    d=ImageDraw.Draw(im);d.line((100,854,1820,854),fill=(170,147,107),width=2)
    label(im,'അവകാശങ്ങൾ',(280,889),45,INK,alpha=ease(p*4))
    label(im,'വ്യാപാരം',(1050,889),45,INK,alpha=ease((p-.30)*3))
    footer(im,'Actual 1928 reproduction · labels summarize context; not a line-by-line translation · H2 / V1',False,False)
    return im

def scene35(p):
    im=backdrop().copy();top(im,'പിന്നീട് പല വഴികൾ')
    label(im,'1653',(120,410),108,GOLD)
    start=(440,550)
    ImageDraw.Draw(im).ellipse((start[0]-9,start[1]-9,start[0]+9,start[1]+9),fill=GOLD)
    for i,(y,col) in enumerate([(330,GOLD),(550,(140,195,173)),(770,(192,145,115))]):
        pts=bezier(start,(890,550),(1470,y))
        route(im,pts,(p-.10-i*.10)/.68,col,7)
    label(im,'പിന്നീടുള്ള നൂറ്റാണ്ടുകൾ',(1110,223),43)
    if p>.65:label(im,'ബന്ധങ്ങൾ - മാറ്റങ്ങൾ',(1070,845),37,MUTED,alpha=ease((p-.65)*4))
    footer(im,'H14 · conceptual branching across centuries; not a complete denominational family tree',False)
    return im

def scene38(p):
    im,pr=regional(p,'അറിവിന് പുതിയ വഴികൾ','Kottayam')
    label(im,'1817',(1030,272),155,GOLD)
    label(im,'സി എം എസ് കോളേജ്',(1025,518),45)
    label(im,'കോട്ടയം',(1030,608),36,MUTED)
    # A line physically extends from the anchored town to the date, giving spatial causality.
    a=pr(*POINTS['Kottayam'][:2]);pts=bezier(a,(860,a[1]),(1000,470));route(im,pts,(p-.15)/.7,width=3)
    footer(im,'CMS College institutional history · GeoNames 1265911 (town area, not campus footprint)')
    return im

def bed(im,x,y,col):
    d=ImageDraw.Draw(im);d.line((x,y-25,x,y+38),fill=col,width=6)
    d.rounded_rectangle((x+9,y,x+91,y+28),radius=5,fill=col)
    d.line((x+91,y+20,x+91,y+38),fill=col,width=6)
    d.ellipse((x+13,y-25,x+36,y-2),fill=col)

def scene48(p):
    im,pr=regional(p,'ഒരു ചെറിയ തുടക്കം','Tiruvalla')
    label(im,'1959',(1030,232),145,GOLD)
    label(im,'പുഷ്പഗിരി',(1030,454),48)
    count=min(8,max(1,int(ease((p-.10)/.70)*8)+1))
    for i in range(8):bed(im,1040+(i%4)*157,620+(i//4)*120,GOLD if i<count else (49,81,85))
    label(im,'8 കിടക്കകൾ',(1034,822),37,CREAM,alpha=ease((p-.25)*3))
    footer(im,'Pushpagiri institutional history: 1959 eight-bed clinic · GeoNames 1254335, town reference')
    return im

def scene55(p):
    im,pr=regional(p,'ഒരേ ഭൂമിയിലെ പല ഓർമ്മകൾ','Chendamangalam')
    label(im,'കോട്ടയിൽ കോവിലകം',(1020,255),43,GOLD)
    # Explicitly a list, not faux geolocated monument pins.
    items=['പള്ളി','ക്ഷേത്രം','മുസ്ലിം ആരാധനാലയം','യഹൂദ പൈതൃകം']
    for i,t in enumerate(items):
        a=ease((p-.05-i*.14)*4)
        y=390+i*105
        if a>0:
            ImageDraw.Draw(im).ellipse((1020,y+15,1032,y+27),fill=mix(TEAL,GOLD,a))
            label(im,t,(1060,y),38,alpha=a)
    footer(im,'Kerala Tourism / Muziris heritage · GeoNames 8181689 town area; exact monument locations not plotted')
    return im

def scene52(p):
    im=backdrop().copy();top(im,'ആഴമുള്ള സ്വന്തം ഇടം')
    # Milestones are deliberately evenly spaced: this is an ordered summary,
    # not a proportional scale or a claim of uninterrupted uniform continuity.
    nodes=[(300,'ആറാം നൂറ്റാണ്ട്','കോസ്മാസിന്റെ രേഖ'),(735,'849','കൊല്ലം ചെപ്പേടുകൾ'),(1170,'1817','സി എം എസ് കോളേജ്'),(1605,'1959','പുഷ്പഗിരിയുടെ തുടക്കം')]
    d=ImageDraw.Draw(im);d.line((300,540,1605,540),fill=(61,95,98),width=4)
    travel=ease((p-.06)/.88);d.line((300,540,300+1305*travel,540),fill=GOLD,width=7)
    for i,(x,date,caption) in enumerate(nodes):
        reveal=ease((p-i*.18)*5)
        if reveal>0:
            d.ellipse((x-13,527,x+13,553),fill=mix(TEAL,GOLD,reveal))
            label(im,date,(x,370),43 if i==0 else 81,GOLD,alpha=reveal,anchor='center')
            label(im,caption,(x,614),29,CREAM,alpha=reveal,anchor='center')
    label(im,'രേഖകൾ - കലകൾ - പഠനം - പരിചരണം',(960,820),40,MUTED,alpha=ease((p-.55)*3),anchor='center')
    footer(im,'H1 / H2 / H16 / H27 · selected chronological milestones; spacing is not proportional to elapsed years',False)
    return im

def scene56(p):
    im=backdrop(False).copy();top(im,'പൊതുജീവിതത്തിലെ സ്ത്രീ',dark=False)
    label(im,'അന്ന ചാണ്ടി',(125,260),104,INK)
    label(im,'1959',(1240,255),172,RUST)
    # Type is the evidence; no invented portrait, quote, courtroom or gavel.
    width=1560*ease((p-.05)/.7)
    ImageDraw.Draw(im).line((130,522,130+width,522),fill=RUST,width=5)
    label(im,'കേരള ഹൈക്കോടതി',(132,590),66,INK,alpha=ease((p-.1)*4))
    label(im,'ഇന്ത്യയിലെ ആദ്യ വനിതാ',(132,730),46,INK,alpha=ease((p-.35)*4))
    label(im,'ഹൈക്കോടതി ജഡ്ജി',(132,800),46,INK,alpha=ease((p-.45)*4))
    footer(im,'Kerala High Court: first woman judge, then High Court judge · Kerala I&PRD,27 Nov2024: appointment1959',False,False)
    return im

def insert(p,kind):
    im=backdrop(False).copy();ispress=kind=='printing1821'
    top(im,'അക്ഷരങ്ങൾക്ക് പുതിയ യാത്ര' if ispress else 'പൊതുജീവിതത്തിലെ സ്ത്രീ',dark=False)
    label(im,'1821' if ispress else '1959',(120,315),200,RUST)
    label(im,'സി എം എസ് അച്ചടിശാല' if ispress else 'അന്ന ചാണ്ടി',(880,330),54,INK)
    label(im,'കോട്ടയം' if ispress else 'കേരള ഹൈക്കോടതി',(880,448),42,INK)
    ImageDraw.Draw(im).line((880,570,880+830*ease(p),570),fill=RUST,width=5)
    label(im,'മലയാള അച്ചടിയുടെ ഒരു പ്രധാന അധ്യായം' if ispress else 'ഇന്ത്യയിലെ ആദ്യ വനിതാ ഹൈക്കോടതി ജഡ്ജി',(880,630),28,INK,alpha=ease((p-.2)*3))
    footer(im,'CMS College2026 prospectus; Malayalam printing existed earlier elsewhere' if ispress else 'Kerala High Court official history + Kerala I&PRD27Nov2024: High Court appointment1959',False,False)
    return im

def frame(scene_id,progress):
    sid=str(scene_id).zfill(2) if str(scene_id).isdigit() else str(scene_id)
    p=max(0,min(1,float(progress)))
    fn=globals().get('scene'+sid)
    if fn:return fn(p)
    if sid in ('printing1821','anna1959'):return insert(p,sid)
    raise ValueError(f'Unknown graphics scene {scene_id}; choose {SCENES}')

def preview(scene_id,output,progress=.5):
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    frame(scene_id,progress).save(output);return str(output)

def render(scene_id,duration_frames,output):
    n=int(duration_frames)
    if n<2:raise ValueError('duration_frames must be >=2')
    output=Path(output).resolve();output.parent.mkdir(parents=True,exist_ok=True)
    ff=shutil.which('ffmpeg')
    if not ff:raise RuntimeError('ffmpeg not found')
    cmd=[ff,'-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(FPS),'-i','-','-an','-c:v','libx264','-preset','fast','-crf','18','-threads','3','-pix_fmt','yuv420p','-color_primaries','bt709','-color_trc','bt709','-colorspace','bt709','-bsf:v','h264_metadata=colour_primaries=1:transfer_characteristics=1:matrix_coefficients=1','-movflags','+faststart',str(output)]
    start=time.time();proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,creationflags=0x08000000)
    try:
        for i in range(n):
            proc.stdin.write(frame(scene_id,i/(n-1)).tobytes())
        proc.stdin.close();rc=proc.wait()
    except BaseException:
        proc.kill();proc.wait();raise
    if rc:raise RuntimeError(f'ffmpeg exited {rc}')
    poster=output.with_suffix('.png');preview(scene_id,poster)
    result={'scene_id':str(scene_id),'frames':n,'fps':FPS,'duration_seconds':n/FPS,'size':[W,H],'render_seconds':round(time.time()-start,2),'output':str(output),'preview':str(poster),'provenance':PROVENANCE}
    output.with_suffix('.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--previews',action='store_true');ap.add_argument('--scene',default='04');ap.add_argument('--frames',type=int,default=180);ap.add_argument('--output',type=Path,default=ROOT/'outputs/film15/graphics/04.mp4');a=ap.parse_args()
    if a.previews:
        for sid in SCENES:print(preview(sid,ROOT/f'outputs/film15/graphics/previews/{sid}.png'))
    else:print(json.dumps(render(a.scene,a.frames,a.output),ensure_ascii=False))
