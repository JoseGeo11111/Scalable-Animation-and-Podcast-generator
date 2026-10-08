"""Finish three real local Wan clips as a 15-second, caption-optional test.

Usage: python finish_local_test.py PEPPER.mp4 PORT.mp4 SAILING.mp4
CPU FFmpeg/libx264 only. Normal Lanczos resizing, not AI super-resolution.
"""
from pathlib import Path
import argparse, hashlib, json, subprocess, struct

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs'
MASTER=OUT/'kerala-local-gpu-test.mp4'
BASE='kerala-local-gpu-test'
SENTENCES=[
    (.25,3.95,"For centuries, pepper drew ships to Kerala’s coast.",
     "നൂറ്റാണ്ടുകളായി, കുരുമുളക് തേടി കപ്പലുകൾ കേരളത്തിന്റെ തീരത്തെത്തി."),
    (3.95,7.75,"But cargo wasn’t the only thing crossing this ocean.",
     "എന്നാൽ ഈ സമുദ്രം കടന്നെത്തിയത് ചരക്കുകൾ മാത്രമായിരുന്നില്ല."),
    (7.75,12.65,"Faith travelled too—and Christianity was here long before the Portuguese.",
     "വിശ്വാസവും കടന്നെത്തി. പോർച്ചുഗീസുകാർ വരുന്നതിനും ഏറെ മുമ്പേ ക്രിസ്തുമതം ഇവിടെയുണ്ടായിരുന്നു."),
]

def run(args):
    p=subprocess.run([str(a) for a in args],text=True,capture_output=True)
    if p.returncode:
        raise RuntimeError(f'Command failed ({p.returncode}): {args[0]}\n{p.stderr[-12000:]}')
    return p

def probe(path,count=False):
    args=['ffprobe','-v','error']
    if count:args+=['-count_frames']
    args+=['-show_streams','-show_format','-of','json',path]
    return json.loads(run(args).stdout)

def stamp(seconds,separator):
    ms=round(seconds*1000)
    return f'{ms//3600000:02}:{ms//60000%60:02}:{ms//1000%60:02}{separator}{ms%1000:03}'

def write_subtitles():
    for language,column in [('en',2),('ml',3)]:
        srt=[];vtt=['WEBVTT','']
        for i,row in enumerate(SENTENCES,1):
            start,end=row[:2];text=row[column]
            srt += [str(i),f'{stamp(start,",")} --> {stamp(end,",")}',text,'']
            vtt += [f'{stamp(start,".")} --> {stamp(end,".")}',text,'']
        (OUT/f'{BASE}.{language}.srt').write_text('\n'.join(srt),encoding='utf-8')
        (OUT/f'{BASE}.{language}.vtt').write_text('\n'.join(vtt),encoding='utf-8')

def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def subtitle_tracks_off(path):
    """Disable subtitle tkhd enabled flags that the MP4 muxer auto-enables.

    Tracks remain in the file and are selectable by the player. No media bytes
    or atom sizes change. MP4 track_enabled is the container's default flag.
    """
    subtitle_ids={int(s['id'],0) for s in probe(path)['streams'] if s['codec_type']=='subtitle'}
    data=bytearray(path.read_bytes())
    def boxes(start,end):
        while start+8<=end:
            size,kind=struct.unpack_from('>I4s',data,start); header=8
            if size==1:size=struct.unpack_from('>Q',data,start+8)[0];header=16
            if size==0:size=end-start
            if size<header or start+size>end:raise ValueError('Invalid MP4 atom bounds')
            yield kind,start+header,start+size
            start+=size
    count=0
    for kind,begin,end in boxes(0,len(data)):
        if kind!=b'moov':continue
        for kind,tbegin,tend in boxes(begin,end):
            if kind!=b'trak':continue
            tkhd=None;handler=None
            for child,cstart,cend in boxes(tbegin,tend):
                if child==b'tkhd':tkhd=cstart
                if child==b'mdia':
                    for sub,sstart,send in boxes(cstart,cend):
                        if sub==b'hdlr':handler=bytes(data[sstart+8:sstart+12])
            track_id=(int.from_bytes(data[tkhd+(20 if data[tkhd]==1 else 12):tkhd+(24 if data[tkhd]==1 else 16)],'big') if tkhd is not None else None)
            # QuickTime chapter tracks also use a text handler; leave those alone.
            if track_id in subtitle_ids and handler in (b'sbtl',b'subt',b'text'):
                flags=int.from_bytes(data[tkhd+1:tkhd+4],'big') & ~1
                data[tkhd+1:tkhd+4]=flags.to_bytes(3,'big');count+=1
    if count!=2:raise RuntimeError(f'Expected2 subtitle tracks, found{count}')
    path.write_bytes(data)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pepper',type=Path)
    parser.add_argument('port',type=Path)
    parser.add_argument('sailing',type=Path)
    args=parser.parse_args()
    clips=[args.pepper.resolve(),args.port.resolve(),args.sailing.resolve()]
    for clip in clips:
        if not clip.is_file():raise FileNotFoundError(clip)
    sources=[]
    for label,path in zip(['pepper','port','sailing'],clips):
        info=probe(path,True)
        video=next(s for s in info['streams'] if s['codec_type']=='video')
        duration=float(video.get('duration',info['format'].get('duration',0)))
        if duration<4.999:raise ValueError(f'{label} is only {duration}s; at least five real seconds required')
        sources.append({'shot':label,'path':str(path),'duration':duration,'width':video['width'],'height':video['height'],'fps':video['avg_frame_rate'],'frames':video.get('nb_read_frames'),'sha256':sha256(path)})
    audio=OUT/'local-gpu-audio/mix.wav'
    if not audio.is_file():raise FileNotFoundError(audio)
    OUT.mkdir(exist_ok=True)
    write_subtitles()
    filters=[]
    for i in range(3):
        filters.append(f'[{i}:v]trim=start=0:duration=5,setpts=PTS-STARTPTS,scale=1920:1080:flags=lanczos:force_original_aspect_ratio=increase:out_color_matrix=bt709,crop=1920:1080,setsar=1,fps=24,format=yuv420p[v{i}]')
    filters.append('[v0][v1][v2]concat=n=3:v=1:a=0[vout]')
    command=['ffmpeg','-hide_banner','-y','-filter_complex_threads','3']
    for clip in clips:command+=['-i',clip]
    command+=['-i',audio,'-i',OUT/f'{BASE}.en.srt','-i',OUT/f'{BASE}.ml.srt',
              '-filter_complex',';'.join(filters),'-map','[vout]','-map','3:a:0','-map','4:s:0','-map','5:s:0',
              '-t','15','-c:v','libx264','-preset','medium','-crf','18','-threads','6','-pix_fmt','yuv420p',
              '-color_primaries','bt709','-color_trc','bt709','-colorspace','bt709','-color_range','tv',
              '-bsf:v','h264_metadata=colour_primaries=1:transfer_characteristics=1:matrix_coefficients=1',
              '-c:a','aac','-b:a','256k','-ar','48000','-ac','2','-c:s','mov_text',
              '-metadata:s:s:0','language=eng','-metadata:s:s:0','title=English',
              '-metadata:s:s:1','language=mal','-metadata:s:s:1','title=Malayalam',
              '-disposition:s:0','0','-disposition:s:1','0',
              '-metadata','title=Kerala — Local GPU Animation Test',
              '-metadata','comment=Built-in cloud ImageGen keyframes; local Wan animation; local CPU Kokoro narration and original procedural score/Foley. Standard Lanczos upscale; subtitles approximate and pending native review.',
              '-movflags','+faststart',MASTER]
    print('Rendering CPU-only 15-second master...',flush=True)
    run(command)
    subtitle_tracks_off(MASTER)
    poster=OUT/f'{BASE}-poster.jpg'
    run(['ffmpeg','-hide_banner','-y','-ss','2.5','-i',MASTER,'-frames:v','1','-q:v','2',poster])
    final=probe(MASTER,True)
    video=next(s for s in final['streams'] if s['codec_type']=='video')
    audio_info=next(s for s in final['streams'] if s['codec_type']=='audio')
    subtitles=[s for s in final['streams'] if s['codec_type']=='subtitle']
    assertions={
        'duration_15s':abs(float(final['format']['duration'])-15)<.025,
        '1920x1080':video['width']==1920 and video['height']==1080,
        '24fps':video['avg_frame_rate']=='24/1',
        '360frames':int(video['nb_read_frames'])==360,
        'audio_stereo_48k':audio_info['channels']==2 and audio_info['sample_rate']=='48000',
        'two_subtitle_tracks':len(subtitles)==2,
        'subtitles_default_off':all(s['disposition']['default']==0 and s['disposition']['forced']==0 for s in subtitles),
        'bt709':all(video.get(key)=='bt709' for key in ['color_space','color_transfer','color_primaries']),
    }
    decode=run(['ffmpeg','-hide_banner','-v','error','-xerror','-i',MASTER,'-map','0:v:0','-map','0:a:0','-f','null','-'])
    assertions['full_audio_video_decode']=decode.returncode==0
    manifest={
        'master':str(MASTER),'poster':str(poster),'duration_seconds':15,'sources_preserved':sources,
        'edit':'First five seconds of pepper, port, sailing; hard cuts at5s and10s. No repeated frames beyond normal fixed-frame-rate conversion.',
        'resolution_note':'Source geometry recorded per shot above. Ordinary aspect-preserving Lanczos upscale and centered crop to1920x1080; not AI upscaling or native1080 generation.',
        'provenance':'Built-in cloud ImageGen keyframes + local Wan animation + CPU Kokoro af_heart narration + original procedural score and Foley.',
        'caption_note':'Three approximate sentence-level cues. Timing is estimated, not forced-aligned. Malayalam translation and Kerala pronunciation pending native review. Both subtitle tracks default off.',
        'finishing':'CPU libx264 CRF18,24fps; AAC256kbps; two mov_text tracks; no NVENC/GPU used for this finishing stage.',
        'sha256':sha256(MASTER),'qa':assertions,'ffprobe':final,
    }
    (OUT/f'{BASE}-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'master':str(MASTER),'qa':assertions},indent=2),flush=True)
    if not all(assertions.values()):raise RuntimeError('One or more final QA checks failed; inspect manifest')

if __name__=='__main__':main()
