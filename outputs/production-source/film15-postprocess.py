"""Resume the already-authorized local film pipeline after native renders finish.
An editor-written lock coordinates GPU ownership and the final selected shots.
It is an internal production checkpoint, not a request for user permission.
"""
import json,time,subprocess,sys,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];WORK=ROOT/'work/film15';SRC=ROOT/'outputs/production-source'
PY=ROOT/'work/gpu-env/Scripts/python.exe';STATUS=WORK/'post-status.json'
def state(stage,**extra):
    STATUS.write_text(json.dumps({'stage':stage,'updated':datetime.datetime.now().isoformat(),**extra},indent=2),encoding='utf8')
    print(stage,flush=True)
def run(script,args,log):
    with (WORK/log).open('w',encoding='utf8') as f:
        p=subprocess.run([str(PY),'-u',str(SRC/script),*map(str,args)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,creationflags=0x08000000)
    if p.returncode:raise RuntimeError(f'{script} failed with {p.returncode}; inspect {log}')
try:
    state('waiting_for_native_queue')
    queue=json.loads((SRC/'film15-video-queue.json').read_text())
    while True:
        try:
            reports=[json.loads((WORK/(s['id']+'-report.json')).read_text()) for s in queue]
            complete=all(r.get('success') and r.get('sampler')=='dmd' for r in reports)
            log=(WORK/'dmd-production.log').read_text(errors='replace')
            if complete and 'BATCH_COMPLETE' in log:break
        except (OSError,json.JSONDecodeError):pass
        time.sleep(15)
    # The native process exits immediately after its completion marker.
    time.sleep(5)
    state('rendering_targeted_retakes')
    retakes=SRC/'film15-retake-queue.json'
    if retakes.exists():run('film15-render.py',[retakes,'--vae-tile','1024','--sampler','dmd'],'retakes.log')
    state('retakes_ready_waiting_for_edit_lock')
    while not (WORK/'edit-locked.json').exists():time.sleep(10)
    state('refreshing_timeline');run('film15-timeline.py',[],'final-timeline.log')
    state('interpolating_30fps');run('film15-interpolate.py',[SRC/'film15-timeline.json'],'interpolation.log')
    state('assembling_and_validating_master');run('film15-finish.py',[],'finish.log')
    state('writing_measured_report');run('film15-report.py',[],'report.log')
    state('complete',master=str(ROOT/'outputs/kerala-roots-and-sea-15min.mp4'))
except Exception as e:
    state('error',error=str(e));raise
