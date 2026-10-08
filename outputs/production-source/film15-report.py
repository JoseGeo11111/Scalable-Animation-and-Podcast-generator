"""Write the final handoff from measured production records, without invented benchmarks."""
import json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs';SRC=OUT/'production-source';WORK=ROOT/'work/film15'
qa=json.loads((OUT/'film15/final-qa.json').read_text(encoding='utf8'))
timeline=json.loads((SRC/'film15-timeline.json').read_text(encoding='utf8'))
selected={s['scene_id'] for s in timeline['shots'] if s['kind']=='video'}
graphic_ids={s['scene_id'] for s in timeline['shots'] if s['kind']!='video'}
queue=json.loads((SRC/'film15-video-queue.json').read_text());reports=[]
for shot in queue:
    if shot['scene_id'] not in selected:continue
    r=json.loads((WORK/(shot['id']+'-report.json')).read_text());assert r['success'];reports.append(r)
maxmem=0;energy=0
for r in reports:
    previous=None
    for sample in r['telemetry']:
        try:
            mem,util,power,temp=[float(x.strip()) for x in sample['gpu'].split(',')]
            maxmem=max(maxmem,mem)
            if previous:energy+=(sample['elapsed']-previous[0])*(power+previous[1])/2/3600000
            previous=(sample['elapsed'],power)
        except (ValueError,KeyError):pass
seconds=sum(r['seconds'] for r in reports)
stats={'successful_scene_jobs':len(reports),'generation_seconds_sum':seconds,'highest_sampled_gpu_memory_mib':maxmem,'approximate_video_generation_gpu_kwh':energy,'samplers':sorted(set(r.get('sampler','unipc') for r in reports))}
(OUT/'film15/render-benchmark.json').write_text(json.dumps(stats,indent=2))
mix=qa['mix']
text=f'''ROOTS AND THE SEA / വേരുകളും കടലും
15-minute Malayalam documentary pilot

DELIVERY
kerala-roots-and-sea-15min.mp4:1920x1080,30fps,27,000frames,900seconds.
Stereo48kHz AAC, optional Malayalam and English meaning captions, six chapters.
kerala-roots-and-sea-podcast.mp3:the complete soundtrack without picture.
kerala-roots-and-sea-storyboard.html:60scene boards, Malayalam narration, English meaning summaries, purposes and source links.
kerala-roots-and-sea-chapters.txt:chapter timestamps.

WHAT WAS MADE
60scenes,{len(selected)}locally generated moving cartoon sequences,{len(graphic_ids)}animated sourced map/evidence sequences, brief animated title/credits.
29original cloud ImageGen illustrations were created for the art direction; not all appear in the final cut. Character animation was generated locally with Wan2.2TI2V5B/FastWan on the RTX4070TiSuper.
Native animation is1280x704 at24fps, subsequently interpolated with RIFE4.26 within each edit shot and conventionally resized to1080p. This is not native1080p generation or30fps model inference. Map/evidence motion is natively rendered30fps.
Malayalam voice uses local Svara-TTS v1 female preset, with18scene retakes and independent Malayalam ASR checks. No person's voice was cloned.
16effects from local MOSS-SoundEffect2.0 and six instrumental themes from local ACE-Step1.5. The score is Kerala-inspired, not a claim of authentic traditional performance. Effects illustrate scenes; they are not archival/location recordings.

MEASURED CHECKS
All final master checks:{json.dumps(qa['checks'])}
Master soundtrack:{mix['integrated_lufs']:.2f}LUFS;{mix['true_peak_dbtp']:.2f}dBTP. All60spoken segments sample-aligned, no overlaps or overruns.
Natural narration:{qa['source_voice_seconds']:.3f}s; pitch-preserving tempo{qa['voice_tempo']:.6f}x; edited narration{qa['retimed_voice_seconds']:.1f}s. Approximately111canonical Malayalam words/minute after retiming.
Successful video-generation jobs:{len(reports)}; summed job time:{seconds/60:.1f}minutes. This excludes setup, research, art generation, rejected tests, sound, interpolation and final encoding.
Highest5second-sampled GPU memory:{maxmem:.0f}MiB. Approximate GPU energy during those video jobs:{energy:.3f}kWh, excluding the rest of the computer and other production stages. Multiply energy by your electricity tariff; no cloud-render purchase was initiated.
File SHA256:{qa['sha256']}

STORY AND EVIDENCE
The narrative moves from pepper and maritime exchange through tradition, documentary evidence, local arts, church authority, education, printing, care and shared belonging. Pride comes from participation and agency, without exclusive credit for Kerala's development.
52CE is presented as tradition, separately from documented later evidence. Cosmas's Male church and Calliana bishop are distinguished. Pattanam/Muziris identification remains qualified. Modern map geometry is orientation, not an exact ancient shoreline. Illustrated people/buildings are composites.
Maps:Natural Earth public-domain geometry; geoBoundaries/DataMeet-ECI Kerala geometry with attribution; GeoNames town coordinates. The copper-plate image is the cited1928published reproduction, not AI-generated evidence.
Sources and scene-specific IDs:production-source/film15-script.json.
Storyboarding research:production-source/film15-storyboard-research.txt, applying Pixar/Khan storytelling and the Academy's animation planning guide.

QUALITY LIMITS
This is an AI-assisted pilot. Native Malayalam listening/editorial approval remains outstanding, especially rare names. ASR is a diagnostic tool and does not certify pronunciation, performance or historical accuracy. English captions convey meaning and are not a certified translation; subtitle timing is editorial, not forced alignment.
Sampled animation findings are recorded separately in film15/animation-visual-qa.json. Technical frame/decode checks do not establish flawless animation.

REPRODUCIBLE MATERIAL
production-source/film15-render.py,film15_dmd_scheduler.py,film15-interpolate.py,film15-timeline.py,film15-finish.py.
production-source/film15_graphics.py,film15_voice.py,film15-mix_film_audio.py,film15-clarity-mix.py. The clarity pass follows the base mixer; original stems are retained in audio/mixes/initial.
film15/audio/voice/final-voice-report.json contains audio hashes, transcripts and retakes.
film15/audio/generation-report.json and mix-report.json contain sound models, settings, cues and measurements.
production-source/film15-dmd-scheduler-research.txt documents sampler equations and pinned upstream sources.
Local model/runtime caches remain under work; the delivered movie does not require those runtimes to play.

LICENSING / PRIMARY PROJECTS
Wan2.2 and FastWan weights:Apache2 declarations. WanGP wrapper:WanGP Community License2.0, not an unrestricted OSI open-source license.
https://github.com/Wan-Video/Wan2.2
https://huggingface.co/FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers
https://github.com/deepbeepmeep/Wan2GP
RIFE:MIT https://github.com/hzwer/Practical-RIFE
MOSS:Apache2 https://huggingface.co/OpenMOSS-Team/MOSS-SoundEffect-v2.0
ACE-Step:MIT https://github.com/ace-step/ACE-Step-1.5
Svara publisher:Apache2; SNAC:MIT; voice report identifies upstream ancestry and model revisions. Publisher labels are not an exhaustive audit of every upstream term.
Font:Noto Sans Malayalam, SIL OFL1.1.
'''
text+='\nVISUAL ATTRIBUTION\n'+(SRC/'film15-visual-credits.txt').read_text(encoding='utf8')
(OUT/'kerala-roots-and-sea-production-notes.txt').write_text(text,encoding='utf8')
print(json.dumps(stats,indent=2))
