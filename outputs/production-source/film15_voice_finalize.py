"""Finalize reproducible voice QA metadata after purposeful retakes."""
import json, statistics
from pathlib import Path
import soundfile as sf

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/film15/audio/voice'
film=json.loads((ROOT/'outputs/production-source/film15-script.json').read_text(encoding='utf-8'))
records=[]; audits=[]
for scene in film['scenes']:
    f=OUT/(scene['id']+'.wav')
    e=json.loads(f.with_suffix('.json').read_text(encoding='utf-8'))
    info=sf.info(f)
    assert info.samplerate==24000 and info.channels==1 and info.frames>0
    assert abs(info.duration-e['duration'])<1/24000
    assert e['text']==scene['narration_ml'] and not e['capped']
    assert e['voice']=='Malayalam (Female)' and e['v1_prompt_framing'] and e['local']
    records.append(e)
    a=json.loads(f.with_suffix('.ml-asr.json').read_text(encoding='utf-8'))
    assert a['expected']==scene['narration_ml']
    audits.append(a)
total=sum(e['duration'] for e in records)
words=sum(len(s['narration_ml'].split()) for s in film['scenes'])
spoken_words=sum(sum(len(u['spoken_text'].split()) for u in e['utterances']) if e.get('utterances') else len(e['text'].split()) for e in records)
factor=total/842.2
assert factor<=1.35, f'Voice exceeds approved tempo budget: {factor}'
summary={
    'scenes':records,'total_duration':total,'complete':True,
    'model':'kenpath/svara-tts-v1','voice':'Malayalam (Female)',
    'canonical_word_count':words,'spoken_variant_whitespace_word_count':spoken_words,
    'rate_basis':'Canonical script whitespace word count, kept consistent across pronunciation spacing variants.',
    'natural_words_per_minute':words*60/total,
    'proposed_tempo_factor_for_842_2_seconds':factor,
    'proposed_words_per_minute':words*60/842.2,
    'note':'Natural WAVs remain unretimed. Timeline/mixer applies pitch-preserving tempo.'
}
(OUT/'durations.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
audit={'count':len(audits),'scenes':audits,'flagged':[r['file'] for r in audits if r['cer']>.15],
       'mean_cer':statistics.mean(a['cer'] for a in audits),
       'median_cer':statistics.median(a['cer'] for a in audits),
       'note':'ASR was independent, with no script prompt. CER includes spelling and spacing variants; flags require semantic review. This is not native human listening approval.'}
(OUT/'malayalam-asr-audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in summary.items() if k!='scenes'},indent=2))
print('ASR',audit['mean_cer'],audit['median_cer'],'flags',len(audit['flagged']))
