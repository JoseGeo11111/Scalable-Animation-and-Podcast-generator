"""Write voice delivery provenance; CPU/file IO only."""
import json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'outputs/film15/audio/voice'
manifest=json.loads((OUT/'durations.json').read_text(encoding='utf-8'))
audit=json.loads((OUT/'malayalam-asr-audit.json').read_text(encoding='utf-8'))
licenses=[
 {'component':'Svara-TTS v1','publisher':'Kenpath','license_declared_by_publisher':'Apache-2.0','url':'https://huggingface.co/kenpath/svara-tts-v1','revision':'db8a02fc1e4eab827ff6dda5bed3b56d4d2dd51e','note':'Model tree lists Llama3.2/Orpheus ancestry. Publisher declaration is recorded; this is not an exhaustive audit of upstream model or training-data terms.'},
 {'component':'SNAC24kHz','publisher':'Hubert Siuzdak','license_declared_by_publisher':'MIT','url':'https://huggingface.co/hubertsiuzdak/snac_24khz','revision':'d73ad176a12188fcf4f360ba3bf2c2fbbe8f58ec'},
 {'component':'Malayalam Whisper Small R-MFT','publisher':'Adalat AI','license_declared_by_publisher':'Apache-2.0','url':'https://huggingface.co/adalat-ai/whisper-small-ml-rmft','revision':'c397b8070ffc8ec3ceec17b7b2e4cfcc84f4b462','citation':'Vividh-ASR, Juvekar et al.,2026, https://arxiv.org/abs/2605.13087'}
]
report={
 'date':'2026-10-06','status':'Full narration generated; production draft requiring native listening review',
 'local_only':True,'voice_cloning':False,'voice':'Malayalam (Female) publisher preset',
 'hardware':'RTX4070TiSUPER16GB','runtime':'Windows, Python3.11.17, PyTorch2.10.0+cu130, Transformers4.57.6',
 'inference':{'dtype':'bfloat16','attention':'SDPA','batch_size':6,'temperature':.65,'top_p':.9,'top_k':50,'repetition_penalty':1.1,'maximum_new_tokens':2600,'seeds':[4070,4071,4072],'framing_source':'https://huggingface.co/spaces/kenpath/svara-tts/blob/main/app.py'},
 'delivery':{k:v for k,v in manifest.items() if k!='scenes'},
 'audio_format':{'sample_rate':24000,'channels':1,'subtype':'PCM_24'},
 'performance':{'original60_generation_wall_seconds':774.34,'first69_retake_utterances_wall_seconds':234.6,'scene45_four_utterances_wall_seconds':22.37,'scene30_two_date_clauses_wall_seconds':12.54,'combined_peak_observed_vram_approx_gb':13.2,'oom_observed':False,'note':'Wall times include model loading; some generation overlapped local ASR. Initial discarded experiments excluded.'},
 'retaken_scenes':[s['id'] for s in manifest['scenes'] if s.get('retake')],
 'retake_reason':'Dates, omissions, negations, repeats, historical names, approved scene45 editorial cleanup and closing emotional message.',
 'canonical_script_sha256':hashlib.sha256((ROOT/'outputs/production-source/film15-script.json').read_bytes()).hexdigest(),
 'qa':{'scene_count':audit['count'],'mean_cer':audit['mean_cer'],'median_cer':audit['median_cer'],'flags_over_0_15':audit['flagged'],'independent_asr_without_script_prompt':True,'native_human_listening_approval':False,'historian_approval':False,'no_capped_generations':all(not s['capped'] for s in manifest['scenes']),'verified_actual_audio_duration':True,'note':audit['note']},
 'known_limits':['Rare proper names including Muziris and Gothuruth remain ambiguous in some ASR transcriptions.','Automatic CER is not word accuracy or a listening-quality score.','Canonical subtitles retain correct names; this alone does not establish audible pronunciation.','Final tempo-adjusted mix quality is owned by the mixer; these WAVs are unretimed.'],
 'licenses':licenses,'license_checked_date':'2026-10-06',
 'cloud_fallback':{'used':False,'note':'A Microsoft EdgeTTS test was rejected by automatic approval review before script transmission; no workaround or cloud synthesis followed.'},
 'scenes':[{**s,'wav_sha256':hashlib.sha256(Path(s['file']).read_bytes()).hexdigest(),'asr':a} for s,a in zip(manifest['scenes'],audit['scenes'])]
}
(OUT/'final-voice-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
text=f'''Malayalam narration:60 final scenes, local Svara-TTS v1 female preset; no voice cloning.
Natural speech:{manifest['total_duration']:.3f}s. Canonical script:{manifest['canonical_word_count']}words.
Tempo proposal:{manifest['proposed_tempo_factor_for_842_2_seconds']:.8f}x to842.2s;83.93→110.85canonical words/minute.
18scenes received purposeful retakes. All60 WAV durations and uncapped-generation metadata verified.
Independent local Malayalam ASR:mean CER{audit['mean_cer']:.2%}, median{audit['median_cer']:.2%}; no scenes above15%. This is not listening certification or word accuracy.
Native human listening review remains outstanding, particularly rare place names.1599/1817/1819/1821dates and key negations were semantically checked after retakes.
Publisher licenses:Svara Apache2; SNAC MIT; Adalat ASR Apache2. Svara lists Llama/Orpheus ancestry; upstream terms have not received an exhaustive legal audit.
Exact sources, revisions, generation parameters, final WAV hashes, per-scene transcript audits and spoken variants:final-voice-report.json.
GPU released; no further voice GPU processes running.
'''
(OUT/'final-voice-report.txt').write_text(text,encoding='utf-8')
print('Saved final-voice-report.json and final-voice-report.txt')
