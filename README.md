# Roots and the Sea · വേരുകളും കടലും

**A Malayalam documentary pilot and a documented production workflow for a single RTX 4070 Ti Super.**

![Film artwork](assets/poster.jpg)

From one peppercorn to a coast connected to the world: a 15-minute film about Kerala Christian history, memory, cultural contribution and shared belonging. Designed to work as both a visual documentary and an audio-only podcast.

**Status:** completed pilot, 7 October 2026. This is a production case study and source handoff, not a turnkey software product or an independently certified historical film.

## Product at a glance

| Item | Delivered |
|---|---|
| Film | 15:00, 1920 × 1080, H.264, 30 fps, 27,000 frames |
| Language | Malayalam narration; optional Malayalam and English meaning subtitles |
| Structure | 60 scenes, six chapters, 49 generated animated sequences, 11 map/evidence scenes |
| Audio | Stereo 48 kHz; podcast MP3; six original generated score themes and 16 generated effects |
| Art direction | 29 original cloud-generated illustrations created; selected assets animated locally |
| Hardware | RTX 4070 Ti Super, 16 GB VRAM; Windows production environment |
| Verification | Duration, frame count, frame rate, resolution, audio format, subtitle options and complete decode passed |

## View and download

Open [the designed product sheet](PRODUCT-SHEET.html), [production notes](kerala-roots-and-sea-production-notes.txt), and [scene overview](assets/scene-overview.jpg).

The [pilot release](https://github.com/JoseGeo11111/Scalable-Animation-and-Podcast-generator/releases/tag/pilot-v1) contains the complete media bundle, a smaller source ZIP, and SHA-256 checksums. The complete bundle includes `kerala-roots-and-sea-15min.mp4`, the podcast, subtitles, storyboard, selected artwork, audio assets, and this source package. Large media is kept outside Git history.

## Why this pilot exists

An affordable channel format that makes Kerala history emotionally engaging while distinguishing documentary evidence from community tradition. The story progresses through maritime exchange, Saint Thomas traditions, later documentary evidence, cultural arts, education, printing, healthcare and shared Kerala identity. Every scene has a stated factual or emotional purpose in the scene manifest.

The film identifies 52 CE as tradition, qualifies Pattanam/Muziris identification, and treats modern maps as geographical orientation rather than exact ancient shorelines. Illustrated characters and settings are composites. Kerala's development is presented as a shared achievement.

## Production stack actually used

| Stage | Tool | Role |
|---|---|---|
| Artwork | Built-in cloud ImageGen | Original cartoon keyframes |
| Local animation | [Wan2.2](https://github.com/Wan-Video/Wan2.2), [FastWan](https://huggingface.co/FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers), [WanGP](https://github.com/deepbeepmeep/Wan2GP) | Image-to-video; corrected DMD sampler adapter |
| Motion finishing | [RIFE](https://github.com/hzwer/Practical-RIFE), FFmpeg | Shot-local interpolation to 30 fps, resize, edit and encode |
| Narration | Kenpath Svara-TTS v1, SNAC | Local Malayalam female preset; no personal voice clone |
| Speech checks | adalat-ai/whisper-small-ml-rmft | Independent local ASR diagnostics |
| Effects | [MOSS-SoundEffect 2.0](https://huggingface.co/OpenMOSS-Team/MOSS-SoundEffect-v2.0) | Local ambience and Foley generation |
| Score | [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5) | Six local instrumental themes |
| Maps and evidence | Natural Earth, geoBoundaries, GeoNames, credited historical/museum images | Sourced geometry and evidence cards |
| Typography and graphics | Python, Pillow, HarfBuzz, FreeType, Noto Sans Malayalam | Native 30 fps graphics and Malayalam layout |

Native AI animation is **1280 × 704 at 24 fps**, interpolated and conventionally resized for the 1080p/30 fps delivery. This is not native 1080p AI inference or AI super-resolution. The soundtrack measurement is for the selected WAV master: **−16 LUFS, −2.27 dBTP**, before final lossy encoding.

## Measured production and budget

| Cost or resource | Evidence / treatment |
|---|---|
| New cloud-render purchase | None initiated for this pilot |
| Existing subscriptions and artwork | Used existing access; marginal subscription allocation was not metered |
| Selected animation jobs | 49 jobs; 116.6 minutes summed generation time |
| Animation GPU energy | Approximately 0.468 kWh sampled; excludes CPU, other stages, setup and rejected attempts |
| Electricity example | At an assumed $0.20/kWh: about $0.094 for those animation GPU jobs only |
| Sampled animation VRAM peak | 12,939 MiB; sound generation separately reached about 14.51 GiB allocated |
| Effects generation | 16 effects / 270 seconds of assets; about 6.8 minutes measured generation |
| Score generation | Six themes / 975 seconds of assets; about 43 seconds measured generation |
| Human/editorial work | Not metered or priced; native Malayalam review remains to be arranged |

**Repeat-episode budget formula:** allocated subscription cost + whole-system electricity + any licensed assets + editorial/review time. The pilot demonstrates a low incremental cash-cost route using owned hardware, not a substantiated zero-cost total or a fixed commercial quote. Hardware purchase, depreciation and human time are excluded from the measured GPU figure.

## Workflow and lessons

1. Research claims and distinguish evidence, tradition and interpretation.
2. Write Malayalam narration, scene purposes and a shot plan; lock narration timing.
3. Create restrained keyframes, then animate sequentially to fit 16 GB VRAM.
4. Generate narration, score and effects locally; use selective voice retakes.
5. Replace unsuitable generated evidence imagery with real credited photographs or maps.
6. Trim material ghosting, apply the DMD sampler fix, interpolate within shots, and mix with speech ducking.
7. Assemble the master and podcast; run technical validation and sparse representative visual checks.

The original prototype, cartoon revision and local GPU tests led to the final film. Major improvements were a sampler correction, a clean veranda retake, real Pattanam imagery, selected clean motion ranges, and stronger speech-focused mixing. Intermediate rejected renders and model caches are deliberately excluded from the handoff.

## Quality and licensing boundaries

Native Malayalam listening/editorial approval remains outstanding, especially rare names. ASR is diagnostic, not pronunciation certification. English captions convey meaning and are not a certified translation. Visual checks were sparse by request; unreviewed shots are identified in the QA record. Some generative motion/style artifacts may remain.

The score is Kerala-inspired, not a claim of authentic traditional performance. Sound effects are illustrative, not archival recordings. No blanket license is granted over all bundled materials: upstream model/runtime terms and third-party asset licenses apply separately. WanGP uses its Community License 2.0 rather than an unrestricted OSI open-source license. See [credits](outputs/production-source/film15-visual-credits.txt) and [production notes](kerala-roots-and-sea-production-notes.txt).

Scene 08 photograph: **KannanVM**, 5 January 2019, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Un_indentified_broken_pottery_found_Muziris_excavation_sites.jpg), [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Resized, cropped, animated and overlaid with typography. The adapted scene 08 visual is offered under CC BY-SA 4.0; no generative alteration to the pictured objects.

## Source handoff

`outputs/production-source/` contains scripts, scene/timing manifests, research, prompts and credits. `outputs/film15/` contains measured QA and sound reports. Original machine paths have been redacted to relative paths in shared text files.

These are **production snapshots**, not a clean one-command installer. Model/runtime environments, dependencies and caches must be provisioned separately; path-bearing manifests need rebasing to the new workspace. Consult the recorded environment/research notes and the [rebuild guide](REBUILD.md) before execution. Do not run the supervisor unchanged against an empty workspace.
