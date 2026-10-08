# Rebuild and handoff

This package preserves the completed production and its source records. Playback requires only the MP4 or MP3. Regeneration requires separately downloaded upstream models and compatible environments.

1. Extract the release media into this repository root, retaining `outputs/film15/` and `outputs/production-source/` structure.
2. Read the voice, sound, DMD and environment notes. Provision the original separate GPU and voice environments; respect upstream terms. No model weights or virtual environments are bundled.
3. Rebase manifest paths to your workspace. Historical reports are evidence records, not executable configuration.
4. Keyframe-to-video generation uses `film15-render.py`, the queue JSON and the DMD adapter. Source native clips and all model caches are excluded; regenerate before interpolation. Expect hardware/runtime differences.
5. Generate or reuse the included voice/effect/score assets. Build the timeline, run the base mixer, clarity pass, and audio selection. Preserve original stems to avoid applying ducking twice.
6. Render graphics, apply selected edit overrides, interpolate each video shot with RIFE, and run the finalizer. Check each script's arguments before running it.
7. Finalizer and report scripts expect prior pipeline outputs; do not treat them as an installer. The supervisor and edit lock were coordination tools for the original run.

No end-to-end clean-machine rebuild has been validated. Final delivery technical validation and representative spot checks are included.
