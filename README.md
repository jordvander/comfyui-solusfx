# ComfyUI Solus FX Nodes

Custom audio-reactive nodes by Solus FX.

- **◐ Audio Prompt Travel (Solus FX)** — builds per-frame travelling CONDITIONING from audio peak indices. Prompts switch on the beat, with a configurable crossfade (`blend_frames`). Accepts a `prompts_override` string input (e.g. from auto-captioning) that takes priority over the typed prompt list. Works on ComfyUI 0.29+ (handles missing pooled_output).
- **◐ Captions → Prompts (Solus FX)** — flattens vision-model captions (e.g. Florence2) into a one-per-line prompt list, with optional prefix/suffix styling and word cap. Feeds Audio Prompt Travel's `prompts_override`.

Install: clone into `ComfyUI/custom_nodes/`. No extra dependencies.
