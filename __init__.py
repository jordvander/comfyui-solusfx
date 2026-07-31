"""Solus FX custom nodes for ComfyUI — audio-driven prompt travel."""
import torch


def _parse_peaks(peaks_index):
    if isinstance(peaks_index, (list, tuple)):
        peaks_index = peaks_index[0] if peaks_index else ""
    if isinstance(peaks_index, str):
        return [int(x.strip()) for x in peaks_index.split(",") if x.strip().lstrip("-").isdigit()]
    return [int(x) for x in peaks_index]


def _encode(clip, text):
    tokens = clip.tokenize(text)
    out = clip.encode_from_tokens(tokens, return_pooled=True)
    if isinstance(out, (tuple, list)):
        return out[0], (out[1] if len(out) > 1 else None)
    return out, None


class SFX_AudioPromptTravel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "peaks_index": ("STRING", {"forceInput": True}),
                "prompts": ("STRING", {"multiline": True, "default": ""}),
                "max_frames": ("INT", {"default": 200, "min": 1, "max": 99999}),
                "blend_frames": ("INT", {"default": 6, "min": 0, "max": 240}),
            },
            "optional": {
                "prompts_override": ("STRING", {"forceInput": True}),
                "pre_text": ("STRING", {"multiline": True, "default": ""}),
                "app_text": ("STRING", {"multiline": True, "default": ""}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "STRING")
    RETURN_NAMES = ("CONDITIONING", "schedule_preview")
    FUNCTION = "travel"
    CATEGORY = "Solus FX"

    def travel(self, clip, peaks_index, prompts, max_frames, blend_frames,
               prompts_override=None, pre_text="", app_text=""):
        if isinstance(prompts_override, (list, tuple)):
            prompts_override = "\n".join(str(x) for x in prompts_override)
        if prompts_override and str(prompts_override).strip():
            prompts = str(prompts_override)

        peaks = _parse_peaks(peaks_index)
        peaks = sorted(set([0] + [p for p in peaks if 0 < p < max_frames]))
        if not peaks:
            peaks = [0]

        lines = [p.strip() for p in prompts.split("\n") if p.strip()]
        if not lines:
            lines = [""]
        seq = [lines[i % len(lines)] for i in range(len(peaks))]
        seq = [" ".join(x for x in (pre_text.strip(), t, app_text.strip()) if x) for t in seq]

        cache = {t: _encode(clip, t) for t in set(seq)}
        maxtok = max(c.shape[1] for c, _ in cache.values())

        def pad(c):
            if c.shape[1] >= maxtok:
                return c[:, :maxtok, :]
            tail = c[:, -77:, :] if c.shape[1] >= 77 else c
            while c.shape[1] < maxtok:
                c = torch.cat([c, tail], dim=1)
            return c[:, :maxtok, :]

        frames = []
        for f in range(max_frames):
            i = 0
            for k, p in enumerate(peaks):
                if f >= p:
                    i = k
            cur = pad(cache[seq[i]][0])
            if blend_frames > 0 and i + 1 < len(peaks):
                dist = peaks[i + 1] - f
                if 0 < dist <= blend_frames:
                    a = 1.0 - (dist / float(blend_frames))
                    cur = cur * (1.0 - a) + pad(cache[seq[i + 1]][0]) * a
            frames.append(cur)

        batch = torch.cat(frames, dim=0)
        extra = {}
        ref_pooled = cache[seq[0]][1]
        if ref_pooled is not None:
            extra["pooled_output"] = (ref_pooled.repeat(max_frames, 1)
                                      if ref_pooled.shape[0] == 1 else ref_pooled)
        preview = "\n".join(f"frame {peaks[i]:>4} | {seq[i]}" for i in range(len(peaks)))
        return ([[batch, extra]], preview)


class SFX_CaptionsToPrompts:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"captions": ("STRING", {"forceInput": True})},
                "optional": {"prefix": ("STRING", {"multiline": True, "default": ""}),
                             "suffix": ("STRING", {"multiline": True, "default": ""}),
                             "max_words": ("INT", {"default": 0, "min": 0, "max": 200})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompts",)
    FUNCTION = "conv"
    CATEGORY = "Solus FX"
    INPUT_IS_LIST = True

    @staticmethod
    def _flatten(x):
        if isinstance(x, (list, tuple)):
            for y in x:
                yield from SFX_CaptionsToPrompts._flatten(y)
        else:
            yield str(x)

    def conv(self, captions, prefix=None, suffix=None, max_words=None):
        pre = (prefix[0] if isinstance(prefix, (list, tuple)) and prefix else (prefix or "")) or ""
        suf = (suffix[0] if isinstance(suffix, (list, tuple)) and suffix else (suffix or "")) or ""
        mw = (max_words[0] if isinstance(max_words, (list, tuple)) and max_words else (max_words or 0)) or 0
        out = []
        for c in self._flatten(captions):
            for line in c.split("\n"):
                line = line.strip().rstrip(".")
                if not line:
                    continue
                if mw:
                    line = " ".join(line.split()[:mw])
                out.append(" ".join(x for x in (str(pre).strip(), line, str(suf).strip()) if x))
        return ("\n".join(out) if out else "",)


NODE_CLASS_MAPPINGS = {
    "SFX_AudioPromptTravel": SFX_AudioPromptTravel,
    "SFX_CaptionsToPrompts": SFX_CaptionsToPrompts,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SFX_AudioPromptTravel": "◐ Audio Prompt Travel (Solus FX)",
    "SFX_CaptionsToPrompts": "◐ Captions → Prompts (Solus FX)",
}
