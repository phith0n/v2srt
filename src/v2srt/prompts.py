"""Prompts for subtitle translation.

The instructions are written in English (models follow English instructions most reliably) and are independent
of the source and target language. Cues are exchanged as JSON so that the model's answer can be validated.
"""

import json
from typing import Sequence

from .subtitle import Cue

_STYLE_NOTES = {
    "zh": (
        "Use Chinese full-width punctuation（，？！：“”）. Do not end a cue with a full stop (。), as is customary "
        "for Chinese subtitles. Do not add spaces between Chinese and Latin characters unless the source has them."
    ),
    "ja": "Follow Japanese subtitle conventions: no sentence-final 。, use full-width ？ and ！ where natural.",
    "ko": "Follow Korean subtitle conventions and use the speech level that fits the relationship between speakers.",
}

SYSTEM_TEMPLATE = """\
You are an expert subtitle translator. You translate subtitles that were produced by automatic speech \
recognition (ASR) from {source} into {target}.

Follow these rules:
1. One-to-one mapping. Return exactly one translation for every input cue, using the same "id". Never merge, \
split, skip or reorder cues. When a sentence runs across several cues, translate it as a whole but distribute \
the meaning over those cues in order, so that each cue still matches what is being said at that moment.
2. Natural, idiomatic {target}. Write what a professional native subtitler would write, not a word-for-word \
rendering. Adapt idioms, jokes and wordplay so they work in {target}.
3. Keep it short. Viewers have only a few seconds per cue. Drop filler words (um, uh, like, you know), \
stutters and false starts unless they matter for meaning or characterization.
4. Fix recognition errors silently. ASR output may contain misheard words, homophones, wrong names or missing \
punctuation. Use the surrounding cues to infer what was actually said and translate that. Never translate \
obvious nonsense literally and never comment on it.
5. Preserve voice and tone: register, politeness, emotion, humor, slang and profanity (at the same intensity). \
Keep each speaker's personality consistent.
6. Names and terminology. Use the established {target} form for well-known people, places, brands and works; \
otherwise keep the original spelling or transliterate. Render the same name or term the same way every time.
7. Non-speech text. Translate sound descriptions such as (laughs) or [applause] and keep the brackets. \
Translate song lyrics and keep a leading ♪ if present.
8. If a cue is already in {target}, or is only a name, number or interjection, return it unchanged or \
minimally adapted.
9. Output only the subtitle text: no notes, explanations, romanization, quotes or source text.{style}
"""


def system_instruction(source_language: str | None, target_language: str, target_code: str | None) -> str:
    source = source_language or "the language of the speech (detect it from the text)"
    style = _STYLE_NOTES.get(target_code or "", "")
    return SYSTEM_TEMPLATE.format(source=source, target=target_language, style=f"\n10. {style}" if style else "")


def user_prompt(
    cues: Sequence[Cue],
    *,
    previous: Sequence[Cue] = (),
    title: str | None = None,
    context: str | None = None,
) -> str:
    parts: list[str] = []
    if title:
        parts.append(f"Video title: {title}")
    if context:
        parts.append(f"Background information from the viewer:\n{context.strip()}")
    if previous:
        lines = []
        for cue in previous:
            line = f"- {cue.text}"
            if cue.translation:
                line += f"  =>  {cue.translation}"
            lines.append(line)
        parts.append(
            "Preceding cues, for context only (do NOT include them in your answer; reuse their wording for "
            "consistency where a translation is given):\n" + "\n".join(lines)
        )
    payload = [{"id": i, "text": cue.text} for i, cue in enumerate(cues, 1)]
    parts.append(
        f"Translate these {len(payload)} cues. Answer with a JSON object of the form "
        '{"translations": [{"id": <id>, "text": "<translation>"}, ...]} containing every id exactly once.\n'
        + json.dumps(payload, ensure_ascii=False, indent=0)
    )
    return "\n\n".join(parts)
