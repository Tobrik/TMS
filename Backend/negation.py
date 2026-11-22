"""
Rule-based denial detection for the regex symptom vocabulary (RU, KK, EN).

retrieval._VOCAB finds where a symptom is mentioned. This module decides, for
each mention, whether the text denies it ("кашля нет", "жөтел жоқ", "no fever")
or states it ("не могу дышать", "cannot breathe", "cough is not getting better").

Precision of "absent" matters more than recall: a wrong denial pushes the model
away from the right disease. When unsure, a mention counts as present.

Scope of a negator or a denial word: the clause that contains the mention. A
clause ends at . ! ? ; newline or a contrast word (но, а, однако, бірақ, алайда,
but, however, though). A comma does not end a clause. At most 5 words before and
4 words after the mention are considered. Numbers also end the scope ("температура
38, нет кашля" leaves the temperature present).
"""

import re
from typing import Dict

_WORD_RE = re.compile(r"[\w'’]+")
_DIGIT_RE = re.compile(r"\d")
_FLAGS = re.DOTALL | re.MULTILINE

_CLAUSE_END_RE = re.compile(
    r"[.!?;\n]|(?<!\w)(?:но|а|однако|бірақ|алайда|but|however|though)(?!\w)"
)


def _w(alternatives: str) -> str:
    """Alternation that must not touch another word character on either side."""
    return r"(?<!\w)(?:" + alternatives + r")(?!\w)"


# Negators before the mention. "Strict" ones do not reach across a comma, so
# "кашля нет, температура 38" does not deny the temperature. "Loose" ones do:
# "без температуры, кашля" denies both.
_BEFORE_STRICT_RE = re.compile(_w(
    r"не|нет|нету|no|not|never|isn['’]?t|aren['’]?t|wasn['’]?t"
    r"|don['’]?t\s+have|doesn['’]?t\s+have|haven['’]?t\s+had|hasn['’]?t\s+had"
))
_BEFORE_LOOSE_RE = re.compile(_w(
    r"без|ни|никак\w*|отсутству\w*|ешқандай|еш|without|denies|deny|free\s+of|negative\s+for"
))

# A negator followed by one of these is a pseudo-negation: "не могу дышать",
# "не проходит", "no relief", "not getting better" state a symptom.
_PSEUDO_AFTER_NEG_RE = re.compile(
    r"\s*(?:могу|могла|могли|могут|мог|смогу|смогла|можем|можете|хватает|хватало"
    r"|проходит|прошло|прекращ\w*|только|получа\w*|становится|улучша\w*|отпуска\w*"
    r"|уходит|стихает|помога\w*|отступает"
    r"|better|relief|improv\w*|stop\w*|only|getting|going|go\s+away"
    r"|cannot|can['’]?t|unable|couldn['’]?t)(?!\w)"
)

# Denial words after the mention, within the first 4 words of the clause.
_AFTER_WINDOW4_RE = re.compile(_w(
    # Russian
    r"нет|нету|не\s+было|отсутству\w*|не\s+беспоко\w*|не\s+наблюда\w*|не\s+замеча\w*"
    r"|в\s+норме|нормальн\w*|в\s+порядке|не\s+трудно|не\s+мешает"
    # Kazakh
    r"|жоқ|емес|болмады|байқалмады|сезілмейді"
    # English
    r"|is\s+fine|are\s+fine|is\s+normal|are\s+normal|is\s+ok(?:ay)?"
    r"|isn['’]?t\s+a\s+problem|is\s+not\s+a\s+problem|not\s+an\s+issue"
))
# Only when the mention is a body part: "горло не болит" yes, "кашель не болит" no.
_AFTER_BODY_RE = re.compile(_w(r"не\s+болит|не\s+болят|не\s+больно"))
# Evaluations directly after the mention (first 2 words): "тәбетім жақсы", "appetite is good".
_AFTER_WINDOW2_RE = re.compile(_w(
    r"fine|normal|ok(?:ay)?|good|great|жақсы|қалыпты|хорош\w*|нормальн\w*|в\s+норме"
))

# Words inside a mention that already deny it, because the pattern consumed them:
# "горло не болит", "тәбетім жақсы", "appetite is fine".
_SPAN_DENIAL_RE = re.compile(_w(
    r"не\s+болит|не\s+болят|не\s+больно|не\s+беспокоит|не\s+беспокоят|не\s+мешает|не\s+трудно"
    r"|в\s+норме|нормальн\w*|в\s+порядке|хорош\w*|жақсы|қалыпты"
    r"|is\s+fine|are\s+fine|is\s+normal|is\s+ok(?:ay)?|fine|normal|good|great|ok|okay"
))

# Kazakh verb negation is a suffix on the word: ауырмайды, жөтелмеймін, құспадым.
_KK_NEG_SUFFIX_RE = re.compile(
    r"(?:ма|ме|ба|бе|па|пе)(?:й|с|ған|ген|ды|ді|ймын|ймін|йды|йді|дым|дім|тым|тім)$"
)
# "алмаймын" is "I cannot" (ability), a symptom, not a denial.
_KK_ABILITY_RE = re.compile(r"ал(?:ма|ме|ба|бе|па|пе)")
_CYRILLIC_WORD_RE = re.compile(r"[а-яёәіңғүұқөһ]+")
_BODY_RE = re.compile(
    r"горл|голов|живот|брюш|спин|поясниц|колен|плеч|ше[еи]|груд|ух[оаё]|глаз|мышц|зуб|бок"
    r"|ног|рук|пальц|затылк|висок|лопат|пах|челюст|суста|кож|язык|сердц|ребр|стоп|лиц|нос"
    r"|кишк|желуд|тело|бедр|щек|тамақ|тамағ|бас|іш|кеуде|буын|жүрек|құрсақ|аяқ|қол|тізе"
    r"|мойын|бел|көз|құлақ|тіл"
)


def _clause_bounds(text: str, start: int, end: int):
    left = 0
    for m in _CLAUSE_END_RE.finditer(text, 0, start):
        left = m.end()
    m = _CLAUSE_END_RE.search(text, end)
    return left, (m.start() if m else len(text))


def _left_window_start(text: str, clause_start: int, start: int) -> int:
    """Start of the 5-word window before the mention, after any number.

    A comma cuts the window only when words sit between it and the mention:
    "без кашля, температура" keeps "без" in scope, "кашля нет, у меня высокая
    температура" does not."""
    lo = clause_start
    digits = list(_DIGIT_RE.finditer(text, clause_start, start))
    if digits:
        lo = digits[-1].end()
    comma = text.rfind(",", lo, start)
    if comma != -1 and _WORD_RE.search(text, comma + 1, start):
        lo = comma + 1
    tokens = list(_WORD_RE.finditer(text, lo, start))
    if len(tokens) > 5:
        lo = tokens[-5].start()
    return lo


def _negated_before(text: str, lo: int, start: int) -> bool:
    for regex, strict in ((_BEFORE_STRICT_RE, True), (_BEFORE_LOOSE_RE, False)):
        for m in regex.finditer(text, lo, start):
            if strict and "," in text[m.end():start]:
                continue
            if _PSEUDO_AFTER_NEG_RE.match(text, m.end()):
                continue
            return True
    return False


# Last letters of the mention's word for a Russian genitive noun (насморка, кашля,
# температуры). Not after д/т: Kazakh verbs end in -ды/-ді/-ты/-ті ("айниды").
_GENITIVE_END_RE = re.compile(r"(?<![дт])[аяыи]$")


def _negated_after(text: str, end: int, right_end: int, body: bool) -> bool:
    # "температуры, кашля нет" denies both: a list of genitive nouns before "нет".
    # Anything else stops at the comma: "кашель сильный, температуры нет",
    # "тошнит, аппетита нет" (a verb) and "ентігемін, и вообще нет сил" keep the
    # first symptom present.
    comma = text.find(",", end, right_end)
    if comma != -1 and (_WORD_RE.search(text, end, comma) or not _GENITIVE_END_RE.search(text, 0, end)):
        right_end = comma
    stop = right_end
    digit = _DIGIT_RE.search(text, end, right_end)
    if digit:
        stop = digit.start()
    tokens = list(_WORD_RE.finditer(text, end, stop))
    end4 = tokens[4].start() if len(tokens) > 4 else stop
    end2 = tokens[2].start() if len(tokens) > 2 else stop
    window4 = text[end:end4]

    # "жөтел жоқ емес" means "there is a cough": not a denial.
    if re.match(r"\s*жоқ\s+емес", window4):
        return False
    if _AFTER_WINDOW4_RE.search(window4):
        return True
    if body and _AFTER_BODY_RE.search(window4):
        return True
    return _AFTER_WINDOW2_RE.search(text[end:end2]) is not None


def _kk_suffix_denied(text: str, start: int, end: int) -> bool:
    """True if the word that contains the end of the mention has a negative verb suffix."""
    if end <= start or not text[end - 1].isalnum():
        return False
    s = end
    while s > 0 and (text[s - 1].isalnum() or text[s - 1] == "_"):
        s -= 1
    e = end
    while e < len(text) and (text[e].isalnum() or text[e] == "_"):
        e += 1
    word = text[s:e]
    if not _CYRILLIC_WORD_RE.fullmatch(word):
        return False
    if _KK_ABILITY_RE.search(word):
        return False
    return _KK_NEG_SUFFIX_RE.search(word) is not None


def is_negated(text: str, start: int, end: int) -> bool:
    """True if the symptom mention text[start:end] is denied in Russian, Kazakh or English.

    `text` must already be lowercased. Pseudo-negations ("не могу дышать",
    "cough is not getting better", "жөтел жоқ емес") count as present.
    """
    if end < start:
        end = start
    # Patterns often stop mid-word ("горло.{0,10}бол" ends at "бол" in "не болит").
    # Take the rest of that word as part of the mention.
    while end > start and end < len(text) and text[end - 1].isalnum() and text[end].isalnum():
        end += 1
    mention = text[start:end]
    if _SPAN_DENIAL_RE.search(mention):
        return True
    # A pattern that contains its own negator states a symptom in negative words
    # ("no appetite", "нет аппетита", "не могу дышать"); an outside negator must
    # not flip it ("lost weight without trying and I have no appetite").
    if _BEFORE_STRICT_RE.search(mention) or _BEFORE_LOOSE_RE.search(mention):
        return False
    if _kk_suffix_denied(text, start, end):
        return True
    # A measured value right after the mention ("температура 38") states it, whatever came before.
    if re.match(r"\s*[:=-]?\s*\d", text[end:end + 12]):
        return False
    clause_start, clause_end = _clause_bounds(text, start, end)
    if _negated_before(text, _left_window_start(text, clause_start, start), start):
        return True
    body = _BODY_RE.search(mention) is not None
    return _negated_after(text, end, clause_end, body)


def vocab_findings(text: str) -> Dict[str, str]:
    """Evidence code -> "present" | "absent" for every vocabulary pattern that matches.

    A pattern is "present" if any of its matches is not denied, "absent" if all
    matches are denied. A code present through one pattern and absent through
    another is "present".
    """
    from retrieval import _VOCAB  # lazy: retrieval imports this module

    # Patients rarely type "ё" ("течет", "отдает"); match both spellings.
    # ё → е keeps every character position, so match offsets stay valid.
    lowered = text.lower().replace("ё", "е")
    verdicts: Dict[str, str] = {}
    for pattern, codes in _VOCAB:
        matches = list(re.finditer(pattern.replace("ё", "е"), lowered, _FLAGS))
        if not matches:
            continue
        verdict = "present" if any(not is_negated(lowered, m.start(), m.end()) for m in matches) else "absent"
        for code in codes:
            if verdicts.get(code) != "present":
                verdicts[code] = verdict
    return verdicts
