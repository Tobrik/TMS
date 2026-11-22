"""Tests for negation.py: rule-based denials for the RU, KK and EN symptom vocabulary.

Each case: (text, evidence code, expected verdict). Expected is "present", "absent",
or None when the code must not appear at all.
"""

import pytest

import negation
import retrieval

# ─── Russian ─────────────────────────────────────────────────────────────────
RU = [
    ("температуры нет", "E_91", "absent"),
    ("без температуры", "E_91", "absent"),
    ("кашель не беспокоит", "E_201", "absent"),
    ("температура, кашля нет", "E_91", "absent"),   # comma does not end the clause: both denied
    ("температура, кашля нет", "E_201", "absent"),
    ("не могу дышать", "E_66", "present"),          # symptom, not a denial
    ("Насморка нет", "E_181", "absent"),
    ("сильный кашель и болит горло", "E_201", "present"),
    ("горло не болит", "E_97", "absent"),
    ("горло болит", "E_97", "present"),
    ("Температуры нет, но сильный кашель", "E_91", "absent"),
    ("Температуры нет, но сильный кашель", "E_201", "present"),
    ("температура 38, нет кашля", "E_91", "present"),  # a number ends the scope
    ("тошноты нет", "E_148", "absent"),
    ("аппетит в норме", "E_161", "absent"),
    ("не могу глотать", "E_65", "present"),
    ("не хватает воздуха", "E_66", "present"),
    ("кашель не проходит", "E_201", "present"),      # "не проходит" is a pseudo-negation
    ("не только кашель, но и температура", "E_201", "present"),
    ("нет сил", "E_175", "present"),                  # "нет сил" = no strength = fatigue
    ("никакой температуры", "E_91", "absent"),
    ("не могу заснуть", "E_89", "present"),           # cannot sleep is a symptom
    ("кашель сильный, температуры нет", "E_201", "present"),  # words before the comma stop the denial
    ("кашель сильный, температуры нет", "E_91", "absent"),
    ("без кашля, у меня высокая температура", "E_91", "present"),
    ("без кашля, у меня высокая температура", "E_201", "absent"),
    ("без кашля, температура 38", "E_91", "present"),       # number ends the scope
]

# ─── Kazakh ──────────────────────────────────────────────────────────────────
KK = [
    ("жөтел жоқ", "E_201", "absent"),
    ("қызуым жоқ", "E_91", "absent"),
    ("тамағым ауырмайды", "E_97", "absent"),          # negative suffix on the verb
    ("құспадым", "E_148", "absent"),                  # past negative -падым
    ("тәбетім жақсы", "E_161", "absent"),             # appetite is good
    ("жөтелмеймін", "E_201", "absent"),
    ("жөтел жоқ емес", "E_201", "present"),           # "not absent" = there is a cough
    ("жұта алмаймын", "E_65", "present"),             # "cannot swallow" is a symptom
    ("жөтелім қатты, қызуым жоқ", "E_201", "present"),
    ("жөтелім қатты, қызуым жоқ", "E_91", "absent"),
    ("Қызуым жоқ, бірақ жөтелім қатты, тамағым ауырмайды.", "E_91", "absent"),
    ("Қызуым жоқ, бірақ жөтелім қатты, тамағым ауырмайды.", "E_97", "absent"),
    ("кеудем ауырмайды", "E_14", "absent"),
    ("басым ауырады", "E_53", "present"),
    ("қанды жөтел", "E_45", "present"),
    ("есінен танды", "E_159", "present"),
    ("тәбетім жоқ", "E_161", "present"),              # no appetite = loss of appetite
    ("ешқандай қызу жоқ", "E_91", "absent"),
]

# ─── English ─────────────────────────────────────────────────────────────────
EN = [
    ("no fever", "E_91", "absent"),
    ("I don't have a cough", "E_201", "absent"),
    ("denies chest pain", "E_14", "absent"),
    ("my appetite is fine", "E_161", "absent"),
    ("can't breathe", "E_66", "present"),
    ("cough is not getting better", "E_201", "present"),   # pseudo-negation: "not getting better"
    ("no fever but bad cough", "E_91", "absent"),
    ("no fever but bad cough", "E_201", "present"),
    ("I have a sore throat", "E_97", "present"),
    ("cannot swallow", "E_65", "present"),
    ("cannot breathe", "E_66", "present"),
    ("no relief from the cough", "E_201", "present"),      # "no relief" is a pseudo-negation
    ("without fever or chills", "E_91", "absent"),
    ("without fever or chills", "E_94", "absent"),
    ("not constantly tired", "E_89", "absent"),
    ("I feel constantly tired", "E_89", "present"),
    ("the pain is worse at night", "E_219", "present"),
    ("my appetite is poor", "E_161", "present"),
    ("no pain", "E_53", "absent"),
    ("I haven't had chest pain", "E_14", "absent"),
    ("negative for fever", "E_91", "absent"),
]

# ─── Present wins across mentions of one code ────────────────────────────────
PRESENT_WINS = [
    ("no cough yesterday but cough now", "E_201", "present"),
    ("кашля нет, но вчера кашлял", "E_201", "present"),
]

# ─── Not mentioned at all: no code expected ──────────────────────────────────
MISSING = [
    ("у меня всё хорошо", "E_91", None),
    ("I feel fine today", "E_91", None),
]

CASES = RU + KK + EN + PRESENT_WINS + MISSING


@pytest.mark.parametrize("text,code,expected", CASES)
def test_verdict(text, code, expected):
    verdicts = negation.vocab_findings(text)
    assert verdicts.get(code) == expected, verdicts


def test_is_negated_takes_lowercased_text_and_span():
    text = "кашля нет"
    start = text.index("кашля")
    assert negation.is_negated(text, start, start + len("кашля")) is True
    text = "кашель сильный"
    assert negation.is_negated(text, 0, len("кашель")) is False


def test_vocab_extract_returns_only_present_codes():
    text = "Температуры нет, но сильный кашель. Тошноты нет."
    present = {c for c, s in negation.vocab_findings(text).items() if s == "present"}
    assert set(retrieval.vocab_extract(text)) == present
    assert "E_201" in present
    assert "E_91" not in present and "E_148" not in present
