import json
import re
from collections import defaultdict
from pathlib import Path

from value_labels import VALUE_LABELS

EVIDENCES = Path(__file__).parent / "training sklearn" / "release_evidences.json"
EIDS = ["E_55", "E_57", "E_54", "E_204"]
LATIN = re.compile(r"[A-Za-z]")


def _evidences():
    return json.loads(EVIDENCES.read_text(encoding="utf-8"))


def test_every_value_id_is_a_key():
    d = _evidences()
    missing = [(e, v) for e in EIDS for v in d[e]["possible-values"] if v not in VALUE_LABELS]
    assert not missing, missing


def test_entries_non_empty_and_no_latin():
    for vid, lab in VALUE_LABELS.items():
        assert set(lab) == {"ru", "kk"}, vid
        for lang in ("ru", "kk"):
            s = lab[lang]
            assert s.strip(), (vid, lang)
            assert not LATIN.search(s), (vid, lang, s)
            assert s == s.strip() and not s.endswith((".", ",", ";")), (vid, lang, s)


def test_no_unjustified_ru_duplicates():
    d = _evidences()
    en = {}
    for e in EIDS:
        for v in d[e]["possible-values"]:
            en.setdefault(v, d[e]["value_meaning"].get(v, {}).get("en"))
    by_ru = defaultdict(list)
    for vid, lab in VALUE_LABELS.items():
        by_ru[lab["ru"]].append(vid)
    collisions = {r: ids for r, ids in by_ru.items()
                  if len(ids) > 1 and len({en.get(i) for i in ids}) > 1}
    for r, ids in collisions.items():
        print("COLLISION", r, [(i, en.get(i)) for i in ids])
    assert not collisions, collisions
    # left/right pairs must differ
    for vid, name in en.items():
        if name and name.endswith("(R)"):
            other = next((v for v, n in en.items() if n == name[:-3] + "(L)"), None)
            if other:
                assert VALUE_LABELS[vid]["ru"] != VALUE_LABELS[other]["ru"]
                assert VALUE_LABELS[vid]["kk"] != VALUE_LABELS[other]["kk"]
