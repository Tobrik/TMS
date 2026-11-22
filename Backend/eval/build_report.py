"""
Build eval/accuracy_report.json: an offline accuracy snapshot shown to doctors
(GET /accuracy_report adds the live real-patient numbers to it).

Two sections:
  synthetic      DDXPlus test patients replayed through the default NB pipeline
                 (eval/simulate.py) with noise 0.3, answer error 0.1, don't know 0.2.
  written_cases  free-text complaints (eval/cases_vN.jsonl) scored from the cached
                 extraction results of one backend/tag (eval/results/). Never calls an API.

Every rate is a metric object {"value", "k", "n", "ci"} with a Wilson 95% interval.

Usage (from Backend/):
    python eval/build_report.py --patients <path/to/release_test_patients> \
        --cases eval/cases_v3.jsonl --backend clef-flash --tag v3 [--out eval/accuracy_report.json] \
        [--n 3000 --seed 7]

The disease labels and MODEL_VERSION are read from "backend TMS.py" with ast, so the
app (and its DATABASE_URL) is never imported.
"""

import argparse
import ast
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

HERE = Path(__file__).resolve().parent
BACKEND = HERE.parent
for _p in (str(BACKEND), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import compare_extraction as ce  # noqa: E402  (also loads Backend/.env, prints nothing)
import ml_model as m  # noqa: E402
import qa_engine  # noqa: E402
import simulate  # noqa: E402
import triage  # noqa: E402
from metrics import metric  # noqa: E402

BACKEND_FILE = BACKEND / "backend TMS.py"
SYNTHETIC_SETTINGS = {"noise": 0.3, "answer_error": 0.1, "dont_know": 0.2}
MIN_CONDITION_N = 20
LANGS = ("ru", "kk", "mix", "en")

CAVEATS = [
    "Synthetic patients follow DDXPlus symptom statistics, and real patients do not, so the synthetic numbers "
    "measure the pipeline and are not a forecast for a clinic.",
    "Simulated patients make mistakes: 30% add one symptom they do not have, 10% of yes/no answers are flipped "
    "and 20% of follow-up questions get \"don't know\".",
    "The written cases were written by an AI model from DDXPlus records, not by patients, so their wording is "
    "cleaner than real complaints.",
    "In the written cases, follow-up answers come from the DDXPlus record and are truthful, and patient_confirms "
    "assumes the patient corrects every extracted symptom; extraction used the {backend} backend.",
    "Real-patient numbers from doctor-confirmed diagnoses are the only real-world evidence, and they are shown "
    "next to this snapshot, not mixed into it.",
    "Samples are small for many conditions, so intervals are wide; read every number together with its ci and n.",
]


# ─── Constants from the app, without importing it ────────────────────────────

def read_backend_constants(path: Path = BACKEND_FILE) -> dict:
    """MODEL_VERSION and disease_labels, parsed from the module source."""
    wanted = {"MODEL_VERSION", "disease_labels"}
    found = {}
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id in wanted):
            found[node.targets[0].id] = ast.literal_eval(node.value)
    missing = wanted - set(found)
    if missing:
        raise RuntimeError(f"{path.name}: no literal {', '.join(sorted(missing))}")
    return found


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(BACKEND).as_posix()
    except ValueError:
        return path.name


# ─── Synthetic section ───────────────────────────────────────────────────────

def default_pipeline() -> simulate.Pipeline:
    found = [p for p in simulate.PIPELINES if "(default)" in p.name]
    if len(found) != 1:
        raise RuntimeError("expected exactly one default pipeline in simulate.PIPELINES")
    return found[0]


# Doctor-facing text for the report page, per UI language. {backend}, {n_synth}
# and {n_written} are filled in by build(). Plain statements, no code names.
I18N = {
    "ru": {
        "synthetic_source": "Синтетические пациенты DDXPlus (тестовая выборка), смоделированный диалог: {n_synth} пациентов",
        "written_source": "{n_written} письменных жалоб на русском, казахском, смешанном и английском, не использовались для настройки",
        "caveats": [
            "Синтетические пациенты построены по статистике симптомов набора DDXPlus. Реальные пациенты отличаются, поэтому эти цифры проверяют работу системы, но не предсказывают точность в клинике.",
            "Симулированные пациенты ошибаются: у 30% появляется один лишний симптом, 10% ответов «да / нет» неверны, на 20% уточняющих вопросов они отвечают «не знаю».",
            "Письменные жалобы составила языковая модель по записям DDXPlus, а не сами пациенты, поэтому они чище настоящих сообщений.",
            "В письменных жалобах ответы на уточняющие вопросы правдивы. Вариант «пациент проверил симптомы» предполагает, что пациент исправил все распознанные симптомы. Распознавание симптомов: {backend}.",
            "Единственное доказательство точности на реальных пациентах: диагнозы, подтверждённые врачами. Они показаны отдельно и не смешиваются с этими расчётами.",
            "По многим заболеваниям случаев мало, поэтому интервалы широкие. Смотрите на каждое число вместе с интервалом и n.",
        ],
    },
    "kk": {
        "synthetic_source": "DDXPlus синтетикалық пациенттері (тест жиыны), модельденген диалог: {n_synth} пациент",
        "written_source": "Орыс, қазақ, аралас және ағылшын тілдеріндегі {n_written} жазбаша шағым, баптауға қолданылмаған",
        "caveats": [
            "Синтетикалық пациенттер DDXPlus жиынының белгілер статистикасы бойынша құрастырылған. Нақты пациенттер одан өзгеше, сондықтан бұл сандар жүйенің жұмысын тексереді, бірақ емханадағы дәлдікті болжамайды.",
            "Симуляциядағы пациенттер қателеседі: 30%-ында бір артық белгі пайда болады, «иә / жоқ» жауаптарының 10%-ы қате, нақтылау сұрақтарының 20%-ына «білмеймін» деп жауап береді.",
            "Жазбаша шағымдарды DDXPlus жазбалары бойынша тілдік модель жазған, пациенттер емес, сондықтан олар нақты хабарламалардан тазарақ.",
            "Жазбаша шағымдарда нақтылау сұрақтарына жауаптар шынайы. «Пациент белгілерді тексерді» нұсқасы пациент танылған белгілердің бәрін түзеткен деп есептейді. Белгілерді тану: {backend}.",
            "Нақты пациенттердегі дәлдіктің жалғыз дәлелі: дәрігер растаған диагноздар. Олар бөлек көрсетіледі және бұл есептерге қосылмайды.",
            "Көптеген аурулар бойынша жағдайлар аз, сондықтан интервалдар кең. Әр санды интервалымен және n-мен бірге қараңыз.",
        ],
    },
    "en": {
        "synthetic_source": "Synthetic DDXPlus test patients, simulated conversation: {n_synth} patients",
        "written_source": "{n_written} written complaints in Russian, Kazakh, mixed Russian-Kazakh and English, never used for tuning",
        "caveats": [
            "Synthetic patients follow the symptom statistics of the DDXPlus dataset. Real patients differ, so these numbers test the system but do not predict accuracy in a clinic.",
            "Simulated patients make mistakes: 30% mention one symptom they do not have, 10% of yes / no answers are wrong, and 20% of follow-up questions get \"don't know\".",
            "The written complaints were written by a language model from DDXPlus records, not by patients, so they are cleaner than real messages.",
            "In the written complaints, answers to follow-up questions are truthful. \"Patient checked the symptoms\" assumes the patient corrected every recognised symptom. Symptom recognition: {backend}.",
            "The only evidence from real patients is diagnoses confirmed by doctors. They are shown separately and never mixed into these figures.",
            "Many conditions have few cases, so the intervals are wide. Read every number together with its interval and n.",
        ],
    },
}


# Only these case files were never looked at while tuning prompts, rules or
# thresholds. Any other file is reported as a development set.
HELD_OUT_FILES = {"cases_v3.jsonl"}

TUNING_SOURCE = {
    "ru": "{n_written} письменных жалоб на русском, казахском, смешанном и английском. Этот набор использовался при настройке системы, поэтому цифры могут быть завышены",
    "kk": "Орыс, қазақ, аралас және ағылшын тілдеріндегі {n_written} жазбаша шағым. Бұл жиын жүйені баптауға қолданылған, сондықтан сандар асыра көрсетілуі мүмкін",
    "en": "{n_written} written complaints in Russian, Kazakh, mixed and English. This set was used while tuning the system, so the numbers may be optimistic",
}


def localized_text(backend: str, n_synth: int, n_written: int, held_out: bool = True) -> dict:
    fill = lambda text: text.format(backend=backend, n_synth=n_synth, n_written=n_written)
    return {lang: {"synthetic_source": fill(t["synthetic_source"]),
                   "written_source": fill(t["written_source"] if held_out else TUNING_SOURCE[lang]),
                   "caveats": [fill(c) for c in t["caveats"]]} for lang, t in I18N.items()}


def synthetic_section(patients_path: str, n: int, seed: int, labels: dict) -> dict:
    patients = simulate.load_patients(patients_path, n, seed)
    saved = (simulate.NOISE, simulate.ANSWER_ERROR, simulate.DONT_KNOW)
    simulate.NOISE = SYNTHETIC_SETTINGS["noise"]
    simulate.ANSWER_ERROR = SYNTHETIC_SETTINGS["answer_error"]
    simulate.DONT_KNOW = SYNTHETIC_SETTINGS["dont_know"]
    try:
        results = simulate.run(default_pipeline(), patients, seed)
    finally:
        simulate.NOISE, simulate.ANSWER_ERROR, simulate.DONT_KNOW = saved
    settings = {"n": n, "seed": seed, **SYNTHETIC_SETTINGS}
    return summarise_synthetic(results, settings, labels)


def summarise_synthetic(results: List[dict], settings: dict, labels: dict) -> dict:
    total = len(results)
    reds = [r for r in results if r["truth_tier"] == "red"]
    greens = [r for r in results if r["truth_tier"] == "green"]

    confs = [r["conf"] for r in results]
    correct = [r["top1"] for r in results]
    calibration = [
        {"bin": f"{lo:.1f}-{hi:.1f}", "n": k,
         "mean_confidence": round(mc, 3) if k else None,
         "accuracy": round(acc, 3) if k else None}
        for lo, hi, k, mc, acc in simulate.reliability(confs, correct)
    ]

    stop = qa_engine.STOP_CONFIDENCE
    low = [r for r in results if r["conf"] < stop]
    rest = [r for r in results if r["conf"] >= stop]

    by_truth: Dict[str, List[dict]] = {}
    for r in results:
        by_truth.setdefault(r["truth"], []).append(r)
    per_condition = []
    for name, group in by_truth.items():
        if len(group) < MIN_CONDITION_N:
            continue
        per_condition.append({
            "name": name,
            "label_ru": labels.get(name, name),
            "tier": triage.tier_of(name),
            "top1": metric(sum(r["top1"] for r in group), len(group)),
            "top3": metric(sum(r["top3"] for r in group), len(group)),
        })
    per_condition.sort(key=lambda c: (c["top1"]["value"], c["name"]))

    return {
        "source": "DDXPlus test patients (synthetic), simulated conversation",
        "settings": settings,
        "metrics": {
            "top1": metric(sum(r["top1"] for r in results), total),
            "top3": metric(sum(r["top3"] for r in results), total),
            "red_sensitivity": metric(sum(r["zone"] == "red" for r in reds), len(reds)),
            "red_to_green": metric(sum(r["zone"] == "green" for r in reds), len(reds)),
            "green_to_red": metric(sum(r["zone"] == "red" for r in greens), len(greens)),
            "zone_exact": metric(sum(r["zone"] == r["truth_tier"] for r in results), total),
        },
        "questions_mean": round(sum(r["asked"] for r in results) / total, 2) if total else None,
        "calibration": calibration,
        "low_confidence": {
            "share": metric(len(low), total),
            "top1_low": metric(sum(r["top1"] for r in low), len(low)),
            "top1_rest": metric(sum(r["top1"] for r in rest), len(rest)),
        },
        "per_condition": per_condition,
    }


# ─── Written-cases section ───────────────────────────────────────────────────

def _predict_after_questions(result: dict, case: dict):
    """(top1, top3, zone) for one accepted/confirmed extraction, after the two
    rounds of follow-up questions answered from the DDXPlus record (as score())."""
    present = [e for e in result["present"] if e in m._feat_idx]
    absent = [e for e in result["absent"] if e in m._feat_idx]
    if present:
        present, absent = ce.simulate_questions(present, absent, result["age"], result["sex"], set(case["record"]))
    dist = m.predict_distribution(present, result["age"], result["sex"], absent) if present else {}
    truth = m._GROUP_MAP.get(case["pathology"], case["pathology"])
    top = list(dist)
    zone = triage.assess(dist, present, result.get("review")).zone
    return bool(top) and top[0] == truth, truth in top[:3], zone


def _extraction_scores(cases: List[dict], accepted: Dict[str, dict]) -> dict:
    """Same matching rules as compare_extraction.score, over every case with a result."""
    tp = fn = fp = absent_tp = absent_gold = red_tp = red_gold = 0
    for case in cases:
        r = accepted[case["id"]]
        present, absent = set(r["present"]), set(r["absent"])
        tp_case = sum(ce._matches(g, present, case) for g in case["present"])
        tp += tp_case
        fn += len(case["present"]) - tp_case
        fp += sum(not ce._acceptable(p, case) for p in present)
        absent_gold += len(case["absent"])
        absent_tp += sum(ce._matches(g, absent, case) for g in case["absent"])
        red = [g for g in case["present"] if g in ce.extraction.SAFETY_NET]
        red_gold += len(red)
        red_tp += sum(ce._matches(g, present, case) for g in red)
    return {
        "precision": metric(tp, tp + fp),
        "recall": metric(tp, tp + fn),
        "denials": metric(absent_tp, absent_gold),
        "red_flag_recall": metric(red_tp, red_gold),
    }


def written_section(cases: List[dict], cache: Dict[str, dict], cases_name: str, backend: str) -> dict:
    done = [c for c in cases if c["id"] in cache]
    if len(done) < len(cases):
        print(f"  {len(cases) - len(done)} of {len(cases)} cases have no cached result; scoring the rest",
              file=sys.stderr)
    by_id = {c["id"]: c for c in done}
    accepted = {}
    for cid, case in by_id.items():
        record = cache[cid]
        if backend.startswith("clef"):  # same re-interpretation as compare_extraction.main
            record = ce.rederive(record, case)
        accepted[cid] = ce.apply_net(record, case)
    confirmed = {cid: ce.confirmed(r, by_id[cid]) for cid, r in accepted.items()}

    in_scope = [c for c in done if c["pathology"] is not None]
    oos = [c for c in done if c["pathology"] is None]

    def variant(results: Dict[str, dict]) -> dict:
        preds = {c["id"]: _predict_after_questions(results[c["id"]], c) for c in in_scope}
        reds = [c for c in in_scope if c["tier"] == "red"]
        greens = [c for c in in_scope if c["tier"] == "green"]
        return {
            "top1": metric(sum(preds[c["id"]][0] for c in in_scope), len(in_scope)),
            "top3": metric(sum(preds[c["id"]][1] for c in in_scope), len(in_scope)),
            "red_sensitivity": metric(sum(preds[c["id"]][2] == "red" for c in reds), len(reds)),
            "red_to_green": metric(sum(preds[c["id"]][2] == "green" for c in reds), len(reds)),
            "green_to_red": metric(sum(preds[c["id"]][2] == "red" for c in greens), len(greens)),
        }, preds

    accepts, _ = variant(accepted)
    confirms, confirm_preds = variant(confirmed)

    by_language = {}
    for lang in list(LANGS) + sorted({c["lang"] for c in in_scope} - set(LANGS)):
        group = [c for c in in_scope if c["lang"] == lang]
        by_language[lang] = {
            "n": len(group),
            "top1": metric(sum(confirm_preds[c["id"]][0] for c in group), len(group)),
        }

    flagged = [bool((accepted[c["id"]].get("review") or {}).get("out_of_scope")) for c in done]
    flags_oos = [f for c, f in zip(done, flagged) if c["pathology"] is None]
    flags_in = [f for c, f in zip(done, flagged) if c["pathology"] is not None]

    return {
        "source": f"{cases_name}: {len(cases)} free-text complaints (ru, kk, ru-kk mixed, en), held out",
        "cases_file": cases_name,
        "n_in_scope": len(in_scope),
        "n_out_of_scope": len(oos),
        "extraction": _extraction_scores(done, accepted),
        "patient_accepts": accepts,
        "patient_confirms": confirms,
        "by_language": by_language,
        "out_of_scope": {
            "flagged": metric(sum(flags_oos), len(flags_oos)),
            "false_flags": metric(sum(flags_in), len(flags_in)),
        },
    }


def load_cases(path: Path) -> List[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# ─── Report ──────────────────────────────────────────────────────────────────

def build_report(patients_path: str, cases_path: Path, backend: str, tag: str, n: int, seed: int) -> dict:
    constants = read_backend_constants()
    labels = constants["disease_labels"]
    synthetic = synthetic_section(patients_path, n, seed, labels)

    cache_name = ce.cache_name(backend, tag)
    cache = ce.load_cached(cache_name)
    if cache:
        written = written_section(load_cases(cases_path), cache, _rel(cases_path), backend)
        written["held_out"] = Path(cases_path).name in HELD_OUT_FILES
    else:
        print(f"  no cached results for {cache_name} ({ce.RESULTS / (cache_name + '.jsonl')}); "
              "written_cases is null", file=sys.stderr)
        written = None

    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_version": constants["MODEL_VERSION"],
        "extraction_backend": backend,
        "synthetic": synthetic,
        "written_cases": written,
        "caveats": [c.replace("{backend}", backend) for c in CAVEATS],
        "i18n": localized_text(backend, n, (written or {}).get("n_in_scope", 0) + (written or {}).get("n_out_of_scope", 0),
                               held_out=bool(written and written.get("held_out"))),
    }


def print_summary(report: dict) -> None:
    def fmt(mo: Optional[dict]) -> str:
        if not mo or mo["value"] is None:
            return "—"
        return f"{mo['value']:.1%} [{mo['ci'][0]:.1%}, {mo['ci'][1]:.1%}] (k={mo['k']}, n={mo['n']})"

    syn = report["synthetic"]
    print("synthetic:")
    for key, mo in syn["metrics"].items():
        print(f"  {key:16} {fmt(mo)}")
    print(f"  questions_mean   {syn['questions_mean']}")
    print(f"  low_confidence   share {fmt(syn['low_confidence']['share'])}  "
          f"top1 low {fmt(syn['low_confidence']['top1_low'])}  rest {fmt(syn['low_confidence']['top1_rest'])}")
    written = report["written_cases"]
    if written:
        print("written_cases (patient_confirms):")
        for key, mo in written["patient_confirms"].items():
            print(f"  {key:16} {fmt(mo)}")
        print("written_cases extraction:")
        for key, mo in written["extraction"].items():
            print(f"  {key:16} {fmt(mo)}")
    else:
        print("written_cases: null")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patients", required=True, help="DDXPlus release_test_patients CSV")
    ap.add_argument("--cases", default=str(HERE / "cases_v3.jsonl"))
    ap.add_argument("--backend", default="clef-flash")
    ap.add_argument("--tag", default="v3", help="cache suffix: eval/results/<backend>-<tag>.jsonl")
    ap.add_argument("--out", default=str(HERE / "accuracy_report.json"))
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    report = build_report(args.patients, Path(args.cases), args.backend, args.tag, args.n, args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print_summary(report)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
