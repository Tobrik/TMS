# Anamnes — TMS (Therapist Machine Support)

Symptom triage prototype for Kazakhstan: a patient describes symptoms in Russian, Kazakh or English, answers a few yes/no questions, and gets a preliminary differential plus a red / yellow / green urgency zone. Doctors see a triage dashboard of their patients. **MVP — not a medical device.**

- `Backend/` — FastAPI + PostgreSQL (`backend TMS.py`, deployed to Vercel via `main.py`)
- `frontend/` — Next.js 16 / React 19

## Pipeline

```
free text ──► /extract_symptoms ──► patient confirms ──► /next_questions ×2 ──► /analys ──► /explain
              LLM, quoted findings     what was understood   expected info gain     model + triage   LLM verbalises
```

| Stage | Module | How |
|---|---|---|
| Extraction | `extraction.py`, `extraction_clef.py` | Groq (default) or Cloudflare Clef (`EXTRACTION_BACKEND=clef`, falls back to Groq). Groq: strict-JSON LLM calls (`gpt-oss-120b`). Every finding is *present / absent / uncertain* and must quote the patient's words; unquotable findings are dropped (1 call by default; `LLM_EXTRACTION_SAMPLES=3` votes, minority findings become "uncertain"). Clef: one probability per symptom (asked with corrected DDXPlus glosses); ≥ 0.8 is pre-ticked for the patient (≥ 0.95 for chest pain at rest, which Clef over-reads), 0.25–0.8 shown as "?", denials pre-marked only at ≥ 0.8. A regex safety net adds missed red flags, and a rule-based denial net (`negation.py`, RU / KK / EN) adds denials the extractor missed ("кашля нет", "жөтел жоқ", "тамағым ауырмайды", "no fever"); it never denies pain or a pain location. Without an LLM, falls back to regex + BM25. |
| Safety review | `extraction.py` | Separate LLM call: emergency? complaint outside the 49 conditions (e.g. appendicitis)? Can only raise the triage zone. |
| Diagnosis | `ml_model.py` | Naive Bayes over per-disease evidence frequencies (`training sklearn/evidence_stats.json`). Uses reported *and* denied answers; age/sex only when stated. The RandomForest is still available (`POSTERIOR = "rf"`). |
| Questions | `qa_engine.py` | Expected information gain over diseases + triage tier; "no" answers count, "don't know" answers are only excluded from later questions. Up to 2 rounds of 3; stops at 70% confidence. A result that ends below 70% is returned with `low_confidence: true` and the patient is told to show all three options to a doctor. |
| Triage | `triage.py` | Worst plausible condition, not the top-1: red if ≥10% probability mass on red-tier diseases, or a red-flag symptom (chest pain at rest E_14, stridor E_194), or a verified LLM emergency. Disease tiers combine the curated lists with DDXPlus severity. |
| Explanations | `/explain` | The LLM is given measured leave-one-out effects of each answer and writes patient / doctor text; it is told not to invent symptom weights. |
| LLM access | `llm.py` | One place for all calls; model fallback lists from env; schema → JSON-mode fallback. |

### Why naive Bayes instead of the RandomForest

`Backend/eval/simulate.py` replays 3,000 held-out DDXPlus test patients: each volunteers their chief complaint + 2 symptoms (+ one pain location), then answers the engine's questions truthfully.

| Pipeline | Top-1 | Top-3 | Red under-triage | Over-triage |
|---|---|---|---|---|
| Full patient record, no questions (upper bound) — RF / naive Bayes | 98.7% / 98.5% | 99.8% / 100% | — | — |
| v1 (JSD questions, "no" dropped, RF, top-1 triage) | 62.7% | 76.4% | 29.7% (9.2% sent green) | 5.7% |
| v2 questions + RF | 65.0% | 78.0% | 0.0% | 67.0% |
| **v2 (naive Bayes)** | **95.8%** | **99.5%** | **0.0%** | **4.9%** |
| v2, one wrongly extracted symptom in 30% of patients | 95.0% | 99.2% | 0.4% | 5.6% |
| v2, realistic patients: that, plus 10% of answers wrong and 20% "don't know" | 90.2% | 97.7% | 0.8% (none sent green) | 6.8% |

Since 2026-11-22 Chagas counts only after travel to Central or South America (`_TRAVEL_ONLY` in `ml_model.py`): its synthetic DDXPlus profile used to top vague headache complaints. The simulation skips DDXPlus Chagas patients without that travel, as it skips Ebola. Rows above that line were measured before this change.

In the realistic setting, 10% of patients end below 70% confidence; they get 54.9% top-1 against 94.1% for the rest, hence the `low_confidence` flag. Asking more (3 rounds, or a 0.8 / 0.9 stop threshold) did not help: with 10% wrong answers, extra questions add as much noise as information.

Red under-triage = truly red-tier patients not sent to the red zone; over-triage = green-tier patients sent to red. The triage threshold (`RED_MASS_THRESHOLD = 0.10`) is tuned for naive Bayes: the RF's posteriors are too flat (mean top-1 probability 0.23), so with the RF the same rule sends two thirds of green patients to red — re-tune with the sweep the script prints before switching back.

The RF was trained on complete DDXPlus records (~20 evidences each, 0 = "absent"), so a chat's handful of evidences reads to it as "denies everything else". Caveat: DDXPlus is synthetic and generated from per-disease symptom probabilities, which suits naive Bayes; none of these numbers say anything about real patients. The LLM extraction step is not part of the simulation.

### Real-world accuracy (doctor feedback)

Doctors confirm the actual diagnosis per diary day on the dashboard ("Подтвердить диагноз": one of the 46 conditions or "other", plus the zone). `POST /doctor_feedback` stores it next to the model's top-1, top-3, zone and `MODEL_VERSION`, append-only. `GET /feedback_stats` gives top-1 / top-3 agreement, zone agreement and under- / over-triage counts per model version (shown in the "Точность модели" panel), and `python eval/export_real_cases.py OUT.jsonl [--with-text]` turns confirmed cases into an eval set in the `eval/cases.jsonl` format. This is the only measure of accuracy on real patients; everything else in this README is synthetic or self-written.

### Accuracy report for doctors

`/accuracy` (doctor login; `/accuracy?mock=1` shows sample data without login) presents three levels of evidence, most trustworthy first, each number with its 95% Wilson interval and n: diagnoses confirmed by doctors (live, from `/doctor_feedback`), held-out written complaints, and the synthetic DDXPlus simulation. It also shows triage safety, results by language, the weakest conditions and calibration, with the caveats in Russian, Kazakh and English. Print / PDF works from the page.

The benchmark part is a snapshot, `Backend/eval/accuracy_report.json`, served with the live numbers by `GET /accuracy_report`. Rebuild it after any change to the model, prompts or thresholds:

```bash
cd Backend
python eval/compare_extraction.py --cases eval/cases_v3.jsonl --backends clef-flash --tag v3   # extraction on the held-out set (API calls)
python eval/build_report.py --patients path/to/release_test_patients --cases eval/cases_v3.jsonl --backend clef-flash --tag v3
```

`eval/cases_v3.jsonl` is the only set that was never used for tuning (`HELD_OUT_FILES` in `build_report.py`); `cases.jsonl` and `cases_v2.jsonl` were used to tune prompts, the denial rules and thresholds. A report built from them is labelled preliminary on the page.

Free-tier limits decide how much can be evaluated (and served) per day: Cloudflare Workers AI gives 10,000 neurons a day, about 290 Clef-flash complaints; Groq gives about 200K tokens a day per model, about 35 complaints with the safety review. A backend that runs out falls back (Clef → Groq → regex), so a busy day silently degrades extraction.

## Running

```bash
# Backend
cd Backend
cp .env.example .env            # fill in DATABASE_URL, JWT_SECRET_KEY, DB_ENCRYPTION_KEY, GROQ_API_KEY
pip install -r requirements.txt  # Python ≥ 3.10 (scikit-learn 1.8 / NumPy 2)
uvicorn main:app --reload

# Frontend (NEXT_PUBLIC_API_URL defaults to http://localhost:8000; no LLM key needed)
cd frontend && npm install && npm run dev
```

## Tests and evaluation

```bash
cd Backend
pip install -r requirements-dev.txt
pytest                          # unit tests + end-to-end API test on a throwaway Postgres (pgserver), fake Groq

# Offline simulation (download DDXPlus from https://figshare.com/articles/dataset/DDXPlus_Dataset/20043374)
python eval/simulate.py path/to/release_test_patients --n 3000 [--noise 0.3]

# Realistic patients: one wrongly extracted symptom in 30% of cases, 10% of answers wrong, 20% "don't know"
python eval/simulate.py path/to/release_test_patients --n 3000 --pipelines NB --noise 0.3 --answer-error 0.1 --dont-know 0.2 --calibration

# Rebuild evidence_stats.json (use the validate split; the test split is for simulate.py)
python "training sklearn/build_evidence_stats.py" path/to/release_validate_patients
```

### Comparing extraction backends (Groq vs Cloudflare Clef)

`eval/cases.jsonl` holds 46 free-text complaints (23 RU, 12 KK, 11 EN) written from held-out DDXPlus test patients, plus 4 outside the 49 conditions. Each lists the codes the text states, the ones it explicitly denies, and the true disease.

```bash
python eval/compare_extraction.py --backends gold,regex,groq,clef-flash,clef --errors
```

It scores findings (precision / recall / denials / negation errors / red flags, per language), then runs each backend's findings through the same naive Bayes + triage, with and without the follow-up questions, and reports tokens, cost and latency. `--confirm` adds each backend after the patient's confirmation step; `--score-only` re-scores cached results without API calls (Clef results are re-derived from saved probabilities, so threshold changes are free to re-test). Results are cached in `eval/results/`; reruns only spend quota on unfinished cases. Free tiers: Groq's base limits (8K tokens/min, 200K/day) cover ~30 cases a day; Clef-flash all 46 in one go.

First run (2026-10-15, the 26 cases both backends had finished; written by me, so cleaner than real messages, and 1 case = 4 points):

| | Groq | Clef-flash, patient accepts defaults | Clef-flash, patient confirms |
|---|---|---|---|
| Findings F1 / recall | 79% / 75% | 80% / 82% | 98% / 96% |
| Explicit denials caught | 19% | 81% | 81% |
| Top-1 / top-3 after questions | 88% / 92% | 88% / 100% | 96% / 100% |
| Green-tier sent red | 6/14 | 3/14 | 1/14 |
| Chips the patient must fix | 0.6 (confirming) | — | 1.7 |
| $ per 1,000 complaints | $1.27 | $0.84 | |

Clef's own emergency score barely separated red-tier from green-tier cases, so it only raises a soft (yellow) warning; red comes from confirmed red-flag symptoms and the model's risk mass, which missed no red case with either backend.

Data: [DDXPlus](https://arxiv.org/abs/2205.09148) (Fansi Tchango et al., NeurIPS 2022), CC BY 4.0 — `release_evidences.json`, `release_conditions.json` and the statistics derived from the validation split.
