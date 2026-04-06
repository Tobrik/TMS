"""
Pipeline test: Text → LLaMA extraction → sklearn prediction
Run: python test_pipeline.py

Requires:
  - GROQ_API_KEY in environment or .env file
  - model.pkl, feature_names.json, label_classes.json in 'training sklearn/'
  - release_evidences.json in 'training sklearn/'
"""

import os
import sys
import json
import httpx
import asyncio
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))
from ml_model import sklearn_predict, LABEL_CLASSES, FEATURE_NAMES

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.3-70b-versatile"

# ─── Test cases: (description, expected_top_disease) ──────────────────────────
TEST_CASES = [
    {
        "id": "flu",
        "lang": "RU",
        "text": "У меня высокая температура 39, ломота в теле, сильная головная боль, насморк и кашель. "
                "Болею уже 3 дня. Мне 28 лет, я мужчина.",
        "expected_hint": "Influenza",
    },
    {
        "id": "appendicitis",
        "lang": "RU",
        "text": "Сильная боль в правом нижнем животе, тошнота, рвота один раз. "
                "Боль усилилась за последние 12 часов. Женщина, 22 года.",
        "expected_hint": "Appendicitis",
    },
    {
        "id": "angina",
        "lang": "EN",
        "text": "I have crushing chest pain radiating to my left arm and jaw. "
                "I'm sweating a lot and feel short of breath. Male, 58 years old.",
        "expected_hint": "STEMI",
    },
    {
        "id": "migraine",
        "lang": "RU",
        "text": "Сильная пульсирующая головная боль с одной стороны, тошнота, "
                "очень чувствителен к свету и звуку. Женщина, 34 года.",
        "expected_hint": "Migraine",
    },
    {
        "id": "pneumonia",
        "lang": "EN",
        "text": "I have fever, productive cough with yellow-green sputum, chest pain when breathing, "
                "and feel very tired. Male, 45 years old.",
        "expected_hint": "Pneumonia",
    },
]


async def extract_evidences(text: str) -> dict:
    """Call Groq to extract DDXPlus evidence tokens from free text (legacy LLaMA path)."""
    if not GROQ_API_KEY:
        return {"error": "GROQ_API_KEY not set", "evidences": [], "age": 25, "sex": "M"}

    system_prompt = (
        "You are a medical symptom extractor. Map patient descriptions to DDXPlus evidence IDs. "
        "Return JSON: {\"age\": int, \"sex\": \"M\"/\"F\", \"evidences\": [\"E_XX\", ...]}"
    )

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text},
                ],
                "temperature": 0.0,
                "max_tokens": 512,
            },
        )

    if resp.status_code != 200:
        return {"error": f"HTTP {resp.status_code}: {resp.text}", "evidences": [], "age": 25, "sex": "M"}

    raw = resp.json()["choices"][0]["message"]["content"].strip()

    # Strip markdown code blocks if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {"error": f"JSON parse failed: {raw[:200]}", "evidences": [], "age": 25, "sex": "M"}

    age = parsed.get("age") or 25
    sex = parsed.get("sex") or "M"
    evidences = [e for e in (parsed.get("evidences") or []) if isinstance(e, str)]

    return {"evidences": evidences, "age": int(age), "sex": str(sex).upper()}


def check_evidences_validity(evidences: list) -> dict:
    """Check how many extracted evidences are valid feature tokens."""
    valid = [e for e in evidences if e in FEATURE_NAMES]
    invalid = [e for e in evidences if e not in FEATURE_NAMES]
    return {"valid": valid, "invalid": invalid, "coverage": len(valid) / max(len(evidences), 1)}


async def run_test(case: dict) -> dict:
    print(f"\n{'='*60}")
    print(f"[{case['id'].upper()}] {case['lang']}")
    print(f"Text: {case['text'][:80]}...")
    print(f"Expected hint: {case['expected_hint']}")
    print("-" * 60)

    # Step 1: Extract
    extraction = await extract_evidences(case["text"])

    if extraction.get("error"):
        print(f"EXTRACTION ERROR: {extraction['error']}")
        return {"id": case["id"], "status": "extraction_error", "error": extraction["error"]}

    evidences = extraction["evidences"]
    age = extraction["age"]
    sex = extraction["sex"]

    print(f"Extracted {len(evidences)} evidences | age={age} | sex={sex}")
    print(f"Evidences: {evidences}")

    # Step 2: Validate tokens
    validity = check_evidences_validity(evidences)
    print(f"Valid tokens: {len(validity['valid'])}/{len(evidences)} ({validity['coverage']*100:.0f}%)")
    if validity["invalid"]:
        print(f"INVALID tokens (not in feature set): {validity['invalid']}")

    if len(validity["valid"]) == 0:
        print("NO VALID EVIDENCES — prediction will be random!")
        return {"id": case["id"], "status": "no_valid_evidences", "evidences": evidences}

    # Step 3: Predict
    predictions = sklearn_predict(validity["valid"], age=age, sex=sex, top_n=5)

    print(f"\nTop-5 predictions:")
    for i, (disease, prob) in enumerate(predictions, 1):
        marker = " ✓" if case["expected_hint"].lower() in disease.lower() else ""
        print(f"  {i}. {disease}: {prob*100:.1f}%{marker}")

    top_disease = predictions[0][0] if predictions else "Unknown"
    top_score = predictions[0][1] if predictions else 0.0
    hit = any(case["expected_hint"].lower() in d.lower() for d, _ in predictions[:3])

    return {
        "id": case["id"],
        "status": "ok",
        "evidences_count": len(evidences),
        "valid_count": len(validity["valid"]),
        "invalid_tokens": validity["invalid"],
        "top_disease": top_disease,
        "top_score": top_score,
        "hit_in_top3": hit,
    }


async def main():
    print("TMS Pipeline Test: Text → LLaMA Extraction → sklearn Prediction")
    print(f"Model: {GROQ_MODEL}")
    print(f"Feature set size: {len(FEATURE_NAMES)}")
    print(f"Disease classes: {len(LABEL_CLASSES)}")

    if not GROQ_API_KEY:
        print("\n⚠ GROQ_API_KEY not found in environment!")
        print("Set it in .env file: GROQ_API_KEY=gsk_...")
        print("Running prediction-only test with hardcoded evidences instead.\n")
        # Test sklearn directly with known DDXPlus evidence IDs for flu
        test_evidences = ["E_91", "E_108", "E_97", "E_83", "E_11"]
        print(f"Direct sklearn test with evidences: {test_evidences}")
        preds = sklearn_predict(test_evidences, age=28, sex="M", top_n=5)
        print("Predictions:")
        for i, (d, p) in enumerate(preds, 1):
            print(f"  {i}. {d}: {p*100:.1f}%")
        return

    results = []
    for case in TEST_CASES:
        result = await run_test(case)
        results.append(result)

    # ─── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    ok = [r for r in results if r["status"] == "ok"]
    hits = [r for r in ok if r["hit_in_top3"]]
    avg_evidences = sum(r.get("valid_count", 0) for r in ok) / max(len(ok), 1)
    any_invalid = [r for r in ok if r.get("invalid_tokens")]

    print(f"Tests run:            {len(results)}")
    print(f"Extraction success:   {len(ok)}/{len(results)}")
    print(f"Expected in Top-3:    {len(hits)}/{len(ok)}")
    print(f"Avg valid evidences:  {avg_evidences:.1f}")

    if any_invalid:
        print(f"\nTests with invalid tokens ({len(any_invalid)}):")
        for r in any_invalid:
            print(f"  [{r['id']}]: {r['invalid_tokens']}")

    print(f"\nConclusion:")
    if len(hits) == len(ok):
        print("  ✓ Pipeline looks solid — all expected diseases in Top-3")
    elif len(hits) >= len(ok) * 0.6:
        print("  ~ Pipeline mostly works — some misses, review invalid tokens above")
    else:
        print("  ✗ Pipeline has issues — LLaMA is not mapping symptoms correctly")
        print("    Check symptom extraction in retrieval.py vocabulary")


if __name__ == "__main__":
    asyncio.run(main())
