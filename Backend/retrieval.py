"""
Retrieval-based symptom extraction for TMS.

Architecture: Vocabulary mapping (boss) + BM25 (fallback for unknowns).

Flow: Patient text → vocabulary lookup → exact E_XX codes → RF classifier.
No LLaMA needed for extraction. LLaMA used ONLY for explanation (frontend).

The vocabulary maps Russian patient terms directly to DDXPlus evidence IDs.
BM25 is kept as a fallback for symptoms not covered by the vocabulary.
"""

import re
import json
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Set

logger = logging.getLogger("tms")

_BASE = Path(__file__).parent
_TRAINING_DIR = _BASE / "training sklearn"
_EVIDENCES_PATH = _TRAINING_DIR / "release_evidences.json"

# ─── Direct vocabulary: Russian term → DDXPlus evidence IDs ──────────────────
# Each entry: (regex_pattern, [evidence_ids]).
# Pattern matched against LOWERCASE patient text.
# No \b word boundaries — they don't work reliably with Cyrillic in Python re.
# Patterns use spacing and character anchors instead.

_VOCAB: List[Tuple[str, List[str]]] = [
    # ── Respiratory ────────────────────────────────────────────────────────────
    (r"насморк|ринит|заложен.{0,10}нос|нос.{0,10}заложен|течёт\s+нос|сопли",
     ["E_181"]),  # nasal congestion / runny nose
    (r"гнойн\w+.{0,15}выдел|жёлт\w+.{0,10}выдел|зелён\w+.{0,10}выдел|гнойн\w+\s+насморк",
     ["E_182"]),  # greenish/yellowish nasal discharge
    (r"каш\w+",
     ["E_201"]),  # cough (каш\w+ covers кашель, кашляет, кашлять)
    (r"мокрот\w+|продуктивн\w+\s+каш\w+|каш\w+\s+с\s+мокрот\w+|жёлт\w+\s+мокрот\w+|зелён\w+\s+мокрот\w+",
     ["E_77"]),   # colored/abundant sputum
    (r"боль.{0,20}при\s+вдох|больно\s+дышать|боль.{0,20}при\s+дыхани|боль.{0,20}вдох|грудь.{0,10}болит.{0,10}дыш|при\s+глубок\w+\s+вдох",
     ["E_220"]),  # pain on deep breathing (pleuritic)
    (r"одышк\w+|задыха\w+|трудно\s+дышать|нехватка\s+воздуха",
     ["E_66"]),   # shortness of breath
    (r"свист\w+.{0,10}дыхани|хрипит|хриплое\s+дыхани|свист\w+.{0,10}выдох|хрипы\s+при\s+выдох",
     ["E_214"]),  # wheezing on exhale
    (r"стридор|свист\w+.{0,10}вдох|шумн\w+\s+дыхани",
     ["E_112"]),  # stridor / wheezing on inhale
    (r"боль.{0,10}горл|горло.{0,10}бол|глотать\s+больно|больно\s+глотать|фарингит|тонзиллит|горло\s+дер\w+|болит\s+горло",
     ["E_97", "E_53", "E_55_@_V_148"]),  # sore throat → pain present + pharynx location
    (r"чихани|чихает|чихаю",
     ["E_181"]),  # sneezing

    # ── Fever / Systemic ───────────────────────────────────────────────────────
    (r"температур\w+|лихорадк\w+|горячк\w+|повышен\w+\s+температур|(?:^|\s)жар(?:\s|,|\.)",
     ["E_91"]),   # fever
    (r"озноб|дрожь|дрожит|знобит|трясёт",
     ["E_94"]),   # chills / shivers
    (r"ломот\w+|мышечн\w+\s+боль|боли\s+в\s+мышц\w+|ноет\s+всё\s+тело|боль\s+во\s+всём\s+тел",
     ["E_144"]),  # diffuse muscle pain
    (r"слабост\w+|усталост\w+|недомогани\w+|нет\s+сил|разбит\s",
     ["E_175"]),  # fatigue / malaise
    (r"потею|потливост\w+|сильно\s+потею|пот\s+льёт|испарина",
     ["E_50"]),   # increased sweating

    # ── Cardiovascular ─────────────────────────────────────────────────────────
    (r"боль.{0,10}груди|грудн\w+\s+боль|боль.{0,10}грудин|стенокарди|грудь\s+болит|сжимает\s+грудь|давит\s+в\s+груди|сжатие\s+в\s+груди",
     ["E_14", "E_53", "E_55_@_V_101"]),  # chest pain → pain present + upper chest location
    (r"сердцебиени\w+|тахикарди\w+|сердце\s+бьётся\s+быстро|учащённ\w+\s+пульс|серд\w+\s+колотится",
     ["E_155"]),  # palpitations / racing heart
    (r"головокружени\w+|кружится\s+голова|голова\s+кружится|вертиго",
     ["E_82"]),   # lightheadedness / dizziness
    (r"обморок|потерял\s+сознани|потеря\s+сознани|упал\s+в\s+обморок",
     ["E_82"]),   # near-syncope
    (r"отдаёт.{0,20}лев\w+.{0,10}(?:рук|плеч|лопатк)|боль.{0,10}лев\w+\s+рук|иррадиир\w+.{0,20}лев\w+|лев\w+.{0,10}(?:рук|плеч).{0,10}(?:боль|отдаёт)",
     ["E_14", "E_53", "E_57_@_V_195"]),  # pain radiates to left arm/shoulder
    (r"отдаёт.{0,20}(?:челюст|jaw|нижн\w+\s+челюст)|боль.{0,10}челюст|челюст\w+\s+боль",
     ["E_53", "E_57_@_V_121"]),  # pain radiates to jaw
    (r"боль\s+отдаёт|иррадиир\w+|отдаёт.{0,15}плеч",
     ["E_14", "E_53"]),  # generic radiation (unspecified direction)

    # ── Gastrointestinal ───────────────────────────────────────────────────────
    (r"тошнот\w+|тошнит|чувство\s+тошнот",
     ["E_148"]),  # nausea
    (r"рвот\w+|вырвало|стошнило|блеват|рвёт",
     ["E_148"]),  # vomiting (E_211 = REPEATED vomiting, only on explicit "несколько раз рвало")
    (r"несколько\s+раз\s+рвало|многократная\s+рвота|рвота\s+несколько\s+раз|повторная\s+рвота",
     ["E_211"]),  # repeatedly vomited
    (r"боль.{0,10}живот|живот.{0,10}бол|болит\s+живот|брюшн\w+\s+боль",
     ["E_53", "E_55_@_V_187"]),  # abdominal pain → pain present + belly location
    (r"боль.{0,15}прав\w+.{0,10}бок|боль.{0,10}справа\s+внизу|прав\w+\s+подвздошн|прав\w+.{0,10}нижн\w+.{0,10}(?:живот|бок)",
     ["E_53", "E_55_@_V_87"]),   # RLQ pain → pain present + right iliac fossa location
    (r"понос|диарея|жидкий\s+стул|частый\s+стул",
     ["E_51"]),   # diarrhea
    (r"изжога|кислый\s+привкус|кислотн\w+\s+рефлюкс",
     ["E_173", "E_53", "E_55_@_V_197", "E_54_@_V_181"]),  # GERD → pain + epigastric + burning
    (r"жжение.{0,10}груди|жжёт\s+в\s+груди|жжение\s+за\s+грудин",
     ["E_173", "E_53", "E_55_@_V_101", "E_54_@_V_181"]),  # burning in chest
    (r"давящ\w+\s+боль|тяжест\w+\s+в\s+груди|давит\s+(?:в\s+груди|на\s+грудь)|сжимающ\w+\s+боль|тяжесть.{0,10}груди",
     ["E_53", "E_54_@_V_183"]),  # pressing/heavy pain character

    # ── Neurological / Musculoskeletal ─────────────────────────────────────────
    (r"нет\s+запаха|потеря.{0,10}(?:запаха|обоняния)|не\s+чувств\w+\s+запах|запах\s+пропал",
     ["E_103"]),  # loss of smell/olfaction
    (r"онемени\w+|покалывани\w+|немеет|онемел",
     ["E_177"]),  # numbness / tingling

    # ── Skin ───────────────────────────────────────────────────────────────────
    (r"сыпь|высыпани\w+|крапивниц\w+|пятна\s+на\s+коже|волдыр\w+|кожн\w+\s+сыпь",
     ["E_129"]),  # skin lesion / rash
    (r"зуд|чешется|сильн\w+\s+зуд",
     ["E_129"]),  # itching

    # ── Eyes / ENT ─────────────────────────────────────────────────────────────
    (r"глаза\s+красн\w+|красн\w+\s+глаза|конъюнктивит|белки\s+красн",
     ["E_74"]),   # eye redness
    (r"лимфоузл\w+|лимфатическ\w+\s+узл|узл\w+\s+увеличен|шея.{0,15}(?:увеличен|опухл)",
     ["E_9"]),    # swollen lymph nodes

    # ── History / Context ─────────────────────────────────────────────────────
    (r"контакт\s+с\s+больн|заразил|больн\w+\s+в\s+семье|общался\s+с\s+больн",
     ["E_41"]),   # contact with person with similar symptoms
    (r"недавно\s+(?:болел|простыл|простудил)|простыл|была\s+простуда",
     ["E_116"]),  # cold in last 2 weeks
    (r"аллерги\w+|аллергическ",
     ["E_169"]),  # allergic context
    (r"астм\w+|бронхиальн\w+\s+астм",
     ["E_123"]),  # asthma/COPD history
    (r"сердечн\w+\s+недостаточност",
     ["E_106"]),  # heart failure history
]

# ─── BM25 synonym expansion (for fallback BM25 only) ─────────────────────────
_BM25_SYNONYMS: Dict[str, List[str]] = {
    "температура": ["fever", "temperature"],
    "жар": ["fever", "temperature"],
    "озноб": ["chills", "shivers"],
    "ломота": ["muscle pain", "body aches"],
    "насморк": ["runny nose", "nasal discharge"],
    "кашель": ["cough"],
    "мокрота": ["sputum", "phlegm"],
    "одышка": ["shortness breath"],
    "тошнота": ["nausea"],
    "рвота": ["vomit"],
    "боль в груди": ["chest pain"],
    "потею": ["sweating"],
    "головокружение": ["dizziness"],
    "горло болит": ["sore throat"],
    "боль в животе": ["abdominal pain"],
    "сыпь": ["rash"],
    "зуд": ["itching"],
    "сердцебиение": ["palpitations"],
    "слабость": ["fatigue", "weakness"],
    "усталость": ["fatigue"],
}


def _tokenize(text: str) -> List[str]:
    """Simple multilingual tokenizer: lowercase, split on non-alphanumeric."""
    text = text.lower()
    tokens = re.findall(r"[a-zа-яёüöäéàèùâêîôûç]+", text)
    return tokens


def _expand_query(text: str) -> str:
    """Expand Russian patient text with English medical synonyms for BM25."""
    text_lower = text.lower()
    extra = []
    for ru_term, expansions in _BM25_SYNONYMS.items():
        if ru_term in text_lower:
            extra.extend(expansions)
    return text + " " + " ".join(extra) if extra else text


def _extract_age_sex(text: str) -> Tuple[Optional[int], Optional[str]]:
    """Extract age and sex from Russian/English patient text using regex."""
    age: Optional[int] = None
    sex: Optional[str] = None

    age_patterns = [
        r"(\d+)\s*лет\b",
        r"(\d+)\s*год[а]?\b",
        r"мне\s+(\d+)",
        r"возраст\s*[:\-]?\s*(\d+)",
        r"age\s*[:\-]?\s*(\d+)",
        r"(\d+)\s*years?\s*old",
        r"(\d+)[- ]year[- ]old",
    ]
    for pat in age_patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            candidate = int(m.group(1))
            if 1 <= candidate <= 120:
                age = candidate
                break

    female_re = r"женщина|женщине|женского|девушка|woman|female|girl"
    male_re = r"мужчина|мужчине|мужского|парень|мальчик|man\b|male|boy"

    if re.search(female_re, text, re.IGNORECASE):
        sex = "F"
    elif re.search(male_re, text, re.IGNORECASE):
        sex = "M"

    return age, sex


def _vocab_extract(text: str) -> List[str]:
    """
    Direct vocabulary lookup: scan patient text for Russian symptom terms,
    return the corresponding DDXPlus evidence IDs.
    High precision — only returns IDs when pattern clearly matches.
    """
    text_lower = text.lower()
    matched: Set[str] = set()

    for pattern, ev_ids in _VOCAB:
        if re.search(pattern, text_lower):
            matched.update(ev_ids)

    return list(matched)


# ─── BM25 retriever (fallback for unknown symptoms) ──────────────────────────

class EvidenceRetriever:
    """
    BM25-based retrieval over DDXPlus evidence questions.
    Used as FALLBACK when vocabulary extraction returns few evidences.
    """

    def __init__(self, evidences_path: Path = _EVIDENCES_PATH):
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError("rank_bm25 not installed. Run: pip install rank-bm25")

        with open(evidences_path, encoding="utf-8") as f:
            self._meta: Dict = json.load(f)

        from ml_model import _QUESTION_TRANSLATIONS
        self._ids: List[str] = []
        corpus_tokens: List[List[str]] = []

        for ev_id, meta in self._meta.items():
            q_en = meta.get("question_en", "")
            ru_trans = _QUESTION_TRANSLATIONS.get(ev_id, {})
            q_ru = ru_trans.get("ru", "") if isinstance(ru_trans, dict) else ""
            q_kk = ru_trans.get("kk", "") if isinstance(ru_trans, dict) else ""

            combined = f"{q_en} {q_ru} {q_kk}"
            tokens = _tokenize(combined)
            if tokens:
                self._ids.append(ev_id)
                corpus_tokens.append(tokens)

        self._bm25 = BM25Okapi(corpus_tokens)
        logger.info("EvidenceRetriever: indexed %d evidences", len(self._ids))

    def retrieve(self, text: str, top_k: int = 20) -> List[Tuple[str, float]]:
        """Retrieve top_k most relevant evidence IDs."""
        expanded = _expand_query(text)
        query_tokens = _tokenize(expanded)

        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)
        results = sorted(
            zip(self._ids, scores.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
        return [(ev_id, score) for ev_id, score in results[:top_k] if score > 0]

    def extract_direct(
        self,
        text: str,
        top_k: int = 25,
        bm25_fallback_threshold: float = 8.0,
        bm25_fallback_max: int = 5,
    ) -> Tuple[List[str], Optional[int], Optional[str]]:
        """
        Main extraction method — vocabulary first, BM25 fallback for unknowns.

        Steps:
          1. Vocabulary lookup: precise Russian→E_XX direct mapping.
          2. If vocabulary returns <4 evidences, add BM25 top candidates
             (only high-confidence: score > 8.0) to fill gaps.
          3. Extract age and sex via regex.

        Returns: (evidences: List[str], age: Optional[int], sex: Optional[str])
        """
        age, sex = _extract_age_sex(text)

        # Step 1: Vocabulary lookup (high precision)
        vocab_evs = _vocab_extract(text)
        matched_set: Set[str] = set(vocab_evs)

        logger.info(
            "vocab_extract: %r -> %d evidences: %s",
            text[:60], len(vocab_evs), vocab_evs,
        )

        # Step 2: BM25 fallback — only if vocabulary found few evidences
        if len(matched_set) < 4:
            try:
                candidates = self.retrieve(text, top_k=top_k)
                added = 0
                for ev_id, score in candidates:
                    if score < bm25_fallback_threshold:
                        break
                    meta = self._meta.get(ev_id, {})
                    # Skip: categorical, antecedent (history), already matched
                    if meta.get("data_type", "B") != "B":
                        continue
                    if meta.get("is_antecedent", False):
                        continue
                    if ev_id in matched_set:
                        continue
                    matched_set.add(ev_id)
                    added += 1
                    if added >= bm25_fallback_max:
                        break
                if added:
                    logger.info("BM25 fallback added %d high-confidence evidences", added)
            except Exception as e:
                logger.warning("BM25 fallback failed: %s", e)

        evidences = list(matched_set)
        logger.info(
            "extract_direct result: %r -> %d evidences (age=%s sex=%s): %s",
            text[:60], len(evidences), age, sex, evidences,
        )
        return evidences, age, sex

    def build_validation_prompt(self, text: str, candidates: List[Tuple[str, float]]) -> str:
        """Build a LLaMA validation prompt (kept for optional hybrid mode)."""
        lines = [
            "You are a medical symptom validator. A patient described their symptoms below.\n"
            "Based ONLY on what the patient explicitly says, select which of the following symptoms are PRESENT.\n\n"
            f"Patient text: \"{text}\"\n\n"
            "Candidate symptoms (only choose from this list):\n"
        ]
        for ev_id, _score in candidates:
            meta = self._meta.get(ev_id, {})
            q_en = meta.get("question_en", ev_id)
            dtype = meta.get("data_type", "B")
            if dtype == "B":
                lines.append(f"- {ev_id}: {q_en}")
            else:
                vals = meta.get("possible-values", [])
                vm = meta.get("value_meaning", {})
                val_list = []
                for v in vals[:6]:
                    m = vm.get(v, {})
                    en = m.get("en", v) if isinstance(m, dict) else v
                    val_list.append(f"{v}={en}")
                lines.append(
                    f"- {ev_id}: {q_en} [values: {', '.join(val_list)}"
                    f"{'...' if len(vals) > 6 else ''}]"
                )
        lines.append(
            "\nRules:\n"
            "- ONLY include evidences clearly mentioned or strongly implied by the patient text\n"
            "- For categorical/multi-choice evidences, use E_XX_@_V_YY format\n"
            "- Do NOT include evidences not supported by the text\n"
            "- Extract age and sex if mentioned\n\n"
            "Return ONLY valid JSON (no markdown):\n"
            '{"age": <integer or null>, "sex": "<M or F or null>", "evidences": ["E_XX", ...]}'
        )
        return "\n".join(lines)


# ─── Singleton ────────────────────────────────────────────────────────────────
_retriever: EvidenceRetriever | None = None


def get_retriever() -> EvidenceRetriever:
    global _retriever
    if _retriever is None:
        _retriever = EvidenceRetriever()
    return _retriever
