"""
QA Engine for two-round diagnostic questioning.

Round 1 — Symptom Exploration: maximise Jensen-Shannon divergence across the
full disease space (no confirmation bias toward any single diagnosis).

Round 2 — Disease Discrimination: maximise separation between top-1 and top-2
diagnoses once the symptom vector is sufficiently populated.

Uses the existing trained RandomForest model without retraining.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np
from scipy.spatial.distance import jensenshannon

logger = logging.getLogger("tms.qa_engine")

# ─── Thresholds ──────────────────────────────────────────────────────────────
EXPLORATION_EXIT_CONFIDENCE = 0.45   # move from round 1 → round 2
STOP_CONFIDENCE = 0.70               # stop asking altogether
MAX_ROUNDS = 2                       # round 1 + round 2
BROAD_SYMPTOM_THRESHOLD = 0.30       # evidence is "broad" if relevant to ≥30% diseases


# ─── Result dataclasses ─────────────────────────────────────────────────────

@dataclass
class QuestionCandidate:
    evidence_id: str
    question_ru: str
    question_kk: str
    score: float
    reason: str


@dataclass
class QAResult:
    questions: List[QuestionCandidate]
    should_stop: bool
    stop_reason: str
    top3: List[Tuple[str, float]]
    confidence: float
    round_type: str


class QAEngine:
    """Two-round question-selection engine on top of an existing RF classifier."""

    def __init__(
        self,
        clf,
        feature_names: List[str],
        label_classes: List[str],
        evidences_meta: dict,
        disease_relevant_evidences: Dict[str, Set[str]],
        question_translations: Dict[str, Dict[str, str]],
        evidence_prerequisites: Dict[str, str],
        never_ask: Set[str],
    ) -> None:
        self.clf = clf
        self.feature_names = feature_names
        self.label_classes = label_classes
        self.evidences_meta = evidences_meta
        self.disease_relevant_evidences = disease_relevant_evidences
        self.question_translations = question_translations
        self.evidence_prerequisites = evidence_prerequisites
        self.never_ask = never_ask

        self._feat_idx: Dict[str, int] = {
            name: i for i, name in enumerate(feature_names)
        }
        self._n_features = len(feature_names)

        # Binary evidence IDs that exist both in meta and in the feature vector
        self._binary_evs: List[str] = [
            ev_id
            for ev_id, meta in evidences_meta.items()
            if meta.get("data_type") == "B" and ev_id in self._feat_idx
        ]

        # Pre-compute "broad" evidences (relevant to ≥30% of diseases)
        self._broad_evidences: Set[str] = self._compute_broad_evidences()

    # ─── Internal helpers ────────────────────────────────────────────────

    def _compute_broad_evidences(self) -> Set[str]:
        """Evidences that appear in the relevant-evidence sets of ≥30% diseases."""
        if not self.disease_relevant_evidences:
            return set()
        n_diseases = len(self.disease_relevant_evidences)
        if n_diseases == 0:
            return set()
        threshold = n_diseases * BROAD_SYMPTOM_THRESHOLD
        ev_count: Dict[str, int] = {}
        for ev_set in self.disease_relevant_evidences.values():
            for ev in ev_set:
                ev_count[ev] = ev_count.get(ev, 0) + 1
        return {ev for ev, cnt in ev_count.items() if cnt >= threshold}

    def _build_feature_vector(
        self, evidences: List[str], age: int, sex: str
    ) -> np.ndarray:
        vec = np.zeros(self._n_features, dtype=np.float32)
        if "__AGE__" in self._feat_idx:
            vec[self._feat_idx["__AGE__"]] = min(max(age, 0), 120) / 100.0
        if sex.upper() == "M" and "__SEX_M__" in self._feat_idx:
            vec[self._feat_idx["__SEX_M__"]] = 1.0
        for ev in evidences:
            if ev in self._feat_idx:
                vec[self._feat_idx[ev]] = 1.0
        return vec

    def _get_proba(self, vec: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(vec.reshape(1, -1))[0]

    def _top3(self, proba: np.ndarray) -> List[Tuple[str, float]]:
        idx = np.argsort(proba)[::-1][:3]
        return [(self.label_classes[i], float(proba[i])) for i in idx]

    def _prereq_ok(self, ev_id: str, ev_set: Set[str]) -> bool:
        for prefix, parent in self.evidence_prerequisites.items():
            if ev_id == parent:
                return True
            if ev_id.startswith(prefix) and parent not in ev_set:
                return False
        return True

    def _base_candidates(self, ev_set: Set[str]) -> List[str]:
        """Return candidate evidence IDs after universal filters."""
        return [
            ev_id
            for ev_id in self._binary_evs
            if ev_id not in ev_set
            and not self.evidences_meta.get(ev_id, {}).get("is_antecedent", False)
            and ev_id not in self.never_ask
            and self._prereq_ok(ev_id, ev_set)
        ]

    # ─── Round 1: Symptom Exploration (JS divergence) ────────────────────

    def _score_exploration(
        self,
        candidates: List[str],
        base_vec: np.ndarray,
        base_proba: np.ndarray,
    ) -> List[Tuple[str, float]]:
        """
        Score each candidate by Jensen-Shannon divergence between
        P(disease | evidence=yes) and P(disease | evidence=no).
        Higher JSD → the answer to this question changes the disease
        distribution more, regardless of direction.
        """
        if not candidates:
            return []

        # Batch: vectors with each candidate turned ON
        batch_on = np.tile(base_vec, (len(candidates), 1))
        for i, ev_id in enumerate(candidates):
            batch_on[i, self._feat_idx[ev_id]] = 1.0
        probas_on = self.clf.predict_proba(batch_on)

        # P(disease | evidence=no) ≈ base_proba (evidence absent in base_vec)
        scores = []
        for i, ev_id in enumerate(candidates):
            p_yes = probas_on[i]
            p_no = base_proba
            # Ensure distributions are valid (non-negative, sum > 0)
            p_yes = np.clip(p_yes, 1e-12, None)
            p_no = np.clip(p_no, 1e-12, None)
            p_yes = p_yes / p_yes.sum()
            p_no = p_no / p_no.sum()
            jsd = float(jensenshannon(p_yes, p_no))
            if np.isnan(jsd):
                jsd = 0.0
            scores.append((ev_id, jsd))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    # ─── Round 2: Disease Discrimination ─────────────────────────────────

    def _score_discrimination(
        self,
        candidates: List[str],
        base_vec: np.ndarray,
        base_proba: np.ndarray,
    ) -> List[Tuple[str, float]]:
        """
        Score by how much the gap between top-1 and top-2 widens when
        evidence=yes.  Larger gap → better discriminator.
        """
        if not candidates:
            return []

        top2_idx = np.argsort(base_proba)[::-1][:2]
        idx1, idx2 = int(top2_idx[0]), int(top2_idx[1])
        base_gap = float(base_proba[idx1] - base_proba[idx2])

        batch = np.tile(base_vec, (len(candidates), 1))
        for i, ev_id in enumerate(candidates):
            batch[i, self._feat_idx[ev_id]] = 1.0
        probas = self.clf.predict_proba(batch)

        scores = []
        for i, ev_id in enumerate(candidates):
            new_gap = float(probas[i][idx1] - probas[i][idx2])
            improvement = new_gap - base_gap
            scores.append((ev_id, max(improvement, 0.0)))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    # ─── Public API ──────────────────────────────────────────────────────

    def get_questions(
        self,
        evidences: List[str],
        age: int = 25,
        sex: str = "M",
        round_num: int = 1,
        n_questions: int = 3,
    ) -> QAResult:
        if self.clf is None or self._n_features == 0:
            return QAResult(
                questions=[], should_stop=True, stop_reason="model_unavailable",
                top3=[], confidence=0.0, round_type="none",
            )

        base_vec = self._build_feature_vector(evidences, age, sex)
        base_proba = self._get_proba(base_vec)
        top3 = self._top3(base_proba)
        confidence = top3[0][1] if top3 else 0.0

        # ── Stop: max rounds exceeded ────────────────────────────────────
        if round_num > MAX_ROUNDS:
            return QAResult(
                questions=[], should_stop=True, stop_reason="max_rounds",
                top3=top3, confidence=confidence, round_type="none",
            )

        # ── Stop: confidence already high enough ─────────────────────────
        if confidence >= STOP_CONFIDENCE:
            return QAResult(
                questions=[], should_stop=True,
                stop_reason="high_confidence",
                top3=top3, confidence=confidence, round_type="none",
            )

        ev_set = set(evidences)
        all_candidates = self._base_candidates(ev_set)

        if not all_candidates:
            return QAResult(
                questions=[], should_stop=True,
                stop_reason="no_candidates",
                top3=top3, confidence=confidence, round_type="none",
            )

        # ── Decide round type ────────────────────────────────────────────
        if round_num == 1 and confidence < EXPLORATION_EXIT_CONFIDENCE:
            round_type = "exploration"
        else:
            round_type = "discrimination"

        if round_type == "exploration":
            scored = self._score_exploration(all_candidates, base_vec, base_proba)
            # Boost broad symptoms to the top by adding a bonus
            broad_bonus = 0.05
            scored_boosted = []
            for ev_id, sc in scored:
                bonus = broad_bonus if ev_id in self._broad_evidences else 0.0
                scored_boosted.append((ev_id, sc + bonus))
            scored_boosted.sort(key=lambda x: x[1], reverse=True)
            scored = scored_boosted
            reason_prefix = "exploration_jsd"
        else:
            # Filter candidates to only those relevant to current top-3
            top3_names = [name for name, _ in top3]
            relevant = set()
            for d_name in top3_names:
                relevant |= self.disease_relevant_evidences.get(d_name, set())
            filtered = [c for c in all_candidates if c in relevant]
            if not filtered:
                return QAResult(
                    questions=[], should_stop=True,
                    stop_reason="no_candidates",
                    top3=top3, confidence=confidence,
                    round_type="discrimination",
                )
            scored = self._score_discrimination(filtered, base_vec, base_proba)
            reason_prefix = "discrimination_gap"

        # ── Build question list ──────────────────────────────────────────
        questions: List[QuestionCandidate] = []
        for ev_id, score in scored[:n_questions]:
            meta = self.evidences_meta.get(ev_id, {})
            q_en = meta.get("question_en", ev_id)
            trans = self.question_translations.get(ev_id, {})
            questions.append(
                QuestionCandidate(
                    evidence_id=ev_id,
                    question_ru=trans.get("ru", q_en),
                    question_kk=trans.get("kk", q_en),
                    score=score,
                    reason=f"{reason_prefix}:{score:.4f}",
                )
            )

        should_stop = len(questions) == 0
        stop_reason = "no_candidates" if should_stop else ""

        return QAResult(
            questions=questions,
            should_stop=should_stop,
            stop_reason=stop_reason,
            top3=top3,
            confidence=confidence,
            round_type=round_type,
        )
