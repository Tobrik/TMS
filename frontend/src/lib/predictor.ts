import {
  MODEL_DICT,
  DISEASE_RECOMMENDATIONS,
  DISEASE_DOCTORS,
  DISEASE_LABELS,
} from "./diseaseWeights";
import { SYMPTOM_LIST } from "./symptoms";
import type { DiseaseSlice, DiagnosisResult, LabContext } from "./types";

// ─── Key Symptom Multiplier ───────────────────────────────────────
// Мультипликативный коэффициент вместо аддитивного бонуса.
// Ключевой симптом есть → score * boost, нет → score * dampen.
// Не ломает maxScore, не раздувает проценты.
const KEY_SYMPTOMS: Record<string, { codes: string[]; minSeverity: number }> = {
  "Appendicitis":  { codes: ["ABDOMINAL_PAIN"], minSeverity: 2 },
  "Chickenpox":    { codes: ["RASH"],            minSeverity: 2 },
  "Scarlet Fever": { codes: ["RASH"],            minSeverity: 2 },
  "Croup":         { codes: ["STRIDOR"],         minSeverity: 1 },
};

const KEY_BOOST  = 1.3;  // +30% если ключевой симптом есть
const KEY_DAMPEN = 0.6;  // -40% если ключевого симптома нет

// ─── Helper: severity по коду симптома ────────────────────────────
function getSeverity(vector: number[], code: string): number {
  const idx = SYMPTOM_LIST.indexOf(code as (typeof SYMPTOM_LIST)[number]);
  return idx >= 0 ? (vector[idx] || 0) : 0;
}

// ─── Main Predict ─────────────────────────────────────────────────
export function predict(symptomsVector: number[], labContext?: LabContext): DiagnosisResult {
  const totalSeverity = symptomsVector.reduce((a, b) => a + b, 0);
  if (totalSeverity === 0) return getEmptyResult();

  const scores: { name: string; score: number }[] = [];

  for (const [disease, model] of Object.entries(MODEL_DICT)) {
    let score = 0;       // raw score (сумма weight * severity)
    let matchCount = 0;  // сколько симптомов совпало

    let maxScore = 0;    // макс. возможный score (все severity=3)

    for (let i = 0; i < symptomsVector.length; i++) {
      const weight = model.weights[i] || 0;
      const severity = symptomsVector[i] || 0;

      if (weight > 0) {
        maxScore += weight * 3;
        if (severity > 0) {
          score += weight * severity;
          matchCount++;
        }
      }
    }

    // Пропускаем болезнь если ни один симптом не совпал
    if (matchCount === 0 || maxScore === 0) continue;

    // ─── Key Symptom Multiplier ─────────────────────────
    const keySym = KEY_SYMPTOMS[disease];
    if (keySym) {
      const hasKey = keySym.codes.some(
        (code) => getSeverity(symptomsVector, code) >= keySym.minSeverity
      );
      score *= hasKey ? KEY_BOOST : KEY_DAMPEN;
    }

    // ─── Lab Context Adjustment ─────────────────────────
    if (labContext?.adjustments[disease] !== undefined) {
      const adj = labContext.adjustments[disease];
      score += adj;
      if (adj > 0) maxScore += adj;
    }

    score = Math.max(score, 0);
    // Процент от максимально возможного score болезни
    const pct = Math.min(score / maxScore, 1);
    scores.push({ name: disease, score: pct });
  }

  scores.sort((a, b) => b.score - a.score);

  if (scores.length === 0 || scores[0].score < 0.32) return getEmptyResult();

  // ─── Build Result ──────────────────────────────────────────
  const top3 = scores.slice(0, 3);

  // Абсолютные проценты — не нормализуем к 100%,
  // чтобы пользователь видел реальную уверенность модели
  const slices: DiseaseSlice[] = top3.map((s) => ({
    name: s.name,
    label: DISEASE_LABELS[s.name] || s.name,
    score: s.score,
  }));

  const top1 = top3[0];

  return {
    diseaseName: top1.name,
    diseaseLabel: DISEASE_LABELS[top1.name] || top1.name,
    doctor: DISEASE_DOCTORS[top1.name] || "Терапевт",
    recommendation:
      DISEASE_RECOMMENDATIONS[top1.name] || "Консультация врача обязательна.",
    slices,
    labInfluences: labContext?.influences.filter((inf) =>
      inf.diseases.some((d) => top3.some((t) => t.name === d))
    ),
  };
}

function getEmptyResult(): DiagnosisResult {
  return {
    diseaseName: "Unknown",
    diseaseLabel: "Не удалось определить",
    doctor: "Терапевт",
    recommendation:
      "Симптомы не специфичны или недостаточно данных. Пожалуйста, опишите состояние подробнее.",
    slices: [],
  };
}
