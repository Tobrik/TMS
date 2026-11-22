import type { AccuracyCondition, AccuracyMetric, AccuracyReportData, Zone } from "@/lib/api";

// Sample report for /accuracy?mock=1. The numbers are invented to show the layout;
// the shape matches the GET /accuracy_report contract. Intervals use the same Wilson formula as the backend.

function round3(x: number): number {
  return Math.round(x * 1000) / 1000;
}

function metric(k: number, n: number): AccuracyMetric {
  if (n === 0) return { value: null, k: 0, n: 0, ci: null };
  const z = 1.96;
  const p = k / n;
  const denom = 1 + (z * z) / n;
  const centre = (p + (z * z) / (2 * n)) / denom;
  const half = (z * Math.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n))) / denom;
  return { value: round3(p), k, n, ci: [round3(centre - half), round3(centre + half)] };
}

// name, label_ru, tier, n, top1 correct, top3 correct
const CONDITIONS: [string, string, Zone, number, number, number][] = [
  ["Possible NSTEMI / STEMI", "Возможный ИМ (NSTEMI / STEMI)", "red", 60, 49, 58],
  ["Pulmonary embolism", "Тромбоэмболия лёгочной артерии", "red", 40, 31, 37],
  ["Anaphylaxis", "Анафилаксия", "red", 35, 33, 35],
  ["Croup", "Круп", "yellow", 45, 37, 43],
  ["Whooping cough", "Коклюш", "yellow", 30, 21, 27],
  ["Pneumonia", "Пневмония", "yellow", 150, 128, 145],
  ["Bronchitis", "Бронхит", "yellow", 180, 163, 176],
  ["Acute rhinosinusitis", "Острый риносинусит", "green", 240, 197, 231],
  ["Viral pharyngitis", "Вирусный фарингит", "green", 300, 277, 295],
  ["Influenza", "Грипп / ОРВИ", "green", 420, 388, 411],
  ["URTI", "ОРВИ (верхние дыхательные пути)", "green", 510, 472, 499],
  ["Allergic sinusitis", "Аллергический синусит", "green", 200, 180, 196],
];

const perCondition: AccuracyCondition[] = CONDITIONS.map(([name, label_ru, tier, n, k1, k3]) => ({
  name,
  label_ru,
  tier,
  top1: metric(k1, n),
  top3: metric(k3, n),
}));

export const SAMPLE_REPORT: AccuracyReportData = {
  generated_at: "2026-11-22T21:00:00Z",
  model_version: "v2.1-nb",
  extraction_backend: "clef-flash",
  synthetic: {
    source: "DDXPlus test patients (synthetic), simulated conversation",
    settings: { n: 3000, seed: 7, noise: 0.3, answer_error: 0.1, dont_know: 0.2 },
    metrics: {
      top1: metric(2706, 3000),
      top3: metric(2934, 3000),
      red_sensitivity: metric(600, 620),
      red_to_green: metric(4, 620),
      green_to_red: metric(88, 1100),
      zone_exact: metric(2793, 3000),
    },
    questions_mean: 1.7,
    calibration: [
      { bin: "0.3-0.5", n: 14, mean_confidence: 0.42, accuracy: 0.36 },
      { bin: "0.5-0.7", n: 286, mean_confidence: 0.62, accuracy: 0.66 },
      { bin: "0.7-0.8", n: 402, mean_confidence: 0.75, accuracy: 0.79 },
      { bin: "0.8-0.9", n: 857, mean_confidence: 0.85, accuracy: 0.88 },
      { bin: "0.9-1.0", n: 1448, mean_confidence: 0.96, accuracy: 0.971 },
    ],
    low_confidence: {
      share: metric(300, 3000),
      top1_low: metric(191, 300),
      top1_rest: metric(2515, 2700),
    },
    per_condition: perCondition.sort((a, b) => (a.top1.value ?? 0) - (b.top1.value ?? 0)),
  },
  written_cases: {
    source: "eval/cases_v3.jsonl: 120 free-text complaints (ru, kk, ru-kk mixed, en), held out",
    cases_file: "eval/cases_v3.jsonl",
    n_in_scope: 110,
    n_out_of_scope: 10,
    extraction: {
      precision: metric(401, 432),
      recall: metric(398, 449),
      denials: metric(37, 40),
      red_flag_recall: metric(29, 30),
    },
    patient_accepts: {
      top1: metric(97, 110),
      top3: metric(105, 110),
      red_sensitivity: metric(17, 18),
      green_to_red: metric(4, 46),
    },
    patient_confirms: {
      top1: metric(103, 110),
      top3: metric(108, 110),
      red_sensitivity: metric(18, 18),
      green_to_red: metric(2, 46),
    },
    by_language: {
      ru: { n: 41, top1: metric(40, 41) },
      kk: { n: 22, top1: metric(19, 22) },
      mix: { n: 19, top1: metric(18, 19) },
      en: { n: 28, top1: metric(26, 28) },
    },
    out_of_scope: {
      flagged: metric(9, 10),
      false_flags: metric(4, 110),
    },
  },
  caveats: [
    "Синтетические пациенты построены по статистике DDXPlus. Реальные пациенты устроены иначе, поэтому эти цифры нельзя переносить на клинику напрямую.",
    "В охвате 110 письменных жалоб. Редкие состояния представлены слабо, интервалы широкие.",
    "Подтверждённых врачами реальных случаев пока нет, поэтому проверка на реальных пациентах ещё не начата.",
    "Уточняющие вопросы в тесте отвечают по записи случая. В живом диалоге пациенты отвечают менее точно.",
    "Система не ставит диагноз. Решение о лечении принимает врач.",
  ],
  real_patients: {
    n: 0,
    other: 0,
    top1: metric(0, 0),
    top3: metric(0, 0),
    zone_exact: metric(0, 0),
    red_sensitivity: metric(0, 0),
    under_triage: metric(0, 0),
    over_triage: metric(0, 0),
    by_version: {},
    since: null,
  },
};
