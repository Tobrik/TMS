import type { Lang } from "./i18n";

export const SYMPTOM_LIST = [
  "ABDOMINAL_PAIN",
  "CHEST_PAIN",
  "COUGH",
  "DEHYDRATION",
  "DIARRHEA",
  "FEVER",
  "HEADACHE",
  "ITCHING",
  "MUSCLE_ACHES",
  "NAUSEA",
  "NECK_STIFFNESS",
  "PHOTOPHOBIA",
  "POLYDIPSIA",
  "POLYURIA",
  "RASH",
  "RESPIRATORY_DISTRESS",
  "RUNNY_NOSE",
  "SNEEZING",
  "SORE_THROAT",
  "STRIDOR",
  "VOMITING",
  "WEIGHT_LOSS",
  "WHEEZING",
] as const;

export type SymptomCode = (typeof SYMPTOM_LIST)[number];

const SYMPTOM_LABELS_I18N: Record<SymptomCode, Record<Lang, string>> = {
  ABDOMINAL_PAIN: { ru: "Боль в животе", en: "Abdominal Pain", kk: "Іш ауруы" },
  CHEST_PAIN: { ru: "Боль в груди", en: "Chest Pain", kk: "Кеуде ауруы" },
  COUGH: { ru: "Кашель", en: "Cough", kk: "Жөтел" },
  DEHYDRATION: { ru: "Обезвоживание", en: "Dehydration", kk: "Сусыздану" },
  DIARRHEA: { ru: "Диарея", en: "Diarrhea", kk: "Диарея" },
  FEVER: { ru: "Температура", en: "Fever", kk: "Температура" },
  HEADACHE: { ru: "Головная боль", en: "Headache", kk: "Бас ауруы" },
  ITCHING: { ru: "Зуд", en: "Itching", kk: "Қышыну" },
  MUSCLE_ACHES: { ru: "Боль в мышцах", en: "Muscle Aches", kk: "Бұлшық ет ауруы" },
  NAUSEA: { ru: "Тошнота", en: "Nausea", kk: "Жүрек айну" },
  NECK_STIFFNESS: { ru: "Ригидность шеи", en: "Neck Stiffness", kk: "Мойын қаттылығы" },
  PHOTOPHOBIA: { ru: "Светобоязнь", en: "Photophobia", kk: "Жарықтан қорқу" },
  POLYDIPSIA: { ru: "Повышенная жажда", en: "Excessive Thirst", kk: "Шөлдеу" },
  POLYURIA: { ru: "Частое мочеиспускание", en: "Frequent Urination", kk: "Жиі зәр шығару" },
  RASH: { ru: "Сыпь", en: "Rash", kk: "Бөртпе" },
  RESPIRATORY_DISTRESS: { ru: "Затруднённое дыхание", en: "Respiratory Distress", kk: "Тыныс алу қиындығы" },
  RUNNY_NOSE: { ru: "Насморк", en: "Runny Nose", kk: "Мұрын бітелу" },
  SNEEZING: { ru: "Чихание", en: "Sneezing", kk: "Түшкіру" },
  SORE_THROAT: { ru: "Боль в горле", en: "Sore Throat", kk: "Тамақ ауруы" },
  STRIDOR: { ru: "Стридор", en: "Stridor", kk: "Стридор" },
  VOMITING: { ru: "Рвота", en: "Vomiting", kk: "Құсу" },
  WEIGHT_LOSS: { ru: "Потеря веса", en: "Weight Loss", kk: "Салмақ жоғалту" },
  WHEEZING: { ru: "Хрипы", en: "Wheezing", kk: "Сырылдау" },
};

export function getSymptomLabel(code: SymptomCode, lang: Lang = "ru"): string {
  return SYMPTOM_LABELS_I18N[code]?.[lang] || SYMPTOM_LABELS_I18N[code]?.ru || code;
}

// Default Russian export for backward compatibility
export const SYMPTOM_LABELS: Record<SymptomCode, string> = Object.fromEntries(
  Object.entries(SYMPTOM_LABELS_I18N).map(([k, v]) => [k, (v as Record<Lang, string>).ru])
) as Record<SymptomCode, string>;
