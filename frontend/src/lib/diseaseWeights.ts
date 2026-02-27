// Индексная карта симптомов (для справки):
// 0: ABDOMINAL_PAIN, 1: CHEST_PAIN, 2: COUGH, 3: DEHYDRATION, 4: DIARRHEA
// 5: FEVER, 6: HEADACHE, 7: ITCHING, 8: MUSCLE_ACHES, 9: NAUSEA
// 10: NECK_STIFFNESS, 11: PHOTOPHOBIA, 12: POLYDIPSIA, 13: POLYURIA, 14: RASH
// 15: RESPIRATORY_DISTRESS, 16: RUNNY_NOSE, 17: SNEEZING, 18: SORE_THROAT
// 19: STRIDOR, 20: VOMITING, 21: WEIGHT_LOSS, 22: WHEEZING

export const MODEL_DICT: Record<string, { bias: number; weights: number[] }> = {
  "Gastroenteritis": {
    bias: 20.0,
    weights: [
      10.0, // 0. ABDOMINAL_PAIN
      0,    // 1. CHEST_PAIN
      0,    // 2. COUGH
      5.0,  // 3. DEHYDRATION
      20.0, // 4. DIARRHEA (Главный)
      3.0,  // 5. FEVER
      0,    // 6. HEADACHE
      0,    // 7. ITCHING
      0,    // 8. MUSCLE_ACHES
      5.0,  // 9. NAUSEA
      0,    // 10. NECK_STIFFNESS
      0,    // 11. PHOTOPHOBIA
      0,    // 12. POLYDIPSIA
      0,    // 13. POLYURIA
      0,    // 14. RASH
      0,    // 15. RESPIRATORY_DISTRESS
      0,    // 16. RUNNY_NOSE
      0,    // 17. SNEEZING
      0,    // 18. SORE_THROAT
      0,    // 19. STRIDOR
      10.0, // 20. VOMITING
      0,    // 21. WEIGHT_LOSS
      0     // 22. WHEEZING
    ]
  },

  "Croup": {
    bias: 15.0,
    weights: [
      0,    // 0. ABDOMINAL_PAIN
      0,    // 1. CHEST_PAIN
      15.0, // 2. COUGH (Лающий кашель)
      0,    // 3. DEHYDRATION
      0,    // 4. DIARRHEA
      5.0,  // 5. FEVER
      0,    // 6. HEADACHE
      0,    // 7. ITCHING
      0,    // 8. MUSCLE_ACHES
      0,    // 9. NAUSEA
      0,    // 10. NECK_STIFFNESS
      0,    // 11. PHOTOPHOBIA
      0,    // 12. POLYDIPSIA
      0,    // 13. POLYURIA
      0,    // 14. RASH
      10.0, // 15. RESPIRATORY_DISTRESS
      0,    // 16. RUNNY_NOSE
      0,    // 17. SNEEZING
      5.0,  // 18. SORE_THROAT
      30.0, // 19. STRIDOR (Главный признак)
      0,    // 20. VOMITING
      0,    // 21. WEIGHT_LOSS
      0     // 22. WHEEZING
    ]
  },

  "Scarlet Fever": {
    bias: 5.0,
    weights: [
      0, 0, 0, 0, 0,
      8.0,  // 5. FEVER
      0, 0, 0, 0, 0, 0, 0, 0,
      25.0, // 14. RASH (ГЛАВНЫЙ)
      0, 0, 0,
      12.0, // 18. SORE_THROAT
      0, 0, 0, 0
    ]
  },

  "Eczema": {
    bias: 15.0,
    weights: [
      0, 0, 0, 0, 0,
      0,     // 5. FEVER
      0,
      15.0,  // 7. ITCHING (Главный)
      0, 0, 0, 0, 0, 0,
      15.0,  // 14. RASH (Главный)
      0, 0, 0, 0, 0, 0, 0, 0
    ]
  },

  "Asthma": {
    bias: 15.0,
    weights: [
      0,
      5.0,   // 1. CHEST_PAIN
      5.0,   // 2. COUGH
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      12.0,  // 15. RESPIRATORY_DISTRESS
      0, 0, 0,
      0,     // 19. STRIDOR
      0, 0,
      22.0   // 22. WHEEZING (Главный)
    ]
  },

  "Type 1 Diabetes": {
    bias: 20.0,
    weights: [
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      8.0,   // 12. POLYDIPSIA
      8.0,   // 13. POLYURIA
      0, 0, 0, 0, 0, 0, 0,
      10.0,  // 21. WEIGHT_LOSS
      0
    ]
  },

  "Bronchiolitis": {
    bias: 12.0,
    weights: [
      0, 0,
      12.0,  // 2. COUGH
      0, 0,
      5.0,   // 5. FEVER
      0, 0, 0, 0, 0, 0, 0, 0, 0,
      5.0,   // 15. RESPIRATORY_DISTRESS
      5.0,   // 16. RUNNY_NOSE
      0, 0,
      0,     // 19. STRIDOR
      0, 0,
      12.0   // 22. WHEEZING
    ]
  },

  "Meningitis": {
    bias: 5.0,
    weights: [
      0,    // 0
      0,    // 1
      0,    // 2
      0,    // 3
      0,    // 4. DIARRHEA
      12.0, // 5. FEVER
      15.0, // 6. HEADACHE
      0,    // 7
      0,    // 8
      5.0,  // 9. NAUSEA
      50.0, // 10. NECK_STIFFNESS (КРИТИЧЕСКИЙ ВЕС)
      10.0, // 11. PHOTOPHOBIA
      0,    // 12
      0,    // 13
      5.0,  // 14. RASH
      0,    // 15
      0,    // 16
      0,    // 17. SNEEZING
      0,    // 18
      0,    // 19
      8.0,  // 20. VOMITING
      0, 0
    ]
  },

  "Influenza": {
    bias: 15.0,
    weights: [
      0, 0,
      10.0,  // 2. COUGH
      0, 0,
      12.0,  // 5. FEVER
      5.0,   // 6. HEADACHE
      0,
      10.0,  // 8. MUSCLE_ACHES
      0, 0, 0, 0, 0, 0, 0,
      5.0,   // 16. RUNNY_NOSE
      3.0,   // 17. SNEEZING
      5.0,   // 18. SORE_THROAT
      0, 0, 0, 0
    ]
  },

  "Pneumonia": {
    bias: 10.0,
    weights: [
      0,
      3.0,   // 1. CHEST_PAIN
      12.0,  // 2. COUGH
      0, 0,
      10.0,  // 5. FEVER
      0, 0, 0, 0, 0, 0, 0, 0, 0,
      15.0,  // 15. RESPIRATORY_DISTRESS
      0, 0, 0, 0, 0, 0,
      5.0    // 22. WHEEZING
    ]
  },

  "Chickenpox": {
    bias: 12.0,
    weights: [
      0, 0, 0, 0, 0,
      3.0,   // 5. FEVER
      0,
      15.0,  // 7. ITCHING
      0, 0, 0, 0, 0, 0,
      15.0,  // 14. RASH
      0, 0, 0, 0, 0, 0, 0, 0
    ]
  },

  "Appendicitis": {
    bias: 15.0,
    weights: [
      20.0,  // 0. ABDOMINAL_PAIN (Главный)
      0, 0, 0,
      0,     // 4. DIARRHEA
      3.0,   // 5. FEVER
      0, 0, 0,
      5.0,   // 9. NAUSEA
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      5.0,   // 20. VOMITING
      0, 0
    ]
  },

  "Common Cold": {
    bias: 15.0,
    weights: [
      0, 0,
      5.0,   // 2. COUGH
      0, 0,
      3.0,   // 5. FEVER
      3.0,   // 6. HEADACHE
      0, 0, 0, 0, 0, 0, 0, 0, 0,
      15.0,  // 16. RUNNY_NOSE (Главный)
      10.0,  // 17. SNEEZING
      12.0,  // 18. SORE_THROAT
      0, 0, 0, 0
    ]
  }
};

import type { Lang } from "./i18n";

const DISEASE_LABELS_I18N: Record<string, Record<Lang, string>> = {
  "Gastroenteritis": { ru: "Гастроэнтерит (Кишечная инфекция)", en: "Gastroenteritis (Intestinal Infection)", kk: "Гастроэнтерит (Ішек инфекциясы)" },
  "Croup": { ru: "Круп (Острый ларинготрахеит)", en: "Croup (Acute Laryngotracheitis)", kk: "Круп (Жіті ларинготрахеит)" },
  "Scarlet Fever": { ru: "Скарлатина", en: "Scarlet Fever", kk: "Қызылша" },
  "Eczema": { ru: "Экзема / Дерматит", en: "Eczema / Dermatitis", kk: "Экзема / Дерматит" },
  "Asthma": { ru: "Бронхиальная астма", en: "Bronchial Asthma", kk: "Бронхиалды астма" },
  "Type 1 Diabetes": { ru: "Сахарный диабет 1 типа (Подозрение)", en: "Type 1 Diabetes (Suspected)", kk: "1-ші типті қант диабеті (Күдік)" },
  "Bronchiolitis": { ru: "Бронхиолит", en: "Bronchiolitis", kk: "Бронхиолит" },
  "Meningitis": { ru: "Менингит", en: "Meningitis", kk: "Менингит" },
  "Influenza": { ru: "Грипп / ОРВИ", en: "Influenza / ARVI", kk: "Тұмау / ЖРВИ" },
  "Pneumonia": { ru: "Пневмония", en: "Pneumonia", kk: "Пневмония" },
  "Chickenpox": { ru: "Ветрянка", en: "Chickenpox", kk: "Су шешек" },
  "Appendicitis": { ru: "Аппендицит", en: "Appendicitis", kk: "Аппендицит" },
  "Common Cold": { ru: "Простуда (ОРЗ)", en: "Common Cold (ARI)", kk: "Суық тию (ЖРА)" },
};

const DISEASE_DOCTORS_I18N: Record<string, Record<Lang, string>> = {
  "Gastroenteritis": { ru: "Гастроэнтеролог / Инфекционист", en: "Gastroenterologist / Infectious Disease Specialist", kk: "Гастроэнтеролог / Инфекционист" },
  "Croup": { ru: "Педиатр / Скорая (если задыхается)", en: "Pediatrician / Emergency (if choking)", kk: "Педиатр / Жедел жәрдем (тұншығу кезінде)" },
  "Scarlet Fever": { ru: "Инфекционист / Педиатр", en: "Infectious Disease Specialist / Pediatrician", kk: "Инфекционист / Педиатр" },
  "Eczema": { ru: "Дерматолог", en: "Dermatologist", kk: "Дерматолог" },
  "Asthma": { ru: "Пульмонолог / Аллерголог", en: "Pulmonologist / Allergist", kk: "Пульмонолог / Аллерголог" },
  "Type 1 Diabetes": { ru: "Эндокринолог (Срочно)", en: "Endocrinologist (Urgent)", kk: "Эндокринолог (Шұғыл)" },
  "Bronchiolitis": { ru: "Педиатр / Пульмонолог", en: "Pediatrician / Pulmonologist", kk: "Педиатр / Пульмонолог" },
  "Meningitis": { ru: "СКОРАЯ ПОМОЩЬ (103) / Невролог", en: "EMERGENCY (911) / Neurologist", kk: "ЖЕДЕЛ ЖӘРДЕМ (103) / Невролог" },
  "Influenza": { ru: "Терапевт", en: "General Practitioner", kk: "Терапевт" },
  "Pneumonia": { ru: "Терапевт / Пульмонолог", en: "GP / Pulmonologist", kk: "Терапевт / Пульмонолог" },
  "Chickenpox": { ru: "Терапевт (вызов на дом)", en: "GP (home visit)", kk: "Терапевт (үйге шақыру)" },
  "Appendicitis": { ru: "СКОРАЯ ПОМОЩЬ (Хирургия)", en: "EMERGENCY (Surgery)", kk: "ЖЕДЕЛ ЖӘРДЕМ (Хирургия)" },
  "Common Cold": { ru: "Терапевт", en: "General Practitioner", kk: "Терапевт" },
};

const DISEASE_RECOMMENDATIONS_I18N: Record<string, Record<Lang, string>> = {
  "Gastroenteritis": { ru: "Регидратация (Регидрон), диета (рис, сухари). При боли - спазмолитик.", en: "Rehydration (ORS), diet (rice, crackers). For pain — antispasmodic.", kk: "Регидратация (Регидрон), диета (күріш, кептірілген нан). Ауырғанда — спазмолитик." },
  "Croup": { ru: "Доступ холодного влажного воздуха (открыть окно). Успокоить ребенка. Если дыхание затруднено — ингаляция Пульмикорт. При удушье — 103.", en: "Cool moist air (open window). Calm the child. If breathing is difficult — Pulmicort inhalation. If choking — call 911.", kk: "Салқын ылғалды ауа (терезені ашыңыз). Баланы тыныштандырыңыз. Тыныс алу қиындаса — Пульмикорт ингаляциясы. Тұншығу кезінде — 103." },
  "Scarlet Fever": { ru: "Изоляция. Обильное питье. Обязательно врач (нужен антибиотик).", en: "Isolation. Plenty of fluids. Doctor required (antibiotics needed).", kk: "Оқшаулау. Мол сұйықтық. Дәрігер міндетті (антибиотик қажет)." },
  "Eczema": { ru: "Увлажнение кожи (эмоленты). Исключить аллергены. Антигистаминные при зуде.", en: "Skin moisturizing (emollients). Avoid allergens. Antihistamines for itching.", kk: "Теріні ылғалдандыру (эмоленттер). Аллергендерді болдырмау. Қышыған кезде антигистаминдер." },
  "Asthma": { ru: "Ингалятор (Сальбутамол/Беродуал). Посадить пациента. Обеспечить воздух.", en: "Inhaler (Salbutamol/Berodual). Sit the patient up. Ensure fresh air.", kk: "Ингалятор (Сальбутамол/Беродуал). Пациентті отырғызыңыз. Ауа қамтамасыз етіңіз." },
  "Type 1 Diabetes": { ru: "Срочный анализ крови на сахар. Обильное питье. Врач немедленно.", en: "Urgent blood sugar test. Plenty of fluids. See doctor immediately.", kk: "Шұғыл қан қантын тексеру. Мол сұйықтық. Дәрігерге дереу жүгініңіз." },
  "Bronchiolitis": { ru: "Увлажнение воздуха, промывание носа. Контроль частоты дыхания.", en: "Humidify air, nasal irrigation. Monitor breathing rate.", kk: "Ауаны ылғалдандыру, мұрынды жуу. Тыныс алу жиілігін бақылау." },
  "Meningitis": { ru: "НЕМЕДЛЕННО СКОРУЮ. Опасно для жизни. Не давать обезболивающие до осмотра.", en: "CALL EMERGENCY IMMEDIATELY. Life-threatening. Do not give painkillers before examination.", kk: "ДЕРЕУ ЖЕДЕЛ ЖӘРДЕМ. Өмірге қауіпті. Тексерілгенге дейін ауырсыну басатын дәрі бермеңіз." },
  "Influenza": { ru: "Покой, обильное питье, Парацетамол/Ибупрофен при t>38.5. Не принимать антибиотики без врача.", en: "Rest, plenty of fluids, Paracetamol/Ibuprofen if temp >38.5. No antibiotics without a doctor.", kk: "Тыныштық, мол сұйықтық, t>38.5 кезінде Парацетамол/Ибупрофен. Дәрігерсіз антибиотик қабылдамаңыз." },
  "Pneumonia": { ru: "Рентген легких. Врач назначит антибиотик. Дыхательная гимнастика.", en: "Chest X-ray. Doctor will prescribe antibiotics. Breathing exercises.", kk: "Өкпе рентгені. Дәрігер антибиотик тағайындайды. Тыныс алу гимнастикасы." },
  "Chickenpox": { ru: "Не чесать (Каламин/Зеленка). Жаропонижающее (НЕ Аспирин!).", en: "Don't scratch (Calamine/antiseptic). Antipyretic (NOT Aspirin!).", kk: "Қасымаңыз (Каламин/Зеленка). Қызуды түсіретін дәрі (Аспирин ЕМЕС!)." },
  "Appendicitis": { ru: "Холод на живот. НЕ пить, НЕ есть, НЕ принимать обезболивающие. Вызвать 103.", en: "Ice on abdomen. DO NOT drink, eat, or take painkillers. Call 911.", kk: "Іш үстіне суық. Ішпеңіз, жемеңіз, ауырсыну басатын дәрі қабылдамаңыз. 103 шақырыңыз." },
  "Common Cold": { ru: "Теплое питье, промывание носа, отдых. Витамин С.", en: "Warm fluids, nasal rinse, rest. Vitamin C.", kk: "Жылы сұйықтық, мұрынды жуу, демалыс. С витамині." },
};

export function getDiseaseLabel(disease: string, lang: Lang = "ru"): string {
  return DISEASE_LABELS_I18N[disease]?.[lang] || DISEASE_LABELS_I18N[disease]?.ru || disease;
}

export function getDiseaseDoctor(disease: string, lang: Lang = "ru"): string {
  return DISEASE_DOCTORS_I18N[disease]?.[lang] || DISEASE_DOCTORS_I18N[disease]?.ru || "";
}

export function getDiseaseRecommendation(disease: string, lang: Lang = "ru"): string {
  return DISEASE_RECOMMENDATIONS_I18N[disease]?.[lang] || DISEASE_RECOMMENDATIONS_I18N[disease]?.ru || "";
}

// Default Russian exports for backward compatibility
export const DISEASE_LABELS: Record<string, string> = Object.fromEntries(
  Object.entries(DISEASE_LABELS_I18N).map(([k, v]) => [k, v.ru])
);
export const DISEASE_DOCTORS: Record<string, string> = Object.fromEntries(
  Object.entries(DISEASE_DOCTORS_I18N).map(([k, v]) => [k, v.ru])
);
export const DISEASE_RECOMMENDATIONS: Record<string, string> = Object.fromEntries(
  Object.entries(DISEASE_RECOMMENDATIONS_I18N).map(([k, v]) => [k, v.ru])
);
