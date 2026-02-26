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

export const DISEASE_LABELS: Record<string, string> = {
  "Gastroenteritis": "Гастроэнтерит (Кишечная инфекция)",
  "Croup": "Круп (Острый ларинготрахеит)",
  "Scarlet Fever": "Скарлатина",
  "Eczema": "Экзема / Дерматит",
  "Asthma": "Бронхиальная астма",
  "Type 1 Diabetes": "Сахарный диабет 1 типа (Подозрение)",
  "Bronchiolitis": "Бронхиолит",
  "Meningitis": "Менингит",
  "Influenza": "Грипп / ОРВИ",
  "Pneumonia": "Пневмония",
  "Chickenpox": "Ветрянка",
  "Appendicitis": "Аппендицит",
  "Common Cold": "Простуда (ОРЗ)"
};

export const DISEASE_DOCTORS: Record<string, string> = {
  "Gastroenteritis": "Гастроэнтеролог / Инфекционист",
  "Croup": "Педиатр / Скорая (если задыхается)",
  "Scarlet Fever": "Инфекционист / Педиатр",
  "Eczema": "Дерматолог",
  "Asthma": "Пульмонолог / Аллерголог",
  "Type 1 Diabetes": "Эндокринолог (Срочно)",
  "Bronchiolitis": "Педиатр / Пульмонолог",
  "Meningitis": "СКОРАЯ ПОМОЩЬ (103) / Невролог",
  "Influenza": "Терапевт",
  "Pneumonia": "Терапевт / Пульмонолог",
  "Chickenpox": "Терапевт (вызов на дом)",
  "Appendicitis": "СКОРАЯ ПОМОЩЬ (Хирургия)",
  "Common Cold": "Терапевт"
};

export const DISEASE_RECOMMENDATIONS: Record<string, string> = {
  "Gastroenteritis": "Регидратация (Регидрон), диета (рис, сухари). При боли - спазмолитик.",
  "Croup": "Доступ холодного влажного воздуха (открыть окно). Успокоить ребенка. Если дыхание затруднено — ингаляция Пульмикорт. При удушье — 103.",
  "Scarlet Fever": "Изоляция. Обильное питье. Обязательно врач (нужен антибиотик).",
  "Eczema": "Увлажнение кожи (эмоленты). Исключить аллергены. Антигистаминные при зуде.",
  "Asthma": "Ингалятор (Сальбутамол/Беродуал). Посадить пациента. Обеспечить воздух.",
  "Type 1 Diabetes": "Срочный анализ крови на сахар. Обильное питье. Врач немедленно.",
  "Bronchiolitis": "Увлажнение воздуха, промывание носа. Контроль частоты дыхания.",
  "Meningitis": "НЕМЕДЛЕННО СКОРУЮ. Опасно для жизни. Не давать обезболивающие до осмотра.",
  "Influenza": "Покой, обильное питье, Парацетамол/Ибупрофен при t>38.5. Не принимать антибиотики без врача.",
  "Pneumonia": "Рентген легких. Врач назначит антибиотик. Дыхательная гимнастика.",
  "Chickenpox": "Не чесать (Каламин/Зеленка). Жаропонижающее (НЕ Аспирин!).",
  "Appendicitis": "Холод на живот. НЕ пить, НЕ есть, НЕ принимать обезболивающие. Вызвать 103.",
  "Common Cold": "Теплое питье, промывание носа, отдых. Витамин С."
};
