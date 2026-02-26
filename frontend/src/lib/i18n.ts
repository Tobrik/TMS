export type Lang = "ru" | "kk";

const translations = {
  ru: {
    // Chat page
    chatTitle: "TMS Assistant",
    online: "Онлайн",
    history: "История",
    logout: "Выйти",
    historyTitle: "История диагнозов",
    loading: "Загрузка...",
    noRecords: "Нет записей",
    confidence: "Уверенность",
    doctor: "Врач",
    inputPlaceholder: "Опишите ваши симптомы...",
    disclaimer: "AI-помощник не заменяет врача. Всегда консультируйтесь со специалистом.",
    welcomeMessage: "Здравствуйте! Я ваш AI-помощник по диагностике. Опишите свои симптомы, и я помогу определить возможное заболевание и порекомендую специалиста.\n\nНапример: \"У меня болит живот и тошнит\"",
    noSymptomsMessage: "Я медицинский ассистент. Пожалуйста, опишите ваши симптомы, и я помогу определить возможное заболевание.\n\nНапример:\n- \"У меня кашель и температура 38\"\n- \"Болит голова и тошнит\"\n- \"Появилась сыпь и зуд\"",
    errorAnalysis: "Произошла ошибка при анализе",
    errorGeneral: "Произошла непредвиденная ошибка. Попробуйте ещё раз.",
    detectedSymptoms: "Обнаруженные симптомы",
    analyzingSymptoms: "Анализирую симптомы...",

    // Login page
    appTitle: "TMS",
    appSubtitle: "Система медицинской диагностики",
    register: "Регистрация",
    login: "Вход",
    fullName: "Полное имя",
    city: "Город",
    pin: "PIN-код (5 цифр)",
    registerBtn: "Зарегистрироваться",
    loginBtn: "Войти",
    patientCode: "Код пациента",
    registrationSuccess: "Регистрация успешна!",
    yourCode: "Ваш код",
    copyCode: "Скопировать код",
    goToChat: "Перейти в чат",
    doctorPortal: "Портал для врачей",

    // Doctor page
    doctorTitle: "TMS - Врач",
    doctorSubtitle: "Портал для врачей",
    selectDoctor: "Выберите врача",
    loginSystem: "Вход в систему",
    backToMain: "Вернуться на главную",
    searchPatient: "Поиск пациента",
    patientCodePlaceholder: "Код пациента (например: OWL-3 или 3)",
    patientNotFound: "Пациент не найден",
    registration: "Регистрация",
    diaryEntries: "записей в дневнике",
    diagnosisHistory: "История диагнозов",
    noEntries: "Записи не найдены",
    confirmed: "Подтверждено",
    patientSymptoms: "Симптомы пациента",
    loadingDots: "Загрузка...",
    noSymptomData: "Нет данных о симптомах",
    preliminaryDiagnosis: "Предварительный диагноз (ML)",
    doctorDiagnosis: "Диагноз врача",
    setDiagnosis: "Установить диагноз...",
    notSet: "Не установлен",
    recept: "Рецепт / Рекомендации",
    writeRecept: "Написать рецепт...",
    noRecommendations: "Нет рекомендаций",
    responsibleDoctor: "Ответственный врач",
    save: "Сохранить",
    cancel: "Отмена",
    edit: "Редактировать",
    downloadPdf: "Скачать PDF",
    symptomTrends: "Динамика симптомов",
    selectSymptom: "Выберите симптом",
    worseTrend: "Тренд ухудшения",

    // Triage
    redZone: "Красная зона — Срочно",
    yellowZone: "Жёлтая зона — Сегодня",
    greenZone: "Зелёная зона — Плановые",
    allPatients: "Все пациенты",
    details: "Подробнее",
    lastDiagnosis: "Последний диагноз",
  },
  kk: {
    // Chat page
    chatTitle: "TMS Көмекші",
    online: "Онлайн",
    history: "Тарих",
    logout: "Шығу",
    historyTitle: "Диагноздар тарихы",
    loading: "Жүктелуде...",
    noRecords: "Жазбалар жоқ",
    confidence: "Сенімділік",
    doctor: "Дәрігер",
    inputPlaceholder: "Симптомдарыңызды сипаттаңыз...",
    disclaimer: "AI-көмекші дәрігерді алмастырмайды. Әрқашан маманмен кеңесіңіз.",
    welcomeMessage: "Сәлеметсіз бе! Мен сіздің AI-диагностика көмекшіңізбін. Симптомдарыңызды сипаттаңыз, мен ықтимал ауруды анықтауға көмектесемін.\n\nМысалы: \"Ішім ауырады және жүрегім айниды\"",
    noSymptomsMessage: "Мен медициналық көмекшімін. Симптомдарыңызды сипаттаңыз.\n\nМысалы:\n- \"Жөтел және температура 38\"\n- \"Басым ауырады және жүрегім айниды\"\n- \"Бөртпе пайда болды және қышиды\"",
    errorAnalysis: "Талдау кезінде қате орын алды",
    errorGeneral: "Күтпеген қате орын алды. Қайтадан көріңіз.",
    detectedSymptoms: "Анықталған симптомдар",
    analyzingSymptoms: "Симптомдар талдануда...",

    // Login page
    appTitle: "TMS",
    appSubtitle: "Медициналық диагностика жүйесі",
    register: "Тіркелу",
    login: "Кіру",
    fullName: "Толық аты",
    city: "Қала",
    pin: "PIN-код (5 сан)",
    registerBtn: "Тіркелу",
    loginBtn: "Кіру",
    patientCode: "Пациент коды",
    registrationSuccess: "Тіркелу сәтті!",
    yourCode: "Сіздің код",
    copyCode: "Кодты көшіру",
    goToChat: "Чатқа өту",
    doctorPortal: "Дәрігерлер порталы",

    // Doctor page
    doctorTitle: "TMS - Дәрігер",
    doctorSubtitle: "Дәрігерлер порталы",
    selectDoctor: "Дәрігерді таңдаңыз",
    loginSystem: "Жүйеге кіру",
    backToMain: "Басты бетке оралу",
    searchPatient: "Пациентті іздеу",
    patientCodePlaceholder: "Пациент коды (мысалы: OWL-3 немесе 3)",
    patientNotFound: "Пациент табылмады",
    registration: "Тіркелу",
    diaryEntries: "күнделік жазба",
    diagnosisHistory: "Диагноздар тарихы",
    noEntries: "Жазбалар табылмады",
    confirmed: "Расталған",
    patientSymptoms: "Пациент симптомдары",
    loadingDots: "Жүктелуде...",
    noSymptomData: "Симптомдар туралы деректер жоқ",
    preliminaryDiagnosis: "Алдын ала диагноз (ML)",
    doctorDiagnosis: "Дәрігер диагнозы",
    setDiagnosis: "Диагноз қою...",
    notSet: "Орнатылмаған",
    recept: "Рецепт / Ұсыныстар",
    writeRecept: "Рецепт жазу...",
    noRecommendations: "Ұсыныстар жоқ",
    responsibleDoctor: "Жауапты дәрігер",
    save: "Сақтау",
    cancel: "Бас тарту",
    edit: "Өңдеу",
    downloadPdf: "PDF жүктеу",
    symptomTrends: "Симптомдар динамикасы",
    selectSymptom: "Симптомды таңдаңыз",
    worseTrend: "Нашарлау трендi",

    // Triage
    redZone: "Қызыл аймақ — Шұғыл",
    yellowZone: "Сары аймақ — Бүгін",
    greenZone: "Жасыл аймақ — Жоспарлы",
    allPatients: "Барлық пациенттер",
    details: "Толығырақ",
    lastDiagnosis: "Соңғы диагноз",
  },
} as const;

export type TranslationKey = keyof typeof translations.ru;

export function getLang(): Lang {
  if (typeof window === "undefined") return "ru";
  return (localStorage.getItem("lang") as Lang) || "ru";
}

export function setLang(lang: Lang): void {
  localStorage.setItem("lang", lang);
}

export function t(key: TranslationKey, lang?: Lang): string {
  const l = lang || getLang();
  return translations[l][key] || translations.ru[key] || key;
}
