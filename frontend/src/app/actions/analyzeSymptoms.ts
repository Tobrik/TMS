"use server";

import { SYMPTOM_LIST } from "@/lib/symptoms";

const GROQ_API_KEY = process.env.GROQ_API_KEY || "";
// Рекомендую оставить 70b, она лучше понимает медицинские нюансы
const GROQ_MODEL = process.env.GROQ_MODEL || "llama-3.3-70b-versatile"; 

const SYSTEM_PROMPT = `
You are an expert medical symptom extractor for a diagnostic system.
Parse user text (usually in Russian) and output a JSON of detected symptoms with severity (1-3).

ALLOWED SYMPTOMS LIST:
${JSON.stringify(SYMPTOM_LIST)}

═══ SEVERITY CALIBRATION (STRICT RULES) ═══

FEVER:
  1 = "температура", "37-37.5", "субфебрильная"
  2 = "37.5-38.5", "жар", "горячий лоб"
  3 = "38.5+", "39", "40", "высокая температура", "горит"

COUGH:
  1 = "покашливает", "немного кашляет", "першит"
  2 = "кашель", "кашляет"
  3 = "сильный кашель", "не может остановиться", "лающий кашель", "надрывный"

PAIN (ABDOMINAL_PAIN, CHEST_PAIN, HEADACHE):
  1 = "дискомфорт", "немного", "ноет", "побаливает"
  2 = "болит", "боль"
  3 = "сильно болит", "невыносимо", "резкая боль", "острая боль", "не могу терпеть"

DIARRHEA:
  1 = "жидкий стул", "разжижение"
  2 = "понос", "диарея"
  3 = "водянистый", "многократный", "не прекращается"

VOMITING:
  1 = "подташнивает", "однократная рвота"
  2 = "рвота", "вырвало"
  3 = "рвота фонтаном", "многократная рвота", "не может остановить"

OTHER SYMPTOMS: 1 = mentioned, 2 = clearly present, 3 = severe/extreme

═══ CRITICAL DIAGNOSTIC MAPPING RULES ═══

1. STRIDOR vs WHEEZING (CRUCIAL — wrong mapping = wrong diagnosis):
   STRIDOR (inspiratory): "Свист на ВДОХЕ", "Шумный вдох", "Лающий кашель", "Осип голос", "стридор"
   WHEEZING (expiratory): "Свист на ВЫДОХЕ", "Хрипы в груди", "свистящее дыхание", "хрипы"
   "Лающий кашель" → COUGH: 3 AND STRIDOR: 2

2. FEVER — NEGATIVE is important:
   "Температуры нет", "Нормальная температура", "Без температуры" → DO NOT include FEVER
   "Температура", "Жар", "Горячий", "37.5+" → FEVER with calibrated severity

3. Pain Locations:
   "Живот" / "Живот справа" / "Живот внизу" → ABDOMINAL_PAIN (right/lower = severity 3)
   "Грудь" / "За грудиной" / "Сердце" → CHEST_PAIN
   "Голова" / "Виски" / "Затылок" → HEADACHE
   "Шея не поворачивается" / "Шея жесткая" / "Затылок напряжен" → NECK_STIFFNESS

4. Breathing:
   "Тяжело дышать", "Задыхается", "Не хватает воздуха", "Одышка" → RESPIRATORY_DISTRESS

5. Skin:
   "Сыпь", "Высыпания", "Пятна на коже" → RASH
   "Чешется", "Зуд" → ITCHING

6. Symptom Combinations (extract ALL that apply):
   "Лающий кашель + осип голос" → COUGH: 3, STRIDOR: 2, SORE_THROAT: 1
   "Болит живот и тошнит" → ABDOMINAL_PAIN + NAUSEA (both!)
   "Температура + сыпь + горло" → FEVER + RASH + SORE_THROAT (all three!)

═══ OUTPUT RULES ═══
1. Return ONLY valid JSON: {"SYMPTOM_NAME": severity}
2. Severity must be integer: 1, 2, or 3. NEVER use 0, decimals, or values > 3
3. ONLY include symptoms EXPLICITLY found or clearly implied in the text
4. If patient DENIES a symptom ("нет кашля"), do NOT include it
5. Extract ALL mentioned symptoms, do not skip any
6. IGNORE symptoms not in the allowed list

Example: "Сильно болит живот, рвота, температуры нет" → {"ABDOMINAL_PAIN": 3, "VOMITING": 2}
(FEVER is NOT included because patient denied it)

═══ NO SYMPTOMS CASE ═══
If the user message contains NO medical symptoms at all (greetings, questions, jokes, unrelated text like "Привет" or "Я люблю котиков"),
return exactly: {"_NO_SYMPTOMS": true}

═══ LANGUAGE SUPPORT ═══
The user may write in Russian, Kazakh, or a mix of both languages.
Internally translate all input to English for symptom extraction.
Output JSON only — the language of user input does not matter.
Examples in Kazakh: "Басым ауырып тұр" = headache, "дене қызуы 39" = fever 39°C, "жөтел" = cough, "іш ауырады" = abdominal pain.
`;

export interface SymptomsResult {
  vector: number[];
  detectedSymptoms: string[];
  error?: string;
  rawResponse?: string; // Для отладки
  noSymptoms?: boolean; // true если пользователь не описал симптомы
}

export async function analyzeSymptoms(userMessage: string): Promise<SymptomsResult> {
  if (!GROQ_API_KEY) {
    console.error("Server Error: GROQ_API_KEY is missing in .env");
    return { vector: [], detectedSymptoms: [], error: "Configuration Error" };
  }

  try {
    const response = await fetch("https://api.groq.com/openai/v1/chat/completions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${GROQ_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: GROQ_MODEL,
        messages: [
          { role: "system", content: SYSTEM_PROMPT },
          { role: "user", content: userMessage },
        ],
        temperature: 0, // 0 = Максимальная детерминированность (строгость)
        response_format: { type: "json_object" },
      }),
    });

    if (!response.ok) {
      const errText = await response.text();
      console.error("Groq API Error:", errText);
      return { vector: [], detectedSymptoms: [], error: `AI Service Error: ${response.status}` };
    }

    const data = await response.json();
    const content = data.choices[0]?.message?.content;

    if (!content) {
      return { vector: [], detectedSymptoms: [], error: "Empty response from AI" };
    }

    // 1. Парсинг JSON
    let parsedSymptoms: Record<string, number> = {};
    try {
      parsedSymptoms = JSON.parse(content);
    } catch (e) {
      console.error("JSON Parse Failed. Raw content:", content);
      return { vector: [], detectedSymptoms: [], error: "Failed to parse symptoms" };
    }

    // 2. Проверяем на отсутствие симптомов
    if (parsedSymptoms._NO_SYMPTOMS) {
      return { vector: [], detectedSymptoms: [], noSymptoms: true };
    }

    // 3. Генерация вектора
    // Создаем массив нужной длины, заполненный нулями
    const vector = new Array(SYMPTOM_LIST.length).fill(0);
    const detectedList: string[] = [];

    // Проходим по ЭТАЛОННОМУ списку SYMPTOM_LIST, чтобы сохранить порядок индексов
    SYMPTOM_LIST.forEach((symptomRef, index) => {
      // Ищем ключ в ответе AI (игнорируя регистр для надежности)
      const foundKey = Object.keys(parsedSymptoms).find(
        (key) => key.toUpperCase() === symptomRef.toUpperCase()
      );

      if (foundKey) {
        let severity = parsedSymptoms[foundKey];
        
        // Приводим к числу и ограничиваем диапазоном 0-3
        severity = Number(severity);
        if (isNaN(severity)) severity = 0;
        if (severity > 3) severity = 3;
        if (severity < 0) severity = 0;

        if (severity > 0) {
          vector[index] = severity;
          detectedList.push(`${symptomRef} (${severity})`);
        }
      }
    });

    // Логирование для отладки (будет видно в терминале сервера)
    console.log("--- Symptom Analysis ---");
    console.log("Input:", userMessage);
    console.log("AI Parsed:", parsedSymptoms);
    console.log("Mapped Vector Indices:", vector.map((v, i) => v > 0 ? `${i}:${v}` : null).filter(Boolean));
    console.log("------------------------");

    return { 
      vector, 
      detectedSymptoms: detectedList,
      rawResponse: content 
    };

  } catch (err) {
    console.error("AnalyzeSymptoms Fatal Error:", err);
    return {
      vector: [],
      detectedSymptoms: [],
      error: err instanceof Error ? err.message : "Unknown error",
    };
  }
}