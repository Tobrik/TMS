"use server";

const GROQ_API_KEY = process.env.GROQ_API_KEY || "";
const GROQ_MODEL = process.env.GROQ_MODEL || "llama-3.3-70b-versatile";

type Lang = "ru" | "en" | "kk";

const LANG_NAMES: Record<Lang, string> = {
  ru: "Russian",
  en: "English",
  kk: "Kazakh",
};

function getPatientPrompt(lang: Lang): string {
  const langName = LANG_NAMES[lang];
  return `
You are a medical AI assistant. The patient described symptoms, the ML system made a preliminary diagnosis.
Your job: write a SHORT explanation in ${langName} for the PATIENT — why this diagnosis was suggested.

RULES:
1. Write 2-4 sentences in ${langName}, simple language (patient-friendly)
2. Explain which symptoms led to this diagnosis
3. Do NOT scare the patient, be calm and professional
4. Mention that this is a preliminary assessment and a doctor's consultation is needed
5. Do NOT output JSON, just plain ${langName} text
6. You MUST write ENTIRELY in ${langName} — do not mix languages
`;
}

function getDoctorPrompt(lang: Lang): string {
  const langName = LANG_NAMES[lang];
  if (lang === "en") {
    return `
You are a clinical decision support system providing detailed analysis for a DOCTOR.
The ML model made a preliminary diagnosis based on patient-reported symptoms.
Provide a thorough clinical reasoning explanation in English.

YOUR RESPONSE MUST INCLUDE ALL OF THESE SECTIONS:

1. **Clinical Reasoning**: Why this specific diagnosis — which symptoms and their combination point to it. Note the weight of each symptom.

2. **Differential Diagnosis**: What other conditions should be considered given the symptoms and why they are less likely (or why they cannot be ruled out).

3. **Key Symptoms**: Which symptoms were decisive for the diagnosis and which were supportive. Note any "red flags".

4. **Recommended Tests**: What laboratory tests and examinations should be ordered to confirm/exclude the diagnosis.

5. **Points of Attention**: Warnings about possible complications, what to monitor.

6. **Laboratory Data**: If lab results are provided, mention specific values and their clinical significance for this diagnosis.

RULES:
- Write in English, medical terminology is OK (this is for doctors)
- Be thorough and detailed (8-15 sentences minimum)
- Use markdown formatting: **bold** for headers, bullet points for lists
- Reference specific symptoms with severity levels
- This is an AI ASSISTANT tool, not a replacement for clinical judgment — state this clearly
- Do NOT output JSON
`;
  }

  if (lang === "kk") {
    return `
You are a clinical decision support system providing detailed analysis for a DOCTOR.
The ML model made a preliminary diagnosis based on patient-reported symptoms.
Provide a thorough clinical reasoning explanation in Kazakh.

YOUR RESPONSE MUST INCLUDE ALL OF THESE SECTIONS:

1. **Клиникалық негіздеме**: Неге дәл осы диагноз — қандай симптомдар мен олардың комбинациясы көрсетеді.

2. **Дифференциалды диагностика**: Осы симптоматикада тағы қандай аурулар қарастырылуы керек.

3. **Негізгі симптомдар**: Қандай симптомдар шешуші болды. "Қызыл жалаушаларды" атаңыз.

4. **Ұсынылатын зерттеулер**: Диагнозды растау/жоққа шығару үшін қандай зерттеулер тағайындау керек.

5. **Назар аударатын жайттар**: Ықтимал асқынулар, нені бақылау керек.

6. **Зертханалық деректер**: Егер сараптама нәтижелері берілсе, нақты көрсеткіштерді атаңыз.

RULES:
- Write in Kazakh, medical terminology is OK (this is for doctors)
- Be thorough and detailed (8-15 sentences minimum)
- Use markdown formatting: **bold** for headers, bullet points for lists
- Reference specific symptoms with severity levels
- This is an AI ASSISTANT tool, not a replacement for clinical judgment — state this clearly
- Do NOT output JSON
`;
  }

  // Default: Russian
  return `
You are a clinical decision support system providing detailed analysis for a DOCTOR.
The ML model made a preliminary diagnosis based on patient-reported symptoms.
Provide a thorough clinical reasoning explanation in Russian.

YOUR RESPONSE MUST INCLUDE ALL OF THESE SECTIONS:

1. **Клиническое обоснование**: Почему именно этот диагноз — какие симптомы и их комбинация указывают на него. Укажите вес каждого симптома.

2. **Дифференциальная диагностика**: Какие ещё заболевания стоит рассмотреть при данной симптоматике и почему они менее вероятны (или наоборот — почему их нельзя исключить).

3. **Ключевые симптомы**: Какие симптомы стали решающими для диагноза, а какие — вспомогательными. Отметьте, если есть «красные флаги» (red flags).

4. **Рекомендуемые обследования**: Какие анализы и обследования стоит назначить для подтверждения/исключения диагноза.

5. **На что обратить внимание**: Предупреждения о возможных осложнениях, что нужно мониторить.

6. **Лабораторные данные**: Если предоставлены результаты анализов, обязательно упомяни конкретные показатели и их клиническое значение для данного диагноза.

RULES:
- Write in Russian, medical terminology is OK (this is for doctors)
- Be thorough and detailed (8-15 sentences minimum)
- Use markdown formatting: **bold** for headers, bullet points for lists
- Reference specific symptoms with severity levels
- This is an AI ASSISTANT tool, not a replacement for clinical judgment — state this clearly
- Do NOT output JSON
`;
}

const FALLBACK_PATIENT: Record<Lang, string> = {
  ru: "Анализ симптомов завершён. Рекомендуем обратиться к врачу.",
  en: "Symptom analysis complete. We recommend consulting a doctor.",
  kk: "Симптомдар талдауы аяқталды. Дәрігерге жүгінуді ұсынамыз.",
};

const FALLBACK_DOCTOR: Record<Lang, string> = {
  ru: "Автоматическое обоснование недоступно.",
  en: "Automatic clinical reasoning is unavailable.",
  kk: "Автоматты клиникалық негіздеме қол жетімсіз.",
};

const FALLBACK_DOCTOR_NO_KEY: Record<Lang, string> = {
  ru: "API ключ не настроен. Автоматическое обоснование недоступно.",
  en: "API key not configured. Automatic reasoning unavailable.",
  kk: "API кілті бапталмаған. Автоматты негіздеме қол жетімсіз.",
};

const FALLBACK_DOCTOR_ERROR: Record<Lang, string> = {
  ru: "Произошла ошибка при генерации обоснования.",
  en: "An error occurred while generating the reasoning.",
  kk: "Негіздеме жасау кезінде қате орын алды.",
};

export interface ExplanationResult {
  patientExplanation: string;
  doctorExplanation: string;
  error?: string;
}

function getContextMessage(
  userMessage: string,
  symptomsText: string,
  diagnosisLabel: string,
  diagnosisName: string,
  topDiseasesText: string,
  labSection: string,
  lang: Lang
): string {
  if (lang === "en") {
    return `Patient wrote: "${userMessage}"
Detected symptoms: ${symptomsText}
Main ML model diagnosis: ${diagnosisLabel} (${diagnosisName})
Top 3 probable conditions:
${topDiseasesText}${labSection}`;
  }
  if (lang === "kk") {
    return `Пациент жазды: "${userMessage}"
Анықталған симптомдар: ${symptomsText}
ML моделінің негізгі диагнозы: ${diagnosisLabel} (${diagnosisName})
Ең ықтимал 3 ауру:
${topDiseasesText}${labSection}`;
  }
  return `Пациент написал: "${userMessage}"
Обнаруженные симптомы: ${symptomsText}
Основной диагноз ML-модели: ${diagnosisLabel} (${diagnosisName})
Топ-3 вероятных заболеваний:
${topDiseasesText}${labSection}`;
}

export async function generateExplanation(
  userMessage: string,
  detectedSymptoms: string[],
  diagnosisName: string,
  diagnosisLabel: string,
  topDiseases: { name: string; label: string; score: number }[],
  labInfluences?: { markerName: string; status: "high" | "low"; direction: string; effect: "boost" | "suppress"; diseases: string[]; delta: number }[],
  lang: Lang = "ru"
): Promise<ExplanationResult> {
  if (!GROQ_API_KEY) {
    return {
      patientExplanation: FALLBACK_PATIENT[lang],
      doctorExplanation: FALLBACK_DOCTOR_NO_KEY[lang],
    };
  }

  const symptomsText = detectedSymptoms.join(", ");
  const topDiseasesText = topDiseases
    .map((d, i) => `${i + 1}. ${d.label} (${d.name}) — ${Math.round(d.score * 100)}%`)
    .join("\n");

  let labSection = "";
  if (labInfluences && labInfluences.length > 0) {
    const boostLabel = lang === "en" ? "boosts" : lang === "kk" ? "күшейтеді" : "усиливает";
    const suppressLabel = lang === "en" ? "reduces" : lang === "kk" ? "төмендетеді" : "снижает";
    const labLines = labInfluences.map(
      (inf) =>
        `- ${inf.markerName}: ${inf.direction} → ${inf.effect === "boost" ? boostLabel : suppressLabel}: ${inf.diseases.join(", ")}`
    );
    const labHeader = lang === "en" ? "\nPatient lab data:" : lang === "kk" ? "\nПациенттің зертханалық деректері:" : "\nДанные лабораторных анализов пациента:";
    labSection = `${labHeader}\n${labLines.join("\n")}`;
  }

  const contextMessage = getContextMessage(userMessage, symptomsText, diagnosisLabel, diagnosisName, topDiseasesText, labSection, lang);

  try {
    const [patientResp, doctorResp] = await Promise.all([
      fetch("https://api.groq.com/openai/v1/chat/completions", {
        method: "POST",
        headers: {
          Authorization: `Bearer ${GROQ_API_KEY}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          model: GROQ_MODEL,
          messages: [
            { role: "system", content: getPatientPrompt(lang) },
            { role: "user", content: contextMessage },
          ],
          temperature: 0.3,
          max_tokens: 300,
        }),
      }),
      fetch("https://api.groq.com/openai/v1/chat/completions", {
        method: "POST",
        headers: {
          Authorization: `Bearer ${GROQ_API_KEY}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          model: GROQ_MODEL,
          messages: [
            { role: "system", content: getDoctorPrompt(lang) },
            { role: "user", content: contextMessage },
          ],
          temperature: 0.3,
          max_tokens: 1500,
        }),
      }),
    ]);

    let patientExplanation = FALLBACK_PATIENT[lang];
    let doctorExplanation = FALLBACK_DOCTOR[lang];

    if (patientResp.ok) {
      const data = await patientResp.json();
      const content = data.choices[0]?.message?.content?.trim();
      if (content) patientExplanation = content;
    }

    if (doctorResp.ok) {
      const data = await doctorResp.json();
      const content = data.choices[0]?.message?.content?.trim();
      if (content) doctorExplanation = content;
    }

    return { patientExplanation, doctorExplanation };
  } catch (err) {
    console.error("generateExplanation error:", err);
    return {
      patientExplanation: FALLBACK_PATIENT[lang],
      doctorExplanation: FALLBACK_DOCTOR_ERROR[lang],
      error: err instanceof Error ? err.message : "Unknown error",
    };
  }
}
