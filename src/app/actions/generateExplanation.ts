"use server";

const GROQ_API_KEY = process.env.GROQ_API_KEY || "";
const GROQ_MODEL = process.env.GROQ_MODEL || "llama-3.3-70b-versatile";

/**
 * Генерирует краткое объяснение для пациента — почему система поставила этот диагноз.
 */
const PATIENT_EXPLANATION_PROMPT = `
You are a medical AI assistant. The patient described symptoms, the ML system made a preliminary diagnosis.
Your job: write a SHORT explanation in Russian for the PATIENT — why this diagnosis was suggested.

RULES:
1. Write 2-4 sentences in Russian, simple language (patient-friendly)
2. Explain which symptoms led to this diagnosis
3. Do NOT scare the patient, be calm and professional
4. Mention that this is a preliminary assessment and a doctor's consultation is needed
5. Do NOT output JSON, just plain Russian text

Example output:
"На основании ваших симптомов — боль в животе (сильная) и рвота — система предположила возможный аппендицит. Эти симптомы характерны для острого воспаления аппендикса, особенно при выраженной боли. Рекомендуем обратиться к хирургу для подтверждения диагноза."
`;

/**
 * Генерирует подробное обоснование для врача — детальный клинический разбор.
 */
const DOCTOR_EXPLANATION_PROMPT = `
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

export interface ExplanationResult {
  patientExplanation: string;
  doctorExplanation: string;
  error?: string;
}

export async function generateExplanation(
  userMessage: string,
  detectedSymptoms: string[],
  diagnosisName: string,
  diagnosisLabel: string,
  topDiseases: { name: string; label: string; score: number }[],
  labInfluences?: { markerName: string; status: "high" | "low"; direction: string; effect: "boost" | "suppress"; diseases: string[]; delta: number }[]
): Promise<ExplanationResult> {
  if (!GROQ_API_KEY) {
    return {
      patientExplanation: "Анализ симптомов завершён. Рекомендуем обратиться к врачу.",
      doctorExplanation: "API ключ не настроен. Автоматическое обоснование недоступно.",
    };
  }

  const symptomsText = detectedSymptoms.join(", ");
  const topDiseasesText = topDiseases
    .map((d, i) => `${i + 1}. ${d.label} (${d.name}) — ${Math.round(d.score * 100)}%`)
    .join("\n");

  let labSection = "";
  if (labInfluences && labInfluences.length > 0) {
    const labLines = labInfluences.map(
      (inf) =>
        `- ${inf.markerName}: ${inf.direction} → ${inf.effect === "boost" ? "усиливает" : "снижает"}: ${inf.diseases.join(", ")}`
    );
    labSection = `\nДанные лабораторных анализов пациента:\n${labLines.join("\n")}`;
  }

  const contextMessage = `Пациент написал: "${userMessage}"
Обнаруженные симптомы: ${symptomsText}
Основной диагноз ML-модели: ${diagnosisLabel} (${diagnosisName})
Топ-3 вероятных заболеваний:
${topDiseasesText}${labSection}`;

  try {
    // Запускаем оба запроса параллельно
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
            { role: "system", content: PATIENT_EXPLANATION_PROMPT },
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
            { role: "system", content: DOCTOR_EXPLANATION_PROMPT },
            { role: "user", content: contextMessage },
          ],
          temperature: 0.3,
          max_tokens: 1500,
        }),
      }),
    ]);

    let patientExplanation = "Анализ симптомов завершён. Рекомендуем обратиться к врачу.";
    let doctorExplanation = "Автоматическое обоснование недоступно.";

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
      patientExplanation: "Анализ симптомов завершён. Рекомендуем обратиться к врачу.",
      doctorExplanation: "Произошла ошибка при генерации обоснования.",
      error: err instanceof Error ? err.message : "Unknown error",
    };
  }
}
