"use server";

const GROQ_API_KEY = process.env.GROQ_API_KEY || "";
const GROQ_MODEL = process.env.GROQ_MODEL || "llama-3.3-70b-versatile";

const FOLLOW_UP_PROMPT = `
You are a medical assistant helping to clarify patient symptoms.
The patient described their symptoms, and the system detected very few symptoms (1-2).
Your job: ask ONE short clarifying question in Russian to help narrow down the diagnosis.

RULES:
1. Ask ONLY ONE question, max 2 sentences
2. The question must be in Russian
3. Be empathetic and professional
4. Focus on the most diagnostically relevant follow-up
5. Examples of good questions:
   - For "болит живот": "Уточните, пожалуйста: боль вверху, внизу или вокруг пупка? Есть ли тошнота или рвота?"
   - For "кашель": "Кашель сухой или с мокротой? Есть ли температура?"
   - For "болит голова": "Где именно болит — в висках, затылке или по всей голове? Есть ли температура или тошнота?"
6. Do NOT output JSON, just plain Russian text
`;

export async function generateFollowUp(
  detectedSymptoms: string[],
  userMessage: string
): Promise<string> {
  if (!GROQ_API_KEY) {
    return "Можете уточнить ваши симптомы подробнее?";
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
          { role: "system", content: FOLLOW_UP_PROMPT },
          {
            role: "user",
            content: `Пациент написал: "${userMessage}"\nОбнаруженные симптомы: ${detectedSymptoms.join(", ")}\n\nЗадай один уточняющий вопрос:`,
          },
        ],
        temperature: 0.3,
        max_tokens: 150,
      }),
    });

    if (!response.ok) {
      return "Можете описать симптомы подробнее? Например, где именно болит и есть ли другие жалобы?";
    }

    const data = await response.json();
    const content = data.choices[0]?.message?.content?.trim();
    return content || "Можете уточнить ваши симптомы подробнее?";
  } catch {
    return "Можете описать симптомы подробнее? Например, где именно болит и есть ли другие жалобы?";
  }
}
