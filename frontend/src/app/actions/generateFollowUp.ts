"use server";

const GROQ_API_KEY = process.env.GROQ_API_KEY || "";
const GROQ_MODEL = process.env.GROQ_MODEL || "llama-3.3-70b-versatile";

type Lang = "ru" | "en" | "kk";

const LANG_NAMES: Record<Lang, string> = {
  ru: "Russian",
  en: "English",
  kk: "Kazakh",
};

function getFollowUpPrompt(lang: Lang): string {
  const langName = LANG_NAMES[lang];
  return `
You are a medical assistant helping to clarify patient symptoms.
The patient described their symptoms, and the system detected very few symptoms (1-2).
Your job: ask ONE short clarifying question in ${langName} to help narrow down the diagnosis.

RULES:
1. Ask ONLY ONE question, max 2 sentences
2. The question must be in ${langName}
3. Be empathetic and professional
4. Focus on the most diagnostically relevant follow-up
5. Do NOT output JSON, just plain ${langName} text
6. You MUST write ENTIRELY in ${langName} — do not mix languages
`;
}

const FALLBACK: Record<Lang, string> = {
  ru: "Можете описать симптомы подробнее? Например, где именно болит и есть ли другие жалобы?",
  en: "Could you describe your symptoms in more detail? For example, where exactly does it hurt and do you have any other complaints?",
  kk: "Симптомдарыңызды толығырақ сипаттай аласыз ба? Мысалы, қай жерде ауырады және басқа шағымдар бар ма?",
};

const FALLBACK_SHORT: Record<Lang, string> = {
  ru: "Можете уточнить ваши симптомы подробнее?",
  en: "Could you clarify your symptoms in more detail?",
  kk: "Симптомдарыңызды толығырақ түсіндіре аласыз ба?",
};

export async function generateFollowUp(
  detectedSymptoms: string[],
  userMessage: string,
  lang: Lang = "ru"
): Promise<string> {
  if (!GROQ_API_KEY) {
    return FALLBACK_SHORT[lang];
  }

  const contextLabel = lang === "en" ? "Patient wrote" : lang === "kk" ? "Пациент жазды" : "Пациент написал";
  const symptomsLabel = lang === "en" ? "Detected symptoms" : lang === "kk" ? "Анықталған симптомдар" : "Обнаруженные симптомы";
  const askLabel = lang === "en" ? "Ask one clarifying question" : lang === "kk" ? "Бір нақтылау сұрағын қойыңыз" : "Задай один уточняющий вопрос";

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
          { role: "system", content: getFollowUpPrompt(lang) },
          {
            role: "user",
            content: `${contextLabel}: "${userMessage}"\n${symptomsLabel}: ${detectedSymptoms.join(", ")}\n\n${askLabel}:`,
          },
        ],
        temperature: 0.3,
        max_tokens: 150,
      }),
    });

    if (!response.ok) {
      return FALLBACK[lang];
    }

    const data = await response.json();
    const content = data.choices[0]?.message?.content?.trim();
    return content || FALLBACK_SHORT[lang];
  } catch {
    return FALLBACK[lang];
  }
}
