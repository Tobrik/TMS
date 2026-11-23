import type { Lang } from "@/lib/i18n";

// Text used only by the accuracy, lab and policy pages. Shared UI text stays in lib/i18n.ts.
export const pageStrings = {
  ru: {
    policyBack: "Вернуться на главную",
    back: "Назад",
  },
  kk: {
    policyBack: "Басты бетке оралу",
    back: "Артқа",
  },
  en: {
    policyBack: "Back to home",
    back: "Back",
  },
} as const;

export type PageStringKey = keyof typeof pageStrings.ru;

export function ps(key: PageStringKey, lang: Lang): string {
  return pageStrings[lang]?.[key] || pageStrings.ru[key];
}
