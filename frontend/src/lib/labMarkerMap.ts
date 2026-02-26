import type { LabResultItem } from "@/lib/api";
import type { LabContext, LabInfluence } from "@/lib/types";

interface LabMarkerRule {
  aliases: string[];
  trigger: "high" | "low";
  directionLabel: string;
  effects: Record<string, number>;
}

const LAB_MARKER_RULES: LabMarkerRule[] = [
  // ─── Inflammatory markers ───────────────────────────────────
  {
    aliases: ["лейкоцит", "wbc", "white blood", "белые кровяные"],
    trigger: "high",
    directionLabel: "Повышены",
    effects: {
      Pneumonia: +5,
      Meningitis: +4,
      Appendicitis: +4,
      "Scarlet Fever": +3,
      Influenza: +2,
    },
  },
  {
    aliases: ["лейкоцит", "wbc", "white blood"],
    trigger: "low",
    directionLabel: "Понижены",
    effects: {
      Pneumonia: -3,
      Meningitis: -2,
    },
  },
  {
    aliases: ["соэ", "esr", "скорость оседания"],
    trigger: "high",
    directionLabel: "Повышена",
    effects: {
      Pneumonia: +5,
      Meningitis: +4,
      "Scarlet Fever": +4,
      Appendicitis: +3,
      Influenza: +2,
    },
  },
  {
    aliases: ["срб", "crp", "c-реактивный", "c реактивный"],
    trigger: "high",
    directionLabel: "Повышен",
    effects: {
      Pneumonia: +5,
      Meningitis: +5,
      Appendicitis: +5,
      "Scarlet Fever": +4,
    },
  },
  // ─── Glucose / Diabetes ─────────────────────────────────────
  {
    aliases: ["глюкоз", "glucose", "сахар крови", "blood sugar"],
    trigger: "high",
    directionLabel: "Повышена",
    effects: {
      "Type 1 Diabetes": +7,
      Gastroenteritis: -2,
    },
  },
  {
    aliases: ["глюкоз", "glucose"],
    trigger: "low",
    directionLabel: "Понижена",
    effects: {
      "Type 1 Diabetes": +3,
    },
  },
  // ─── Hemoglobin ─────────────────────────────────────────────
  {
    aliases: ["гемоглоб", "hemoglobin", "haemoglobin", "hgb", "hb "],
    trigger: "low",
    directionLabel: "Понижен",
    effects: {
      Influenza: -2,
      "Common Cold": -2,
      "Type 1 Diabetes": +2,
    },
  },
  // ─── Platelets ──────────────────────────────────────────────
  {
    aliases: ["тромбоцит", "platelet", "plt"],
    trigger: "low",
    directionLabel: "Понижены",
    effects: {
      Meningitis: +3,
    },
  },
  // ─── Eosinophils (allergy / asthma / eczema) ────────────────
  {
    aliases: ["эозинофил", "eosinophil"],
    trigger: "high",
    directionLabel: "Повышены",
    effects: {
      Asthma: +5,
      Eczema: +4,
      Bronchiolitis: +2,
    },
  },
  // ─── IgE ────────────────────────────────────────────────────
  {
    aliases: ["иге", "ige", "immunoglobulin e"],
    trigger: "high",
    directionLabel: "Повышен",
    effects: {
      Asthma: +4,
      Eczema: +5,
    },
  },
  // ─── Procalcitonin ─────────────────────────────────────────
  {
    aliases: ["прокальцитонин", "procalcitonin", "pct"],
    trigger: "high",
    directionLabel: "Повышен",
    effects: {
      Pneumonia: +6,
      Meningitis: +5,
      Appendicitis: +4,
      "Scarlet Fever": +3,
    },
  },
  // ─── Urine: ketones ─────────────────────────────────────────
  {
    aliases: ["кетон", "ketone", "ацетон"],
    trigger: "high",
    directionLabel: "Повышены",
    effects: {
      "Type 1 Diabetes": +6,
    },
  },
  // ─── Potassium ──────────────────────────────────────────────
  {
    aliases: ["калий", "potassium", "k+"],
    trigger: "low",
    directionLabel: "Понижен",
    effects: {
      Gastroenteritis: +3,
    },
  },
  // ─── Cholesterol (general health marker, mild) ──────────────
  {
    aliases: ["холестерин", "cholesterol"],
    trigger: "high",
    directionLabel: "Повышен",
    effects: {
      "Type 1 Diabetes": +2,
    },
  },
  // ─── Bilirubin ──────────────────────────────────────────────
  {
    aliases: ["билируб", "bilirubin"],
    trigger: "high",
    directionLabel: "Повышен",
    effects: {
      Gastroenteritis: +2,
    },
  },
];

function matchesAlias(markerName: string, aliases: string[]): boolean {
  const normalized = markerName.toLowerCase();
  return aliases.some((alias) => normalized.includes(alias.toLowerCase()));
}

export function applyLabContext(labItems: LabResultItem[]): LabContext {
  if (!labItems || labItems.length === 0) {
    return { adjustments: {}, influences: [], hasData: false };
  }

  const adjustments: Record<string, number> = {};
  const influences: LabInfluence[] = [];
  const seen = new Set<string>(); // deduplicate: "alias:trigger"

  for (const item of labItems) {
    if (item.status === "normal" || !item.status) continue;

    for (const rule of LAB_MARKER_RULES) {
      if (rule.trigger !== item.status) continue;
      if (!matchesAlias(item.name, rule.aliases)) continue;

      const dedupeKey = `${rule.aliases[0]}:${rule.trigger}`;
      if (seen.has(dedupeKey)) continue;
      seen.add(dedupeKey);

      const affectedDiseases: string[] = [];

      for (const [disease, delta] of Object.entries(rule.effects)) {
        adjustments[disease] = (adjustments[disease] ?? 0) + delta;
        affectedDiseases.push(disease);
      }

      if (affectedDiseases.length > 0) {
        const firstDelta = Object.values(rule.effects)[0];
        influences.push({
          markerName: item.name,
          status: item.status as "high" | "low",
          direction: rule.directionLabel,
          effect: firstDelta > 0 ? "boost" : "suppress",
          diseases: affectedDiseases,
          delta: firstDelta,
        });
      }
    }
  }

  return {
    adjustments,
    influences,
    hasData: influences.length > 0,
  };
}
