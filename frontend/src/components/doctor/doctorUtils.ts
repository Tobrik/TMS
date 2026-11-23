import { dt } from "@/lib/strings/doctor";
import type { Lang } from "@/lib/i18n";
import { shortFinding } from "@/lib/findingLabels";

export function parseIdFromCode(input: string): number | null {
  const trimmed = input.trim();
  if (!trimmed) return null;
  const last = trimmed.includes("-") ? trimmed.split("-").pop() ?? "" : trimmed;
  if (!/^\d+$/.test(last)) return null;
  const num = parseInt(last, 10);
  return num > 0 ? num : null;
}

const ANIMALS = [
  "LION", "BEAR", "WOLF", "HAWK", "DEER", "FOX", "OWL", "LYNX",
  "PIKE", "CROW", "SEAL", "HARE", "SWAN", "DOVE", "COLT",
];

/** The same `ANIMAL-id` code the patient sees. */
export function makeCode(patientId: number): string {
  return `${ANIMALS[patientId % ANIMALS.length]}-${patientId}`;
}

/** Backend timestamps are naive `YYYY-MM-DD HH:MM:SS.ffffff` in UTC (the database runs in UTC),
 *  or ISO strings with an offset. Naive ones are read as UTC, not as the browser's local time. */
export function parseServerTime(value: string | null | undefined): number | null {
  if (!value) return null;
  let iso = value.trim().replace(" ", "T").replace(/(\.\d{3})\d+/, "$1");
  if (!/(Z|[+-]\d{2}:?\d{2})$/.test(iso)) iso += "Z";
  const ms = Date.parse(iso);
  return Number.isNaN(ms) ? null : ms;
}

const DATE_LOCALE: Record<Lang, string> = { ru: "ru-RU", kk: "kk-KZ", en: "en-GB" };

/** A server timestamp as a short local date and time ("23 нояб. 2026 г., 12:05"). */
export function formatServerTime(value: string | null | undefined, lang: Lang): string {
  const ms = parseServerTime(value);
  if (ms === null) return value ?? "";
  return new Intl.DateTimeFormat(DATE_LOCALE[lang], { dateStyle: "medium", timeStyle: "short" }).format(ms);
}

export function timeAgo(value: string | null | undefined, lang: Lang, now: number = Date.now()): string {
  const ms = parseServerTime(value);
  if (ms === null) return "";
  const minutes = Math.floor((now - ms) / 60000);
  if (minutes < 1) return dt("justNow", lang);
  if (minutes < 60) return dt("minAgo", lang).replace("{n}", String(minutes));
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return dt("hourAgo", lang).replace("{n}", String(hours));
  return dt("dayAgo", lang).replace("{n}", String(Math.floor(hours / 24)));
}

export function pct(value: number): string {
  return `${Math.round(value * 100)}%`;
}

export const ZONE_EDGE = {
  red: "border-l-urgent",
  yellow: "border-l-warn",
  green: "border-l-ok",
  none: "border-l-border",
} as const;

const QUESTION_LEAD = /^(у вас есть |у вас |вас |do you have |do you |are you |have you |is there |сізде )/i;

/** Short clinical finding for an evidence code; when the code has no curated label, falls back to
 *  trimming the chat question the backend stored for it. */
export function findingLabel(code: string, lang: Lang, question?: string): string {
  const short = shortFinding(code, lang);
  if (short) return short;
  const raw = question ?? code;
  const core = raw.trim().replace(/[?？]+$/, "").replace(QUESTION_LEAD, "").trim();
  return core ? core.charAt(0).toUpperCase() + core.slice(1) : raw;
}
