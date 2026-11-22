import type { AccuracyMetric, AccuracyReport } from "@/lib/api";
import { t, type Lang, type TranslationKey } from "@/lib/i18n";

export const LOCALE: Record<Lang, string> = { ru: "ru-RU", en: "en-GB", kk: "kk-KZ" };

/** Translation with {placeholders} filled in. */
export function tf(key: TranslationKey, lang: Lang, vars: Record<string, string | number> = {}): string {
  return t(key, lang).replace(/\{(\w+)\}/g, (_, name: string) => String(vars[name] ?? ""));
}

// Percentages get one decimal only below 10%, so small rates (0.4%) stay readable
function digitsFor(share: number): number {
  return share * 100 < 10 ? 1 : 0;
}

function fixed(value: number, lang: Lang, digits: number): string {
  return new Intl.NumberFormat(LOCALE[lang], {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  }).format(value);
}

/** A share (0..1) as a number in percent, without the % sign. */
export function pctPlain(share: number | null | undefined, lang: Lang): string {
  if (share === null || share === undefined) return "—";
  return fixed(share * 100, lang, digitsFor(share));
}

/** A share (0..1) as "88%". */
export function formatPct(share: number | null | undefined, lang: Lang): string {
  if (share === null || share === undefined) return "—";
  return `${pctPlain(share, lang)}%`;
}

/** The 95% interval as "88–97%", or null when there is none. */
export function formatCi(metric: AccuracyMetric | undefined, lang: Lang): string | null {
  if (!metric || !metric.ci) return null;
  const [lo, hi] = metric.ci;
  return `${pctPlain(lo, lang)}–${pctPlain(hi, lang)}%`;
}

export function formatNumber(x: number, lang: Lang, maxDigits = 1): string {
  return new Intl.NumberFormat(LOCALE[lang], { maximumFractionDigits: maxDigits }).format(x);
}

/** "0.9-1.0" as "0,9–1" in the locale's style. */
export function formatBin(bin: string, lang: Lang): string {
  const parts = bin.split("-").map(Number);
  if (parts.length !== 2 || parts.some((p) => Number.isNaN(p))) return bin;
  return parts.map((p) => formatNumber(p, lang, 2)).join("–");
}

/** ISO timestamp, or a plain YYYY-MM-DD (read as a local date, so it does not shift a day). */
export function formatDate(value: string | null | undefined, lang: Lang): string {
  if (!value) return "—";
  const dateOnly = /^(\d{4})-(\d{2})-(\d{2})$/.exec(value);
  const date = dateOnly
    ? new Date(Number(dateOnly[1]), Number(dateOnly[2]) - 1, Number(dateOnly[3]))
    : new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  const options: Intl.DateTimeFormatOptions = dateOnly
    ? { day: "numeric", month: "long", year: "numeric" }
    : { day: "numeric", month: "long", year: "numeric", hour: "2-digit", minute: "2-digit" };
  return date.toLocaleString(LOCALE[lang], options);
}

/** The report with caveats and source lines in the UI language, when the report carries them. */
export function localizeReport<R extends { caveats: string[]; i18n?: AccuracyReport["i18n"];
  synthetic?: { source: string } | null; written_cases?: { source: string } | null }>(report: R, lang: "ru" | "en" | "kk"): R {
  const text = report.i18n?.[lang];
  if (!text) return report;
  return {
    ...report,
    caveats: text.caveats,
    synthetic: report.synthetic ? { ...report.synthetic, source: text.synthetic_source } : report.synthetic,
    written_cases: report.written_cases ? { ...report.written_cases, source: text.written_source } : report.written_cases,
  };
}
