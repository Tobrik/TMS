"use client";

import { useState } from "react";
import type { AccuracyCondition, Zone } from "@/lib/api";
import { getDiseaseLabel } from "@/lib/diseaseWeights";
import { t, type Lang, type TranslationKey } from "@/lib/i18n";
import { CiBar, MetricText } from "./CiBar";
import { tf } from "./format";

const SHOW_FIRST = 10;

const ZONE_DOT: Record<Zone, { dot: string; label: TranslationKey }> = {
  red: { dot: "bg-urgent", label: "zoneShortRed" },
  yellow: { dot: "bg-warn", label: "zoneShortYellow" },
  green: { dot: "bg-ok", label: "zoneShortGreen" },
};

// Weakest first; unknown values go last
function weakestFirst(list: AccuracyCondition[]): AccuracyCondition[] {
  return [...list].sort((a, b) => (a.top1.value ?? 2) - (b.top1.value ?? 2));
}

export function ConditionBars({ conditions, lang }: { conditions: AccuracyCondition[]; lang: Lang }) {
  const [expanded, setExpanded] = useState(false);
  const sorted = weakestFirst(conditions);
  const hasMore = sorted.length > SHOW_FIRST;

  return (
    <section className="space-y-3">
      <h2 className="text-lg font-semibold text-foreground sm:text-xl">{t("accConditionsTitle", lang)}</h2>
      <p className="text-sm text-foreground/85">{t("accConditionsIntro", lang)}</p>
      {sorted.length === 0 ? (
        <p className="text-sm text-muted-foreground">{t("accNotAvailable", lang)}</p>
      ) : (
        <ul className="space-y-4">
          {sorted.map((c, i) => {
            // Rows past the first ten are hidden on screen, but always printed
            const hiddenRow = !expanded && i >= SHOW_FIRST;
            const label = lang === "ru" ? c.label_ru : getDiseaseLabel(c.name, lang);
            const zone = ZONE_DOT[c.tier];
            return (
              <li
                key={c.name}
                className={`${hiddenRow ? "hidden print:flex" : "flex"} flex-col gap-2 sm:flex-row sm:items-center sm:gap-4 print:break-inside-avoid`}
              >
                <div className="min-w-0 sm:w-64 sm:shrink-0">
                  <p className="text-sm text-foreground">{label}</p>
                  <p className="flex items-center gap-1.5 text-xs text-muted-foreground">
                    <span aria-hidden="true" className={`inline-block h-2 w-2 rounded-full ${zone.dot}`} />
                    {t(zone.label, lang)}
                  </p>
                </div>
                <div className="flex min-w-0 flex-1 flex-col gap-1 sm:flex-row sm:items-center sm:gap-3">
                  <div className="min-w-0 flex-1">
                    <CiBar metric={c.top1} />
                  </div>
                  <p className="whitespace-nowrap text-sm">
                    <MetricText metric={c.top1} lang={lang} />
                  </p>
                </div>
              </li>
            );
          })}
        </ul>
      )}
      {hasMore && (
        <button
          type="button"
          onClick={() => setExpanded((e) => !e)}
          aria-expanded={expanded}
          className="text-sm font-medium text-primary underline-offset-4 hover:underline focus-visible:outline-none focus-visible:ring-[3px] focus-visible:ring-ring rounded print:hidden"
        >
          {expanded ? t("accShowLess", lang) : tf("accShowAll", lang, { count: sorted.length })}
        </button>
      )}
    </section>
  );
}
