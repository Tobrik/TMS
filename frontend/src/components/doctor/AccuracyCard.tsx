"use client";

import { useState } from "react";
import Link from "next/link";
import { Activity, ChevronDown, ChevronUp, FileText } from "lucide-react";
import { Button } from "@/components/ui/button";
import { StatTile } from "@/components/brand";
import { getDiseaseLabel } from "@/lib/diseaseWeights";
import { t, type Lang } from "@/lib/i18n";
import type { FeedbackStats } from "@/lib/api";
import { pct } from "./doctorUtils";

// Ratios are weighted by rows per version, so the totals match the backend's counts
export function summarizeStats(stats: FeedbackStats) {
  const versions = Object.values(stats.by_version);
  const rows = versions.reduce((s, v) => s + v.n, 0);
  const weighted = (key: "top1" | "top3" | "zone_exact") =>
    rows === 0 ? 0 : versions.reduce((s, v) => s + v[key] * v.n, 0) / rows;
  return {
    top1: weighted("top1"),
    top3: weighted("top3"),
    zone: weighted("zone_exact"),
    underTriage: versions.reduce((s, v) => s + v.under_triage, 0),
    overTriage: versions.reduce((s, v) => s + v.over_triage, 0),
  };
}

export function AccuracyCard({ stats, lang, defaultOpen = false }: { stats: FeedbackStats; lang: Lang; defaultOpen?: boolean }) {
  const [open, setOpen] = useState(defaultOpen);
  const s = stats.n > 0 ? summarizeStats(stats) : null;
  return (
    <div className="rounded-xl border border-border bg-card shadow-card">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        className="flex min-h-11 w-full items-center justify-between gap-2 rounded-xl px-4 py-2.5 text-left text-sm font-semibold text-foreground outline-none transition-colors hover:bg-muted focus-visible:ring-[3px] focus-visible:ring-ring"
      >
        <span className="flex items-center gap-2">
          <Activity className="size-4 text-primary" aria-hidden="true" />
          {t("modelAccuracy", lang)}
        </span>
        {open ? <ChevronUp className="size-4 text-muted-foreground" /> : <ChevronDown className="size-4 text-muted-foreground" />}
      </button>
      {open && (
        <div className="space-y-4 border-t border-border px-4 py-4">
          {!s ? (
            <p className="text-sm text-muted-foreground">{t("noFeedbackYet", lang)}</p>
          ) : (
            <>
              <div className="grid grid-cols-2 gap-3">
                <StatTile label={t("accuracyN", lang)} value={stats.n} />
                <StatTile label={t("accuracyTop1", lang)} value={pct(s.top1)} />
                <StatTile label={t("accuracyTop3", lang)} value={pct(s.top3)} />
                <StatTile label={t("accuracyZone", lang)} value={pct(s.zone)} />
                <StatTile
                  label={t("underTriage", lang)}
                  value={<span className={s.underTriage > 0 ? "text-urgent" : undefined}>{s.underTriage}</span>}
                />
                <StatTile label={t("overTriage", lang)} value={s.overTriage} />
              </div>
              {stats.confusions.length > 0 && (
                <div className="space-y-1.5">
                  <p className="text-[13px] font-semibold text-muted-foreground">{t("confusions", lang)}</p>
                  <ul className="space-y-1">
                    {stats.confusions.map((c) => (
                      <li key={`${c.model}-${c.doctor}`} className="flex justify-between gap-2 text-sm text-foreground">
                        <span className="min-w-0 truncate">
                          {getDiseaseLabel(c.model, lang)} → {getDiseaseLabel(c.doctor, lang)}
                        </span>
                        <span className="shrink-0 tabular-nums text-muted-foreground">×{c.count}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </>
          )}
          <Button asChild variant="outline" size="sm">
            <Link href="/accuracy">
              <FileText />
              {t("accuracyReportLink", lang)}
            </Link>
          </Button>
        </div>
      )}
    </div>
  );
}
