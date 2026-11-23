"use client";

import { useMemo, useRef } from "react";
import { CircleHelp, Loader2, Search, Users } from "lucide-react";
import { Input } from "@/components/ui/input";
import { EmptyState, ZoneBadge } from "@/components/brand";
import { getDiseaseLabel } from "@/lib/diseaseWeights";
import { t, type Lang } from "@/lib/i18n";
import { dt } from "@/lib/strings/doctor";
import type { TriagePatient, Zone } from "@/lib/api";
import { cn } from "@/lib/utils";
import { makeCode, parseIdFromCode, timeAgo, ZONE_EDGE } from "./doctorUtils";

export interface QueuePaneProps {
  lang: Lang;
  patients: TriagePatient[];
  loading: boolean;
  selectedId: number | null;
  /** patient_id -> true while the "new" highlight is on. */
  newIds: Set<number>;
  now: number;
  search: string;
  onSearch: (value: string) => void;
  onSelect: (patientId: number) => void;
}

const GROUPS: { zone: Zone | null; titleKey: "redZone" | "yellowZone" | "greenZone" | "notAssessedGroup"; text: string }[] = [
  { zone: "red", titleKey: "redZone", text: "text-urgent" },
  { zone: "yellow", titleKey: "yellowZone", text: "text-warn" },
  { zone: "green", titleKey: "greenZone", text: "text-ok" },
  { zone: null, titleKey: "notAssessedGroup", text: "text-muted-foreground" },
];

export function QueuePane({ lang, patients, loading, selectedId, newIds, now, search, onSearch, onSelect }: QueuePaneProps) {
  const listRef = useRef<HTMLDivElement>(null);
  const query = search.trim().toLowerCase();

  const filtered = useMemo(() => {
    if (!query) return patients;
    return patients.filter(
      (p) => p.full_name.toLowerCase().includes(query) || makeCode(p.patient_id).toLowerCase().includes(query)
    );
  }, [patients, query]);

  const codeId = parseIdFromCode(search);
  const codeInQueue = codeId !== null && patients.some((p) => p.patient_id === codeId);

  const submitSearch = () => {
    if (codeId !== null && (codeInQueue || filtered.length === 0)) {
      onSelect(codeId);
    } else if (filtered.length > 0) {
      onSelect(filtered[0].patient_id);
    }
  };

  const onListKeyDown = (e: React.KeyboardEvent) => {
    if (e.key !== "ArrowDown" && e.key !== "ArrowUp") return;
    const rows = Array.from(listRef.current?.querySelectorAll<HTMLButtonElement>("button[data-row]") ?? []);
    const idx = rows.indexOf(document.activeElement as HTMLButtonElement);
    if (idx < 0) return;
    e.preventDefault();
    rows[Math.max(0, Math.min(rows.length - 1, idx + (e.key === "ArrowDown" ? 1 : -1)))]?.focus();
  };

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="relative">
        <Search className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" aria-hidden="true" />
        <Input
          type="search"
          value={search}
          onChange={(e) => onSearch(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") submitSearch();
            if (e.key === "ArrowDown") (listRef.current?.querySelector("button[data-row]") as HTMLElement | null)?.focus();
          }}
          placeholder={dt("searchPlaceholder", lang)}
          aria-label={t("searchPatient", lang)}
          className="pl-9"
        />
      </div>

      <div
        ref={listRef}
        onKeyDown={onListKeyDown}
        className="relative mt-3 min-h-0 flex-1 space-y-5 lg:overflow-y-auto lg:pr-1"
        aria-label={dt("queueTitle", lang)}
      >
        {codeId !== null && !codeInQueue && (
          <button
            type="button"
            onClick={() => onSelect(codeId)}
            className="flex min-h-11 w-full items-center gap-2 rounded-xl border border-dashed border-input bg-card px-4 py-2 text-left text-sm font-medium text-primary outline-none transition-colors hover:bg-accent focus-visible:ring-[3px] focus-visible:ring-ring"
          >
            <Search className="size-4" aria-hidden="true" />
            {dt("searchOpenCode", lang)}: {makeCode(codeId)}
          </button>
        )}

        {loading && patients.length === 0 && (
          <div className="flex justify-center py-10"><Loader2 className="size-6 animate-spin text-muted-foreground" aria-label={t("loadingDots", lang)} /></div>
        )}

        {!loading && patients.length === 0 && (
          <EmptyState icon={Users} title={dt("queueEmpty", lang)} text={dt("queueEmptyText", lang)} />
        )}

        {patients.length > 0 && filtered.length === 0 && !(codeId !== null && !codeInQueue) && (
          <p className="py-8 text-center text-sm text-muted-foreground">{dt("queueNoMatch", lang)}</p>
        )}

        {GROUPS.map((g) => {
          const rows = filtered.filter((p) => p.zone === g.zone);
          if (rows.length === 0) return null;
          return (
            <section key={g.zone ?? "none"} aria-label={t(g.titleKey, lang)}>
              <h2 className="mb-2 flex items-center gap-2 px-1 text-sm font-semibold">
                <span className={g.text}>{t(g.titleKey, lang)}</span>
                <span className="rounded-full bg-muted px-2 py-0.5 text-xs font-medium tabular-nums text-muted-foreground">{rows.length}</span>
              </h2>
              <ul className="space-y-2">
                {rows.map((p) => (
                  <li key={p.patient_id}>
                    <QueueRow
                      p={p}
                      lang={lang}
                      now={now}
                      selected={p.patient_id === selectedId}
                      isNew={newIds.has(p.patient_id)}
                      onSelect={onSelect}
                    />
                  </li>
                ))}
              </ul>
            </section>
          );
        })}
      </div>
    </div>
  );
}

function QueueRow({ p, lang, now, selected, isNew, onSelect }: {
  p: TriagePatient; lang: Lang; now: number; selected: boolean; isNew: boolean; onSelect: (id: number) => void;
}) {
  const edge = ZONE_EDGE[p.zone ?? "none"];
  const ago = timeAgo(p.diag_date ?? p.created_at, lang, now);
  return (
    <button
      type="button"
      data-row
      onClick={() => onSelect(p.patient_id)}
      aria-current={selected ? "true" : undefined}
      className={cn(
        "block w-full rounded-xl border border-l-4 bg-card px-4 py-3 text-left shadow-card outline-none transition-colors duration-150 focus-visible:ring-[3px] focus-visible:ring-ring",
        edge,
        selected ? "border-primary/40 bg-accent" : isNew ? "border-border bg-accent/60 hover:bg-accent" : "border-border hover:bg-muted/60",
        isNew && "animate-in fade-in slide-in-from-top-2 duration-500"
      )}
    >
      <span className="flex items-start justify-between gap-2">
        <span className="flex min-w-0 items-center gap-2">
          <span className="truncate text-[15px] font-semibold text-foreground">{p.full_name}</span>
          {isNew && (
            <span className="shrink-0 rounded-full bg-primary px-2 py-0.5 text-xs font-semibold text-primary-foreground">
              {dt("newTag", lang)}
            </span>
          )}
        </span>
        <ZoneBadge zone={p.zone} size="sm" lang={lang} />
      </span>
      {p.last_disease && (
        <span className="mt-1 flex items-baseline justify-between gap-2 text-sm">
          <span className="min-w-0 truncate font-medium text-foreground">{getDiseaseLabel(p.last_disease, lang)}</span>
          {p.last_score > 0 && (
            <span className="shrink-0 tabular-nums text-muted-foreground">{Math.round(p.last_score * 100)}%</span>
          )}
        </span>
      )}
      {p.complaint && (
        <span className="mt-1 line-clamp-2 text-[13px] leading-snug text-muted-foreground">{p.complaint}</span>
      )}
      <span className="mt-2 flex items-center justify-between gap-2 text-[13px] text-muted-foreground">
        <span className="tabular-nums">{ago}</span>
        <span className="flex items-center gap-2">
          {p.low_confidence && (
            <span title={dt("lowConfidence", lang)} className="inline-flex items-center">
              <CircleHelp className="size-4" aria-hidden="true" />
              <span className="sr-only">{dt("lowConfidence", lang)}</span>
            </span>
          )}
          {p.feedback_pending && (
            <span title={dt("awaiting", lang)} className="inline-flex items-center gap-1">
              <span className="size-2 rounded-full bg-primary" aria-hidden="true" />
              <span className="sr-only">{dt("awaiting", lang)}</span>
            </span>
          )}
        </span>
      </span>
    </button>
  );
}
