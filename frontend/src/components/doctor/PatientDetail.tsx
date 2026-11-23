"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import axios from "axios";
import dynamic from "next/dynamic";
import {
  ArrowLeft, Brain, ChevronDown, ChevronUp, Download, Edit3, Loader2, MapPin, Quote, Save, X,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { ProbabilityBar, ZoneBadge, type ZoneValue } from "@/components/brand";
import { MarkdownText } from "@/components/MarkdownText";
import { LabResultCard } from "@/components/LabResultCard";
import {
  getConditions, getDaySymptoms, getHistory, getPatientInfo, updateByDoctor,
  type Condition, type DaySymptoms, type DoctorProfile, type HistoryEntry, type LabResult,
  type PatientInfo, type TriagePatient, type TriageReason,
} from "@/lib/api";
import { getDiseaseLabel } from "@/lib/diseaseWeights";
import { generatePatientPDF } from "@/lib/pdfExport";
import { getSymptomLabel, type SymptomCode } from "@/lib/symptoms";
import { t, type Lang } from "@/lib/i18n";
import { dt, type DoctorStringKey } from "@/lib/strings/doctor";
import { cn } from "@/lib/utils";
import { FeedbackForm } from "./FeedbackForm";
import { findingLabel, formatServerTime, makeCode } from "./doctorUtils";

// recharts is only needed for the symptom trend card, so it is split out of the first load.
const SymptomTrend = dynamic(
  () => import("@/components/SymptomTrend").then((m) => m.SymptomTrend),
  { ssr: false, loading: () => <div className="h-56" aria-hidden="true" /> }
);

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

async function fetchLabResults(patientId: number): Promise<LabResult[]> {
  const token = localStorage.getItem("access_token");
  const { data } = await axios.post<{ results: LabResult[] }>(
    `${API_URL}/get_lab_results`,
    { patient_id: patientId },
    token ? { headers: { Authorization: `Bearer ${token}` } } : undefined
  );
  return data.results ?? [];
}

const OUTCOME_KEY = { agrees: "fbAgrees", top3: "fbInTop3", miss: "fbMiss" } as const;
const ZONE_TEXT = { red: "text-urgent", yellow: "text-warn", green: "text-ok" } as const;
const ZONE_KEYS = { red: "zoneShortRed", yellow: "zoneShortYellow", green: "zoneShortGreen" } as const;

function sectionTitle(text: string, icon?: React.ReactNode) {
  return (
    <h3 className="flex items-center gap-2 text-[15px] font-semibold text-foreground">
      {icon}{text}
    </h3>
  );
}

function isEntry(r: unknown): r is TriageReason {
  return typeof r === "object" && r !== null && "code" in r;
}

export interface PatientDetailProps {
  patientId: number;
  /** The queue row, when this patient is in the queue (gives the complaint and the triage reasons). */
  queueRow?: TriagePatient;
  doctor: DoctorProfile;
  lang: Lang;
  /** Mobile only: shown as a back button above the header. */
  onBack?: () => void;
  /** Changes when the queue sees a new diary day for this patient, so the detail reloads. */
  refreshKey?: number | null;
  /** Called after feedback is saved so the page can refresh the queue and stats. */
  onChanged: () => void;
}

export function PatientDetail({ patientId, queueRow, doctor, lang, onBack, refreshKey, onChanged }: PatientDetailProps) {
  const [info, setInfo] = useState<PatientInfo | null>(null);
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [failed, setFailed] = useState(false);
  const [notFound, setNotFound] = useState(false);
  const [labs, setLabs] = useState<LabResult[]>([]);
  const [pdfError, setPdfError] = useState("");

  const [daySymptoms, setDaySymptoms] = useState<Record<number, DaySymptoms>>({});
  const [expanded, setExpanded] = useState<Set<number>>(new Set());

  const [editingDayId, setEditingDayId] = useState<number | null>(null);
  const [editRecept, setEditRecept] = useState("");
  const [editDiagnosis, setEditDiagnosis] = useState("");
  const [editError, setEditError] = useState("");
  const [saving, setSaving] = useState(false);

  const [conditions, setConditions] = useState<Condition[] | null>(null);
  const [fbDayId, setFbDayId] = useState<number | null>(null);
  const [outcome, setOutcome] = useState<Record<number, "agrees" | "top3" | "miss">>({});

  const loadDaySymptoms = useCallback(async (dayId: number) => {
    try {
      const resp = await getDaySymptoms(dayId);
      setDaySymptoms((prev) => ({ ...prev, [dayId]: resp }));
    } catch { /* the section shows "no symptom data" */ }
  }, []);

  const load = useCallback(async (initial: boolean) => {
    if (initial) { setLoading(true); setFailed(false); setNotFound(false); }
    try {
      const [infoResp, histResp] = await Promise.all([getPatientInfo(patientId), getHistory(patientId)]);
      if (infoResp.found && infoResp.patient) {
        setInfo(infoResp.patient);
        const sorted = [...histResp.history].sort((a, b) => b.day_id - a.day_id);
        setHistory(sorted);
        if (sorted[0]) {
          loadDaySymptoms(sorted[0].day_id);
          if (initial) setExpanded(new Set([sorted[0].day_id]));
        }
      } else {
        setInfo(null); setHistory([]); setNotFound(true);
      }
    } catch {
      if (initial) { setInfo(null); setHistory([]); setFailed(true); }
    } finally {
      setLoading(false);
    }
    fetchLabResults(patientId).then(setLabs).catch(() => setLabs([]));
  }, [patientId, loadDaySymptoms]);

  // A different patient: reset everything that belongs to the previous one.
  useEffect(() => {
    setInfo(null); setHistory([]); setDaySymptoms({}); setExpanded(new Set()); setLabs([]);
    setEditingDayId(null); setFbDayId(null); setOutcome({}); setPdfError("");
    load(true);
  }, [patientId, load]);

  // The queue saw a newer diary day for the same patient: reload quietly.
  useEffect(() => {
    if (refreshKey != null && info && history[0] && history[0].day_id !== refreshKey) {
      load(false).then(() => setExpanded((prev) => new Set(prev).add(refreshKey)));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [refreshKey]);

  const latest = history[0];
  const latestSymptoms = latest ? daySymptoms[latest.day_id] : undefined;

  const labelFor = useCallback(
    (day: DaySymptoms | undefined, code: string) =>
      findingLabel(code, lang, day?.labels?.[code]?.[lang] ?? getSymptomLabel(code as SymptomCode, lang)),
    [lang]
  );

  const groups = useMemo(() => {
    const present: string[] = [], absent: string[] = [], unknown: string[] = [];
    for (const [code, value] of Object.entries(latestSymptoms?.symptoms ?? {})) {
      const label = labelFor(latestSymptoms, code);
      if (value === 1) present.push(label);
      else if (value === 0) absent.push(label);
      else unknown.push(label);
    }
    return { present, absent, unknown };
  }, [latestSymptoms, labelFor]);

  const zone: ZoneValue = queueRow?.zone ?? latest?.zone ?? null;

  // One line per reason code: the evidence of all entries with that code is merged
  const reasonLines = useMemo(() => {
    const out: string[] = [];
    const byCode = new Map<string, Set<string>>();
    for (const r of (queueRow?.reasons ?? latest?.reasons ?? []) as unknown[]) {
      if (typeof r === "string") { if (!out.includes(r)) out.push(r); continue; }
      if (!isEntry(r) || (zone && r.zone !== zone)) continue;
      const names = byCode.get(r.code) ?? new Set<string>();
      for (const c of r.evidences ?? (r.evidence ? [r.evidence] : [])) names.add(labelFor(latestSymptoms, c));
      byCode.set(r.code, names);
    }
    for (const [code, names] of byCode) {
      const base = dt(`reasonCode_${code}` as DoctorStringKey, lang);
      out.push(names.size > 0 ? `${base}: ${[...names].join("; ")}` : base);
    }
    return out;
  }, [queueRow, latest?.reasons, zone, latestSymptoms, labelFor, lang]);

  const toggle = async (dayId: number) => {
    const open = expanded.has(dayId);
    setExpanded((prev) => { const n = new Set(prev); if (open) n.delete(dayId); else n.add(dayId); return n; });
    if (!open && !daySymptoms[dayId]) loadDaySymptoms(dayId);
  };

  const startEdit = (entry: HistoryEntry) => {
    setEditingDayId(entry.day_id);
    setEditRecept(entry.recept || "");
    // "Nothing" is the backend's "no diagnosis set" marker: show an empty field instead
    setEditDiagnosis(entry.disease_setup && entry.disease_setup !== "Nothing" ? entry.disease_setup : "");
    setEditError("");
  };

  const saveEdit = async (entry: HistoryEntry) => {
    setSaving(true);
    setEditError("");
    try {
      let ok = false;
      try {
        const resp = await updateByDoctor(patientId, entry.day_id, doctor.doctor_id, editRecept, editDiagnosis.trim() || "Nothing");
        ok = resp.ok === true;
      } catch { ok = false; }
      if (!ok) { setEditError(t("saveFailed", lang)); return; }
      try {
        const resp = await getHistory(patientId);
        setHistory([...resp.history].sort((a, b) => b.day_id - a.day_id));
      } catch { /* saved; the list keeps its previous state until reload */ }
      setEditingDayId(null);
    } finally { setSaving(false); }
  };

  const openFeedback = async (entry: HistoryEntry) => {
    setFbDayId(entry.day_id);
    if (!conditions) {
      try {
        const resp = await getConditions();
        setConditions([...resp.conditions].sort((a, b) =>
          getDiseaseLabel(a.name, lang).localeCompare(getDiseaseLabel(b.name, lang), lang)));
      } catch { setFbDayId(null); }
    }
  };

  const feedbackSaved = async (entry: HistoryEntry, resp: { agrees_top1: boolean; in_top3: boolean }) => {
    setOutcome((prev) => ({ ...prev, [entry.day_id]: resp.agrees_top1 ? "agrees" : resp.in_top3 ? "top3" : "miss" }));
    setFbDayId(null);
    try {
      const hist = await getHistory(patientId);
      setHistory([...hist.history].sort((a, b) => b.day_id - a.day_id));
    } catch { /* the entry refreshes on the next load */ }
    onChanged();
  };

  const exportPdf = async () => {
    if (!info) return;
    setPdfError("");
    try {
      await generatePatientPDF(info, history, doctor.full_name, lang);
    } catch (err) {
      console.error("PDF export failed", err);
      setPdfError(dt("pdfError", lang));
    }
  };

  const backButton = onBack && (
    <Button variant="ghost" size="sm" onClick={onBack} className="-ml-2 mb-1 min-h-11 w-fit lg:hidden">
      <ArrowLeft />{dt("back", lang)}
    </Button>
  );

  if (loading) {
    return (
      <div>
        {backButton}
        <div className="flex justify-center py-20"><Loader2 className="size-7 animate-spin text-muted-foreground" aria-label={t("loadingDots", lang)} /></div>
      </div>
    );
  }

  if (failed || notFound || !info) {
    return (
      <div>
        {backButton}
        <Card className="items-center gap-3 p-8 text-center">
          <p className="text-sm text-muted-foreground">{notFound ? t("patientNotFound", lang) : dt("loadError", lang)}</p>
          {failed && <Button variant="outline" size="sm" onClick={() => load(true)}>{dt("retry", lang)}</Button>}
        </Card>
      </div>
    );
  }

  // The full text of the latest day beats the queue's 160-character preview.
  const complaint = latest?.complaint ?? queueRow?.complaint ?? null;
  const top3 = (latest?.top3?.length ? latest.top3 : queueRow?.top3) ?? [];
  const lowConfidence = queueRow?.low_confidence ?? latest?.low_confidence ?? false;
  const top1 = latest?.disease_predict ?? queueRow?.last_disease ?? "";
  const top1Score = latest?.score ?? queueRow?.last_score ?? 0;

  return (
    <div className="space-y-4">
      {backButton}

      <header className="space-y-3">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div className="min-w-0">
            <h2 className="font-display text-2xl font-semibold leading-tight text-foreground sm:text-3xl">{info.full_name}</h2>
            <div className="mt-1.5 flex flex-wrap items-center gap-x-3 gap-y-1 text-sm text-muted-foreground">
              <span className="rounded-md bg-muted px-2 py-0.5 font-mono text-[13px] text-foreground">{makeCode(info.patient_id)}</span>
              {info.city && <span className="inline-flex items-center gap-1"><MapPin className="size-3.5" aria-hidden="true" />{info.city}</span>}
              <span className="tabular-nums">{dt("entryCount", lang)}: {history.length}</span>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <ZoneBadge zone={zone} size="md" lang={lang} />
            {history.length > 0 && (
              <Button variant="outline" size="sm" onClick={exportPdf}>
                <Download />{t("downloadPdf", lang)}
              </Button>
            )}
          </div>
        </div>
        {pdfError && <p role="alert" className="text-sm text-urgent">{pdfError}</p>}
      </header>

      {latest && (
        <Card className="gap-3 p-5">
          {sectionTitle(dt("complaint", lang), <Quote className="size-4 text-primary" aria-hidden="true" />)}
          {complaint ? (
            <blockquote className="border-l-4 border-primary/30 pl-4 text-[17px] leading-relaxed text-foreground">
              {complaint}
            </blockquote>
          ) : (
            <p className="text-sm italic text-muted-foreground">{dt("noComplaint", lang)}</p>
          )}
        </Card>
      )}

      {latest && (
        <Card className="gap-4 p-5">
          {sectionTitle(dt("symptomsFound", lang))}
          {!latestSymptoms ? (
            <Loader2 className="size-4 animate-spin text-muted-foreground" aria-label={t("loadingDots", lang)} />
          ) : (
            <div className="space-y-3">
              <SymptomRow title={dt("symptomsPresent", lang)} items={groups.present} tone="present" empty={dt("symptomsNone", lang)} />
              <SymptomRow title={dt("symptomsAbsent", lang)} items={groups.absent} tone="absent" empty={dt("symptomsNone", lang)} />
              {groups.unknown.length > 0 && (
                <SymptomRow title={dt("symptomsUnknown", lang)} items={groups.unknown} tone="absent" empty="" />
              )}
            </div>
          )}
        </Card>
      )}

      {latest && top1 && (
        <Card className="gap-4 p-5">
          {sectionTitle(dt("modelResult", lang))}
          <div className="space-y-3">
            {top3.length > 0 ? (
              top3.map((c, i) => (
                <ProbabilityBar key={c.name} label={getDiseaseLabel(c.name, lang)} percent={c.p * 100} rank={i} />
              ))
            ) : (
              <>
                <ProbabilityBar label={getDiseaseLabel(top1, lang)} percent={top1Score * 100} rank={0} />
                <p className="text-[13px] text-muted-foreground">{dt("modelTop3Missing", lang)}</p>
              </>
            )}
          </div>
          {lowConfidence && (
            <p className="rounded-lg bg-muted px-3 py-2 text-sm text-foreground">{dt("lowConfidenceNote", lang)}</p>
          )}
          {reasonLines.length > 0 && (
            <div className="space-y-1.5">
              <p className={cn("text-sm font-semibold", zone ? ZONE_TEXT[zone] : "text-foreground")}>
                {dt("triageReasons", lang)}{zone ? ` (${t(ZONE_KEYS[zone], lang)})` : ""}
              </p>
              <ul className="list-disc space-y-1 pl-5 text-sm text-foreground">
                {reasonLines.map((r) => <li key={r}>{r}</li>)}
              </ul>
            </div>
          )}
        </Card>
      )}

      <section className="space-y-3">
        <h3 className="px-1 text-lg font-semibold text-foreground">{dt("historyTitle", lang)}</h3>
        {history.length === 0 ? (
          <Card className="p-6 text-center text-sm text-muted-foreground">{t("noEntries", lang)}</Card>
        ) : (
          history.map((entry, idx) => {
            const isOpen = expanded.has(entry.day_id);
            const isEditing = editingDayId === entry.day_id;
            const fb = entry.feedback;
            const fbLabel = fb ? (fb.diagnosis === "other" ? fb.other_text || t("otherDiagnosis", lang) : getDiseaseLabel(fb.diagnosis, lang)) : "";
            const confirmedDx = entry.disease_setup && entry.disease_setup !== "Nothing" ? entry.disease_setup : null;
            const day = daySymptoms[entry.day_id];
            const out = outcome[entry.day_id];
            return (
              <Card key={entry.day_id} className="gap-0 overflow-hidden p-0">
                <button
                  type="button"
                  onClick={() => toggle(entry.day_id)}
                  aria-expanded={isOpen}
                  className="flex min-h-14 w-full items-center justify-between gap-3 px-4 py-3 text-left outline-none transition-colors hover:bg-muted/60 focus-visible:ring-[3px] focus-visible:ring-inset focus-visible:ring-ring"
                >
                  <span className="min-w-0">
                    <span className="block truncate text-[15px] font-semibold text-foreground">{getDiseaseLabel(entry.disease_predict, lang)}</span>
                    <span className="block text-[13px] tabular-nums text-muted-foreground">
                      {formatServerTime(entry.created_at, lang)}
                      {entry.score != null && ` · ${Math.round(entry.score * 100)}%`}
                    </span>
                  </span>
                  <span className="flex shrink-0 items-center gap-2">
                    {entry.zone && <ZoneBadge zone={entry.zone} size="sm" lang={lang} />}
                    {(confirmedDx || fb) && (
                      <span className="rounded-full bg-ok-bg px-2 py-0.5 text-xs font-medium text-ok">{t("confirmed", lang)}</span>
                    )}
                    {isOpen ? <ChevronUp className="size-4 text-muted-foreground" /> : <ChevronDown className="size-4 text-muted-foreground" />}
                  </span>
                </button>

                {isOpen && (
                  <div className="space-y-5 border-t border-border bg-muted/30 px-4 py-4">
                    {idx > 0 && (
                      <div className="space-y-2">
                        <p className="text-[13px] font-semibold text-muted-foreground">{t("patientSymptoms", lang)}</p>
                        {!day ? (
                          <Loader2 className="size-4 animate-spin text-muted-foreground" aria-label={t("loadingDots", lang)} />
                        ) : Object.keys(day.symptoms).length === 0 ? (
                          <p className="text-sm italic text-muted-foreground">{t("noSymptomData", lang)}</p>
                        ) : (
                          <div className="flex flex-wrap gap-1.5">
                            {Object.entries(day.symptoms)
                              .sort((a, b) => b[1] - a[1])
                              .map(([code, v]) => (
                                <Chip key={code} tone={v === 1 ? "present" : "absent"}>
                                  {v === 0 && <span>{t("denied", lang)}: </span>}{labelFor(day, code)}
                                </Chip>
                              ))}
                          </div>
                        )}
                      </div>
                    )}

                    <div className="space-y-2">
                      <p className="text-[13px] font-semibold text-muted-foreground">{t("preliminaryDiagnosis", lang)}</p>
                      <p className="text-[15px] text-foreground">{getDiseaseLabel(entry.disease_predict, lang)}</p>
                      {fb && (
                        <div className="flex flex-wrap items-center gap-2">
                          <span className="rounded-full bg-ok-bg px-2.5 py-0.5 text-xs font-medium text-ok">
                            {t("confirmedAs", lang).replace("{label}", fbLabel)}
                          </span>
                          <ZoneBadge zone={fb.zone} size="sm" lang={lang} />
                        </div>
                      )}
                      {out && <p className="text-sm text-muted-foreground">{t(OUTCOME_KEY[out], lang)}</p>}
                      {fbDayId === entry.day_id ? (
                        <FeedbackForm
                          entry={entry}
                          conditions={conditions}
                          lang={lang}
                          onSaved={(resp) => feedbackSaved(entry, resp)}
                          onCancel={() => setFbDayId(null)}
                        />
                      ) : (
                        <Button size="sm" variant={fb ? "outline" : "default"} onClick={() => openFeedback(entry)}>
                          {fb ? t("changeDiagnosis", lang) : t("confirmDiagnosis", lang)}
                        </Button>
                      )}
                    </div>

                    {entry.doctor_explanation && (
                      <div className="space-y-2 rounded-xl border border-border bg-accent p-4">
                        <p className="flex items-center gap-1.5 text-[13px] font-semibold text-accent-foreground">
                          <Brain className="size-4" aria-hidden="true" />{t("aiClinicalAnalysis", lang)}
                        </p>
                        <MarkdownText text={entry.doctor_explanation} className="space-y-2 text-sm leading-relaxed text-foreground" />
                        <p className="text-[13px] italic text-muted-foreground">{t("aiDisclaimer", lang)}</p>
                      </div>
                    )}

                    <div className="space-y-1.5">
                      <p className="text-[13px] font-semibold text-muted-foreground">{t("doctorDiagnosis", lang)}</p>
                      {isEditing ? (
                        <Input value={editDiagnosis} onChange={(e) => setEditDiagnosis(e.target.value)} aria-label={t("doctorDiagnosis", lang)} placeholder={t("setDiagnosis", lang)} />
                      ) : (
                        <p className="text-[15px] text-foreground">{confirmedDx ?? <span className="italic text-muted-foreground">{t("notSet", lang)}</span>}</p>
                      )}
                    </div>

                    <div className="space-y-1.5">
                      <p className="text-[13px] font-semibold text-muted-foreground">{t("recept", lang)}</p>
                      {isEditing ? (
                        <Textarea value={editRecept} onChange={(e) => setEditRecept(e.target.value)} aria-label={t("recept", lang)} placeholder={t("writeRecept", lang)} rows={3} />
                      ) : (
                        <p className="whitespace-pre-wrap text-[15px] text-foreground">
                          {entry.recept || <span className="italic text-muted-foreground">{t("noRecommendations", lang)}</span>}
                        </p>
                      )}
                    </div>

                    {entry.doctor_name && (
                      <div className="space-y-1">
                        <p className="text-[13px] font-semibold text-muted-foreground">{t("responsibleDoctor", lang)}</p>
                        <p className="text-[15px] text-primary">{entry.doctor_name}</p>
                      </div>
                    )}

                    {isEditing && editError && <p role="alert" className="text-sm text-urgent">{editError}</p>}

                    <div className="flex gap-2">
                      {isEditing ? (
                        <>
                          <Button size="sm" onClick={() => saveEdit(entry)} disabled={saving}>
                            {saving ? <Loader2 className="animate-spin" /> : <Save />}{t("save", lang)}
                          </Button>
                          <Button size="sm" variant="outline" onClick={() => setEditingDayId(null)} disabled={saving}>
                            <X />{t("cancel", lang)}
                          </Button>
                        </>
                      ) : (
                        <Button size="sm" variant="outline" onClick={() => startEdit(entry)}>
                          <Edit3 />{t("edit", lang)}
                        </Button>
                      )}
                    </div>
                  </div>
                )}
              </Card>
            );
          })
        )}
      </section>

      {/* The trend endpoint only knows the v1 symptom codes, so the chart is shown only for patients with v1 days. */}
      {history.length >= 2 && history.some((h) => !h.top3?.length) && (
        <Card className="p-5"><SymptomTrend patientId={patientId} lang={lang} /></Card>
      )}

      <section className="space-y-3">
        <h3 className="px-1 text-lg font-semibold text-foreground">{dt("labsTitle", lang)}</h3>
        {labs.length === 0 ? (
          <Card className="p-5 text-sm text-muted-foreground">{dt("labsNone", lang)}</Card>
        ) : (
          labs.map((r) => <LabResultCard key={r.result_id} result={r} lang={lang} />)
        )}
      </section>
    </div>
  );
}

function Chip({ tone, children }: { tone: "present" | "absent"; children: React.ReactNode }) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full border px-2.5 py-1 text-[13px] font-medium",
        tone === "present" ? "border-transparent bg-accent text-accent-foreground" : "border-border bg-muted text-muted-foreground"
      )}
    >
      {children}
    </span>
  );
}

function SymptomRow({ title, items, tone, empty }: { title: string; items: string[]; tone: "present" | "absent"; empty: string }) {
  return (
    <div className="space-y-1.5">
      <p className="text-[13px] font-semibold text-muted-foreground">{title} <span className="tabular-nums">({items.length})</span></p>
      {items.length === 0 ? (
        <p className="text-sm text-muted-foreground">{empty}</p>
      ) : (
        <div className="flex flex-wrap gap-1.5">
          {items.map((l) => <Chip key={l} tone={tone}>{l}</Chip>)}
        </div>
      )}
    </div>
  );
}
