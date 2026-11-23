"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import axios from "axios";
import { Loader2, UserRound } from "lucide-react";
import { AppHeader, EmptyState, LanguageSwitcher, StatTile } from "@/components/brand";
import { DoctorAuth } from "@/components/doctor/DoctorAuth";
import { AccountMenu } from "@/components/doctor/AccountMenu";
import { QueuePane } from "@/components/doctor/QueuePane";
import { PatientDetail } from "@/components/doctor/PatientDetail";
import { DemoPanel } from "@/components/doctor/DemoPanel";
import { AccuracyCard, summarizeStats } from "@/components/doctor/AccuracyCard";
import { pct } from "@/components/doctor/doctorUtils";
import {
  getFeedbackStats, listPatientsTriage,
  type DoctorProfile, type FeedbackStats, type TriagePatient,
} from "@/lib/api";
import { getLang, type Lang } from "@/lib/i18n";
import { dt } from "@/lib/strings/doctor";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// GET /doctor/me, called directly: the shared axios client turns any 401 into "clear the session and
// go to the landing page", but an expired doctor session should land on the sign-in screen instead.
async function fetchDoctorMe(token: string): Promise<DoctorProfile> {
  const { data } = await axios.get<DoctorProfile>(`${API_URL}/doctor/me`, { headers: { Authorization: `Bearer ${token}` } });
  return data;
}

const REFRESH_MS = 10_000;
const NEW_MS = 60_000;

type Session = "checking" | "out" | "in";

export default function DoctorPage() {
  // "ru" on the server and first render; the stored language is read after mount
  const [lang, setLangState] = useState<Lang>("ru");
  const [session, setSession] = useState<Session>("checking");
  const [doctor, setDoctor] = useState<DoctorProfile | null>(null);

  const [patients, setPatients] = useState<TriagePatient[]>([]);
  const [queueLoading, setQueueLoading] = useState(true);
  const [stats, setStats] = useState<FeedbackStats | null>(null);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [search, setSearch] = useState("");
  const [now, setNow] = useState(() => Date.now());
  const [newAt, setNewAt] = useState<Record<number, number>>({});

  // patient_id -> last_day_id seen by the previous refresh; null until the first load finishes
  const seen = useRef<Map<number, number | null> | null>(null);

  const setLang = (l: Lang) => {
    setLangState(l);
    document.documentElement.lang = l;
  };

  useEffect(() => {
    const current = getLang();
    setLangState(current);
    document.documentElement.lang = current;

    const token = localStorage.getItem("access_token");
    const id = localStorage.getItem("doctor_id");
    if (!token || !id) { setSession("out"); return; }
    fetchDoctorMe(token)
      .then((me) => { setDoctor(me); setSession("in"); })
      .catch((err) => {
        // an expired or foreign token is dropped; a network error keeps it for the next try
        if (axios.isAxiosError(err) && err.response?.status === 401) {
          ["access_token", "doctor_id", "doctor_name", "doctor_specialty", "doctor_is_demo"].forEach((k) => localStorage.removeItem(k));
        }
        setSession("out");
      });
  }, []);

  const refreshQueue = useCallback(async () => {
    try {
      const { patients: rows } = await listPatientsTriage();
      const prev = seen.current;
      const stamp = Date.now();
      if (prev) {
        const fresh: Record<number, number> = {};
        for (const p of rows) {
          const before = prev.get(p.patient_id);
          if (before === undefined || (p.last_day_id != null && (before == null || p.last_day_id > before))) {
            fresh[p.patient_id] = stamp;
          }
        }
        if (Object.keys(fresh).length > 0) setNewAt((cur) => ({ ...cur, ...fresh }));
      }
      seen.current = new Map(rows.map((p) => [p.patient_id, p.last_day_id]));
      setPatients(rows);
      setNow(stamp);
    } catch { /* keep the previous queue; the next tick retries */ }
    finally { setQueueLoading(false); }
  }, []);

  const refreshStats = useCallback(async () => {
    try { setStats(await getFeedbackStats()); } catch { setStats(null); }
  }, []);

  // First load plus a refresh every 10 s while the tab is visible
  useEffect(() => {
    if (session !== "in") return;
    refreshQueue();
    refreshStats();
    const tick = () => {
      if (document.visibilityState !== "visible") return;
      refreshQueue();
      refreshStats();
    };
    const timer = setInterval(tick, REFRESH_MS);
    const onVisible = () => { if (document.visibilityState === "visible") tick(); };
    document.addEventListener("visibilitychange", onVisible);
    return () => { clearInterval(timer); document.removeEventListener("visibilitychange", onVisible); };
  }, [session, refreshQueue, refreshStats]);

  const newIds = useMemo(
    () => new Set(Object.entries(newAt).filter(([, at]) => now - at < NEW_MS).map(([id]) => Number(id))),
    [newAt, now]
  );

  const handleAuthenticated = (me: DoctorProfile) => {
    seen.current = null;
    setPatients([]); setNewAt({}); setSelectedId(null); setSearch(""); setQueueLoading(true);
    setDoctor(me);
    setSession("in");
  };

  const handleLogout = () => {
    setSession("out");
    setDoctor(null); setPatients([]); setStats(null); setSelectedId(null); setNewAt({}); setSearch("");
    seen.current = null;
  };

  if (session === "checking") {
    return (
      <div className="flex min-h-dvh items-center justify-center bg-background">
        <Loader2 className="size-7 animate-spin text-muted-foreground" aria-label={dt("sessionChecking", lang)} />
      </div>
    );
  }

  if (session === "out" || !doctor) {
    return <DoctorAuth lang={lang} onAuthenticated={handleAuthenticated} />;
  }

  const redNow = patients.filter((p) => p.zone === "red").length;
  const pending = patients.filter((p) => p.feedback_pending).length;
  const agreement = stats && stats.n > 0 ? pct(summarizeStats(stats).top1) : "—";
  const selectedRow = patients.find((p) => p.patient_id === selectedId);

  const tileClass = "p-3 sm:p-4 [&>p:nth-child(2)]:text-2xl sm:[&>p:nth-child(2)]:text-3xl [&>p:nth-child(3)]:hidden sm:[&>p:nth-child(3)]:block";
  const tiles = (
    <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
      <StatTile className={tileClass} label={dt("statQueue", lang)} value={patients.length} />
      <StatTile
        className={tileClass}
        label={dt("statRed", lang)}
        value={<span className={redNow > 0 ? "text-urgent" : undefined}>{redNow}</span>}
      />
      <StatTile className={tileClass} label={dt("statPending", lang)} value={pending} />
      <StatTile
        className={tileClass}
        label={dt("statAgreement", lang)}
        value={agreement}
        hint={stats && stats.n > 0 ? `${dt("statAgreementHint", lang)} · ${stats.n}` : dt("statAgreementNone", lang)}
      />
    </div>
  );

  return (
    <div className="flex min-h-dvh flex-col bg-background lg:h-dvh">
      <AppHeader
        title={
          <span className="flex items-center gap-2">
            <span className="hidden sm:inline">{dt("pageTitle", lang)}</span>
            {doctor.is_demo && (
              <span className="hidden rounded-full border border-warn-border bg-warn-bg px-2 py-0.5 text-xs font-semibold text-warn sm:inline">
                {dt("demoPill", lang)}
              </span>
            )}
          </span>
        }
        actions={
          <>
            <LanguageSwitcher lang={lang} onChange={setLang} />
            <AccountMenu lang={lang} doctor={doctor} onLogout={handleLogout} />
          </>
        }
      />

      <main className="mx-auto flex w-full max-w-[1600px] flex-1 flex-col gap-4 px-4 py-4 sm:px-6 lg:min-h-0">
        {/* Below 1024px the stat row belongs to the queue view only */}
        <div className={selectedId !== null ? "hidden lg:block" : undefined}>
          <h1 className="mb-3 flex items-center gap-2 font-display text-2xl font-semibold text-foreground sm:hidden">
            {dt("pageTitle", lang)}
            {doctor.is_demo && (
              <span className="rounded-full border border-warn-border bg-warn-bg px-2 py-0.5 font-sans text-xs font-semibold text-warn">
                {dt("demoPill", lang)}
              </span>
            )}
          </h1>
          {tiles}
        </div>

        <div className="flex flex-1 flex-col gap-6 lg:min-h-0 lg:flex-row">
          <aside
            className={`flex-col gap-3 lg:flex lg:min-h-0 lg:w-[400px] lg:shrink-0 ${selectedId !== null ? "hidden" : "flex"}`}
            aria-label={dt("queueTitle", lang)}
          >
            {doctor.is_demo && <DemoPanel lang={lang} variant="card" />}
            <QueuePane
              lang={lang}
              patients={patients}
              loading={queueLoading}
              selectedId={selectedId}
              newIds={newIds}
              now={now}
              search={search}
              onSearch={setSearch}
              onSelect={setSelectedId}
            />
            {stats && (
              <div className="lg:hidden">
                <AccuracyCard stats={stats} lang={lang} />
              </div>
            )}
          </aside>

          <section
            className={`relative min-w-0 flex-1 lg:block lg:min-h-0 lg:overflow-y-auto lg:pr-1 ${selectedId !== null ? "block" : "hidden"}`}
            aria-live="polite"
          >
            {selectedId !== null ? (
              <div className="mx-auto w-full max-w-[860px] pb-8">
                <PatientDetail
                  key={selectedId}
                  patientId={selectedId}
                  queueRow={selectedRow}
                  doctor={doctor}
                  lang={lang}
                  onBack={() => setSelectedId(null)}
                  refreshKey={selectedRow?.last_day_id ?? null}
                  onChanged={() => { refreshQueue(); refreshStats(); }}
                />
              </div>
            ) : (
              <div className="mx-auto flex w-full max-w-[860px] flex-col gap-6 pb-8">
                {doctor.is_demo && <DemoPanel lang={lang} variant="hero" />}
                <EmptyState icon={UserRound} title={dt("selectPatientTitle", lang)} text={dt("selectPatientText", lang)} />
                {stats && <AccuracyCard stats={stats} lang={lang} />}
              </div>
            )}
          </section>
        </div>
      </main>
    </div>
  );
}
