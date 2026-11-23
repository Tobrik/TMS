"use client";

import { useEffect, useState, useSyncExternalStore, type ReactNode } from "react";
import { useRouter } from "next/navigation";
import { AlertTriangle, ArrowLeft, Loader2, Printer } from "lucide-react";
import { Button } from "@/components/ui/button";
import { AppHeader } from "@/components/brand/AppHeader";
import { LanguageSwitcher } from "@/components/brand/LanguageSwitcher";
import { Logo } from "@/components/brand/Logo";
import { getAccuracyReport, type AccuracyReportResponse } from "@/lib/api";
import { getLang, t, type Lang } from "@/lib/i18n";
import { ps } from "@/lib/strings/pages";
import { HeadlineCards } from "@/components/accuracy/HeadlineCards";
import { RealPatientsBlock, SyntheticBlock, WrittenBlock } from "@/components/accuracy/EvidenceLevels";
import { TriageTable } from "@/components/accuracy/TriageTable";
import { LanguageTable } from "@/components/accuracy/LanguageTable";
import { ConditionBars } from "@/components/accuracy/ConditionBars";
import { CalibrationTable } from "@/components/accuracy/CalibrationTable";
import { formatDate, localizeReport, tf } from "@/components/accuracy/format";
import { SAMPLE_REPORT } from "@/components/accuracy/sampleReport";

// A4 page on white paper; the header and buttons are hidden with print:hidden
const PRINT_CSS = `
@page { size: A4; margin: 14mm; }
@media print {
  html, body { background: #ffffff !important; color: var(--foreground) !important; }
  * { -webkit-print-color-adjust: exact; print-color-adjust: exact; box-shadow: none !important; }
}`;

type Status = "checking" | "need_login" | "loading" | "ready" | "error";

// The stored language is read without an effect. "ru" is used on the server and during hydration.
function noSubscribe() {
  return () => {};
}
function readStoredLang(): Lang {
  return getLang();
}
function serverLang(): Lang {
  return "ru";
}

export default function AccuracyPage() {
  const router = useRouter();
  const storedLang = useSyncExternalStore(noSubscribe, readStoredLang, serverLang);
  const [langOverride, setLangOverride] = useState<Lang | null>(null);
  const lang = langOverride ?? storedLang;
  const [status, setStatus] = useState<Status>("checking");
  const [report, setReport] = useState<AccuracyReportResponse | null>(null);
  const [isMock, setIsMock] = useState(false);
  const [reloadKey, setReloadKey] = useState(0);

  useEffect(() => {
    document.documentElement.lang = lang;
  }, [lang]);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      // Sample data for layout review; no login and no API call
      if (new URLSearchParams(window.location.search).get("mock") === "1") {
        setIsMock(true);
        setReport(SAMPLE_REPORT);
        setStatus("ready");
        return;
      }

      const token = localStorage.getItem("access_token");
      const doctorId = localStorage.getItem("doctor_id");
      if (!token || !doctorId) {
        setStatus("need_login");
        return;
      }

      setStatus("loading");
      try {
        const data = await getAccuracyReport();
        if (cancelled) return;
        setReport(data);
        setStatus("ready");
      } catch {
        if (!cancelled) setStatus("error");
      }
    }

    void load();
    return () => {
      cancelled = true;
    };
  }, [reloadKey]);

  return (
    <div className="min-h-screen bg-background text-foreground print:bg-white">
      <style>{PRINT_CSS}</style>

      <AppHeader
        className="print:hidden"
        logoSlot={
          <div className="flex items-center gap-1">
            <Button
              variant="ghost"
              size="icon"
              onClick={() => router.push("/doctor")}
              aria-label={ps("back", lang)}
              title={t("accBackToDoctor", lang)}
            >
              <ArrowLeft />
            </Button>
            <Logo />
          </div>
        }
        actions={
          <>
            {status === "ready" && (
              <Button variant="outline" size="sm" onClick={() => window.print()} aria-label={t("accPrint", lang)}>
                <Printer />
                <span className="hidden sm:inline">{t("accPrint", lang)}</span>
              </Button>
            )}
            <LanguageSwitcher lang={lang} onChange={setLangOverride} />
          </>
        }
      />

      <main className="mx-auto max-w-4xl space-y-10 px-4 py-6 sm:px-6 print:max-w-none print:space-y-6 print:px-0 print:py-0">
        <div className="space-y-3">
          <h1 className="font-display text-3xl font-semibold leading-tight text-foreground sm:text-[32px]">
            {t("accTitle", lang)}
          </h1>
          {report && !report.report_missing && (
            <p className="text-sm text-muted-foreground">
              {tf("accSubtitle", lang, {
                model: report.model_version,
                backend: report.extraction_backend,
                date: formatDate(report.generated_at, lang),
              })}
            </p>
          )}
          <p className="text-sm text-foreground/85">{t("accDisclaimer", lang)}</p>
          {isMock && (
            <WarnNote className="print:hidden">{t("accMockBanner", lang)}</WarnNote>
          )}
        </div>

        {(status === "checking" || status === "loading") && (
          <p className="flex items-center gap-2 text-sm text-muted-foreground">
            <Loader2 className="h-4 w-4 animate-spin" />
            {t("accLoading", lang)}
          </p>
        )}

        {status === "need_login" && (
          <div className="space-y-4 rounded-xl border border-border bg-card p-6 shadow-card">
            <p className="text-sm text-foreground/85">{t("accNeedDoctor", lang)}</p>
            <Button onClick={() => router.push("/doctor")}>{t("accGoToDoctor", lang)}</Button>
          </div>
        )}

        {status === "error" && (
          <div className="space-y-4 rounded-xl border border-urgent-border border-l-4 border-l-urgent bg-card p-6 shadow-card">
            <p className="text-sm text-urgent">{t("accLoadError", lang)}</p>
            <Button variant="outline" onClick={() => setReloadKey((k) => k + 1)}>
              {t("accRetry", lang)}
            </Button>
          </div>
        )}

        {status === "ready" && report && (
          <ReportBody report={"caveats" in report ? localizeReport(report, lang) : report} lang={lang} />
        )}
      </main>
    </div>
  );
}

function ReportBody({ report, lang }: { report: AccuracyReportResponse; lang: Lang }) {
  // Benchmark not built yet: only the real-patient numbers are available
  if (report.report_missing) {
    return (
      <>
        <WarnNote>{t("accMissingNote", lang)}</WarnNote>
        <section className="space-y-4">
          <h2 className="text-lg font-semibold text-foreground sm:text-xl">{t("accEvidence", lang)}</h2>
          <RealPatientsBlock real={report.real_patients} lang={lang} />
        </section>
      </>
    );
  }

  return (
    <>
      {report.written_cases?.held_out === false && (
        <WarnNote role="note">{t("accTuningSetNote", lang)}</WarnNote>
      )}
      <HeadlineCards written={report.written_cases} lang={lang} />

      <section className="space-y-4">
        <h2 className="text-lg font-semibold text-foreground sm:text-xl">{t("accEvidence", lang)}</h2>
        <p className="text-sm text-muted-foreground">{t("accEvidenceIntro", lang)}</p>
        <RealPatientsBlock real={report.real_patients} lang={lang} />
        <WrittenBlock written={report.written_cases} lang={lang} />
        <SyntheticBlock synthetic={report.synthetic} lang={lang} />
      </section>

      <TriageTable written={report.written_cases} synthetic={report.synthetic} lang={lang} />

      <LanguageTable written={report.written_cases} lang={lang} />

      <ConditionBars conditions={report.synthetic.per_condition} lang={lang} />

      <CalibrationTable bins={report.synthetic.calibration} lang={lang} />

      <section className="space-y-2">
        <h2 className="text-lg font-semibold text-foreground sm:text-xl">{t("accCaveats", lang)}</h2>
        <ul className="list-disc space-y-2 pl-5 text-sm leading-relaxed text-foreground/85 marker:text-muted-foreground">
          {report.caveats.map((caveat, i) => (
            <li key={i}>{caveat}</li>
          ))}
        </ul>
      </section>

      <section className="space-y-2">
        <h2 className="text-lg font-semibold text-foreground sm:text-xl">{t("accMethod", lang)}</h2>
        <p className="text-sm leading-relaxed text-foreground/85">{t("accMethodText", lang)}</p>
      </section>
    </>
  );
}

// Calm note: soft warm background, thin edge, no alarm colour
function WarnNote({
  children,
  className,
  role,
}: {
  children: ReactNode;
  className?: string;
  role?: "note";
}) {
  return (
    <div
      role={role}
      className={`flex gap-3 rounded-xl border border-warn-border border-l-4 border-l-warn bg-warn-bg p-4 text-sm leading-relaxed text-foreground ${className ?? ""}`}
    >
      <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-warn" aria-hidden="true" />
      <p>{children}</p>
    </div>
  );
}
