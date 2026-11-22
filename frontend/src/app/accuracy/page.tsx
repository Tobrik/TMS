"use client";

import { useEffect, useState, useSyncExternalStore } from "react";
import { useRouter } from "next/navigation";
import { ArrowLeft, Loader2, Printer } from "lucide-react";
import { Button } from "@/components/ui/button";
import { getAccuracyReport, type AccuracyReportResponse } from "@/lib/api";
import { getLang, setLang, t, type Lang } from "@/lib/i18n";
import { HeadlineCards } from "@/components/accuracy/HeadlineCards";
import { RealPatientsBlock, SyntheticBlock, WrittenBlock } from "@/components/accuracy/EvidenceLevels";
import { TriageTable } from "@/components/accuracy/TriageTable";
import { LanguageTable } from "@/components/accuracy/LanguageTable";
import { ConditionBars } from "@/components/accuracy/ConditionBars";
import { CalibrationTable } from "@/components/accuracy/CalibrationTable";
import { formatDate, localizeReport, tf } from "@/components/accuracy/format";
import { SAMPLE_REPORT } from "@/components/accuracy/sampleReport";

const LANG_CYCLE: Lang[] = ["ru", "en", "kk"];
const LANG_LABELS: Record<Lang, string> = { ru: "RU", en: "EN", kk: "KZ" };

// A4 page, no dark backgrounds; the buttons and navigation are hidden with print:hidden
const PRINT_CSS = `
@page { size: A4; margin: 14mm; }
@media print {
  html, body { background: #ffffff !important; color: #111827 !important; }
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

  const toggleLang = () => {
    const next = LANG_CYCLE[(LANG_CYCLE.indexOf(lang) + 1) % LANG_CYCLE.length];
    setLang(next);
    setLangOverride(next);
  };

  return (
    <div className="min-h-screen bg-gray-50 text-gray-900 print:bg-white">
      <style>{PRINT_CSS}</style>

      <header className="sticky top-0 z-10 border-b border-gray-200 bg-white print:hidden">
        <div className="mx-auto flex max-w-4xl items-center justify-between gap-2 px-4 py-3">
          <Button variant="ghost" size="sm" onClick={() => router.push("/doctor")}>
            <ArrowLeft />
            {t("accBackToDoctor", lang)}
          </Button>
          <div className="flex items-center gap-1">
            <Button
              variant="ghost"
              size="sm"
              onClick={toggleLang}
              className="px-2 text-xs font-semibold text-gray-600"
              title={t("changeLang", lang)}
            >
              {LANG_LABELS[lang]}
            </Button>
            {status === "ready" && (
              <Button variant="outline" size="sm" onClick={() => window.print()}>
                <Printer />
                {t("accPrint", lang)}
              </Button>
            )}
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-4xl space-y-10 px-4 py-6 sm:px-6 print:max-w-none print:space-y-6 print:px-0 print:py-0">
        <div className="space-y-2">
          <h1 className="text-2xl font-semibold text-gray-900">{t("accTitle", lang)}</h1>
          {report && !report.report_missing && (
            <p className="text-sm text-gray-700">
              {tf("accSubtitle", lang, {
                model: report.model_version,
                backend: report.extraction_backend,
                date: formatDate(report.generated_at, lang),
              })}
            </p>
          )}
          <p className="text-sm text-gray-800">{t("accDisclaimer", lang)}</p>
          {isMock && (
            <p className="rounded-md border border-amber-300 bg-amber-50 p-3 text-sm text-amber-900 print:hidden">
              {t("accMockBanner", lang)}
            </p>
          )}
        </div>

        {(status === "checking" || status === "loading") && (
          <p className="flex items-center gap-2 text-sm text-gray-700">
            <Loader2 className="h-4 w-4 animate-spin" />
            {t("accLoading", lang)}
          </p>
        )}

        {status === "need_login" && (
          <div className="space-y-4 rounded-xl border border-gray-200 bg-white p-6">
            <p className="text-sm text-gray-800">{t("accNeedDoctor", lang)}</p>
            <Button onClick={() => router.push("/doctor")}>{t("accGoToDoctor", lang)}</Button>
          </div>
        )}

        {status === "error" && (
          <div className="space-y-4 rounded-xl border border-red-200 bg-white p-6">
            <p className="text-sm text-red-800">{t("accLoadError", lang)}</p>
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
        <p className="rounded-md border border-amber-300 bg-amber-50 p-3 text-sm text-amber-900">
          {t("accMissingNote", lang)}
        </p>
        <section className="space-y-4">
          <h2 className="text-lg font-semibold text-gray-900">{t("accEvidence", lang)}</h2>
          <RealPatientsBlock real={report.real_patients} lang={lang} />
        </section>
      </>
    );
  }

  return (
    <>
      {report.written_cases?.held_out === false && (
        <p role="note" className="rounded-lg border border-amber-300 bg-amber-50 p-4 text-sm text-amber-900">
          {t("accTuningSetNote", lang)}
        </p>
      )}
      <HeadlineCards written={report.written_cases} lang={lang} />

      <section className="space-y-4">
        <h2 className="text-lg font-semibold text-gray-900">{t("accEvidence", lang)}</h2>
        <p className="text-sm text-gray-700">{t("accEvidenceIntro", lang)}</p>
        <RealPatientsBlock real={report.real_patients} lang={lang} />
        <WrittenBlock written={report.written_cases} lang={lang} />
        <SyntheticBlock synthetic={report.synthetic} lang={lang} />
      </section>

      <TriageTable written={report.written_cases} synthetic={report.synthetic} lang={lang} />

      <LanguageTable written={report.written_cases} lang={lang} />

      <ConditionBars conditions={report.synthetic.per_condition} lang={lang} />

      <CalibrationTable bins={report.synthetic.calibration} lang={lang} />

      <section className="space-y-2">
        <h2 className="text-lg font-semibold text-gray-900">{t("accCaveats", lang)}</h2>
        <ul className="list-disc space-y-2 pl-5 text-sm leading-relaxed text-gray-800">
          {report.caveats.map((caveat, i) => (
            <li key={i}>{caveat}</li>
          ))}
        </ul>
      </section>

      <section className="space-y-2">
        <h2 className="text-lg font-semibold text-gray-900">{t("accMethod", lang)}</h2>
        <p className="text-sm leading-relaxed text-gray-800">{t("accMethodText", lang)}</p>
      </section>
    </>
  );
}
