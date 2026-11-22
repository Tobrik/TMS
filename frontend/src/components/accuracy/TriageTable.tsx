import type { AccuracySynthetic, AccuracyWritten } from "@/lib/api";
import { t, type Lang } from "@/lib/i18n";
import { MetricText } from "./CiBar";

// Triage safety: the written column uses patient_confirms (after the patient's follow-up answers)
export function TriageTable({
  written,
  synthetic,
  lang,
}: {
  written: AccuracyWritten | null;
  synthetic: AccuracySynthetic;
  lang: Lang;
}) {
  const confirms = written?.patient_confirms;
  const rows = [
    { label: "accTriageRedSens" as const, w: confirms?.red_sensitivity, s: synthetic.metrics.red_sensitivity },
    { label: "accTriageRedToGreen" as const, w: confirms?.red_to_green, s: synthetic.metrics.red_to_green },
    { label: "accTriageGreenToRed" as const, w: confirms?.green_to_red, s: synthetic.metrics.green_to_red },
  ];

  return (
    <section className="space-y-3">
      <h2 className="text-lg font-semibold text-gray-900">{t("accTriageTitle", lang)}</h2>
      <p className="text-sm text-gray-700">{t("accTriageIntro", lang)}</p>
      <div className="overflow-x-auto print:overflow-visible">
        <table className="w-full min-w-[32rem] text-left text-sm">
          <caption className="sr-only">{t("accTriageTitle", lang)}</caption>
          <thead>
            <tr className="border-b border-gray-200 text-gray-700">
              <th scope="col" className="py-2 pr-3 font-medium">{t("accColMetric", lang)}</th>
              <th scope="col" className="py-2 pr-3 font-medium">{t("accLevelWritten", lang)}</th>
              <th scope="col" className="py-2 font-medium">{t("accLevelSynthetic", lang)}</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.label} className="border-b border-gray-100 align-top">
                <th scope="row" className="py-2 pr-3 font-normal text-gray-800">{t(row.label, lang)}</th>
                <td className="py-2 pr-3"><MetricText metric={row.w} lang={lang} /></td>
                <td className="py-2"><MetricText metric={row.s} lang={lang} /></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
