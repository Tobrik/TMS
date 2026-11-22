import type { AccuracyCalibrationBin } from "@/lib/api";
import { t, type Lang } from "@/lib/i18n";
import { formatBin, formatPct } from "./format";

const MIN_N = 20;

export function CalibrationTable({ bins, lang }: { bins: AccuracyCalibrationBin[]; lang: Lang }) {
  const shown = bins.filter((b) => b.n >= MIN_N);
  return (
    <section className="space-y-3">
      <h2 className="text-lg font-semibold text-gray-900">{t("accCalibTitle", lang)}</h2>
      <p className="text-sm text-gray-700">{t("accCalibIntro", lang)}</p>
      {shown.length === 0 ? (
        <p className="text-sm text-gray-600">{t("accNotAvailable", lang)}</p>
      ) : (
        <div className="overflow-x-auto print:overflow-visible">
          <table className="w-full min-w-[20rem] text-left text-sm">
            <caption className="sr-only">{t("accCalibTitle", lang)}</caption>
            <thead>
              <tr className="border-b border-gray-200 text-gray-700">
                <th scope="col" className="py-2 pr-3 font-medium">{t("accCalibBin", lang)}</th>
                <th scope="col" className="py-2 pr-3 font-medium">{t("accCalibN", lang)}</th>
                <th scope="col" className="py-2 font-medium">{t("accCalibAcc", lang)}</th>
              </tr>
            </thead>
            <tbody>
              {shown.map((b) => (
                <tr key={b.bin} className="border-b border-gray-100">
                  <th scope="row" className="py-2 pr-3 font-normal tabular-nums text-gray-800">{formatBin(b.bin, lang)}</th>
                  <td className="py-2 pr-3 tabular-nums">{b.n}</td>
                  <td className="py-2 font-semibold tabular-nums text-gray-900">{formatPct(b.accuracy, lang)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
