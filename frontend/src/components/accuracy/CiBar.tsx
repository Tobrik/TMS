import type { AccuracyMetric } from "@/lib/api";
import { t, type Lang } from "@/lib/i18n";
import { formatCi, formatPct, tf } from "./format";

/**
 * Bar filled to the value, with the 95% interval drawn as a thin line.
 * The bar is decorative: the numbers printed beside it carry the same information.
 */
export function CiBar({ metric }: { metric: AccuracyMetric }) {
  const { value, ci } = metric;
  return (
    <div aria-hidden="true" className="relative h-3 w-full rounded-full bg-gray-100 print:border print:border-gray-300">
      {value !== null && (
        <div
          className="absolute inset-y-0 left-0 rounded-full bg-indigo-300 print:bg-gray-300"
          style={{ width: `${value * 100}%` }}
        />
      )}
      {ci && (
        <div
          className="absolute top-1/2 h-0.5 -translate-y-1/2 bg-gray-900"
          style={{ left: `${ci[0] * 100}%`, width: `${(ci[1] - ci[0]) * 100}%` }}
        />
      )}
    </div>
  );
}

/** "94% · 95% ДИ 88–97% · n = 110", as text so it survives print and screen readers. */
export function MetricText({
  metric,
  lang,
  showN = true,
}: {
  metric: AccuracyMetric | undefined;
  lang: Lang;
  showN?: boolean;
}) {
  if (!metric) return <span className="text-gray-500">{t("accNotAvailable", lang)}</span>;
  const ci = formatCi(metric, lang);
  return (
    <span className="tabular-nums">
      <span className="font-semibold text-gray-900">{formatPct(metric.value, lang)}</span>
      {ci && <span className="text-gray-700"> · {tf("accCi", lang, { ci })}</span>}
      {showN && <span className="text-gray-600"> · {tf("accN", lang, { n: metric.n })}</span>}
    </span>
  );
}
