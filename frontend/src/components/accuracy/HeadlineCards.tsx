import type { AccuracyMetric, AccuracyWritten } from "@/lib/api";
import { t, type Lang, type TranslationKey } from "@/lib/i18n";
import { formatCi, formatPct, tf } from "./format";

interface HeadlineItem {
  title: TranslationKey;
  help: TranslationKey;
  metric: AccuracyMetric | undefined;
}

// Written complaints after patient confirmation: the closest thing to real use we have
export function HeadlineCards({ written, lang }: { written: AccuracyWritten | null; lang: Lang }) {
  const confirms = written?.patient_confirms;
  const items: HeadlineItem[] = [
    { title: "accCardRedTitle", help: "accCardRedHelp", metric: confirms?.red_sensitivity },
    { title: "accCardTop1Title", help: "accCardTop1Help", metric: confirms?.top1 },
    { title: "accCardTop3Title", help: "accCardTop3Help", metric: confirms?.top3 },
    { title: "accCardGreenRedTitle", help: "accCardGreenRedHelp", metric: confirms?.green_to_red },
  ];

  return (
    <section className="space-y-4">
      <h2 className="text-lg font-semibold text-gray-900">{t("accHeadline", lang)}</h2>
      {!written && <p className="text-sm text-gray-600">{t("accNoWritten", lang)}</p>}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 print:grid-cols-2">
        {items.map((item) => {
          const ci = formatCi(item.metric, lang);
          return (
            <article
              key={item.title}
              className="flex flex-col rounded-xl border border-gray-200 bg-white p-5 shadow-sm print:break-inside-avoid print:shadow-none"
            >
              <h3 className="text-sm font-semibold text-gray-800">{t(item.title, lang)}</h3>
              <p className="mt-2 text-4xl font-semibold tabular-nums text-gray-900">
                {formatPct(item.metric?.value ?? null, lang)}
              </p>
              {ci && <p className="mt-1 text-sm text-gray-700">{tf("accCi", lang, { ci })}</p>}
              <p className="text-xs text-gray-600">{tf("accN", lang, { n: item.metric?.n ?? 0 })}</p>
              <p className="mt-3 text-sm leading-relaxed text-gray-700">{t(item.help, lang)}</p>
            </article>
          );
        })}
      </div>
    </section>
  );
}
