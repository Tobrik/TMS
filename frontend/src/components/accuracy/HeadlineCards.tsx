import type { AccuracyMetric, AccuracyWritten } from "@/lib/api";
import { t, type Lang, type TranslationKey } from "@/lib/i18n";
import { StatTile } from "@/components/brand/StatTile";
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
      <h2 className="text-lg font-semibold text-foreground sm:text-xl">{t("accHeadline", lang)}</h2>
      {!written && <p className="text-sm text-muted-foreground">{t("accNoWritten", lang)}</p>}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 print:grid-cols-2">
        {items.map((item) => {
          const ci = formatCi(item.metric, lang);
          const hint = [ci && tf("accCi", lang, { ci }), tf("accN", lang, { n: item.metric?.n ?? 0 })]
            .filter(Boolean)
            .join(" · ");
          return (
            <div key={item.title} className="flex flex-col gap-2 print:break-inside-avoid">
              <StatTile
                label={t(item.title, lang)}
                value={formatPct(item.metric?.value ?? null, lang)}
                hint={hint}
              />
              <p className="px-1 text-[13px] leading-relaxed text-muted-foreground">{t(item.help, lang)}</p>
            </div>
          );
        })}
      </div>
    </section>
  );
}
