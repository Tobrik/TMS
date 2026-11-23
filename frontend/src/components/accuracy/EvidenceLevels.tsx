import type { ReactNode } from "react";
import type { AccuracyMetric, AccuracyRealPatients, AccuracySynthetic, AccuracyWritten } from "@/lib/api";
import { t, type Lang, type TranslationKey } from "@/lib/i18n";
import { MetricText } from "./CiBar";
import { formatDate, formatNumber, formatPct, pctPlain, tf } from "./format";

function LevelBlock({ title, source, children }: { title: string; source: string; children: ReactNode }) {
  return (
    <section className="space-y-4 rounded-xl border border-border border-l-4 border-l-primary bg-card p-5 shadow-card print:break-inside-avoid print:shadow-none">
      <div>
        <h3 className="text-base font-semibold text-foreground">{title}</h3>
        {source && <p className="mt-1 text-xs text-muted-foreground">{source}</p>}
      </div>
      {children}
    </section>
  );
}

function MetricRows({ rows, lang }: { rows: [TranslationKey, AccuracyMetric | undefined][]; lang: Lang }) {
  return (
    <dl className="divide-y divide-border text-sm">
      {rows.map(([key, metric]) => (
        <div key={key} className="grid grid-cols-1 gap-1 py-2 sm:grid-cols-[minmax(0,1fr)_auto] sm:gap-4">
          <dt className="text-foreground/85">{t(key, lang)}</dt>
          <dd>
            <MetricText metric={metric} lang={lang} />
          </dd>
        </div>
      ))}
    </dl>
  );
}

export function RealPatientsBlock({ real, lang }: { real: AccuracyRealPatients; lang: Lang }) {
  return (
    <LevelBlock title={t("accLevelReal", lang)} source={`${t("accSource", lang)}: ${t("accRealSource", lang)}`}>
      {real.n === 0 ? (
        <p className="text-sm text-foreground/85">{t("accRealEmpty", lang)}</p>
      ) : (
        <>
          <p className="text-sm text-muted-foreground">
            {tf("accN", lang, { n: real.n })}
            {real.since && ` · ${tf("accSince", lang, { date: formatDate(real.since, lang) })}`}
          </p>
          <MetricRows
            lang={lang}
            rows={[
              ["accTop1", real.top1],
              ["accTop3", real.top3],
              ["accZoneExact", real.zone_exact],
              ["accTriageRedSens", real.red_sensitivity],
              ["underTriage", real.under_triage],
              ["overTriage", real.over_triage],
            ]}
          />
        </>
      )}
    </LevelBlock>
  );
}

export function WrittenBlock({ written, lang }: { written: AccuracyWritten | null; lang: Lang }) {
  if (!written) {
    return (
      <LevelBlock title={t("accLevelWritten", lang)} source="">
        <p className="text-sm text-foreground/85">{t("accNoWritten", lang)}</p>
      </LevelBlock>
    );
  }
  const rows: [TranslationKey, AccuracyMetric, AccuracyMetric][] = [
    ["accTop1", written.patient_accepts.top1, written.patient_confirms.top1],
    ["accTop3", written.patient_accepts.top3, written.patient_confirms.top3],
    ["accTriageRedSens", written.patient_accepts.red_sensitivity, written.patient_confirms.red_sensitivity],
    ["accTriageGreenToRed", written.patient_accepts.green_to_red, written.patient_confirms.green_to_red],
  ];
  return (
    <LevelBlock
      title={t("accLevelWritten", lang)}
      source={`${t("accSource", lang)}: ${written.source}`}
    >
      <p className="text-sm text-muted-foreground">
        {tf("accWrittenScope", lang, { inScope: written.n_in_scope, outScope: written.n_out_of_scope })}
      </p>

      <div className="overflow-x-auto print:overflow-visible">
        <table className="w-full min-w-[32rem] text-left text-sm">
          <caption className="sr-only">{t("accLevelWritten", lang)}</caption>
          <thead>
            <tr className="border-b border-border text-muted-foreground">
              <th scope="col" className="py-2 pr-3 font-medium">{t("accColMetric", lang)}</th>
              <th scope="col" className="py-2 pr-3 font-medium">{t("accColAccepts", lang)}</th>
              <th scope="col" className="py-2 font-medium">{t("accColConfirms", lang)}</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(([label, accepts, confirms]) => (
              <tr key={label} className="border-b border-border align-top">
                <th scope="row" className="py-2 pr-3 font-normal text-foreground">{t(label, lang)}</th>
                <td className="py-2 pr-3"><MetricText metric={accepts} lang={lang} /></td>
                <td className="py-2"><MetricText metric={confirms} lang={lang} /></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="space-y-1">
        <h4 className="text-sm font-semibold text-foreground">{t("accExtraction", lang)}</h4>
        <MetricRows
          lang={lang}
          rows={[
            ["accExtPrecision", written.extraction.precision],
            ["accExtRecall", written.extraction.recall],
            ["accExtDenials", written.extraction.denials],
            ["accExtRedRecall", written.extraction.red_flag_recall],
          ]}
        />
      </div>

      <div className="space-y-1">
        <h4 className="text-sm font-semibold text-foreground">{t("accOutOfScope", lang)}</h4>
        <MetricRows
          lang={lang}
          rows={[
            ["accOosFlagged", written.out_of_scope.flagged],
            ["accOosFalse", written.out_of_scope.false_flags],
          ]}
        />
      </div>
    </LevelBlock>
  );
}

export function SyntheticBlock({ synthetic, lang }: { synthetic: AccuracySynthetic; lang: Lang }) {
  const low = synthetic.low_confidence;
  return (
    <LevelBlock title={t("accLevelSynthetic", lang)} source={`${t("accSource", lang)}: ${synthetic.source}`}>
      <p className="text-sm text-muted-foreground">
        {tf("accSynthMeta", lang, {
          n: synthetic.settings.n,
          q: formatNumber(synthetic.questions_mean, lang, 1),
        })}
      </p>
      <MetricRows
        lang={lang}
        rows={[
          ["accTop1", synthetic.metrics.top1],
          ["accTop3", synthetic.metrics.top3],
          ["accZoneExact", synthetic.metrics.zone_exact],
        ]}
      />
      <div className="rounded-lg bg-muted p-4 text-sm text-foreground/85">
        <h4 className="font-semibold text-foreground">{t("accLowConfTitle", lang)}</h4>
        <p className="mt-1 leading-relaxed">
          {tf("accLowConfText", lang, {
            share: formatPct(low.share.value, lang),
            low: pctPlain(low.top1_low.value, lang),
            rest: pctPlain(low.top1_rest.value, lang),
          })}
        </p>
      </div>
    </LevelBlock>
  );
}
