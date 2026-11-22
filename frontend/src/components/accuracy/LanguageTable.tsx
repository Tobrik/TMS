import type { AccuracyLanguage, AccuracyWritten } from "@/lib/api";
import { t, type Lang, type TranslationKey } from "@/lib/i18n";
import { MetricText } from "./CiBar";

const MIN_N = 30;

const LANGUAGES: { code: AccuracyLanguage; label: TranslationKey }[] = [
  { code: "ru", label: "accLangRu" },
  { code: "kk", label: "accLangKk" },
  { code: "mix", label: "accLangMix" },
  { code: "en", label: "accLangEn" },
];

export function LanguageTable({ written, lang }: { written: AccuracyWritten | null; lang: Lang }) {
  return (
    <section className="space-y-3">
      <h2 className="text-lg font-semibold text-gray-900">{t("accByLangTitle", lang)}</h2>
      {!written ? (
        <p className="text-sm text-gray-700">{t("accNoWritten", lang)}</p>
      ) : (
        <>
          <p className="text-sm text-gray-700">{t("accByLangIntro", lang)}</p>
          <div className="overflow-x-auto print:overflow-visible">
            <table className="w-full min-w-[28rem] text-left text-sm">
              <caption className="sr-only">{t("accByLangTitle", lang)}</caption>
              <thead>
                <tr className="border-b border-gray-200 text-gray-700">
                  <th scope="col" className="py-2 pr-3 font-medium">{t("accColLang", lang)}</th>
                  <th scope="col" className="py-2 pr-3 font-medium">{t("accColN", lang)}</th>
                  <th scope="col" className="py-2 font-medium">{t("accColTop1", lang)}</th>
                </tr>
              </thead>
              <tbody>
                {LANGUAGES.map(({ code, label }) => {
                  const entry = written.by_language[code];
                  const n = entry?.n ?? 0;
                  return (
                    <tr key={code} className="border-b border-gray-100 align-top">
                      <th scope="row" className="py-2 pr-3 font-normal text-gray-800">{t(label, lang)}</th>
                      <td className="py-2 pr-3 tabular-nums">{n}</td>
                      <td className="py-2">
                        <MetricText metric={entry?.top1} lang={lang} showN={false} />
                        {n < MIN_N && (
                          <span className="ml-2 rounded bg-amber-100 px-1.5 py-0.5 text-xs text-amber-900">
                            {t("accFewData", lang)}
                          </span>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </>
      )}
    </section>
  );
}
