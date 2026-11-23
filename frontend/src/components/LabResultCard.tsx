"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import type { LabResult } from "@/lib/api";
import { t, getLang, type Lang } from "@/lib/i18n";

function formatDate(dateStr: string | undefined, lang: Lang): string {
  if (!dateStr) return "";
  try {
    const locale = lang === "en" ? "en-US" : lang === "kk" ? "kk-KZ" : "ru-RU";
    return new Date(dateStr).toLocaleDateString(locale, {
      day: "numeric",
      month: "long",
      year: "numeric",
    });
  } catch {
    return dateStr;
  }
}

interface Props {
  result: LabResult;
  lang?: Lang;
}

export function LabResultCard({ result, lang }: Props) {
  const l = lang || getLang();
  const { test_type, test_date, results, interpretation, created_at } = result;
  const displayDate = test_date || created_at;

  const statusConfig: Record<string, { label: string; className: string }> = {
    normal: { label: t("statusNormal", l), className: "bg-ok-bg text-ok border-ok-border" },
    high: { label: t("statusHigh", l), className: "bg-urgent-bg text-urgent border-urgent-border" },
    low: { label: t("statusLow", l), className: "bg-warn-bg text-warn border-warn-border" },
  };

  // Unknown or empty status is neutral, never shown as "normal"
  const unknownStatus = { label: t("labStatusUnknown", l), className: "bg-muted text-muted-foreground border-border" };

  return (
    <Card className="gap-4 py-5">
      <CardHeader className="px-5">
        <div className="flex items-center justify-between">
          <CardTitle className="text-base font-semibold text-foreground">
            {test_type || t("analysisDefault", l)}
          </CardTitle>
          {displayDate && (
            <span className="text-[13px] tabular-nums text-muted-foreground">{formatDate(displayDate, l)}</span>
          )}
        </div>
      </CardHeader>
      <CardContent className="space-y-4 px-5">
        {results.length > 0 && (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border">
                  <th className="text-left py-2 pr-3 font-medium text-muted-foreground">{t("colIndicator", l)}</th>
                  <th className="text-left py-2 pr-3 font-medium text-muted-foreground">{t("colValue", l)}</th>
                  <th className="text-left py-2 pr-3 font-medium text-muted-foreground">{t("colNorm", l)}</th>
                  <th className="text-left py-2 font-medium text-muted-foreground">{t("colStatus", l)}</th>
                </tr>
              </thead>
              <tbody>
                {results.map((item, idx) => {
                  const cfg = statusConfig[item.status] || unknownStatus;
                  return (
                    <tr key={idx} className="border-b border-border/60 last:border-0">
                      <td className="py-2 pr-3 text-foreground">{item.name}</td>
                      <td className="py-2 pr-3 font-medium tabular-nums text-foreground">
                        {item.value}
                        {item.unit ? ` ${item.unit}` : ""}
                      </td>
                      <td className="py-2 pr-3 text-muted-foreground">{item.reference_range || "—"}</td>
                      <td className="py-2">
                        <Badge variant="outline" className={`text-xs ${cfg.className}`}>
                          {cfg.label}
                        </Badge>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}

        {interpretation && (
          <div className="bg-accent border border-border rounded-lg p-3">
            <p className="text-[13px] font-semibold text-accent-foreground mb-1">{t("aiInterpretation", l)}</p>
            <p className="text-sm text-foreground">{interpretation}</p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
