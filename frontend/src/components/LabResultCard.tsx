"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import type { LabResult } from "@/lib/api";

const statusConfig: Record<string, { label: string; className: string }> = {
  normal: { label: "Норма", className: "bg-emerald-100 text-emerald-700 border-emerald-200" },
  high: { label: "Выше нормы", className: "bg-red-100 text-red-700 border-red-200" },
  low: { label: "Ниже нормы", className: "bg-amber-100 text-amber-700 border-amber-200" },
};

function formatDate(dateStr: string | undefined): string {
  if (!dateStr) return "";
  try {
    return new Date(dateStr).toLocaleDateString("ru-RU", {
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
}

export function LabResultCard({ result }: Props) {
  const { test_type, test_date, results, interpretation, created_at } = result;
  const displayDate = test_date || created_at;

  return (
    <Card className="border border-gray-200 shadow-sm">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-base font-semibold text-gray-900">
            {test_type || "Анализ"}
          </CardTitle>
          {displayDate && (
            <span className="text-xs text-gray-500">{formatDate(displayDate)}</span>
          )}
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {results.length > 0 && (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-100">
                  <th className="text-left py-2 pr-3 font-medium text-gray-500">Показатель</th>
                  <th className="text-left py-2 pr-3 font-medium text-gray-500">Значение</th>
                  <th className="text-left py-2 pr-3 font-medium text-gray-500">Норма</th>
                  <th className="text-left py-2 font-medium text-gray-500">Статус</th>
                </tr>
              </thead>
              <tbody>
                {results.map((item, idx) => {
                  const cfg = statusConfig[item.status] || statusConfig.normal;
                  return (
                    <tr key={idx} className="border-b border-gray-50 last:border-0">
                      <td className="py-2 pr-3 text-gray-800">{item.name}</td>
                      <td className="py-2 pr-3 font-medium text-gray-900">
                        {item.value}
                        {item.unit ? ` ${item.unit}` : ""}
                      </td>
                      <td className="py-2 pr-3 text-gray-500">{item.reference_range || "—"}</td>
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
          <div className="bg-blue-50 border border-blue-100 rounded-lg p-3">
            <p className="text-xs font-medium text-blue-700 mb-1">AI-интерпретация</p>
            <p className="text-sm text-blue-900">{interpretation}</p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
