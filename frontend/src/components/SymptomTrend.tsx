"use client";

import { useState, useEffect } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { TrendingUp, AlertTriangle } from "lucide-react";
import { getSymptomGraph, type SymptomGraphEntry } from "@/lib/api";
import { SYMPTOM_LIST, type SymptomCode } from "@/lib/symptoms";
import { getSymptomLabel } from "@/lib/symptoms";
import { t, type Lang } from "@/lib/i18n";

interface SymptomTrendProps {
  patientId: number;
  lang?: Lang;
}

export function SymptomTrend({ patientId, lang = "ru" }: SymptomTrendProps) {
  const [selectedSymptom, setSelectedSymptom] = useState<string>(SYMPTOM_LIST[0]);
  const [loaded, setLoaded] = useState<{ key: string; data: SymptomGraphEntry[] } | null>(null);
  const key = `${patientId}:${selectedSymptom}`;
  const loading = loaded?.key !== key;
  const data = loaded?.key === key ? loaded.data : [];

  useEffect(() => {
    if (!patientId || !selectedSymptom) return;
    let cancelled = false;
    getSymptomGraph(patientId, selectedSymptom)
      .then((resp) => resp.symptoms_arr)
      .catch(() => [] as SymptomGraphEntry[])
      .then((rows) => { if (!cancelled) setLoaded({ key: `${patientId}:${selectedSymptom}`, data: rows }); });
    return () => { cancelled = true; };
  }, [patientId, selectedSymptom]);

  const chartData = data.map((d) => ({
    date: d.created_at?.split(" ")[0] || d.created_at?.split("T")[0] || "",
    value: d.value,
  }));

  // Detect worsening trend: last 3 entries going up
  const isWorsening =
    chartData.length >= 3 &&
    chartData[chartData.length - 1].value > chartData[chartData.length - 2].value &&
    chartData[chartData.length - 2].value > chartData[chartData.length - 3].value;

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        <TrendingUp className="size-5 text-primary" />
        <h3 className="font-semibold text-foreground text-[15px]">{t("symptomDynamics", lang)}</h3>
      </div>

      <select
        value={selectedSymptom}
        onChange={(e) => setSelectedSymptom(e.target.value)}
        className="h-11 w-full rounded-md border border-input bg-card px-3 text-base text-foreground outline-none transition-[color,box-shadow] focus-visible:border-ring focus-visible:ring-[3px] focus-visible:ring-ring/60 md:text-sm"
      >
        {SYMPTOM_LIST.map((code) => (
          <option key={code} value={code}>
            {getSymptomLabel(code as SymptomCode, lang)} ({code})
          </option>
        ))}
      </select>

      {loading ? (
        <div className="h-48 flex items-center justify-center text-sm text-muted-foreground">
          {t("loadingEllipsis", lang)}
        </div>
      ) : chartData.length === 0 ? (
        <div className="h-48 flex items-center justify-center text-sm text-muted-foreground">
          {t("noDataToDisplay", lang)}
        </div>
      ) : (
        <>
          {isWorsening && (
            <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-urgent-bg border border-urgent-border">
              <AlertTriangle className="size-4 text-urgent" />
              <span className="text-sm text-urgent">{t("worseningTrend", lang)}</span>
            </div>
          )}

          <div className="h-48">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                <XAxis
                  dataKey="date"
                  tick={{ fontSize: 12, fill: "var(--muted-foreground)" }}
                  tickLine={false}
                  axisLine={{ stroke: "var(--border)" }}
                />
                <YAxis
                  domain={[0, 3]}
                  ticks={[0, 1, 2, 3]}
                  tick={{ fontSize: 12, fill: "var(--muted-foreground)" }}
                  tickLine={false}
                  axisLine={{ stroke: "var(--border)" }}
                  width={30}
                />
                <Tooltip
                  formatter={(value) => [`${t("severity", lang)}: ${value}`, ""]}
                  labelFormatter={(label) => `${t("date", lang)}: ${label}`}
                  contentStyle={{
                    borderRadius: "8px",
                    border: "1px solid var(--border)",
                    fontSize: "12px",
                  }}
                />
                <Line
                  type="monotone"
                  dataKey="value"
                  stroke="var(--chart-1)"
                  strokeWidth={2}
                  dot={{ fill: "var(--chart-1)", r: 4 }}
                  activeDot={{ r: 6 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </>
      )}
    </div>
  );
}
