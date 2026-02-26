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
import { SYMPTOM_LIST, SYMPTOM_LABELS, type SymptomCode } from "@/lib/symptoms";

interface SymptomTrendProps {
  patientId: number;
}

export function SymptomTrend({ patientId }: SymptomTrendProps) {
  const [selectedSymptom, setSelectedSymptom] = useState<string>(SYMPTOM_LIST[0]);
  const [data, setData] = useState<SymptomGraphEntry[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!patientId || !selectedSymptom) return;
    setLoading(true);
    getSymptomGraph(patientId, selectedSymptom)
      .then((resp) => setData(resp.symptoms_arr))
      .catch(() => setData([]))
      .finally(() => setLoading(false));
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
        <TrendingUp className="h-5 w-5 text-gray-400" />
        <h3 className="font-semibold text-gray-900 text-sm">Динамика симптомов</h3>
      </div>

      <select
        value={selectedSymptom}
        onChange={(e) => setSelectedSymptom(e.target.value)}
        className="w-full px-3 py-2 rounded-lg border border-gray-200 text-sm bg-white focus:border-indigo-400 focus:ring-1 focus:ring-indigo-400 outline-none"
      >
        {SYMPTOM_LIST.map((code) => (
          <option key={code} value={code}>
            {SYMPTOM_LABELS[code as SymptomCode]} ({code})
          </option>
        ))}
      </select>

      {loading ? (
        <div className="h-48 flex items-center justify-center text-sm text-gray-400">
          Загрузка...
        </div>
      ) : chartData.length === 0 ? (
        <div className="h-48 flex items-center justify-center text-sm text-gray-400">
          Нет данных для отображения
        </div>
      ) : (
        <>
          {isWorsening && (
            <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-red-50 border border-red-200">
              <AlertTriangle className="h-4 w-4 text-red-500" />
              <span className="text-sm text-red-700">Тренд ухудшения</span>
            </div>
          )}

          <div className="h-48">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis
                  dataKey="date"
                  tick={{ fontSize: 11 }}
                  tickLine={false}
                  axisLine={{ stroke: "#e5e7eb" }}
                />
                <YAxis
                  domain={[0, 3]}
                  ticks={[0, 1, 2, 3]}
                  tick={{ fontSize: 11 }}
                  tickLine={false}
                  axisLine={{ stroke: "#e5e7eb" }}
                  width={30}
                />
                <Tooltip
                  formatter={(value) => [`Тяжесть: ${value}`, ""]}
                  labelFormatter={(label) => `Дата: ${label}`}
                  contentStyle={{
                    borderRadius: "8px",
                    border: "1px solid #e5e7eb",
                    fontSize: "12px",
                  }}
                />
                <Line
                  type="monotone"
                  dataKey="value"
                  stroke="#6366f1"
                  strokeWidth={2}
                  dot={{ fill: "#6366f1", r: 4 }}
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
