"use client";

import type { DiseaseSlice } from "@/lib/types";

interface PieChartProps {
  slices: DiseaseSlice[];
  size?: number;
}

const COLORS = ["#6366f1", "#f59e0b", "#10b981"]; // indigo, amber, emerald

export function PieChart({ slices, size = 140 }: PieChartProps) {
  const total = slices.reduce((sum, s) => sum + Math.max(s.score, 0), 0);
  if (total === 0) return null;

  const radius = size / 2;
  const center = radius;
  const innerRadius = radius * 0.55; // donut hole

  let currentAngle = -90; // start from top

  const paths = slices.map((slice, i) => {
    const pct = Math.max(slice.score, 0) / total;
    const angle = pct * 360;
    const startAngle = currentAngle;
    const endAngle = currentAngle + angle;
    currentAngle = endAngle;

    const startRad = (startAngle * Math.PI) / 180;
    const endRad = (endAngle * Math.PI) / 180;

    const x1 = center + radius * Math.cos(startRad);
    const y1 = center + radius * Math.sin(startRad);
    const x2 = center + radius * Math.cos(endRad);
    const y2 = center + radius * Math.sin(endRad);

    const ix1 = center + innerRadius * Math.cos(startRad);
    const iy1 = center + innerRadius * Math.sin(startRad);
    const ix2 = center + innerRadius * Math.cos(endRad);
    const iy2 = center + innerRadius * Math.sin(endRad);

    const largeArc = angle > 180 ? 1 : 0;

    const d = [
      `M ${x1} ${y1}`,
      `A ${radius} ${radius} 0 ${largeArc} 1 ${x2} ${y2}`,
      `L ${ix2} ${iy2}`,
      `A ${innerRadius} ${innerRadius} 0 ${largeArc} 0 ${ix1} ${iy1}`,
      "Z",
    ].join(" ");

    return (
      <path key={i} d={d} fill={COLORS[i]} className="transition-all duration-300" />
    );
  });

  return (
    <div className="flex flex-col items-center gap-3">
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        {paths}
        {/* Center text */}
        <text
          x={center}
          y={center - 4}
          textAnchor="middle"
          className="fill-gray-900 text-[11px] font-bold"
        >
          TOP 3
        </text>
        <text
          x={center}
          y={center + 10}
          textAnchor="middle"
          className="fill-gray-400 text-[9px]"
        >
          диагноза
        </text>
      </svg>

      {/* Legend */}
      <div className="w-full space-y-1.5">
        {slices.map((slice, i) => {
          const pct = total > 0 ? Math.round((Math.max(slice.score, 0) / total) * 100) : 0;
          return (
            <div key={i} className="flex items-center gap-2">
              <div
                className="w-3 h-3 rounded-sm shrink-0"
                style={{ backgroundColor: COLORS[i] }}
              />
              <span className="text-xs text-gray-700 flex-1 truncate">
                {slice.label}
              </span>
              <span className="text-xs font-semibold text-gray-900 tabular-nums">
                {pct}%
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
