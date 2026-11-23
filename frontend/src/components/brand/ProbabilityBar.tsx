import { cn } from "@/lib/utils";

export interface ProbabilityBarProps {
  label: string;
  /** Percent, 0-100. */
  percent: number;
  /** 0 = top-1 (darkest), 1 = second, 2+ = third and below. */
  rank?: number;
  className?: string;
}

const FILL = ["bg-chart-1", "bg-chart-2", "bg-chart-3"];

export function ProbabilityBar({ label, percent, rank = 0, className }: ProbabilityBarProps) {
  const pct = Math.max(0, Math.min(100, percent));
  return (
    <div className={cn("space-y-1", className)}>
      <div className="flex items-baseline justify-between gap-3 text-sm">
        <span className={cn("min-w-0 truncate", rank === 0 ? "font-semibold text-foreground" : "text-foreground/80")}>
          {label}
        </span>
        <span className="shrink-0 font-medium tabular-nums text-foreground">{Math.round(pct)}%</span>
      </div>
      <div
        className="h-1.5 w-full overflow-hidden rounded-full bg-muted"
        role="progressbar"
        aria-valuenow={Math.round(pct)}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-label={label}
      >
        <div className={cn("h-full rounded-full", FILL[Math.min(rank, 2)])} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}
