import type { ReactNode } from "react";
import { cn } from "@/lib/utils";

export interface StatTileProps {
  label: string;
  value: ReactNode;
  hint?: string;
  className?: string;
}

export function StatTile({ label, value, hint, className }: StatTileProps) {
  return (
    <div className={cn("rounded-xl border border-border bg-card p-4 shadow-card", className)}>
      <p className="text-[13px] font-medium text-muted-foreground">{label}</p>
      <p className="mt-1 text-3xl font-semibold leading-none tabular-nums text-foreground">{value}</p>
      {hint && <p className="mt-2 text-[13px] text-muted-foreground">{hint}</p>}
    </div>
  );
}
