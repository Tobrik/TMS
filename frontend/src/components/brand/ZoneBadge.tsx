"use client";

import { cn } from "@/lib/utils";
import { t, type Lang } from "@/lib/i18n";

export type ZoneValue = "red" | "yellow" | "green" | null;

export interface ZoneBadgeProps {
  zone: ZoneValue;
  size?: "sm" | "md";
  lang?: Lang;
  className?: string;
}

const STYLES = {
  red: { box: "bg-urgent-bg text-urgent border-urgent-border", dot: "bg-urgent" },
  yellow: { box: "bg-warn-bg text-warn border-warn-border", dot: "bg-warn" },
  green: { box: "bg-ok-bg text-ok border-ok-border", dot: "bg-ok" },
  none: { box: "bg-muted text-muted-foreground border-border", dot: "bg-muted-foreground/60" },
} as const;

export function ZoneBadge({ zone, size = "md", lang, className }: ZoneBadgeProps) {
  const s = STYLES[zone ?? "none"];
  const label =
    zone === "red" ? t("zoneShortRed", lang)
    : zone === "yellow" ? t("zoneShortYellow", lang)
    : zone === "green" ? t("zoneShortGreen", lang)
    : t("notAssessed", lang);
  return (
    <span
      className={cn(
        "inline-flex w-fit shrink-0 items-center gap-1.5 whitespace-nowrap rounded-full border font-medium",
        size === "sm" ? "px-2 py-0.5 text-xs" : "px-3 py-1 text-[13px]",
        s.box,
        className
      )}
    >
      <span className={cn("size-2 rounded-full", s.dot)} aria-hidden="true" />
      {label}
    </span>
  );
}
