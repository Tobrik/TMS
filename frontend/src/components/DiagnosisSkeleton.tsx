"use client";

import { Logo } from "@/components/brand";
import { t, type Lang } from "@/lib/i18n";

interface DiagnosisSkeletonProps {
  lang?: Lang;
}

export function DiagnosisSkeleton({ lang = "ru" }: DiagnosisSkeletonProps) {
  return (
    <div className="flex items-end gap-2.5" role="status">
      <Logo size={28} withText={false} className="mb-0.5 shrink-0" />
      <div className="w-full max-w-[85%] space-y-2 rounded-2xl rounded-bl-md border border-border bg-card p-4 shadow-card sm:max-w-[80%]">
        <div className="space-y-2 motion-safe:animate-pulse">
          <div className="h-3.5 w-2/3 rounded bg-muted" />
          <div className="h-3.5 w-1/2 rounded bg-muted" />
          <div className="h-3.5 w-3/4 rounded bg-muted" />
        </div>
        <p className="text-[13px] text-muted-foreground">{t("analyzingSymptomsDots", lang)}</p>
      </div>
    </div>
  );
}
