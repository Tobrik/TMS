"use client";

import { Bot } from "lucide-react";
import { t, type Lang } from "@/lib/i18n";

interface DiagnosisSkeletonProps {
  lang?: Lang;
}

export function DiagnosisSkeleton({ lang = "ru" }: DiagnosisSkeletonProps) {
  return (
    <div className="flex gap-3 items-start">
      <div className="w-8 h-8 rounded-full bg-gradient-to-br from-teal-400 to-emerald-500 flex items-center justify-center shrink-0">
        <Bot className="h-4 w-4 text-white" />
      </div>
      <div className="flex-1 space-y-3">
        <div className="bg-white border border-gray-200 rounded-2xl p-4 space-y-4 animate-pulse">
          {/* Title skeleton */}
          <div className="flex items-center gap-3">
            <div className="h-5 w-40 bg-gray-200 rounded" />
            <div className="h-5 w-20 bg-gray-200 rounded-full" />
          </div>

          {/* Pie chart skeleton */}
          <div className="flex items-center justify-center py-4">
            <div className="w-32 h-32 bg-gray-200 rounded-full" />
          </div>

          {/* Text lines skeleton */}
          <div className="space-y-2">
            <div className="h-4 w-3/4 bg-gray-200 rounded" />
            <div className="h-4 w-1/2 bg-gray-200 rounded" />
            <div className="h-4 w-5/6 bg-gray-200 rounded" />
          </div>

          {/* Badge skeleton */}
          <div className="h-8 w-48 bg-gray-200 rounded-lg" />
        </div>

        <p className="text-xs text-gray-400 ml-1">{t("analyzingSymptomsDots", lang)}</p>
      </div>
    </div>
  );
}
