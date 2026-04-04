"use client";

import { Bot } from "lucide-react";
import { t, type Lang } from "@/lib/i18n";

interface LoadingIndicatorProps {
  lang?: Lang;
}

export function LoadingIndicator({ lang = "ru" }: LoadingIndicatorProps) {
  return (
    <div className="flex gap-3">
      <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-gradient-to-br from-teal-500 to-emerald-600 text-white">
        <Bot className="h-4 w-4" />
      </div>
      <div className="bg-white border border-gray-100 rounded-2xl rounded-bl-md px-4 py-3 shadow-sm">
        <div className="flex items-center gap-1.5">
          <div className="h-2 w-2 rounded-full bg-gray-400 animate-bounce [animation-delay:0ms]" />
          <div className="h-2 w-2 rounded-full bg-gray-400 animate-bounce [animation-delay:150ms]" />
          <div className="h-2 w-2 rounded-full bg-gray-400 animate-bounce [animation-delay:300ms]" />
        </div>
        <p className="text-xs text-gray-400 mt-1">{t("analyzingSymptomsDots", lang)}</p>
      </div>
    </div>
  );
}
