"use client";

import { Logo } from "@/components/brand";
import { UrgencyBanner } from "@/components/brand/UrgencyBanner";
import { cn } from "@/lib/utils";
import { tc } from "@/lib/strings/chat";
import type { Lang } from "@/lib/i18n";

interface ChatMessageProps {
  role: "user" | "assistant";
  content: string;
  timestamp?: string;
  /** Emergency alert (red flag / stridor / LLM emergency): rendered as the solid urgent banner */
  alert?: boolean;
  lang?: Lang;
}

export function ChatMessage({ role, content, timestamp, alert, lang = "ru" }: ChatMessageProps) {
  const isUser = role === "user";

  if (alert && !isUser) {
    return (
      <UrgencyBanner
        title={tc("alertTitle", lang)}
        message={content}
        action={
          <a
            href="tel:103"
            className="inline-flex min-h-11 items-center rounded-[10px] bg-white px-4 text-sm font-semibold text-urgent outline-none focus-visible:ring-[3px] focus-visible:ring-white/70"
          >
            {tc("alertAction", lang)}
          </a>
        }
      />
    );
  }

  return (
    <div className={cn("flex items-end gap-2.5", isUser ? "justify-end" : "justify-start")}>
      {!isUser && <Logo size={28} withText={false} className="mb-0.5 shrink-0" />}
      <div
        className={cn(
          "max-w-[85%] px-4 py-2.5 sm:max-w-[80%]",
          isUser
            ? "rounded-2xl rounded-br-md bg-primary text-primary-foreground"
            : "rounded-2xl rounded-bl-md border border-border bg-card text-foreground shadow-card"
        )}
      >
        <p className="whitespace-pre-wrap break-words text-[15px] leading-relaxed">{content}</p>
        {timestamp && (
          <p
            className={cn(
              "mt-1 text-xs tabular-nums",
              isUser ? "text-right text-primary-foreground/70" : "text-muted-foreground"
            )}
          >
            {timestamp}
          </p>
        )}
      </div>
    </div>
  );
}
