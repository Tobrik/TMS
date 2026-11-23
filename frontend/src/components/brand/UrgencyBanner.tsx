import type { ReactNode } from "react";
import { TriangleAlert } from "lucide-react";
import { cn } from "@/lib/utils";

export interface UrgencyBannerProps {
  title: string;
  message?: ReactNode;
  /** Optional action, e.g. a "Call 103" link or button. */
  action?: ReactNode;
  className?: string;
}

/** The one loud element: solid urgent background, white text. Emergencies only. */
export function UrgencyBanner({ title, message, action, className }: UrgencyBannerProps) {
  return (
    <div
      role="alert"
      className={cn("flex items-start gap-3 rounded-xl bg-urgent p-4 text-white", className)}
    >
      <TriangleAlert className="mt-0.5 size-6 shrink-0" aria-hidden="true" />
      <div className="min-w-0 flex-1">
        <p className="text-base font-semibold leading-snug">{title}</p>
        {message && <div className="mt-1 text-[15px] leading-relaxed text-white/95">{message}</div>}
        {action && <div className="mt-3">{action}</div>}
      </div>
    </div>
  );
}
