import type { ReactNode } from "react";
import type { LucideIcon } from "lucide-react";
import { cn } from "@/lib/utils";

export interface EmptyStateProps {
  icon?: LucideIcon;
  title: string;
  text?: string;
  action?: ReactNode;
  className?: string;
}

export function EmptyState({ icon: Icon, title, text, action, className }: EmptyStateProps) {
  return (
    <div className={cn("flex flex-col items-center px-4 py-10 text-center", className)}>
      {Icon && (
        <span className="mb-3 inline-flex size-12 items-center justify-center rounded-full bg-accent text-accent-foreground">
          <Icon className="size-6" aria-hidden="true" />
        </span>
      )}
      <p className="text-base font-semibold text-foreground">{title}</p>
      {text && <p className="mt-1 max-w-sm text-sm text-muted-foreground">{text}</p>}
      {action && <div className="mt-4">{action}</div>}
    </div>
  );
}
