import type { ReactNode } from "react";
import { Logo } from "./Logo";
import { cn } from "@/lib/utils";

export interface AppHeaderProps {
  /** Optional title next to the logo (e.g. page name). */
  title?: ReactNode;
  subtitle?: ReactNode;
  /** Right side: actions and the LanguageSwitcher. */
  actions?: ReactNode;
  /** Wrap logo in a link/button if needed. */
  logoSlot?: ReactNode;
  className?: string;
  /** Max width of the inner row, e.g. "max-w-[720px]". Default is full width. */
  innerClassName?: string;
}

export function AppHeader({ title, subtitle, actions, logoSlot, className, innerClassName }: AppHeaderProps) {
  return (
    <header className={cn("sticky top-0 z-30 border-b border-border bg-card", className)}>
      <div className={cn("mx-auto flex h-14 w-full items-center justify-between gap-3 px-4 sm:px-6", innerClassName)}>
        <div className="flex min-w-0 items-center gap-3">
          {logoSlot ?? <Logo />}
          {(title || subtitle) && (
            <div className="min-w-0 border-l border-border pl-3">
              {title && <div className="truncate text-sm font-semibold leading-tight text-foreground">{title}</div>}
              {subtitle && <div className="truncate text-xs leading-tight text-muted-foreground">{subtitle}</div>}
            </div>
          )}
        </div>
        {actions && <div className="flex shrink-0 items-center gap-2 sm:gap-3">{actions}</div>}
      </div>
    </header>
  );
}
