"use client";

import { cn } from "@/lib/utils";
import { setLang, t, type Lang } from "@/lib/i18n";

const OPTIONS: { value: Lang; label: string; name: string }[] = [
  { value: "ru", label: "RU", name: "Русский" },
  { value: "kk", label: "ҚАЗ", name: "Қазақша" },
  { value: "en", label: "EN", name: "English" },
];

export interface LanguageSwitcherProps {
  lang: Lang;
  onChange?: (lang: Lang) => void;
  className?: string;
}

export function LanguageSwitcher({ lang, onChange, className }: LanguageSwitcherProps) {
  const select = (next: Lang) => {
    if (next === lang) return;
    setLang(next);
    onChange?.(next);
  };

  const onKeyDown = (e: React.KeyboardEvent, index: number) => {
    let target = -1;
    if (e.key === "ArrowRight" || e.key === "ArrowDown") target = (index + 1) % OPTIONS.length;
    if (e.key === "ArrowLeft" || e.key === "ArrowUp") target = (index - 1 + OPTIONS.length) % OPTIONS.length;
    if (target < 0) return;
    e.preventDefault();
    select(OPTIONS[target].value);
    (e.currentTarget.parentElement?.children[target] as HTMLElement | undefined)?.focus();
  };

  return (
    <div
      role="radiogroup"
      aria-label={t("languageLabel", lang)}
      className={cn("inline-flex items-center rounded-full border border-border bg-muted p-0.5", className)}
    >
      {OPTIONS.map((o, i) => {
        const active = o.value === lang;
        return (
          <button
            key={o.value}
            type="button"
            role="radio"
            aria-checked={active}
            aria-label={o.name}
            tabIndex={active ? 0 : -1}
            onClick={() => select(o.value)}
            onKeyDown={(e) => onKeyDown(e, i)}
            className={cn(
              "min-h-8 min-w-10 rounded-full px-2.5 text-xs font-semibold outline-none transition-colors duration-150 focus-visible:ring-[3px] focus-visible:ring-ring",
              active
                ? "bg-card text-foreground shadow-card"
                : "text-muted-foreground hover:text-foreground"
            )}
          >
            {o.label}
          </button>
        );
      })}
    </div>
  );
}
