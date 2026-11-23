"use client";

import { useState } from "react";
import { Check, ChevronDown, ChevronUp, Copy, Smartphone } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import type { Lang } from "@/lib/i18n";
import { dt } from "@/lib/strings/doctor";

export interface DemoPanelProps {
  lang: Lang;
  /** "hero": large, for the empty state. "card": collapsible card above the queue. */
  variant: "hero" | "card";
  className?: string;
}

export function DemoPanel({ lang, variant, className }: DemoPanelProps) {
  // This panel only mounts after the session check, so it never renders on the server.
  const [url] = useState(() => (typeof window === "undefined" ? "" : `${window.location.origin}/?demo=1`));
  const [copied, setCopied] = useState(false);
  const [open, setOpen] = useState(false);

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(url);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      /* clipboard blocked: the link is selectable on screen */
    }
  };

  const hero = variant === "hero";
  const body = (
    <div className={cn("flex flex-col gap-4", hero && "items-center text-center")}>
      <div className={cn("flex flex-col gap-4", hero && "sm:flex-row sm:items-center sm:gap-8")}>
        <div className={cn("flex min-w-0 flex-col gap-3", hero && "items-center sm:items-start sm:text-left")}>
          <p
            className={cn(
              "break-all font-semibold tabular-nums text-foreground select-all",
              hero ? "text-3xl leading-tight sm:text-4xl" : "text-lg leading-snug"
            )}
          >
            {url || " "}
          </p>
          <Button variant="secondary" size={hero ? "lg" : "sm"} onClick={copy} className="w-fit">
            {copied ? <Check /> : <Copy />}
            {copied ? dt("demoCopied", lang) : dt("demoCopy", lang)}
          </Button>
          <p className={cn("text-muted-foreground", hero ? "max-w-md text-base" : "text-sm")}>
            {dt("demoLine", lang)}
          </p>
        </div>
        {/* QR_SLOT: the orchestrator renders the QR code for `url` here (about 160px square). */}
        <div data-slot="demo-qr" className="hidden" aria-hidden="true">
          {dt("demoQrNote", lang)}
        </div>
      </div>
    </div>
  );

  if (hero) {
    return (
      <section
        aria-label={dt("demoTitle", lang)}
        className={cn("rounded-xl border border-border bg-card p-6 shadow-card sm:p-8", className)}
      >
        <h2 className="mb-4 flex items-center justify-center gap-2 font-display text-2xl font-semibold text-foreground">
          <Smartphone className="size-6 text-primary" aria-hidden="true" />
          {dt("demoTitle", lang)}
        </h2>
        {body}
      </section>
    );
  }

  return (
    <section
      aria-label={dt("demoTitle", lang)}
      className={cn("rounded-xl border border-border bg-card shadow-card", className)}
    >
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        className="flex min-h-11 w-full items-center justify-between gap-2 rounded-xl px-4 py-2.5 text-left text-sm font-semibold text-foreground outline-none transition-colors hover:bg-muted focus-visible:ring-[3px] focus-visible:ring-ring"
      >
        <span className="flex items-center gap-2">
          <Smartphone className="size-4 text-primary" aria-hidden="true" />
          {dt("demoTitle", lang)}
        </span>
        {open ? <ChevronUp className="size-4 text-muted-foreground" /> : <ChevronDown className="size-4 text-muted-foreground" />}
      </button>
      {open && <div className="border-t border-border px-4 py-4">{body}</div>}
    </section>
  );
}
