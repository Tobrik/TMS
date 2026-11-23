"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import { Popover, Dialog } from "radix-ui";
import { BarChart3, Check, ChevronDown, Copy, KeyRound, Loader2, LogOut, UserPlus, X } from "lucide-react";
import type { Lang } from "@/lib/i18n";
import { apiErrorCode, changeDoctorPassword, createDoctorInvite, logout, type DoctorProfile } from "@/lib/api";
import { da, LOCALE } from "@/lib/strings/doctorAuth";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";

export interface AccountMenuProps {
  lang: Lang;
  doctor: DoctorProfile;
  /** The dashboard clears its own state; the menu has already called api logout(). */
  onLogout: () => void;
}

type Panel = null | "invite" | "password";

const PASSWORD_RE = /^(?=.*[A-Za-z])(?=.*\d).{8,64}$/; // same rule as the backend

function initials(name: string): string {
  const parts = name.trim().split(/\s+/).filter(Boolean);
  const letters = parts.slice(0, 2).map((p) => Array.from(p)[0] ?? "");
  return (letters.join("") || "?").toUpperCase();
}

const itemClass =
  "flex min-h-11 w-full items-center gap-3 rounded-[10px] px-3 text-left text-sm font-medium text-foreground outline-none transition-colors duration-150 hover:bg-muted focus-visible:bg-muted focus-visible:ring-[3px] focus-visible:ring-ring";

export function AccountMenu({ lang, doctor, onLogout }: AccountMenuProps) {
  const s = da(lang);
  const [open, setOpen] = useState(false);
  const [panel, setPanel] = useState<Panel>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);

  const openPanel = (p: Exclude<Panel, null>) => {
    setOpen(false);
    setPanel(p);
  };

  const doLogout = () => {
    setOpen(false);
    logout();
    onLogout();
  };

  return (
    <>
      <Popover.Root open={open} onOpenChange={setOpen}>
        <Popover.Trigger asChild>
          <button
            ref={triggerRef}
            type="button"
            aria-label={`${s.account}: ${doctor.full_name}`}
            className="inline-flex h-11 items-center gap-2 rounded-full border border-border bg-card py-1 pl-1 pr-2 outline-none transition-colors duration-150 hover:bg-muted focus-visible:ring-[3px] focus-visible:ring-ring sm:pr-3"
          >
            <span aria-hidden="true" className="flex size-9 items-center justify-center rounded-full bg-accent text-sm font-semibold text-accent-foreground">
              {initials(doctor.full_name)}
            </span>
            <span className="hidden max-w-40 truncate text-sm font-medium text-foreground sm:inline">{doctor.full_name}</span>
            <ChevronDown className="size-4 text-muted-foreground" aria-hidden="true" />
          </button>
        </Popover.Trigger>
        <Popover.Portal>
          <Popover.Content
            align="end"
            sideOffset={8}
            collisionPadding={12}
            className="z-50 w-[min(20rem,calc(100vw-1.5rem))] rounded-[14px] border border-border bg-popover p-2 text-popover-foreground shadow-card outline-none"
            onCloseAutoFocus={(e) => {
              e.preventDefault();
              if (!panel) triggerRef.current?.focus();
            }}
          >
            <div className="px-3 pb-3 pt-2">
              <div className="flex items-center gap-2">
                <p className="truncate font-semibold text-foreground">{doctor.full_name}</p>
                {doctor.is_demo ? (
                  <span className="shrink-0 rounded-full border border-warn-border bg-warn-bg px-2 py-0.5 text-xs font-semibold text-warn">
                    {s.demoPill}
                  </span>
                ) : null}
              </div>
              <p className="truncate text-[13px] text-muted-foreground">{doctor.specialty}</p>
              <p className="truncate text-[13px] text-muted-foreground">@{doctor.username}</p>
            </div>
            <div className="border-t border-border pt-2">
              {doctor.is_demo ? (
                <p className="px-3 pb-2 text-[13px] text-muted-foreground">{s.demoNote}</p>
              ) : (
                <>
                  <button type="button" className={itemClass} onClick={() => openPanel("invite")}>
                    <UserPlus className="size-4 text-muted-foreground" aria-hidden="true" />
                    {s.invite}
                  </button>
                  <button type="button" className={itemClass} onClick={() => openPanel("password")}>
                    <KeyRound className="size-4 text-muted-foreground" aria-hidden="true" />
                    {s.changePassword}
                  </button>
                </>
              )}
              <Link href="/accuracy" className={itemClass} onClick={() => setOpen(false)}>
                <BarChart3 className="size-4 text-muted-foreground" aria-hidden="true" />
                {s.accuracy}
              </Link>
              <button type="button" className={cn(itemClass, "text-urgent")} onClick={doLogout}>
                <LogOut className="size-4" aria-hidden="true" />
                {s.logout}
              </button>
            </div>
          </Popover.Content>
        </Popover.Portal>
      </Popover.Root>

      <Dialog.Root open={panel !== null} onOpenChange={(o) => { if (!o) setPanel(null); }}>
        <Dialog.Portal>
          <Dialog.Overlay className="fixed inset-0 z-50 bg-foreground/40" />
          <Dialog.Content
            aria-describedby={undefined}
            onCloseAutoFocus={(e) => {
              e.preventDefault();
              triggerRef.current?.focus();
            }}
            className="fixed left-1/2 top-1/2 z-50 max-h-[calc(100dvh-2rem)] w-[calc(100vw-2rem)] max-w-md -translate-x-1/2 -translate-y-1/2 overflow-y-auto rounded-[14px] border border-border bg-card p-5 shadow-card outline-none sm:p-6"
          >
            {panel === "invite" ? <InvitePanel lang={lang} /> : null}
            {panel === "password" ? <PasswordPanel lang={lang} /> : null}
            <Dialog.Close asChild>
              <button
                type="button"
                aria-label={s.close}
                className="absolute right-2 top-2 flex size-11 items-center justify-center rounded-full text-muted-foreground outline-none hover:bg-muted hover:text-foreground focus-visible:ring-[3px] focus-visible:ring-ring"
              >
                <X className="size-4" aria-hidden="true" />
              </button>
            </Dialog.Close>
          </Dialog.Content>
        </Dialog.Portal>
      </Dialog.Root>
    </>
  );
}

function InvitePanel({ lang }: { lang: Lang }) {
  const s = da(lang);
  const [invite, setInvite] = useState<{ code: string; expires_at: string } | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);
  const [copied, setCopied] = useState(false);

  const generate = useCallback(async () => {
    setLoading(true);
    setError(false);
    setCopied(false);
    try {
      setInvite(await createDoctorInvite());
    } catch {
      setError(true);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void generate();
  }, [generate]);

  const copy = async () => {
    if (!invite) return;
    try {
      await navigator.clipboard.writeText(invite.code);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      /* clipboard blocked: the code is selectable on screen */
    }
  };

  const expires = invite
    ? new Intl.DateTimeFormat(LOCALE[lang], { dateStyle: "long", timeStyle: "short" }).format(new Date(invite.expires_at))
    : "";

  return (
    <div>
      <Dialog.Title className="pr-10 font-display text-xl font-bold text-foreground">{s.inviteTitle}</Dialog.Title>
      <div className="mt-4 space-y-4" aria-live="polite">
        {loading ? (
          <p className="flex items-center gap-2 text-sm text-muted-foreground">
            <Loader2 className="size-4 animate-spin" aria-hidden="true" />
            {s.inviteGenerating}
          </p>
        ) : null}
        {error ? (
          <p role="alert" className="rounded-[10px] border border-urgent-border bg-urgent-bg px-3 py-2.5 text-sm text-urgent">{s.inviteErr}</p>
        ) : null}
        {invite && !loading ? (
          <>
            <div className="rounded-[10px] border border-border bg-muted px-3 py-4 text-center">
              <p className="select-all break-all font-mono text-2xl font-semibold tracking-widest text-foreground sm:text-3xl">{invite.code}</p>
            </div>
            <Button type="button" variant="outline" className="w-full" onClick={copy}>
              {copied ? <Check aria-hidden="true" /> : <Copy aria-hidden="true" />}
              {copied ? s.inviteCopied : s.inviteCopy}
            </Button>
            <p className="text-sm text-muted-foreground">
              {s.inviteExpires}: <span className="tabular-nums text-foreground">{expires}</span>
            </p>
            <p className="text-sm text-muted-foreground">{s.inviteRule}</p>
          </>
        ) : null}
        {!loading ? (
          <Button type="button" variant="ghost" size="sm" onClick={() => void generate()}>
            {s.inviteNew}
          </Button>
        ) : null}
      </div>
    </div>
  );
}

function PasswordPanel({ lang }: { lang: Lang }) {
  const s = da(lang);
  const [oldPw, setOldPw] = useState("");
  const [newPw, setNewPw] = useState("");
  const [repeat, setRepeat] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [done, setDone] = useState(false);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (busy) return;
    setDone(false);
    if (!oldPw) return setError(s.pwWrong);
    if (!PASSWORD_RE.test(newPw)) return setError(s.pwWeak);
    if (newPw !== repeat) return setError(s.pwMismatch);
    setBusy(true);
    setError(null);
    try {
      await changeDoctorPassword(oldPw, newPw);
      setDone(true);
      setOldPw("");
      setNewPw("");
      setRepeat("");
    } catch (err) {
      const status = (err as { response?: { status?: number } })?.response?.status;
      const c = apiErrorCode(err);
      if (c === "wrong_password") setError(s.pwWrong);
      else if (status === 422) setError(s.pwWeak);
      else setError(s.pwErr);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div>
      <Dialog.Title className="pr-10 font-display text-xl font-bold text-foreground">{s.pwTitle}</Dialog.Title>
      <form onSubmit={submit} className="mt-4 space-y-4" noValidate>
        {[
          { id: "pw-old", label: s.pwOld, value: oldPw, set: setOldPw, ac: "current-password" },
          { id: "pw-new", label: s.pwNew, value: newPw, set: setNewPw, ac: "new-password" },
          { id: "pw-repeat", label: s.pwRepeat, value: repeat, set: setRepeat, ac: "new-password" },
        ].map((f) => (
          <div key={f.id} className="space-y-1.5">
            <label htmlFor={f.id} className="block text-sm font-medium text-foreground">{f.label}</label>
            <Input id={f.id} type="password" value={f.value} onChange={(e) => f.set(e.target.value)} autoComplete={f.ac} />
            {f.id === "pw-new" ? <p className="text-[13px] text-muted-foreground">{s.passwordHint}</p> : null}
          </div>
        ))}
        <div aria-live="polite">
          {error ? (
            <p role="alert" className="rounded-[10px] border border-urgent-border bg-urgent-bg px-3 py-2.5 text-sm text-urgent">{error}</p>
          ) : null}
          {done ? (
            <p role="status" className="rounded-[10px] border border-ok-border bg-ok-bg px-3 py-2.5 text-sm text-ok">{s.pwSuccess}</p>
          ) : null}
        </div>
        <Button type="submit" size="lg" className="w-full" disabled={busy}>
          {busy ? <><Loader2 className="animate-spin" aria-hidden="true" />{s.pwSaving}</> : s.pwSave}
        </Button>
      </form>
    </div>
  );
}
