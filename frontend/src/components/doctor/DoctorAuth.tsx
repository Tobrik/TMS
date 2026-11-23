"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { AlertCircle, Eye, EyeOff, Loader2, Play } from "lucide-react";
import type { Lang } from "@/lib/i18n";
import {
  apiErrorCode,
  getDemoStatus,
  loginDoctorByUsername,
  registerDoctor,
  startDemoDoctor,
  type DoctorAuthResponse,
  type DoctorProfile,
} from "@/lib/api";
import { da, type DoctorAuthStrings } from "@/lib/strings/doctorAuth";
import { Logo, LanguageSwitcher } from "@/components/brand";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";

export interface DoctorAuthProps {
  lang: Lang;
  /** Called after a successful password login, invite registration or demo login, once the
   *  session keys (access_token, doctor_id, doctor_name, doctor_specialty, doctor_is_demo) are stored. */
  onAuthenticated: (doctor: DoctorProfile) => void;
}

type Mode = "signin" | "join";

const PASSWORD_RE = /^(?=.*[A-Za-z])(?=.*\d).{8,64}$/; // same rule as the backend: a Latin letter and a digit
const USERNAME_RE = /^[a-z0-9._-]{3,32}$/;

function statusOf(err: unknown): number | null {
  const s = (err as { response?: { status?: number } })?.response?.status;
  return typeof s === "number" ? s : null;
}

/** Name of the field a 422 refers to, from FastAPI's `detail[].loc`. */
function invalidField(err: unknown): string | null {
  const detail = (err as { response?: { data?: { detail?: unknown } } })?.response?.data?.detail;
  if (!Array.isArray(detail)) return null;
  const loc = (detail[0] as { loc?: unknown[] } | undefined)?.loc;
  const f = Array.isArray(loc) ? loc[loc.length - 1] : null;
  return typeof f === "string" ? f : null;
}

function fieldMessage(field: string | null, s: DoctorAuthStrings): string {
  switch (field) {
    case "invite_code": return s.errFieldInvite;
    case "username": return s.errFieldUsername;
    case "full_name": return s.errFieldFullName;
    case "specialty": return s.errFieldSpecialty;
    case "password": return s.errFieldPassword;
    default: return s.errFieldRequired;
  }
}

function storeSession(res: DoctorAuthResponse): DoctorProfile {
  const d = res.doctor;
  try {
    localStorage.setItem("access_token", res.access_token);
    localStorage.setItem("doctor_id", String(d.doctor_id));
    localStorage.setItem("doctor_name", d.full_name);
    localStorage.setItem("doctor_specialty", d.specialty);
    localStorage.setItem("doctor_is_demo", d.is_demo ? "1" : "0");
  } catch {
    /* storage blocked: the dashboard still gets the profile */
  }
  return d;
}

function Field({
  id, label, hint, children,
}: { id: string; label: string; hint?: string; children: React.ReactNode }) {
  return (
    <div className="space-y-1.5">
      <label htmlFor={id} className="block text-sm font-medium text-foreground">{label}</label>
      {children}
      {hint ? <p id={`${id}-hint`} className="text-[13px] text-muted-foreground">{hint}</p> : null}
    </div>
  );
}

function PasswordInput({
  id, value, onChange, s, autoComplete, describedBy,
}: {
  id: string; value: string; onChange: (v: string) => void; s: DoctorAuthStrings;
  autoComplete: string; describedBy?: string;
}) {
  const [show, setShow] = useState(false);
  return (
    <div className="relative">
      <Input
        id={id}
        type={show ? "text" : "password"}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        autoComplete={autoComplete}
        aria-describedby={describedBy}
        className="pr-12"
      />
      <button
        type="button"
        onClick={() => setShow((v) => !v)}
        aria-label={show ? s.hidePassword : s.showPassword}
        aria-pressed={show}
        className="absolute right-0 top-0 flex h-11 w-11 items-center justify-center rounded-md text-muted-foreground outline-none transition-colors hover:text-foreground focus-visible:ring-[3px] focus-visible:ring-ring"
      >
        {show ? <EyeOff className="size-4" aria-hidden="true" /> : <Eye className="size-4" aria-hidden="true" />}
      </button>
    </div>
  );
}

export function DoctorAuth({ lang, onAuthenticated }: DoctorAuthProps) {
  const s = da(lang);
  const [mode, setMode] = useState<Mode>("signin");
  const [demoEnabled, setDemoEnabled] = useState(false);
  const [busy, setBusy] = useState<null | "form" | "demo">(null);
  const [error, setError] = useState<string | null>(null);

  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [code, setCode] = useState("");
  const [regUsername, setRegUsername] = useState("");
  const [fullName, setFullName] = useState("");
  const [specialty, setSpecialty] = useState("");
  const [regPassword, setRegPassword] = useState("");

  useEffect(() => {
    let alive = true;
    getDemoStatus().then((r) => { if (alive) setDemoEnabled(r.enabled); });
    return () => { alive = false; };
  }, []);

  const switchMode = (m: Mode) => { setMode(m); setError(null); };

  const mapError = (err: unknown, kind: "signin" | "join" | "demo"): string => {
    const status = statusOf(err);
    if (status === null) return s.errNetwork;
    if (status === 429) return s.errTooMany;
    const c = apiErrorCode(err);
    if (kind === "signin" && (status === 401 || c === "Invalid credentials")) return s.errWrongCredentials;
    if (kind === "join") {
      if (c === "invalid_invite") return s.errInvalidInvite;
      if (c === "username_taken") return s.errUsernameTaken;
      if (status === 422) return fieldMessage(invalidField(err), s);
    }
    if (kind === "signin" && status === 422) return s.errWrongCredentials;
    if (kind === "demo" && (status === 404 || c === "demo_disabled")) return s.errDemoUnavailable;
    return s.errGeneric;
  };

  const run = async (kind: "signin" | "join" | "demo", call: () => Promise<DoctorAuthResponse>) => {
    setBusy(kind === "demo" ? "demo" : "form");
    setError(null);
    try {
      const res = await call();
      onAuthenticated(storeSession(res));
    } catch (err) {
      setError(mapError(err, kind));
      setBusy(null);
    }
  };

  const submitSignIn = (e: React.FormEvent) => {
    e.preventDefault();
    if (busy) return;
    if (!username.trim() || !password) { setError(s.errWrongCredentials); return; }
    void run("signin", () => loginDoctorByUsername(username.trim().toLowerCase(), password));
  };

  const submitJoin = (e: React.FormEvent) => {
    e.preventDefault();
    if (busy) return;
    const normCode = code.replace(/[\s-]/g, "");
    const uname = regUsername.trim().toLowerCase();
    if (normCode.length < 4) return setError(s.errFieldInvite);
    if (!USERNAME_RE.test(uname)) return setError(s.errFieldUsername);
    if (!fullName.trim()) return setError(s.errFieldFullName);
    if (!specialty.trim()) return setError(s.errFieldSpecialty);
    if (!PASSWORD_RE.test(regPassword)) return setError(s.errFieldPassword);
    void run("join", () =>
      registerDoctor({
        invite_code: code.trim().toUpperCase(),
        username: uname,
        full_name: fullName.trim(),
        specialty: specialty.trim(),
        password: regPassword,
      })
    );
  };

  const tabs: { id: Mode; label: string }[] = [
    { id: "signin", label: s.tabSignIn },
    { id: "join", label: s.tabJoin },
  ];

  return (
    <div className="min-h-dvh bg-background">
      <div className="mx-auto flex min-h-dvh w-full max-w-md flex-col px-4 py-4 sm:py-6">
        <div className="flex items-center justify-end">
          <LanguageSwitcher lang={lang} />
        </div>

        <main className="flex flex-1 flex-col justify-center py-6">
          <div className="mb-6 flex flex-col items-center gap-3 text-center">
            <Logo size={40} />
            <h1 className="font-display text-2xl font-bold text-foreground sm:text-[28px]">{s.title}</h1>
            <p className="text-[15px] text-muted-foreground">{s.subtitle}</p>
          </div>

          <div className="rounded-[14px] border border-border bg-card p-5 shadow-card sm:p-6">
            <div role="tablist" aria-label={s.title} className="mb-5 grid grid-cols-2 gap-1 rounded-[10px] bg-muted p-1">
              {tabs.map((tab) => (
                <button
                  key={tab.id}
                  type="button"
                  role="tab"
                  id={`tab-${tab.id}`}
                  aria-selected={mode === tab.id}
                  aria-controls="doctor-auth-panel"
                  onClick={() => switchMode(tab.id)}
                  className={cn(
                    "min-h-10 rounded-[8px] px-2 text-sm font-medium outline-none transition-colors duration-150 focus-visible:ring-[3px] focus-visible:ring-ring",
                    mode === tab.id ? "bg-card text-foreground shadow-card" : "text-muted-foreground hover:text-foreground"
                  )}
                >
                  {tab.label}
                </button>
              ))}
            </div>

            <div id="doctor-auth-panel" role="tabpanel" aria-labelledby={`tab-${mode}`}>
              {mode === "signin" ? (
                <form onSubmit={submitSignIn} className="space-y-4" noValidate>
                  <Field id="da-username" label={s.username}>
                    <Input
                      id="da-username"
                      value={username}
                      onChange={(e) => setUsername(e.target.value)}
                      autoComplete="username"
                      autoCapitalize="none"
                      autoCorrect="off"
                      spellCheck={false}
                    />
                  </Field>
                  <Field id="da-password" label={s.password}>
                    <PasswordInput id="da-password" value={password} onChange={setPassword} s={s} autoComplete="current-password" />
                  </Field>
                  {error ? <ErrorLine text={error} /> : null}
                  <Button type="submit" size="lg" className="w-full" disabled={busy !== null}>
                    {busy === "form" ? <><Loader2 className="animate-spin" aria-hidden="true" />{s.signingIn}</> : s.signIn}
                  </Button>
                </form>
              ) : (
                <form onSubmit={submitJoin} className="space-y-4" noValidate>
                  <Field id="da-code" label={s.inviteCode} hint={s.inviteCodeHint}>
                    <Input
                      id="da-code"
                      value={code}
                      onChange={(e) => setCode(e.target.value.toUpperCase())}
                      className="font-mono uppercase tracking-wider"
                      autoCapitalize="characters"
                      autoComplete="off"
                      spellCheck={false}
                      aria-describedby="da-code-hint"
                    />
                  </Field>
                  <Field id="da-reg-username" label={s.username} hint={s.usernameHint}>
                    <Input
                      id="da-reg-username"
                      value={regUsername}
                      onChange={(e) => setRegUsername(e.target.value)}
                      autoComplete="username"
                      autoCapitalize="none"
                      autoCorrect="off"
                      spellCheck={false}
                      aria-describedby="da-reg-username-hint"
                    />
                  </Field>
                  <Field id="da-fullname" label={s.fullName}>
                    <Input id="da-fullname" value={fullName} onChange={(e) => setFullName(e.target.value)} autoComplete="name" />
                  </Field>
                  <Field id="da-specialty" label={s.specialty}>
                    <Input id="da-specialty" value={specialty} onChange={(e) => setSpecialty(e.target.value)} autoComplete="organization-title" />
                  </Field>
                  <Field id="da-reg-password" label={s.password} hint={s.passwordHint}>
                    <PasswordInput
                      id="da-reg-password"
                      value={regPassword}
                      onChange={setRegPassword}
                      s={s}
                      autoComplete="new-password"
                      describedBy="da-reg-password-hint"
                    />
                  </Field>
                  {error ? <ErrorLine text={error} /> : null}
                  <Button type="submit" size="lg" className="w-full" disabled={busy !== null}>
                    {busy === "form" ? <><Loader2 className="animate-spin" aria-hidden="true" />{s.joining}</> : s.join}
                  </Button>
                </form>
              )}
            </div>
          </div>

          {demoEnabled ? (
            <section aria-labelledby="da-demo-title" className="mt-4 rounded-[14px] border border-dashed border-input bg-muted/60 p-5">
              <h2 id="da-demo-title" className="text-base font-semibold text-foreground">{s.demoTitle}</h2>
              <p className="mt-1 text-sm text-muted-foreground">{s.demoText}</p>
              <Button
                type="button"
                variant="outline"
                size="lg"
                className="mt-4 w-full border-primary text-primary hover:bg-accent"
                disabled={busy !== null}
                onClick={() => void run("demo", startDemoDoctor)}
              >
                {busy === "demo" ? <><Loader2 className="animate-spin" aria-hidden="true" />{s.demoLoading}</> : <><Play aria-hidden="true" />{s.demoButton}</>}
              </Button>
            </section>
          ) : null}

          <p className="mt-6 text-center">
            <Link href="/" className="inline-flex min-h-11 items-center text-sm font-medium text-primary underline-offset-4 hover:underline focus-visible:ring-[3px] focus-visible:ring-ring outline-none rounded-md px-1">
              {s.backToPatient}
            </Link>
          </p>
        </main>
      </div>
    </div>
  );
}

function ErrorLine({ text }: { text: string }) {
  return (
    <div role="alert" className="flex items-start gap-2 rounded-[10px] border border-urgent-border bg-urgent-bg px-3 py-2.5 text-sm text-urgent">
      <AlertCircle className="mt-0.5 size-4 shrink-0" aria-hidden="true" />
      <span>{text}</span>
    </div>
  );
}
