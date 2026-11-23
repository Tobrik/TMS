"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import {
  UserPlus,
  LogIn,
  Loader2,
  Copy,
  Check,
  MessageSquareText,
  ListChecks,
  ClipboardList,
  PlayCircle,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { registerPatient, loginPatient, getDemoStatus, startDemoPatient } from "@/lib/api";
import { t, getLang, type Lang } from "@/lib/i18n";
import { AppHeader, LanguageSwitcher } from "@/components/brand";

const ANIMALS = [
  "LION", "BEAR", "WOLF", "HAWK", "DEER", "FOX", "OWL", "LYNX",
  "PIKE", "CROW", "SEAL", "HARE", "SWAN", "DOVE", "COLT",
];

function makeCode(patientId: number): string {
  const animal = ANIMALS[patientId % ANIMALS.length];
  return `${animal}-${patientId}`;
}

function parseIdFromCode(input: string): number | null {
  const trimmed = input.trim();
  if (trimmed.includes("-")) {
    const parts = trimmed.split("-");
    const num = parseInt(parts[parts.length - 1], 10);
    return isNaN(num) ? null : num;
  }
  const num = parseInt(trimmed, 10);
  return isNaN(num) ? null : num;
}

const GUEST_NAME: Record<Lang, string> = { ru: "Гость", kk: "Қонақ", en: "Guest" };

export default function LoginPage() {
  const router = useRouter();
  // "ru" on the server and first render; the stored language is read after mount
  const [lang, setLangState] = useState<Lang>("ru");

  const [demoEnabled, setDemoEnabled] = useState(false);
  const [demoHighlight, setDemoHighlight] = useState(false);
  const [demoLoading, setDemoLoading] = useState(false);
  const [demoError, setDemoError] = useState("");

  useEffect(() => {
    const currentLang = getLang();
    setLangState(currentLang);
    document.documentElement.lang = currentLang;
    setDemoHighlight(new URLSearchParams(window.location.search).get("demo") === "1");
    let alive = true;
    getDemoStatus().then((s) => {
      if (alive) setDemoEnabled(s.enabled);
    });
    return () => {
      alive = false;
    };
  }, []);

  const [regName, setRegName] = useState("");
  const [regCity, setRegCity] = useState("");
  const [regPin, setRegPin] = useState("");
  const [regLoading, setRegLoading] = useState(false);
  const [regError, setRegError] = useState("");
  const [registeredCode, setRegisteredCode] = useState("");
  const [copied, setCopied] = useState(false);

  const [loginId, setLoginId] = useState("");
  const [loginPin, setLoginPin] = useState("");
  const [loginLoading, setLoginLoading] = useState(false);
  const [loginError, setLoginError] = useState("");

  const handleLangChange = (next: Lang) => {
    setLangState(next);
    document.documentElement.lang = next;
  };

  const handleDemo = async () => {
    setDemoError("");
    setDemoLoading(true);
    try {
      const resp = await startDemoPatient(undefined, lang);
      localStorage.setItem("patient_id", String(resp.patient_id));
      localStorage.setItem("patient_name", GUEST_NAME[lang]);
      localStorage.setItem("pairing_code", makeCode(resp.patient_id));
      localStorage.setItem("patient_is_demo", "1");
      router.push("/chat");
    } catch {
      setDemoError(t("errorDemo", lang));
      setDemoLoading(false);
    }
  };

  const handleCopy = async () => {
    await navigator.clipboard.writeText(registeredCode);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleRegister = async () => {
    if (!regName.trim()) {
      setRegError(t("errorEnterName", lang));
      return;
    }
    if (!regPin || regPin.length < 8 || !/(?=.*[A-Za-z])(?=.*\d)/.test(regPin)) {
      setRegError(t("errorPassword", lang));
      return;
    }
    setRegError("");
    setRegLoading(true);
    try {
      const resp = await registerPatient(regName.trim(), regCity.trim(), regPin);
      const code = makeCode(resp.patient_id);
      localStorage.setItem("patient_id", String(resp.patient_id));
      localStorage.setItem("patient_name", regName.trim());
      localStorage.setItem("pairing_code", code);
      localStorage.removeItem("patient_is_demo");
      setRegisteredCode(code);
    } catch {
      setRegError(t("errorRegistration", lang));
    } finally {
      setRegLoading(false);
    }
  };

  const handleGoToChat = () => {
    router.push("/chat");
  };

  const handleLogin = async () => {
    const id = parseIdFromCode(loginId);
    if (!id || id <= 0) {
      setLoginError(t("errorEnterCode", lang));
      return;
    }
    if (!loginPin || loginPin.length < 1) {
      setLoginError(t("errorEnterPassword", lang));
      return;
    }
    setLoginError("");
    setLoginLoading(true);
    try {
      const resp = await loginPatient(id, loginPin);
      if (resp.login) {
        localStorage.setItem("patient_id", String(id));
        localStorage.setItem("pairing_code", makeCode(id));
        localStorage.removeItem("patient_is_demo");
        router.push("/chat");
      } else {
        setLoginError(t("errorWrongCredentials", lang));
      }
    } catch (err: unknown) {
      const status = (err as { response?: { status?: number } })?.response?.status;
      if (status === 401) {
        setLoginError(t("errorWrongCredentials", lang));
      } else if (status === 429) {
        setLoginError(t("errorTooManyAttempts", lang));
      } else {
        setLoginError(t("errorConnection", lang));
      }
    } finally {
      setLoginLoading(false);
    }
  };

  const steps = [
    { icon: MessageSquareText, title: t("step1Title", lang), text: t("step1Text", lang) },
    { icon: ListChecks, title: t("step2Title", lang), text: t("step2Text", lang) },
    { icon: ClipboardList, title: t("step3Title", lang), text: t("step3Text", lang) },
  ];

  const demoButton = (variant: "default" | "secondary", full = true) => (
    <Button
      type="button"
      variant={variant}
      size="lg"
      className={full ? "w-full" : undefined}
      onClick={handleDemo}
      disabled={demoLoading}
    >
      {demoLoading ? <Loader2 className="animate-spin" /> : <PlayCircle />}
      {demoLoading ? t("demoStarting", lang) : t("tryDemo", lang)}
    </Button>
  );

  const label = "text-sm font-medium text-foreground";
  const errorText = "text-sm text-urgent";

  return (
    <div className="min-h-screen bg-background">
      <AppHeader
        innerClassName="max-w-6xl"
        actions={
          <>
            <Link
              href="/doctor"
              className="inline-flex min-h-11 items-center rounded-md px-1 text-sm font-medium text-primary underline-offset-4 hover:underline focus-visible:outline-none focus-visible:ring-[3px] focus-visible:ring-ring"
            >
              {t("forDoctors", lang)}
            </Link>
            <LanguageSwitcher lang={lang} onChange={handleLangChange} />
          </>
        }
      />

      <main className="mx-auto grid w-full max-w-6xl gap-x-16 gap-y-8 px-4 py-8 sm:px-6 lg:grid-cols-[1.2fr_1fr] lg:grid-rows-[auto_1fr] lg:items-start lg:py-16">
        <section className="lg:col-start-1 lg:row-start-1">
          <p className="mb-3 text-[13px] font-semibold uppercase tracking-wider text-primary">
            {t("heroEyebrow", lang)}
          </p>
          <h1 className="font-display text-[28px] font-semibold leading-[1.15] text-foreground sm:text-4xl lg:text-[44px]">
            {t("heroTitle", lang)}
          </h1>
          <p className="mt-4 max-w-xl text-base text-muted-foreground lg:text-lg">
            {t("heroText", lang)}
          </p>
        </section>

        <section className="min-w-0 space-y-4 lg:col-start-2 lg:row-span-2 lg:row-start-1">
          {demoEnabled && demoHighlight && (
            <div className="rounded-xl border border-primary/30 bg-accent p-5 shadow-card">
              <p className="text-lg font-semibold text-accent-foreground">{t("demoPanelTitle", lang)}</p>
              <p className="mt-1 mb-4 text-[15px] text-accent-foreground/80">{t("demoPanelText", lang)}</p>
              {demoButton("default")}
              {demoError && <p className={`${errorText} mt-3`} role="alert">{demoError}</p>}
            </div>
          )}

          <Card>
            <CardContent>
              <Tabs defaultValue="register" className="w-full">
                <TabsList className="mb-5 grid w-full grid-cols-2">
                  <TabsTrigger value="register" className="text-sm">
                    <UserPlus />
                    {t("newPatient", lang)}
                  </TabsTrigger>
                  <TabsTrigger value="login" className="text-sm">
                    <LogIn />
                    {t("haveCode", lang)}
                  </TabsTrigger>
                </TabsList>

                <TabsContent value="register" className="space-y-4">
                  {registeredCode ? (
                    <div className="space-y-4 py-1 text-center">
                      <div className="mx-auto flex size-12 items-center justify-center rounded-full bg-ok-bg text-ok">
                        <Check className="size-6" />
                      </div>
                      <div>
                        <p className="mb-3 text-[15px] text-muted-foreground">{t("registrationSuccess", lang)}</p>
                        <p className="mb-2 text-sm font-medium text-foreground">{t("yourLoginCode", lang)}</p>
                        <div className="flex items-center justify-center gap-2">
                          <div className="rounded-xl border border-primary/30 bg-accent px-4 py-3 font-mono text-2xl font-bold tracking-wider text-accent-foreground sm:px-6">
                            {registeredCode}
                          </div>
                          <Button variant="outline" size="icon-lg" onClick={handleCopy} className="shrink-0" aria-label={t("copyCode", lang)}>
                            {copied ? <Check className="text-ok" /> : <Copy />}
                          </Button>
                        </div>
                        <p className="mt-3 rounded-lg border border-warn-border bg-warn-bg px-3 py-2 text-[13px] font-medium text-warn">
                          {t("rememberCode", lang)}
                        </p>
                      </div>
                      <Button size="lg" className="w-full" onClick={handleGoToChat}>
                        {t("goToChat", lang)}
                      </Button>
                    </div>
                  ) : (
                    <>
                      <div className="space-y-1.5">
                        <label htmlFor="reg-name" className={label}>{t("yourName", lang)}</label>
                        <Input id="reg-name" autoComplete="name" placeholder={t("namePlaceholder", lang)} value={regName} onChange={(e) => setRegName(e.target.value)} />
                      </div>
                      <div className="space-y-1.5">
                        <label htmlFor="reg-city" className={label}>{t("city", lang)}</label>
                        <Input id="reg-city" autoComplete="address-level2" placeholder={t("cityPlaceholder", lang)} value={regCity} onChange={(e) => setRegCity(e.target.value)} />
                      </div>
                      <div className="space-y-1.5">
                        <label htmlFor="reg-pin" className={label}>{t("passwordLabel", lang)}</label>
                        <Input id="reg-pin" type="password" autoComplete="new-password" maxLength={64} placeholder={t("passwordPlaceholder", lang)} value={regPin} onChange={(e) => setRegPin(e.target.value.slice(0, 64))} />
                      </div>
                      {regError && <p className={errorText} role="alert">{regError}</p>}
                      <Button size="lg" className="w-full" onClick={handleRegister} disabled={regLoading}>
                        {regLoading ? <Loader2 className="animate-spin" /> : <UserPlus />}
                        {t("createCard", lang)}
                      </Button>
                    </>
                  )}
                </TabsContent>

                <TabsContent value="login" className="space-y-4">
                  <div className="space-y-1.5">
                    <label htmlFor="login-code" className={label}>{t("yourCode", lang)}</label>
                    <Input id="login-code" type="text" autoComplete="username" placeholder={t("codePlaceholder", lang)} value={loginId} onChange={(e) => setLoginId(e.target.value)} />
                  </div>
                  <div className="space-y-1.5">
                    <label htmlFor="login-pin" className={label}>{t("password", lang)}</label>
                    <Input id="login-pin" type="password" autoComplete="current-password" maxLength={64} placeholder={t("passwordPlaceholder", lang)} value={loginPin} onChange={(e) => setLoginPin(e.target.value.slice(0, 64))} />
                  </div>
                  {loginError && <p className={errorText} role="alert">{loginError}</p>}
                  <Button size="lg" className="w-full" onClick={handleLogin} disabled={loginLoading}>
                    {loginLoading ? <Loader2 className="animate-spin" /> : <LogIn />}
                    {t("loginBtn", lang)}
                  </Button>
                </TabsContent>
              </Tabs>

              {demoEnabled && !demoHighlight && (
                <div className="mt-5 border-t border-border pt-5">
                  {demoButton("secondary")}
                  {demoError && <p className={`${errorText} mt-3`} role="alert">{demoError}</p>}
                </div>
              )}
            </CardContent>
          </Card>

          <p className="px-1 text-center text-[13px] text-muted-foreground">
            {t("noDiagnosisNote", lang)} {t("mvpDescription", lang)}
          </p>
          <p className="px-1 text-center text-[13px] text-muted-foreground">
            <Link href="/policy" className="underline underline-offset-2 hover:text-primary focus-visible:outline-none focus-visible:ring-[3px] focus-visible:ring-ring rounded">
              {t("privacyPolicy", lang)}
            </Link>
          </p>
        </section>

        <section className="lg:col-start-1 lg:row-start-2" aria-labelledby="how-title">
          <h2 id="how-title" className="sr-only">{t("howItWorks", lang)}</h2>
          <ol className="space-y-5">
            {steps.map((s, i) => (
              <li key={i} className="flex items-start gap-4">
                <span className="mt-0.5 inline-flex size-10 shrink-0 items-center justify-center rounded-full bg-accent text-accent-foreground">
                  <s.icon className="size-5" aria-hidden="true" />
                </span>
                <div>
                  <p className="font-semibold text-foreground">{i + 1}. {s.title}</p>
                  <p className="text-[15px] text-muted-foreground">{s.text}</p>
                </div>
              </li>
            ))}
          </ol>
        </section>
      </main>
    </div>
  );
}
