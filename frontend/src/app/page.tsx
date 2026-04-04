"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import {
  HeartPulse,
  UserPlus,
  LogIn,
  Loader2,
  Copy,
  Check,
  ShieldCheck,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { registerPatient, loginPatient } from "@/lib/api";
import { t, getLang, setLang, type Lang } from "@/lib/i18n";

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

const LANG_CYCLE: Lang[] = ["ru", "en", "kk"];
const LANG_LABELS: Record<Lang, string> = { ru: "RU", en: "EN", kk: "KZ" };

export default function LoginPage() {
  const router = useRouter();
  const [lang, setLangState] = useState<Lang>(getLang());

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

  const toggleLang = () => {
    const idx = LANG_CYCLE.indexOf(lang);
    const newLang = LANG_CYCLE[(idx + 1) % LANG_CYCLE.length];
    setLang(newLang);
    setLangState(newLang);
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

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-teal-50 flex items-center justify-center p-4">
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-40 -right-40 w-96 h-96 bg-blue-100 rounded-full opacity-30 blur-3xl" />
        <div className="absolute -bottom-40 -left-40 w-96 h-96 bg-teal-100 rounded-full opacity-30 blur-3xl" />
      </div>

      <div className="relative w-full max-w-md space-y-6">
        <div className="flex justify-end">
          <Button variant="ghost" size="sm" onClick={toggleLang} className="text-xs font-semibold text-gray-500 px-2">
            {LANG_LABELS[lang]}
          </Button>
        </div>

        <div className="text-center space-y-2">
          <div className="inline-flex items-center justify-center w-16 h-16 bg-gradient-to-br from-blue-600 to-teal-500 rounded-2xl shadow-lg shadow-blue-200">
            <HeartPulse className="h-8 w-8 text-white" />
          </div>
          <h1 className="text-2xl font-bold text-gray-900">{t("appTitle", lang)}</h1>
          <p className="text-sm text-gray-500">{t("appSubtitle", lang)}</p>
        </div>

        <Card className="shadow-xl border-0 bg-white/80 backdrop-blur-sm">
          <CardHeader className="pb-2 pt-6">
            <div className="flex items-center gap-2 justify-center text-sm text-gray-500">
              <ShieldCheck className="h-4 w-4 text-emerald-500" />
              <span>{t("secureLogin", lang)}</span>
            </div>
          </CardHeader>
          <CardContent className="pt-2">
            <Tabs defaultValue="register" className="w-full">
              <TabsList className="grid w-full grid-cols-2 mb-6">
                <TabsTrigger value="register" className="text-sm data-[state=active]:bg-blue-600 data-[state=active]:text-white">
                  <UserPlus className="h-4 w-4 mr-1.5" />
                  {t("newPatient", lang)}
                </TabsTrigger>
                <TabsTrigger value="login" className="text-sm data-[state=active]:bg-blue-600 data-[state=active]:text-white">
                  <LogIn className="h-4 w-4 mr-1.5" />
                  {t("returningPatient", lang)}
                </TabsTrigger>
              </TabsList>

              <TabsContent value="register" className="space-y-4">
                {registeredCode ? (
                  <div className="space-y-4 text-center py-2">
                    <div className="w-14 h-14 bg-emerald-100 rounded-full flex items-center justify-center mx-auto">
                      <Check className="h-7 w-7 text-emerald-600" />
                    </div>
                    <div>
                      <p className="text-sm text-gray-600 mb-2">{t("registrationSuccess", lang)}</p>
                      <p className="text-sm font-medium text-gray-700 mb-1">{t("yourLoginCode", lang)}</p>
                      <div className="flex items-center justify-center gap-2">
                        <div className="bg-gray-50 border-2 border-blue-200 rounded-xl px-6 py-3 font-mono text-2xl font-bold text-blue-700 tracking-wider">
                          {registeredCode}
                        </div>
                        <Button variant="outline" size="icon" onClick={handleCopy} className="shrink-0">
                          {copied ? <Check className="h-4 w-4 text-emerald-500" /> : <Copy className="h-4 w-4" />}
                        </Button>
                      </div>
                      <p className="text-xs text-red-500 mt-2 font-medium">{t("rememberCode", lang)}</p>
                    </div>
                    <Button className="w-full bg-gradient-to-r from-blue-600 to-teal-500 hover:from-blue-700 hover:to-teal-600 text-white shadow-lg shadow-blue-200" onClick={handleGoToChat}>
                      {t("goToChat", lang)}
                    </Button>
                  </div>
                ) : (
                  <>
                    <div className="space-y-2">
                      <label className="text-sm font-medium text-gray-700">{t("yourName", lang)}</label>
                      <Input placeholder={t("namePlaceholder", lang)} value={regName} onChange={(e) => setRegName(e.target.value)} />
                    </div>
                    <div className="space-y-2">
                      <label className="text-sm font-medium text-gray-700">{t("city", lang)}</label>
                      <Input placeholder={t("cityPlaceholder", lang)} value={regCity} onChange={(e) => setRegCity(e.target.value)} />
                    </div>
                    <div className="space-y-2">
                      <label className="text-sm font-medium text-gray-700">{t("passwordLabel", lang)}</label>
                      <Input type="password" maxLength={64} placeholder={t("passwordPlaceholder", lang)} value={regPin} onChange={(e) => setRegPin(e.target.value.slice(0, 64))} />
                    </div>
                    {regError && <p className="text-sm text-red-500 text-center">{regError}</p>}
                    <Button className="w-full bg-gradient-to-r from-blue-600 to-teal-500 hover:from-blue-700 hover:to-teal-600 text-white shadow-lg shadow-blue-200" onClick={handleRegister} disabled={regLoading}>
                      {regLoading ? <Loader2 className="h-4 w-4 mr-2 animate-spin" /> : <UserPlus className="h-4 w-4 mr-2" />}
                      {t("createCard", lang)}
                    </Button>
                  </>
                )}
              </TabsContent>

              <TabsContent value="login" className="space-y-4">
                <div className="space-y-2">
                  <label className="text-sm font-medium text-gray-700">{t("yourCode", lang)}</label>
                  <Input type="text" placeholder={t("codePlaceholder", lang)} value={loginId} onChange={(e) => setLoginId(e.target.value)} />
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium text-gray-700">{t("password", lang)}</label>
                  <Input type="password" maxLength={64} placeholder={t("passwordPlaceholder", lang)} value={loginPin} onChange={(e) => setLoginPin(e.target.value.slice(0, 64))} />
                </div>
                {loginError && <p className="text-sm text-red-500 text-center">{loginError}</p>}
                <Button className="w-full bg-gradient-to-r from-blue-600 to-teal-500 hover:from-blue-700 hover:to-teal-600 text-white shadow-lg shadow-blue-200" onClick={handleLogin} disabled={loginLoading}>
                  {loginLoading ? <Loader2 className="h-4 w-4 mr-2 animate-spin" /> : <LogIn className="h-4 w-4 mr-2" />}
                  {t("loginBtn", lang)}
                </Button>
              </TabsContent>
            </Tabs>
          </CardContent>
        </Card>

        <div className="text-center">
          <button onClick={() => router.push("/doctor")} className="text-sm text-indigo-500 hover:text-indigo-700 underline underline-offset-2 transition-colors">
            {t("doctorPortalLink", lang)}
          </button>
        </div>

        <div className="bg-amber-50/80 border border-amber-200 rounded-xl px-4 py-3 text-center space-y-1">
          <p className="text-xs font-semibold text-amber-700">{t("mvpTitle", lang)}</p>
          <p className="text-xs text-amber-600">{t("mvpDescription", lang)}</p>
        </div>

        <div className="text-center space-y-1">
          <p className="text-xs text-gray-400">{t("dataProtected", lang)}</p>
          <button onClick={() => router.push("/policy")} className="text-xs text-gray-400 hover:text-blue-500 underline underline-offset-2 transition-colors">
            {t("privacyPolicy", lang)}
          </button>
        </div>
      </div>
    </div>
  );
}
