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
  // Принимаем "OWL-3" или просто "3"
  if (trimmed.includes("-")) {
    const parts = trimmed.split("-");
    const num = parseInt(parts[parts.length - 1], 10);
    return isNaN(num) ? null : num;
  }
  const num = parseInt(trimmed, 10);
  return isNaN(num) ? null : num;
}

export default function LoginPage() {
  const router = useRouter();

  // Register state
  const [regName, setRegName] = useState("");
  const [regCity, setRegCity] = useState("");
  const [regPin, setRegPin] = useState("");
  const [regLoading, setRegLoading] = useState(false);
  const [regError, setRegError] = useState("");
  const [registeredCode, setRegisteredCode] = useState("");
  const [copied, setCopied] = useState(false);

  // Login state
  const [loginId, setLoginId] = useState("");
  const [loginPin, setLoginPin] = useState("");
  const [loginLoading, setLoginLoading] = useState(false);
  const [loginError, setLoginError] = useState("");

  const handleCopy = async () => {
    await navigator.clipboard.writeText(registeredCode);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleRegister = async () => {
    if (!regName.trim()) {
      setRegError("Введите ваше имя");
      return;
    }
    if (!regPin || regPin.length < 8 || !/(?=.*[A-Za-z])(?=.*\d)/.test(regPin)) {
      setRegError("Пароль: мин. 8 символов, буквы + цифры");
      return;
    }
    setRegError("");
    setRegLoading(true);
    try {
      const resp = await registerPatient(regName.trim(), regCity.trim(), regPin);
      const code = makeCode(resp.patient_id);
      // access_token is saved automatically by api.ts
      localStorage.setItem("patient_id", String(resp.patient_id));
      localStorage.setItem("patient_name", regName.trim());
      localStorage.setItem("pairing_code", code);
      setRegisteredCode(code);
    } catch {
      setRegError("Ошибка регистрации. Проверьте подключение к серверу.");
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
      setLoginError("Введите ваш код (например: OWL-3) или числовой ID");
      return;
    }
    if (!loginPin || loginPin.length < 1) {
      setLoginError("Введите пароль");
      return;
    }
    setLoginError("");
    setLoginLoading(true);
    try {
      const resp = await loginPatient(id, loginPin);
      if (resp.login) {
        // access_token is saved automatically by api.ts
        localStorage.setItem("patient_id", String(id));
        localStorage.setItem("pairing_code", makeCode(id));
        router.push("/chat");
      } else {
        setLoginError("Неверный код или пароль");
      }
    } catch (err: unknown) {
      const status = (err as { response?: { status?: number } })?.response?.status;
      if (status === 401) {
        setLoginError("Неверный код или пароль");
      } else if (status === 429) {
        setLoginError("Слишком много попыток. Подождите минуту.");
      } else {
        setLoginError("Ошибка подключения к серверу.");
      }
    } finally {
      setLoginLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-teal-50 flex items-center justify-center p-4">
      {/* Background decoration */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-40 -right-40 w-96 h-96 bg-blue-100 rounded-full opacity-30 blur-3xl" />
        <div className="absolute -bottom-40 -left-40 w-96 h-96 bg-teal-100 rounded-full opacity-30 blur-3xl" />
      </div>

      <div className="relative w-full max-w-md space-y-6">
        {/* Logo & Title */}
        <div className="text-center space-y-2">
          <div className="inline-flex items-center justify-center w-16 h-16 bg-gradient-to-br from-blue-600 to-teal-500 rounded-2xl shadow-lg shadow-blue-200">
            <HeartPulse className="h-8 w-8 text-white" />
          </div>
          <h1 className="text-2xl font-bold text-gray-900">TMS</h1>
          <p className="text-sm text-gray-500">
            Therapist Machine Support
          </p>
        </div>

        {/* Main Card */}
        <Card className="shadow-xl border-0 bg-white/80 backdrop-blur-sm">
          <CardHeader className="pb-2 pt-6">
            <div className="flex items-center gap-2 justify-center text-sm text-gray-500">
              <ShieldCheck className="h-4 w-4 text-emerald-500" />
              <span>Безопасный вход</span>
            </div>
          </CardHeader>
          <CardContent className="pt-2">
            <Tabs defaultValue="register" className="w-full">
              <TabsList className="grid w-full grid-cols-2 mb-6">
                <TabsTrigger
                  value="register"
                  className="text-sm data-[state=active]:bg-blue-600 data-[state=active]:text-white"
                >
                  <UserPlus className="h-4 w-4 mr-1.5" />
                  Новый пациент
                </TabsTrigger>
                <TabsTrigger
                  value="login"
                  className="text-sm data-[state=active]:bg-blue-600 data-[state=active]:text-white"
                >
                  <LogIn className="h-4 w-4 mr-1.5" />
                  Я уже был здесь
                </TabsTrigger>
              </TabsList>

              {/* Register Tab */}
              <TabsContent value="register" className="space-y-4">
                {registeredCode ? (
                  /* Success screen after registration */
                  <div className="space-y-4 text-center py-2">
                    <div className="w-14 h-14 bg-emerald-100 rounded-full flex items-center justify-center mx-auto">
                      <Check className="h-7 w-7 text-emerald-600" />
                    </div>
                    <div>
                      <p className="text-sm text-gray-600 mb-2">Регистрация прошла успешно!</p>
                      <p className="text-sm font-medium text-gray-700 mb-1">Ваш код для входа:</p>
                      <div className="flex items-center justify-center gap-2">
                        <div className="bg-gray-50 border-2 border-blue-200 rounded-xl px-6 py-3 font-mono text-2xl font-bold text-blue-700 tracking-wider">
                          {registeredCode}
                        </div>
                        <Button
                          variant="outline"
                          size="icon"
                          onClick={handleCopy}
                          className="shrink-0"
                        >
                          {copied ? (
                            <Check className="h-4 w-4 text-emerald-500" />
                          ) : (
                            <Copy className="h-4 w-4" />
                          )}
                        </Button>
                      </div>
                      <p className="text-xs text-red-500 mt-2 font-medium">
                        Запомните этот код и пароль! Они нужны для повторного входа.
                      </p>
                    </div>
                    <Button
                      className="w-full bg-gradient-to-r from-blue-600 to-teal-500 hover:from-blue-700 hover:to-teal-600 text-white shadow-lg shadow-blue-200"
                      onClick={handleGoToChat}
                    >
                      Перейти в чат
                    </Button>
                  </div>
                ) : (
                  /* Registration form */
                  <>
                    <div className="space-y-2">
                      <label className="text-sm font-medium text-gray-700">
                        Ваше имя
                      </label>
                      <Input
                        placeholder="Иван Петров"
                        value={regName}
                        onChange={(e) => setRegName(e.target.value)}
                      />
                    </div>

                    <div className="space-y-2">
                      <label className="text-sm font-medium text-gray-700">
                        Город
                      </label>
                      <Input
                        placeholder="Москва"
                        value={regCity}
                        onChange={(e) => setRegCity(e.target.value)}
                      />
                    </div>

                    <div className="space-y-2">
                      <label className="text-sm font-medium text-gray-700">
                        Пароль (мин. 8 символов, буквы + цифры)
                      </label>
                      <Input
                        type="password"
                        maxLength={64}
                        placeholder="Введите пароль"
                        value={regPin}
                        onChange={(e) =>
                          setRegPin(e.target.value.slice(0, 64))
                        }
                      />
                    </div>

                    {regError && (
                      <p className="text-sm text-red-500 text-center">{regError}</p>
                    )}

                    <Button
                      className="w-full bg-gradient-to-r from-blue-600 to-teal-500 hover:from-blue-700 hover:to-teal-600 text-white shadow-lg shadow-blue-200"
                      onClick={handleRegister}
                      disabled={regLoading}
                    >
                      {regLoading ? (
                        <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                      ) : (
                        <UserPlus className="h-4 w-4 mr-2" />
                      )}
                      Создать карту
                    </Button>
                  </>
                )}
              </TabsContent>

              {/* Login Tab */}
              <TabsContent value="login" className="space-y-4">
                <div className="space-y-2">
                  <label className="text-sm font-medium text-gray-700">
                    Ваш код
                  </label>
                  <Input
                    type="text"
                    placeholder="Например: OWL-3 или 3"
                    value={loginId}
                    onChange={(e) => setLoginId(e.target.value)}
                  />
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium text-gray-700">
                    Пароль
                  </label>
                  <Input
                    type="password"
                    maxLength={64}
                    placeholder="Введите пароль"
                    value={loginPin}
                    onChange={(e) =>
                      setLoginPin(e.target.value.slice(0, 64))
                    }
                  />
                </div>

                {loginError && (
                  <p className="text-sm text-red-500 text-center">
                    {loginError}
                  </p>
                )}

                <Button
                  className="w-full bg-gradient-to-r from-blue-600 to-teal-500 hover:from-blue-700 hover:to-teal-600 text-white shadow-lg shadow-blue-200"
                  onClick={handleLogin}
                  disabled={loginLoading}
                >
                  {loginLoading ? (
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  ) : (
                    <LogIn className="h-4 w-4 mr-2" />
                  )}
                  Войти
                </Button>
              </TabsContent>
            </Tabs>
          </CardContent>
        </Card>

        {/* Doctor Portal Link */}
        <div className="text-center">
          <button
            onClick={() => router.push("/doctor")}
            className="text-sm text-indigo-500 hover:text-indigo-700 underline underline-offset-2 transition-colors"
          >
            Вход для врачей
          </button>
        </div>

        {/* MVP Disclaimer */}
        <div className="bg-amber-50/80 border border-amber-200 rounded-xl px-4 py-3 text-center space-y-1">
          <p className="text-xs font-semibold text-amber-700">
            MVP — Демонстрационный прототип
          </p>
          <p className="text-xs text-amber-600">
            Все диагнозы носят предположительный характер и не являются медицинскими рекомендациями.
          </p>
        </div>

        {/* Footer */}
        <div className="text-center space-y-1">
          <p className="text-xs text-gray-400">
            Ваши данные защищены и хранятся в безопасности
          </p>
          <button
            onClick={() => router.push("/policy")}
            className="text-xs text-gray-400 hover:text-blue-500 underline underline-offset-2 transition-colors"
          >
            Политика конфиденциальности и Условия использования
          </button>
        </div>
      </div>
    </div>
  );
}
