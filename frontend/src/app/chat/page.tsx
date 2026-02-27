"use client";

import { useState, useRef, useEffect } from "react";
import { useRouter } from "next/navigation";
import {
  Send,
  HeartPulse,
  LogOut,
  History,
  X,
  FlaskConical,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ChatMessage } from "@/components/ChatMessage";
import { DiagnosisCard } from "@/components/DiagnosisCard";
import { DiagnosisSkeleton } from "@/components/DiagnosisSkeleton";
import { analyzeSymptoms } from "@/app/actions/analyzeSymptoms";
import { generateExplanation } from "@/app/actions/generateExplanation";
import { getSymptomLabel, type SymptomCode } from "@/lib/symptoms";
import { sendAnalysis, saveExplanation, getHistory, logout as apiLogout, type HistoryEntry } from "@/lib/api";
import { getDiseaseLabel } from "@/lib/diseaseWeights";
import type { DiagnosisResult } from "@/lib/types";
import { t, getLang, setLang, type Lang } from "@/lib/i18n";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: string;
  diagnosis?: DiagnosisResult;
}

function getTime(): string {
  return new Date().toLocaleTimeString("ru-RU", {
    hour: "2-digit",
    minute: "2-digit",
  });
}

const CHAT_STORAGE_PREFIX = "chat_messages_";

function getWelcomeMessage(lang: Lang = "ru"): Message {
  return {
    id: "welcome",
    role: "assistant",
    content: t("welcomeMessage", lang),
    timestamp: getTime(),
  };
}

function loadMessages(patientId: number, lang: Lang): Message[] {
  try {
    const raw = localStorage.getItem(`${CHAT_STORAGE_PREFIX}${patientId}`);
    if (raw) {
      const parsed = JSON.parse(raw) as Message[];
      if (Array.isArray(parsed) && parsed.length > 0) return parsed;
    }
  } catch {
    // corrupted data
  }
  return [getWelcomeMessage(lang)];
}

function saveMessages(patientId: number, msgs: Message[]): void {
  try {
    localStorage.setItem(`${CHAT_STORAGE_PREFIX}${patientId}`, JSON.stringify(msgs));
  } catch {
    // localStorage full
  }
}

// LLM (Groq) используется только для парсинга текста → вектор симптомов.
// Предикт делается на бэкенде через /analys (ML модель).

const LANG_CYCLE: Lang[] = ["ru", "en", "kk"];
const LANG_LABELS: Record<Lang, string> = { ru: "RU", en: "EN", kk: "KZ" };

export default function ChatPage() {
  const router = useRouter();
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const [messages, setMessages] = useState<Message[]>([getWelcomeMessage()]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [patientId, setPatientId] = useState<number | null>(null);
  const [showHistory, setShowHistory] = useState(false);
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [cooldown, setCooldown] = useState(0);
  const [lang, setLangState] = useState<Lang>("ru");

  useEffect(() => {
    const id = localStorage.getItem("patient_id");
    const token = localStorage.getItem("access_token");
    if (!id || !token) {
      router.push("/");
      return;
    }
    const numId = parseInt(id, 10);
    const currentLang = getLang();
    setPatientId(numId);
    setMessages(loadMessages(numId, currentLang));
    setLangState(currentLang);
  }, [router]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  useEffect(() => {
    if (patientId !== null && messages.length > 0) {
      saveMessages(patientId, messages);
    }
  }, [messages, patientId]);

  useEffect(() => {
    if (cooldown <= 0) return;
    const timer = setTimeout(() => setCooldown((c) => c - 1), 1000);
    return () => clearTimeout(timer);
  }, [cooldown]);

  const handleLogout = () => {
    if (patientId !== null) {
      localStorage.removeItem(`${CHAT_STORAGE_PREFIX}${patientId}`);
    }
    apiLogout();
    router.push("/");
  };

  const loadHistoryData = async () => {
    if (!patientId) return;
    setShowHistory(true);
    setHistoryLoading(true);
    try {
      const resp = await getHistory(patientId);
      setHistory(resp.history);
    } catch {
      setHistory([]);
    } finally {
      setHistoryLoading(false);
    }
  };

  const handleSend = async () => {
    const text = input.trim();
    if (!text || isLoading || cooldown > 0 || !patientId) return;

    const userMsg: Message = {
      id: Date.now().toString(),
      role: "user",
      content: text,
      timestamp: getTime(),
    };

    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setIsLoading(true);

    try {
      // Step 1: LLM extracts symptoms from text
      const symptomResult = await analyzeSymptoms(text);

      const { vector, detectedSymptoms, error, noSymptoms } = symptomResult;

      if (error) {
        setMessages((prev) => [
          ...prev,
          {
            id: Date.now().toString(),
            role: "assistant",
            content: `${t("errorAnalysis", lang)}: ${error}`,
            timestamp: getTime(),
          },
        ]);
        setIsLoading(false);
        return;
      }

      if (noSymptoms || detectedSymptoms.length === 0) {
        setMessages((prev) => [
          ...prev,
          {
            id: Date.now().toString(),
            role: "assistant",
            content: t("noSymptomsMessage", lang),
            timestamp: getTime(),
          },
        ]);
        setIsLoading(false);
        return;
      }

      // Step 2: Send symptoms to backend ML model for prediction + storage
      const analysisResult = await sendAnalysis(patientId, vector, "Nothing").catch(() => null);

      // Build DiagnosisResult from backend response
      const diagnosis: DiagnosisResult = analysisResult
        ? {
            diseaseName: analysisResult.diseaseName,
            diseaseLabel: getDiseaseLabel(analysisResult.diseaseName, lang),
            doctor: analysisResult.doctor,
            recommendation: analysisResult.recommendation,
            slices: analysisResult.slices.map((s) => ({
              ...s,
              label: getDiseaseLabel(s.name, lang),
            })),
          }
        : {
            diseaseName: "Unknown",
            diseaseLabel: lang === "en" ? "Could not determine" : lang === "kk" ? "Анықтау мүмкін болмады" : "Не удалось определить",
            doctor: lang === "en" ? "General Practitioner" : "Терапевт",
            recommendation: t("errorDiagnosis", lang),
            slices: [],
          };

      // Build symptom labels for display
      const symptomNames = detectedSymptoms
        .map((s) => {
          const match = s.match(/^(\w+)\s*\((\d+)\)$/);
          if (match) {
            const code = match[1] as SymptomCode;
            const sev = match[2];
            const label = getSymptomLabel(code, lang);
            return `${label} (${sev}/3)`;
          }
          return getSymptomLabel(s as SymptomCode, lang);
        })
        .join(", ");

      // Step 4: Generate AI explanation (why this diagnosis was chosen)
      const topDiseases = diagnosis.slices.map((s) => ({
        name: s.name,
        label: s.label,
        score: s.score,
      }));

      const { patientExplanation, doctorExplanation } = await generateExplanation(
        text,
        detectedSymptoms,
        diagnosis.diseaseName,
        diagnosis.diseaseLabel,
        topDiseases,
        undefined,
        lang,
      );

      diagnosis.patientExplanation = patientExplanation;
      diagnosis.doctorExplanation = doctorExplanation;

      // Step 4b: Save explanations to DB (fire-and-forget)
      if (analysisResult?.day) {
        saveExplanation(analysisResult.day, patientExplanation, doctorExplanation).catch(() => {});
      }

      // Step 5: Show results with explanation
      const assistantMsg: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: `${t("detectedSymptoms", lang)}: ${symptomNames}`,
        timestamp: getTime(),
        diagnosis,
      };

      setMessages((prev) => [...prev, assistantMsg]);
    } catch {
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now().toString(),
          role: "assistant",
          content: t("errorGeneral", lang),
          timestamp: getTime(),
        },
      ]);
    } finally {
      setIsLoading(false);
      setCooldown(3);
    }
  };

  const toggleLang = () => {
    const idx = LANG_CYCLE.indexOf(lang);
    const newLang = LANG_CYCLE[(idx + 1) % LANG_CYCLE.length];
    setLang(newLang);
    setLangState(newLang);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="flex flex-col h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 px-4 py-3 flex items-center justify-between shrink-0">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-gradient-to-br from-blue-600 to-teal-500 rounded-xl flex items-center justify-center shadow-md">
            <HeartPulse className="h-5 w-5 text-white" />
          </div>
          <div>
            <h1 className="font-semibold text-gray-900 text-sm">
              {t("chatTitle", lang)}
            </h1>
            <p className="text-xs text-emerald-500 flex items-center gap-1">
              <span className="w-1.5 h-1.5 bg-emerald-500 rounded-full inline-block" />
              {t("online", lang)}
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant="ghost"
            size="sm"
            onClick={toggleLang}
            className="text-xs font-semibold text-gray-500 px-2"
            title={t("changeLang", lang)}
          >
            {LANG_LABELS[lang]}
          </Button>
          <Button
            variant="ghost"
            size="icon"
            onClick={() => router.push("/lab")}
            title={t("myAnalyses", lang)}
          >
            <FlaskConical className="h-5 w-5 text-gray-500" />
          </Button>
          <Button
            variant="ghost"
            size="icon"
            onClick={loadHistoryData}
            title={t("history", lang)}
          >
            <History className="h-5 w-5 text-gray-500" />
          </Button>
          <Button
            variant="ghost"
            size="icon"
            onClick={handleLogout}
            title={t("logout", lang)}
          >
            <LogOut className="h-5 w-5 text-gray-500" />
          </Button>
        </div>
      </header>

      {/* History Sidebar */}
      {showHistory && (
        <div className="absolute inset-0 z-50 flex">
          <div
            className="absolute inset-0 bg-black/30"
            onClick={() => setShowHistory(false)}
          />
          <div className="relative ml-auto w-full max-w-sm bg-white h-full shadow-2xl flex flex-col">
            <div className="flex items-center justify-between p-4 border-b">
              <h2 className="font-semibold text-gray-900">
                {t("historyTitle", lang)}
              </h2>
              <Button
                variant="ghost"
                size="icon"
                onClick={() => setShowHistory(false)}
              >
                <X className="h-5 w-5" />
              </Button>
            </div>
            <div className="flex-1 overflow-y-auto p-4 space-y-3">
              {historyLoading ? (
                <p className="text-sm text-gray-500 text-center py-8">
                  {t("loading", lang)}
                </p>
              ) : history.length === 0 ? (
                <p className="text-sm text-gray-500 text-center py-8">
                  {t("noRecords", lang)}
                </p>
              ) : (
                history.map((entry) => (
                  <div
                    key={entry.day_id}
                    className="bg-gray-50 rounded-xl p-3 space-y-1 border border-gray-100"
                  >
                    <div className="flex justify-between items-center">
                      <span className="text-sm font-medium text-gray-900">
                        {getDiseaseLabel(entry.disease_predict, lang)}
                      </span>
                      <span className="text-xs text-gray-400">
                        {entry.created_at?.split("T")[0] || entry.created_at}
                      </span>
                    </div>
                    {entry.score !== null && (
                      <p className="text-xs text-gray-500">
                        {t("confidence", lang)}: {Math.round(entry.score * 100)}%
                      </p>
                    )}
                    {entry.recept && (
                      <p className="text-xs text-gray-600">{entry.recept}</p>
                    )}
                    {entry.doctor_name && (
                      <p className="text-xs text-blue-600">
                        {t("doctor", lang)}: {entry.doctor_name}
                      </p>
                    )}
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      )}

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto px-4 py-4 space-y-4">
        {messages.map((msg) => (
          <div key={msg.id} className="space-y-2">
            <ChatMessage
              role={msg.role}
              content={msg.content}
              timestamp={msg.timestamp}
            />
            {msg.diagnosis && (
              <div className="flex">
                <div className="ml-11">
                  <DiagnosisCard diagnosis={msg.diagnosis} lang={lang} />
                </div>
              </div>
            )}
          </div>
        ))}
        {isLoading && <DiagnosisSkeleton lang={lang} />}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="bg-white border-t border-gray-200 p-4 shrink-0">
        <div className="max-w-3xl mx-auto flex gap-2">
          <Input
            ref={inputRef}
            placeholder={t("inputPlaceholder", lang)}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={isLoading}
            className="flex-1 bg-gray-50 border-gray-200 focus:border-blue-400 focus:ring-blue-400"
          />
          <Button
            onClick={handleSend}
            disabled={isLoading || cooldown > 0 || !input.trim()}
            className="bg-blue-600 hover:bg-blue-700 text-white shadow-md px-4"
          >
            {cooldown > 0 ? (
              <span className="text-xs font-mono">{cooldown}</span>
            ) : (
              <Send className="h-4 w-4" />
            )}
          </Button>
        </div>
        <p className="text-center text-xs text-gray-400 mt-2">
          {t("mvpDisclaimer", lang)}
        </p>
      </div>
    </div>
  );
}
