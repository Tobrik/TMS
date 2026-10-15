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
  Check,
  Minus,
  HelpCircle,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ChatMessage } from "@/components/ChatMessage";
import { DiagnosisCard } from "@/components/DiagnosisCard";
import { DiagnosisSkeleton } from "@/components/DiagnosisSkeleton";
import {
  extractSymptoms,
  sendAnalysis,
  explainDiagnosis,
  getHistory,
  getNextQuestions,
  logout as apiLogout,
  type HistoryEntry,
  type ClarifyingQuestion,
  type CriticalAlert,
  type Finding,
} from "@/lib/api";
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
  if (typeof window === "undefined") return "";
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
      if (Array.isArray(parsed) && parsed.length > 0) {
        // An explanation still loading when the page was closed will never arrive
        return parsed.map((m) =>
          m.diagnosis?.explanationPending ? { ...m, diagnosis: { ...m.diagnosis, explanationPending: false } } : m
        );
      }
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

// Flow: free text → /extract_symptoms (LLM, quoted findings) → patient confirms
// what was understood → up to 2 rounds of yes/no questions (/next_questions)
// → /analys (model + triage) → /explain (filled in when ready).

const LANG_CYCLE: Lang[] = ["ru", "en", "kk"];
const LANG_LABELS: Record<Lang, string> = { ru: "RU", en: "EN", kk: "KZ" };
const MAX_ROUNDS = 2;

type FindingState = "present" | "absent" | "unknown";
const NEXT_STATE: Record<FindingState, FindingState> = { present: "absent", absent: "unknown", unknown: "present" };

interface Session {
  intakeId: number | null;
  age: number | null;
  sex: "M" | "F" | null;
  evidences: string[];
  absent: string[];
  asked: string[]; // every question shown, including "don't know"
  round: number;
}

function questionText(q: ClarifyingQuestion, lang: Lang): string {
  return lang === "en" ? q.question_en : lang === "kk" ? q.question_kk : q.question_ru;
}

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

  // Current complaint: confirmation step, then clarifying questions
  const [session, setSession] = useState<Session | null>(null);
  const [findings, setFindings] = useState<Finding[]>([]);
  const [findingStates, setFindingStates] = useState<Record<string, FindingState>>({});
  const [clarifyingQuestions, setClarifyingQuestions] = useState<ClarifyingQuestion[]>([]);
  const [roundAnswers, setRoundAnswers] = useState<{ yes: string[]; no: string[] }>({ yes: [], no: [] });
  const [alertShown, setAlertShown] = useState(false);

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
  }, [messages, isLoading, findings, clarifyingQuestions]);

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

  const pushAssistant = (content: string, diagnosis?: DiagnosisResult): string => {
    const id = `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
    setMessages((prev) => [...prev, { id, role: "assistant", content, timestamp: getTime(), diagnosis }]);
    return id;
  };

  const pushUser = (content: string) => {
    setMessages((prev) => [...prev, { id: `${Date.now()}-u`, role: "user", content, timestamp: getTime() }]);
  };

  const showAlert = (alert: CriticalAlert | null, alreadyShown: boolean): boolean => {
    if (!alert || alreadyShown) return alreadyShown;
    pushAssistant(alert.message);
    setAlertShown(true);
    return true;
  };

  const resetComplaint = () => {
    setSession(null);
    setFindings([]);
    setFindingStates({});
    setClarifyingQuestions([]);
    setRoundAnswers({ yes: [], no: [] });
  };

  const handleSend = async () => {
    const text = input.trim();
    if (!text || isLoading || cooldown > 0 || !patientId) return;

    resetComplaint();
    setAlertShown(false);
    pushUser(text);
    setInput("");
    setIsLoading(true);

    try {
      // Step 1: backend extracts evidence tokens, each with the patient's own words
      const extracted = await extractSymptoms(text);

      if (extracted.error) {
        pushAssistant(`${t("errorAnalysis", lang)}: ${extracted.error}`);
        return;
      }
      showAlert(extracted.critical_alert, false);

      if (extracted.noSymptoms || extracted.findings.length === 0) {
        pushAssistant(t("noSymptomsMessage", lang));
        return;
      }

      // Step 2: show what was understood; the patient corrects it before we go on
      setSession({
        intakeId: extracted.intake_id,
        age: extracted.age,
        sex: extracted.sex,
        evidences: [],
        absent: [],
        asked: [],
        round: 1,
      });
      setFindings(extracted.findings);
      // "uncertain" findings (Groq: models disagreed; Clef: confidence 0.25–0.8)
      // start as "?": patients tend to accept defaults, so only sure ones are pre-ticked.
      setFindingStates(
        Object.fromEntries(
          extracted.findings.map((f) => [f.code, f.status === "uncertain" ? "unknown" : f.status])
        ) as Record<string, FindingState>
      );
      pushAssistant(t("confirmTitle", lang));
    } catch {
      pushAssistant(t("errorGeneral", lang));
    } finally {
      setIsLoading(false);
      setCooldown(3);
    }
  };

  const confirmFindings = async () => {
    if (!session) return;
    const evidences = findings.filter((f) => findingStates[f.code] === "present").map((f) => f.code);
    const absent = findings.filter((f) => findingStates[f.code] === "absent").map((f) => f.code);
    setFindings([]);
    if (evidences.length === 0) {
      pushAssistant(t("nothingConfirmed", lang));
      resetComplaint();
      return;
    }
    const next = { ...session, evidences, absent };
    setSession(next);
    await askRound(next, 1);
  };

  // Step 3: up to MAX_ROUNDS rounds of questions chosen by expected information gain
  const askRound = async (current: Session, round: number) => {
    setIsLoading(true);
    const resp = await getNextQuestions(
      current.evidences, current.absent, current.age, current.sex, round, 3
    ).catch(() => null);
    setIsLoading(false);

    showAlert(resp?.critical_alert ?? null, alertShown);
    const questions = (resp?.questions ?? []).filter((q) => !current.asked.includes(q.evidence_id));
    if (!resp || resp.should_stop || questions.length === 0) {
      await finalizeDiagnosis(current);
      return;
    }
    setSession({ ...current, round, asked: [...current.asked, ...questions.map((q) => q.evidence_id)] });
    setRoundAnswers({ yes: [], no: [] });
    setClarifyingQuestions(questions);
    if (round === 1) pushAssistant(t("clarifyIntro", lang));
  };

  const handleClarifyAnswer = async (question: ClarifyingQuestion, answer: "yes" | "no" | "unknown") => {
    if (!session) return;
    const answerLabel = answer === "yes" ? t("yes", lang) : answer === "no" ? t("no", lang) : t("dontKnow", lang);
    pushUser(`${questionText(question, lang)} — ${answerLabel}`);

    // "No" is information too: it goes back to the backend as `absent`
    const yes = answer === "yes" ? [...roundAnswers.yes, question.evidence_id] : roundAnswers.yes;
    const no = answer === "no" ? [...roundAnswers.no, question.evidence_id] : roundAnswers.no;
    const remaining = clarifyingQuestions.filter((q) => q.evidence_id !== question.evidence_id);
    setRoundAnswers({ yes, no });
    setClarifyingQuestions(remaining);
    if (remaining.length > 0) return;

    const next = { ...session, evidences: [...session.evidences, ...yes], absent: [...session.absent, ...no] };
    setSession(next);
    if (session.round < MAX_ROUNDS) {
      await askRound(next, session.round + 1);
    } else {
      await finalizeDiagnosis(next);
    }
  };

  // Step 4: model + triage, then the explanation fills in when it's ready
  const finalizeDiagnosis = async (current: Session) => {
    setIsLoading(true);
    setClarifyingQuestions([]);
    try {
      const result = await sendAnalysis(
        current.evidences, current.absent, current.age, current.sex, current.intakeId
      ).catch(() => null);
      if (!result) {
        pushAssistant(t("errorDiagnosis", lang));
        return;
      }
      const diagnosis: DiagnosisResult = {
        diseaseName: result.diseaseName,
        diseaseLabel: getDiseaseLabel(result.diseaseName, lang),
        doctor: result.doctor,
        recommendation: result.recommendation,
        slices: result.slices.map((s) => ({ ...s, label: getDiseaseLabel(s.name, lang) })),
        zone: result.zone,
        reasons: result.reasons,
        outOfScope: result.out_of_scope,
        explanationPending: true,
      };
      const count = current.evidences.length;
      const summary = lang === "en"
        ? `${count} symptoms detected`
        : lang === "kk" ? `${count} белгі анықталды` : `Обнаружено признаков: ${count}`;
      const messageId = pushAssistant(`${t("detectedSymptoms", lang)}: ${summary}`, diagnosis);

      const setExplanation = (patch: Partial<DiagnosisResult>) =>
        setMessages((prev) => prev.map((m) =>
          m.id === messageId && m.diagnosis ? { ...m, diagnosis: { ...m.diagnosis, ...patch } } : m
        ));
      explainDiagnosis(result.day, lang)
        .then((e) => setExplanation({
          patientExplanation: e.patientExplanation,
          doctorExplanation: e.doctorExplanation,
          explanationPending: false,
        }))
        .catch(() => setExplanation({ explanationPending: false }));
    } finally {
      setIsLoading(false);
      setCooldown(3);
      resetComplaint();
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
        {findings.length > 0 && (
          <div className="ml-11 bg-white border border-blue-100 rounded-xl p-3 shadow-sm space-y-2 max-w-[90%]">
            <div className="flex flex-wrap gap-2">
              {findings.map((f) => {
                const state = findingStates[f.code] ?? "unknown";
                const style = state === "present"
                  ? "bg-blue-50 border-blue-300 text-blue-800"
                  : state === "absent"
                  ? "bg-gray-50 border-gray-300 text-gray-500 line-through decoration-gray-400"
                  : "bg-amber-50 border-amber-300 text-amber-800";
                const Icon = state === "present" ? Check : state === "absent" ? Minus : HelpCircle;
                const stateLabel = state === "present"
                  ? t("statusPresent", lang)
                  : state === "absent" ? t("statusAbsent", lang) : t("statusUnknown", lang);
                return (
                  <button
                    key={f.code}
                    onClick={() => setFindingStates((prev) => ({ ...prev, [f.code]: NEXT_STATE[state] }))}
                    className={`text-left text-xs border rounded-lg px-2.5 py-1.5 transition-colors ${style}`}
                    title={stateLabel}
                  >
                    <span className="flex items-center gap-1.5 font-medium">
                      <Icon className="h-3.5 w-3.5 shrink-0" />
                      {f.label[lang]}
                    </span>
                    {f.quote && (
                      <span className="block text-[11px] opacity-70 mt-0.5">
                        {t("youSaid", lang)}: «{f.quote}»
                      </span>
                    )}
                  </button>
                );
              })}
            </div>
            <Button
              size="sm"
              className="bg-blue-600 hover:bg-blue-700 text-white"
              onClick={confirmFindings}
              disabled={isLoading}
            >
              {t("confirmBtn", lang)}
            </Button>
          </div>
        )}
        {clarifyingQuestions.length > 0 && (
          <div className="flex flex-col gap-3 ml-11">
            {clarifyingQuestions.map((q) => (
              <div key={q.evidence_id} className="bg-white border border-blue-100 rounded-xl p-3 shadow-sm">
                <p className="text-sm text-gray-800 mb-2">{questionText(q, lang)}</p>
                <div className="flex gap-2">
                  <Button
                    size="sm"
                    className="bg-blue-600 hover:bg-blue-700 text-white px-4"
                    onClick={() => handleClarifyAnswer(q, "yes")}
                  >
                    {t("yes", lang)}
                  </Button>
                  <Button
                    size="sm"
                    variant="outline"
                    className="border-gray-300 text-gray-600 px-4"
                    onClick={() => handleClarifyAnswer(q, "no")}
                  >
                    {t("no", lang)}
                  </Button>
                  <Button
                    size="sm"
                    variant="ghost"
                    className="text-gray-400 px-3"
                    onClick={() => handleClarifyAnswer(q, "unknown")}
                  >
                    {t("dontKnow", lang)}
                  </Button>
                </div>
              </div>
            ))}
          </div>
        )}
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
