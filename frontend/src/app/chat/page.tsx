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
  HelpCircle,
  type LucideIcon,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ChatMessage } from "@/components/ChatMessage";
import { DiagnosisCard, type DiagnosisCardData } from "@/components/DiagnosisCard";
import { DiagnosisSkeleton } from "@/components/DiagnosisSkeleton";
import {
  ALERT_CODES,
  extractSymptoms,
  sendAnalysis,
  explainDiagnosis,
  getEvidenceCatalog,
  getHistory,
  getNextQuestions,
  logout as apiLogout,
  type AnalysResponse,
  type EvidenceCatalogEntry,
  type HistoryEntry,
  type ClarifyingQuestion,
  type CriticalAlert,
  type Finding,
  type NextQuestionsResponse,
} from "@/lib/api";
import { getDiseaseLabel } from "@/lib/diseaseWeights";
import { t, getLang, setLang, type Lang, type TranslationKey } from "@/lib/i18n";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: string;
  diagnosis?: DiagnosisCardData;
}

function getTime(): string {
  if (typeof window === "undefined") return "";
  return new Date().toLocaleTimeString("ru-RU", {
    hour: "2-digit",
    minute: "2-digit",
  });
}

const CHAT_STORAGE_PREFIX = "chat_messages_";

// The timestamp is passed in: getTime() differs on server and client, so it is never read during render
function getWelcomeMessage(lang: Lang = "ru", timestamp: string = ""): Message {
  return {
    id: "welcome",
    role: "assistant",
    content: t("welcomeMessage", lang),
    timestamp,
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
  return [getWelcomeMessage(lang, getTime())];
}

function saveMessages(patientId: number, msgs: Message[]): void {
  try {
    localStorage.setItem(`${CHAT_STORAGE_PREFIX}${patientId}`, JSON.stringify(msgs));
  } catch {
    // localStorage full
  }
}

// Flow: free text → /extract_symptoms (LLM, quoted findings) → patient confirms
// what was understood (and can add missed symptoms) → up to 2 rounds of yes/no
// questions (/next_questions) → /analys (model + triage) → /explain (filled in when ready).

const LANG_CYCLE: Lang[] = ["ru", "en", "kk"];
const LANG_LABELS: Record<Lang, string> = { ru: "RU", en: "EN", kk: "KZ" };
const MAX_ROUNDS = 2;
const MAX_SUGGESTIONS = 8;

type FindingState = "present" | "absent" | "unknown";

// Three explicit choices per finding; "unknown" is "not sure" and counts as "don't know"
const FINDING_OPTIONS: { state: FindingState; labelKey: TranslationKey; Icon: LucideIcon }[] = [
  { state: "present", labelKey: "chipYes", Icon: Check },
  { state: "absent", labelKey: "chipNo", Icon: X },
  { state: "unknown", labelKey: "chipUnsure", Icon: HelpCircle },
];

const OPTION_SELECTED: Record<FindingState, string> = {
  present: "bg-blue-600 border-blue-600 text-white font-semibold",
  absent: "bg-gray-700 border-gray-700 text-white font-semibold",
  unknown: "bg-amber-100 border-amber-500 text-amber-900 font-semibold",
};
const OPTION_IDLE = "bg-white border-gray-300 text-gray-600 hover:bg-gray-50";

function initialFindingState(f: Finding): FindingState {
  // "uncertain" (Groq: models disagreed; Clef: confidence 0.25–0.8) starts as "not sure":
  // patients tend to accept defaults, so only sure findings are pre-selected.
  if (f.status === "present") return "present";
  if (f.status === "absent") return "absent";
  return "unknown";
}

function initialStates(findings: Finding[]): Record<string, FindingState> {
  return Object.fromEntries(
    findings.map((f): [string, FindingState] => [f.code, initialFindingState(f)])
  );
}

interface Session {
  intakeId: number | null;
  age: number | null;
  sex: "M" | "F" | null;
  evidences: string[];
  absent: string[];
  unknown: string[]; // "don't know" / "not sure" answers: excluded from later questions
  asked: string[]; // every question shown, including "don't know"
  round: number;
}

// A step that failed on the network; the session is kept so the patient can retry
type Failure =
  | { kind: "questions"; session: Session; round: number }
  | { kind: "analysis"; session: Session };

function questionText(q: ClarifyingQuestion, lang: Lang): string {
  return lang === "en" ? q.question_en : lang === "kk" ? q.question_kk : q.question_ru;
}

// The backend sends Russian text; with a known code we show the translated one
function alertText(alert: CriticalAlert, lang: Lang): string {
  if (alert.code && (ALERT_CODES as readonly string[]).includes(alert.code)) {
    return t(`alert_${alert.code}` as TranslationKey, lang);
  }
  return alert.message;
}

// Kazakh users often type Russian, so kk also matches the Russian label
function catalogMatches(entry: EvidenceCatalogEntry, query: string, lang: Lang): boolean {
  const q = query.trim().toLowerCase();
  const labels = lang === "kk" ? [entry.label.kk, entry.label.ru] : [entry.label[lang]];
  return labels.some((label) => label.toLowerCase().includes(q));
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
  const [roundAnswers, setRoundAnswers] = useState<{ yes: string[]; no: string[]; unknown: string[] }>({
    yes: [],
    no: [],
    unknown: [],
  });
  const [alertShown, setAlertShown] = useState(false);
  const [failure, setFailure] = useState<Failure | null>(null);

  // Symptom catalog for "add a symptom"; fetched once per session, hidden if it fails
  const [catalog, setCatalog] = useState<EvidenceCatalogEntry[]>([]);
  const [addOpen, setAddOpen] = useState(false);
  const [addQuery, setAddQuery] = useState("");

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
    document.documentElement.lang = currentLang;
  }, [router]);

  useEffect(() => {
    if (patientId === null) return;
    let cancelled = false;
    getEvidenceCatalog()
      .then((resp) => {
        if (!cancelled) setCatalog(resp.evidences);
      })
      .catch(() => {
        // No catalog: the "add a symptom" button stays hidden, the rest of the flow works
      });
    return () => {
      cancelled = true;
    };
  }, [patientId]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading, findings, clarifyingQuestions, failure]);

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

  const pushAssistant = (content: string, diagnosis?: DiagnosisCardData): string => {
    const id = `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
    setMessages((prev) => [...prev, { id, role: "assistant", content, timestamp: getTime(), diagnosis }]);
    return id;
  };

  const pushUser = (content: string) => {
    setMessages((prev) => [...prev, { id: `${Date.now()}-u`, role: "user", content, timestamp: getTime() }]);
  };

  const showAlert = (alert: CriticalAlert | null, alreadyShown: boolean): boolean => {
    if (!alert || alreadyShown) return alreadyShown;
    pushAssistant(alertText(alert, lang));
    setAlertShown(true);
    return true;
  };

  const setFindingState = (code: string, state: FindingState) => {
    setFindingStates((prev) => ({ ...prev, [code]: state }));
  };

  const resetComplaint = () => {
    setSession(null);
    setFindings([]);
    setFindingStates({});
    setClarifyingQuestions([]);
    setRoundAnswers({ yes: [], no: [], unknown: [] });
    setFailure(null);
    setAddOpen(false);
    setAddQuery("");
  };

  const handleSend = async () => {
    const text = input.trim();
    if (!text || isLoading || cooldown > 0 || !patientId) return;

    // While the patient is still confirming symptoms, more text adds to that list
    // instead of wiping it
    const addingToConfirmation = session !== null && findings.length > 0;
    if (!addingToConfirmation) {
      resetComplaint();
      setAlertShown(false);
    }
    pushUser(text);
    setInput("");
    setIsLoading(true);

    try {
      // Step 1: backend extracts evidence tokens, each with the patient's own words
      const extracted = await extractSymptoms(text);

      if (extracted.error) {
        pushAssistant(`${t("errorAnalysis", lang)}: ${t(extracted.error, lang)}`);
        return;
      }
      showAlert(extracted.critical_alert, addingToConfirmation && alertShown);

      if (extracted.noSymptoms || extracted.findings.length === 0) {
        pushAssistant(t("noSymptomsMessage", lang));
        return;
      }

      if (addingToConfirmation) {
        const known = new Set(findings.map((f) => f.code));
        const fresh = extracted.findings.filter((f) => !known.has(f.code));
        setFindings((prev) => [...prev, ...fresh]);
        setFindingStates((prev) => ({ ...prev, ...initialStates(fresh) }));
        return;
      }

      // Step 2: show what was understood; the patient corrects it before we go on
      setSession({
        intakeId: extracted.intake_id,
        age: extracted.age,
        sex: extracted.sex,
        evidences: [],
        absent: [],
        unknown: [],
        asked: [],
        round: 1,
      });
      setFindings(extracted.findings);
      setFindingStates(initialStates(extracted.findings));
      pushAssistant(t("confirmTitle", lang));
    } catch {
      pushAssistant(t("errorGeneral", lang));
    } finally {
      setIsLoading(false);
      setCooldown(3);
    }
  };

  const closeAddSymptom = () => {
    setAddOpen(false);
    setAddQuery("");
  };

  const addFromCatalog = (entry: EvidenceCatalogEntry) => {
    // Added by the patient: "present", no quote
    setFindings((prev) => [
      ...prev,
      {
        code: entry.code,
        status: "present",
        quote: "",
        votes: 0,
        source: "manual",
        confidence: null,
        label: entry.label,
      },
    ]);
    setFindingState(entry.code, "present");
    closeAddSymptom();
  };

  const confirmFindings = async () => {
    if (!session) return;
    const codesIn = (state: FindingState) =>
      findings.filter((f) => (findingStates[f.code] ?? "unknown") === state).map((f) => f.code);
    const evidences = codesIn("present");
    const absent = codesIn("absent");
    const unknown = codesIn("unknown");
    setFindings([]);
    closeAddSymptom();
    if (evidences.length === 0) {
      pushAssistant(t("nothingConfirmed", lang));
      resetComplaint();
      return;
    }
    const next = { ...session, evidences, absent, unknown };
    setSession(next);
    await askRound(next, 1);
  };

  // Step 3: up to MAX_ROUNDS rounds of questions chosen by expected information gain
  const askRound = async (current: Session, round: number) => {
    setFailure(null);
    setIsLoading(true);
    let resp: NextQuestionsResponse | null = null;
    try {
      resp = await getNextQuestions(
        current.evidences, current.absent, current.age, current.sex, round, 3, current.intakeId, current.unknown
      );
    } catch {
      resp = null;
    }
    setIsLoading(false);

    if (!resp) {
      // Keep the session: the patient can retry this round or skip to the result
      setFailure({ kind: "questions", session: current, round });
      pushAssistant(t("questionsFailed", lang));
      return;
    }

    showAlert(resp.critical_alert, alertShown);
    const questions = (resp.questions ?? []).filter((q) => !current.asked.includes(q.evidence_id));
    if (resp.should_stop || questions.length === 0) {
      await finalizeDiagnosis(current);
      return;
    }
    setSession({ ...current, round, asked: [...current.asked, ...questions.map((q) => q.evidence_id)] });
    setRoundAnswers({ yes: [], no: [], unknown: [] });
    setClarifyingQuestions(questions);
    if (round === 1) pushAssistant(t("clarifyIntro", lang));
  };

  const handleClarifyAnswer = async (question: ClarifyingQuestion, answer: "yes" | "no" | "unknown") => {
    if (!session) return;
    const answerLabel = answer === "yes" ? t("yes", lang) : answer === "no" ? t("no", lang) : t("dontKnow", lang);
    pushUser(`${questionText(question, lang)} — ${answerLabel}`);

    // "No" is information too: it goes back to the backend as `absent`;
    // "don't know" goes as `unknown` (not evidence, only excluded from later questions)
    const id = question.evidence_id;
    const yes = answer === "yes" ? [...roundAnswers.yes, id] : roundAnswers.yes;
    const no = answer === "no" ? [...roundAnswers.no, id] : roundAnswers.no;
    const unknown = answer === "unknown" ? [...roundAnswers.unknown, id] : roundAnswers.unknown;
    const remaining = clarifyingQuestions.filter((q) => q.evidence_id !== id);
    setRoundAnswers({ yes, no, unknown });
    setClarifyingQuestions(remaining);
    if (remaining.length > 0) return;

    const next: Session = {
      ...session,
      evidences: [...session.evidences, ...yes],
      absent: [...session.absent, ...no],
      unknown: [...session.unknown, ...unknown],
    };
    setSession(next);
    if (session.round < MAX_ROUNDS) {
      await askRound(next, session.round + 1);
    } else {
      await finalizeDiagnosis(next);
    }
  };

  // Step 4: model + triage, then the explanation fills in when it's ready
  const finalizeDiagnosis = async (current: Session) => {
    setFailure(null);
    setIsLoading(true);
    setClarifyingQuestions([]);
    let result: AnalysResponse | null = null;
    try {
      result = await sendAnalysis(
        current.evidences, current.absent, current.age, current.sex, current.intakeId, "Nothing", current.unknown
      );
    } catch {
      result = null;
    }
    setIsLoading(false);
    setCooldown(3);

    if (!result) {
      // Keep the session and every answer; nothing is reset on an error
      setFailure({ kind: "analysis", session: current });
      pushAssistant(t("analysisFailed", lang));
      return;
    }

    const diagnosis: DiagnosisCardData = {
      diseaseName: result.diseaseName,
      diseaseLabel: getDiseaseLabel(result.diseaseName, lang),
      doctor: result.doctor,
      recommendation: result.recommendation,
      slices: result.slices.map((s) => ({ ...s, label: getDiseaseLabel(s.name, lang) })),
      zone: result.zone,
      reasons: result.reasons,
      outOfScope: result.out_of_scope,
      lowConfidence: result.low_confidence ?? false,
      explanationPending: true,
    };
    const count = current.evidences.length;
    const summary = lang === "en"
      ? `${count} symptoms detected`
      : lang === "kk" ? `${count} белгі анықталды` : `Обнаружено признаков: ${count}`;
    const messageId = pushAssistant(`${t("detectedSymptoms", lang)}: ${summary}`, diagnosis);

    const setExplanation = (patch: Partial<DiagnosisCardData>) =>
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
    resetComplaint();
  };

  const retryFailed = () => {
    if (!failure) return;
    if (failure.kind === "questions") {
      askRound(failure.session, failure.round);
    } else {
      finalizeDiagnosis(failure.session);
    }
  };

  const skipToResult = () => {
    if (failure?.kind === "questions") {
      finalizeDiagnosis(failure.session);
    }
  };

  const toggleLang = () => {
    const idx = LANG_CYCLE.indexOf(lang);
    const newLang = LANG_CYCLE[(idx + 1) % LANG_CYCLE.length];
    setLang(newLang);
    setLangState(newLang);
  };

  // Clarifying questions are on screen: the text input is locked until they are answered
  const answeringQuestions = clarifyingQuestions.length > 0;

  // Deliberate reset of the current complaint (the only way to start over mid-questions)
  const startNewComplaint = () => {
    resetComplaint();
    setAlertShown(false);
    setInput("");
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const findingCodes = new Set(findings.map((f) => f.code));
  const suggestions = addQuery.trim()
    ? catalog
        .filter((e) => !findingCodes.has(e.code) && catalogMatches(e, addQuery, lang))
        .slice(0, MAX_SUGGESTIONS)
    : [];

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
          <div className="ml-11 bg-white border border-blue-100 rounded-xl p-3 shadow-sm space-y-3 max-w-[90%]">
            <div className="space-y-2">
              {findings.map((f) => {
                const state = findingStates[f.code] ?? "unknown";
                return (
                  <div key={f.code} className="border border-gray-100 rounded-lg p-2.5 space-y-2">
                    <div>
                      <p className="text-sm font-medium text-gray-800">{f.label[lang]}</p>
                      {f.quote && (
                        <p className="text-[11px] text-gray-500 mt-0.5">
                          {t("youSaid", lang)}: «{f.quote}»
                        </p>
                      )}
                    </div>
                    <div role="group" aria-label={f.label[lang]} className="flex flex-wrap gap-1.5">
                      {FINDING_OPTIONS.map(({ state: option, labelKey, Icon }) => {
                        const selected = state === option;
                        const OptionIcon = selected ? Check : Icon;
                        return (
                          <button
                            key={option}
                            type="button"
                            aria-pressed={selected}
                            onClick={() => setFindingState(f.code, option)}
                            className={`inline-flex items-center gap-1 text-xs border rounded-lg px-2.5 py-1.5 transition-colors ${
                              selected ? OPTION_SELECTED[option] : OPTION_IDLE
                            }`}
                          >
                            <OptionIcon className="h-3.5 w-3.5 shrink-0" aria-hidden="true" />
                            {t(labelKey, lang)}
                          </button>
                        );
                      })}
                    </div>
                  </div>
                );
              })}
            </div>

            {catalog.length > 0 && (
              addOpen ? (
                <div className="space-y-2">
                  <Input
                    autoFocus
                    value={addQuery}
                    onChange={(e) => setAddQuery(e.target.value)}
                    placeholder={t("addSymptomSearch", lang)}
                    aria-label={t("addSymptom", lang)}
                    className="bg-gray-50 border-gray-200 focus:border-blue-400 focus:ring-blue-400"
                  />
                  {addQuery.trim() && (
                    suggestions.length > 0 ? (
                      <ul className="flex flex-col gap-1">
                        {suggestions.map((e) => (
                          <li key={e.code}>
                            <button
                              type="button"
                              onClick={() => addFromCatalog(e)}
                              className="w-full text-left text-xs border border-gray-200 rounded-lg px-2.5 py-1.5 hover:bg-blue-50 hover:border-blue-300 transition-colors"
                            >
                              {e.label[lang]}
                            </button>
                          </li>
                        ))}
                      </ul>
                    ) : (
                      <p className="text-xs text-gray-500">{t("addSymptomNoMatch", lang)}</p>
                    )
                  )}
                  <Button variant="ghost" size="sm" className="text-gray-500" onClick={closeAddSymptom}>
                    {t("cancel", lang)}
                  </Button>
                </div>
              ) : (
                <Button
                  variant="outline"
                  size="sm"
                  className="border-blue-300 text-blue-700"
                  onClick={() => setAddOpen(true)}
                  disabled={isLoading}
                >
                  {t("addSymptom", lang)}
                </Button>
              )
            )}

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
        {failure && !isLoading && (
          <div className="flex flex-wrap gap-2 ml-11">
            <Button
              size="sm"
              className="bg-blue-600 hover:bg-blue-700 text-white px-4"
              onClick={retryFailed}
            >
              {t("retry", lang)}
            </Button>
            {failure.kind === "questions" && (
              <Button
                size="sm"
                variant="outline"
                className="border-gray-300 text-gray-600"
                onClick={skipToResult}
              >
                {t("skipToResult", lang)}
              </Button>
            )}
          </div>
        )}
        {isLoading && <DiagnosisSkeleton lang={lang} />}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="bg-white border-t border-gray-200 p-4 shrink-0">
        {answeringQuestions && (
          <div className="max-w-3xl mx-auto mb-2">
            <Button
              variant="link"
              size="sm"
              className="h-auto p-0 text-xs text-blue-600"
              onClick={startNewComplaint}
            >
              {t("newComplaint", lang)}
            </Button>
          </div>
        )}
        <div className="max-w-3xl mx-auto flex gap-2">
          <Input
            ref={inputRef}
            placeholder={answeringQuestions ? t("answerQuestionsFirst", lang) : t("inputPlaceholder", lang)}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={isLoading || answeringQuestions}
            className="flex-1 bg-gray-50 border-gray-200 focus:border-blue-400 focus:ring-blue-400"
          />
          <Button
            onClick={handleSend}
            disabled={isLoading || answeringQuestions || cooldown > 0 || !input.trim()}
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
