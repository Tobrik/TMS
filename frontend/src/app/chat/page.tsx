"use client";

import { useState, useRef, useEffect } from "react";
import { useRouter } from "next/navigation";
import {
  Send,
  LogOut,
  History,
  X,
  FlaskConical,
  Check,
  HelpCircle,
  Plus,
  Loader2,
  type LucideIcon,
} from "lucide-react";
import { AppHeader, LanguageSwitcher, Logo } from "@/components/brand";
import { cn } from "@/lib/utils";
import { tc } from "@/lib/strings/chat";
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
import { t, getLang, type Lang, type TranslationKey } from "@/lib/i18n";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: string;
  diagnosis?: DiagnosisCardData;
  alert?: boolean;
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

const MAX_ROUNDS = 2;
const MAX_SUGGESTIONS = 8;

type FindingState = "present" | "absent" | "unknown";

// Three explicit choices per finding; "unknown" is "not sure" and counts as "don't know"
const FINDING_OPTIONS: { state: FindingState; labelKey: TranslationKey; Icon: LucideIcon }[] = [
  { state: "present", labelKey: "chipYes", Icon: Check },
  { state: "absent", labelKey: "chipNo", Icon: X },
  { state: "unknown", labelKey: "chipUnsure", Icon: HelpCircle },
];

// Selected states differ by fill AND icon, never colour alone:
// yes = solid primary + check, no = muted + struck-through label + cross, don't know = outlined + question mark
const OPTION_SELECTED: Record<FindingState, string> = {
  present: "border-primary bg-primary text-primary-foreground font-semibold",
  absent: "border-foreground/30 bg-muted text-foreground font-semibold line-through decoration-2",
  unknown: "border-2 border-primary bg-card text-primary font-semibold",
};
const OPTION_IDLE = "border-input bg-card text-foreground/80 hover:bg-accent hover:text-accent-foreground";

function AnswerGroup({
  label,
  selected,
  onPick,
  options,
}: {
  label: string;
  selected: FindingState | null;
  onPick: (state: FindingState) => void;
  options: { state: FindingState; text: string; Icon: LucideIcon }[];
}) {
  return (
    <div role="group" aria-label={label} className="grid grid-cols-3 gap-2">
      {options.map(({ state, text, Icon }) => {
        const isSelected = selected === state;
        return (
          <button
            key={state}
            type="button"
            aria-pressed={isSelected}
            onClick={() => onPick(state)}
            className={cn(
              "inline-flex min-h-11 items-center justify-center gap-1.5 rounded-[10px] border px-2 text-sm outline-none transition-colors duration-150 focus-visible:ring-[3px] focus-visible:ring-ring",
              isSelected ? OPTION_SELECTED[state] : OPTION_IDLE
            )}
          >
            {isSelected && <Icon className="size-4 shrink-0" aria-hidden="true" />}
            <span className="min-w-0 py-1.5 text-center leading-tight">{text}</span>
          </button>
        );
      })}
    </div>
  );
}

function HeaderAction({
  label,
  onClick,
  children,
}: {
  label: string;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <div className="group relative">
      <button
        type="button"
        onClick={onClick}
        aria-label={label}
        className="inline-flex size-10 items-center justify-center rounded-[10px] text-muted-foreground outline-none transition-colors duration-150 hover:bg-muted hover:text-foreground focus-visible:ring-[3px] focus-visible:ring-ring"
      >
        {children}
      </button>
      <span
        role="tooltip"
        className="pointer-events-none absolute right-0 top-full z-40 mt-1 hidden whitespace-nowrap rounded-md bg-foreground px-2 py-1 text-xs text-background group-hover:block group-focus-within:block"
      >
        {label}
      </span>
    </div>
  );
}

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
  const [isDemo, setIsDemo] = useState(false);

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
    setIsDemo(localStorage.getItem("patient_is_demo") === "1");
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

  const pushAssistant = (content: string, diagnosis?: DiagnosisCardData, alert?: boolean): string => {
    const id = `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
    setMessages((prev) => [...prev, { id, role: "assistant", content, timestamp: getTime(), diagnosis, alert }]);
    return id;
  };

  const pushUser = (content: string) => {
    setMessages((prev) => [...prev, { id: `${Date.now()}-u`, role: "user", content, timestamp: getTime() }]);
  };

  const showAlert = (alert: CriticalAlert | null, alreadyShown: boolean): boolean => {
    if (!alert || alreadyShown) return alreadyShown;
    pushAssistant(alertText(alert, lang).replace(/^\u26A0\uFE0F?\s*/, ""), undefined, true);
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
    const messageId = pushAssistant(`${t("detectedSymptoms", lang)}: ${current.evidences.length}`, diagnosis);

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

  const onLangChange = (newLang: Lang) => {
    setLangState(newLang);
    document.documentElement.lang = newLang;
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


  const findingOptions = FINDING_OPTIONS.map(({ state, labelKey, Icon }) => ({ state, text: t(labelKey, lang), Icon }));
  const questionOptions: { state: FindingState; text: string; Icon: LucideIcon }[] = [
    { state: "present", text: t("yes", lang), Icon: Check },
    { state: "absent", text: t("no", lang), Icon: X },
    { state: "unknown", text: t("dontKnow", lang), Icon: HelpCircle },
  ];
  const toAnswer = (state: FindingState): "yes" | "no" | "unknown" =>
    state === "present" ? "yes" : state === "absent" ? "no" : "unknown";

  const cardClass = "rounded-xl border border-border bg-card p-3.5 shadow-card sm:p-4";
  // Everything the assistant offers (findings, questions, retry) lines up with the bubbles' text, past the avatar
  const indent = "sm:ml-[38px]";

  return (
    <div className="relative flex h-dvh flex-col overflow-hidden bg-background">
      <AppHeader
        innerClassName="max-w-[720px]"
        logoSlot={
          <div className="flex min-w-0 items-center gap-3">
            <Logo withText={false} className="sm:hidden" />
            <Logo className="hidden sm:inline-flex" />
            <div className="hidden min-w-0 border-l border-border pl-3 sm:block">
              <div className="truncate text-sm font-semibold leading-tight text-foreground">{tc("headerTitle", lang)}</div>
              <div className="truncate text-xs leading-tight text-muted-foreground">{tc("headerStatus", lang)}</div>
            </div>
            {isDemo && (
              <span className="shrink-0 rounded-full border border-warn-border bg-warn-bg px-2 py-0.5 text-xs font-semibold text-warn">
                {tc("demoPill", lang)}
              </span>
            )}
          </div>
        }
        actions={
          <>
            <div className="flex items-center">
              <HeaderAction label={t("myAnalyses", lang)} onClick={() => router.push("/lab")}>
                <FlaskConical className="size-5" aria-hidden="true" />
              </HeaderAction>
              <HeaderAction label={t("history", lang)} onClick={loadHistoryData}>
                <History className="size-5" aria-hidden="true" />
              </HeaderAction>
              <HeaderAction label={t("logout", lang)} onClick={handleLogout}>
                <LogOut className="size-5" aria-hidden="true" />
              </HeaderAction>
            </div>
            <LanguageSwitcher lang={lang} onChange={onLangChange} />
          </>
        }
      />

      {/* History drawer */}
      {showHistory && (
        <div className="fixed inset-0 z-50 flex" role="dialog" aria-modal="true" aria-label={t("historyTitle", lang)}>
          <div className="absolute inset-0 bg-foreground/30" onClick={() => setShowHistory(false)} />
          <div className="relative ml-auto flex h-full w-full max-w-sm flex-col border-l border-border bg-card shadow-card">
            <div className="flex items-center justify-between border-b border-border p-4">
              <h2 className="font-display text-lg font-semibold text-foreground">{t("historyTitle", lang)}</h2>
              <Button
                variant="ghost"
                size="icon"
                onClick={() => setShowHistory(false)}
                aria-label={t("closeHistory", lang)}
              >
                <X className="size-5" />
              </Button>
            </div>
            <div className="relative flex-1 space-y-3 overflow-y-auto p-4">
              {historyLoading ? (
                <p className="py-8 text-center text-sm text-muted-foreground">{t("loading", lang)}</p>
              ) : history.length === 0 ? (
                <p className="py-8 text-center text-sm text-muted-foreground">{t("noRecords", lang)}</p>
              ) : (
                history.map((entry) => (
                  <div key={entry.day_id} className="space-y-1 rounded-xl border border-border bg-background p-3">
                    <div className="flex items-baseline justify-between gap-2">
                      <span className="text-sm font-semibold text-foreground">
                        {getDiseaseLabel(entry.disease_predict, lang)}
                      </span>
                      <span className="shrink-0 text-xs tabular-nums text-muted-foreground">
                        {entry.created_at?.split("T")[0] || entry.created_at}
                      </span>
                    </div>
                    {entry.score !== null && (
                      <p className="text-[13px] text-muted-foreground">
                        {t("confidence", lang)}: <span className="tabular-nums">{Math.round(entry.score * 100)}%</span>
                      </p>
                    )}
                    {entry.recept && <p className="text-[13px] text-foreground/80">{entry.recept}</p>}
                    {entry.doctor_name && (
                      <p className="text-[13px] text-primary">
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

      {/* Conversation */}
      <div className="relative flex-1 overflow-y-auto" role="log" aria-live="polite" aria-relevant="additions">
        <div className="mx-auto w-full max-w-[720px] space-y-4 px-4 py-5 sm:px-6">
          {messages.map((msg) => (
            <div key={msg.id} className="space-y-3">
              <ChatMessage
                role={msg.role}
                content={msg.content}
                timestamp={msg.timestamp}
                alert={msg.alert}
                lang={lang}
              />
              {msg.diagnosis && (
                <div className={indent}>
                  <DiagnosisCard diagnosis={msg.diagnosis} lang={lang} demoQueued={isDemo} />
                </div>
              )}
            </div>
          ))}

          {findings.length > 0 && (
            <div className={cn(indent, cardClass, "space-y-4")}>
              <p className="text-[13px] text-muted-foreground">{tc("confirmHint", lang)}</p>
              <div className="space-y-3">
                {findings.map((f) => {
                  const state = findingStates[f.code] ?? "unknown";
                  return (
                    <div key={f.code} className="space-y-2.5 rounded-xl border border-border p-3">
                      <div>
                        <p className="text-[15px] font-medium leading-snug text-foreground">{f.label[lang]}</p>
                        {f.quote && (
                          <p className="mt-0.5 text-[13px] text-muted-foreground">
                            {t("youSaid", lang)}: «{f.quote}»
                          </p>
                        )}
                      </div>
                      <AnswerGroup
                        label={f.label[lang]}
                        selected={state}
                        onPick={(next) => setFindingState(f.code, next)}
                        options={findingOptions}
                      />
                    </div>
                  );
                })}
              </div>

              {catalog.length > 0 &&
                (addOpen ? (
                  <div className="space-y-2">
                    <Input
                      autoFocus
                      value={addQuery}
                      onChange={(e) => setAddQuery(e.target.value)}
                      placeholder={t("addSymptomSearch", lang)}
                      aria-label={t("addSymptom", lang)}
                    />
                    {addQuery.trim() &&
                      (suggestions.length > 0 ? (
                        <ul className="flex flex-col gap-1.5">
                          {suggestions.map((e) => (
                            <li key={e.code}>
                              <button
                                type="button"
                                onClick={() => addFromCatalog(e)}
                                className="min-h-11 w-full rounded-[10px] border border-input bg-card px-3 text-left text-sm outline-none transition-colors hover:bg-accent focus-visible:ring-[3px] focus-visible:ring-ring"
                              >
                                {e.label[lang]}
                              </button>
                            </li>
                          ))}
                        </ul>
                      ) : (
                        <p className="text-[13px] text-muted-foreground">{t("addSymptomNoMatch", lang)}</p>
                      ))}
                    <Button variant="ghost" className="min-h-11 text-muted-foreground" onClick={closeAddSymptom}>
                      {t("cancel", lang)}
                    </Button>
                  </div>
                ) : (
                  <Button variant="outline" className="min-h-11 w-full sm:w-auto" onClick={() => setAddOpen(true)} disabled={isLoading}>
                    <Plus aria-hidden="true" />
                    {t("addSymptom", lang)}
                  </Button>
                ))}

              <div>
                <Button className="min-h-11 w-full sm:w-auto" onClick={confirmFindings} disabled={isLoading}>
                  {t("confirmBtn", lang)}
                </Button>
              </div>
            </div>
          )}

          {clarifyingQuestions.length > 0 && session && (
            <div className={cn(indent, "space-y-3")}>
              {clarifyingQuestions.map((q) => {
                const n = session.asked.indexOf(q.evidence_id) + 1;
                return (
                  <div key={q.evidence_id} className={cardClass}>
                    {n > 0 && (
                      <p className="mb-1 text-[13px] font-medium tabular-nums text-primary">
                        {tc("questionN", lang, { n })}
                      </p>
                    )}
                    <p className="mb-3 text-[15px] leading-snug text-foreground">{questionText(q, lang)}</p>
                    <AnswerGroup
                      label={questionText(q, lang)}
                      selected={null}
                      onPick={(state) => handleClarifyAnswer(q, toAnswer(state))}
                      options={questionOptions}
                    />
                  </div>
                );
              })}
            </div>
          )}

          {failure && !isLoading && (
            <div className={cn(indent, "flex flex-wrap gap-2")}>
              <Button className="min-h-11 px-5" onClick={retryFailed}>
                {t("retry", lang)}
              </Button>
              {failure.kind === "questions" && (
                <Button variant="outline" className="min-h-11" onClick={skipToResult}>
                  {t("skipToResult", lang)}
                </Button>
              )}
            </div>
          )}

          {isLoading && <DiagnosisSkeleton lang={lang} />}
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Composer */}
      <div className="shrink-0 border-t border-border bg-card pb-[max(0.75rem,env(safe-area-inset-bottom))] pt-3">
        <div className="mx-auto w-full max-w-[720px] px-4 sm:px-6">
          {answeringQuestions && (
            <div className="mb-2">
              <Button variant="link" className="h-auto min-h-11 p-0 text-[13px]" onClick={startNewComplaint}>
                {t("newComplaint", lang)}
              </Button>
            </div>
          )}
          <div className="flex items-center gap-1 rounded-full border border-input bg-background py-1 pl-4 pr-1 transition-shadow focus-within:border-ring focus-within:ring-[3px] focus-within:ring-ring/40">
            <input
              ref={inputRef}
              type="text"
              aria-label={tc("composerLabel", lang)}
              placeholder={answeringQuestions ? t("answerQuestionsFirst", lang) : t("inputPlaceholder", lang)}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={isLoading || answeringQuestions}
              className="h-10 min-w-0 flex-1 bg-transparent text-base text-foreground outline-none placeholder:text-muted-foreground disabled:cursor-not-allowed disabled:opacity-50 md:text-[15px]"
            />
            <button
              type="button"
              onClick={handleSend}
              aria-label={t("sendMessage", lang)}
              title={cooldown > 0 ? tc("waitSeconds", lang, { n: cooldown }) : t("sendMessage", lang)}
              disabled={isLoading || answeringQuestions || cooldown > 0 || !input.trim()}
              className="inline-flex size-11 shrink-0 items-center justify-center rounded-full bg-primary text-primary-foreground outline-none transition-colors duration-150 hover:bg-primary-hover focus-visible:ring-[3px] focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-40"
            >
              {isLoading ? (
                <Loader2 className="size-5 motion-safe:animate-spin" aria-hidden="true" />
              ) : cooldown > 0 ? (
                <span className="text-sm font-medium tabular-nums">{cooldown}</span>
              ) : (
                <Send className="size-5" aria-hidden="true" />
              )}
            </button>
          </div>
          <p className="mt-2 text-center text-xs text-muted-foreground">{t("mvpDisclaimer", lang)}</p>
        </div>
      </div>
    </div>
  );
}
