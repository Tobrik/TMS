"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import {
  Stethoscope, LogIn, LogOut, Loader2, Search, FileText, Edit3, Save,
  X, ChevronDown, ChevronUp, User, Calendar, Activity, ArrowLeft,
  MapPin, Thermometer, AlertTriangle, Clock, CheckCircle, Download, Brain,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Textarea } from "@/components/ui/textarea";
import {
  loginDoctor, listDoctors, getHistory, getPatientInfo, getDaySymptoms,
  updateByDoctor, listPatientsTriage, logout as apiLogout,
  type Doctor, type HistoryEntry, type PatientInfo, type TriagePatient,
} from "@/lib/api";
import { getDiseaseLabel } from "@/lib/diseaseWeights";
import { SymptomTrend } from "@/components/SymptomTrend";
import { generatePatientPDF } from "@/lib/pdfExport";
import { getSymptomLabel, type SymptomCode } from "@/lib/symptoms";
import { t, getLang, type Lang } from "@/lib/i18n";

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

const ANIMALS = [
  "LION", "BEAR", "WOLF", "HAWK", "DEER", "FOX", "OWL", "LYNX",
  "PIKE", "CROW", "SEAL", "HARE", "SWAN", "DOVE", "COLT",
];

function makeCode(patientId: number): string {
  return `${ANIMALS[patientId % ANIMALS.length]}-${patientId}`;
}

type View = "login" | "dashboard";

export default function DoctorPage() {
  const router = useRouter();
  const [lang] = useState<Lang>(getLang());
  const [view, setView] = useState<View>("login");

  const [doctors, setDoctors] = useState<Doctor[]>([]);
  const [selectedDoctorId, setSelectedDoctorId] = useState<number | null>(null);
  const [pin, setPin] = useState("");
  const [loginLoading, setLoginLoading] = useState(false);
  const [loginError, setLoginError] = useState("");
  const [doctorsLoading, setDoctorsLoading] = useState(true);

  const [doctorInfo, setDoctorInfo] = useState<Doctor | null>(null);
  const [patientIdInput, setPatientIdInput] = useState("");
  const [patientInfo, setPatientInfo] = useState<PatientInfo | null>(null);
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [searchedPatientId, setSearchedPatientId] = useState<number | null>(null);
  const [patientNotFound, setPatientNotFound] = useState(false);

  const [editingDayId, setEditingDayId] = useState<number | null>(null);
  const [editRecept, setEditRecept] = useState("");
  const [editDiagnosis, setEditDiagnosis] = useState("");
  const [saving, setSaving] = useState(false);

  const [expandedDayId, setExpandedDayId] = useState<number | null>(null);
  const [daySymptoms, setDaySymptoms] = useState<Record<number, Record<string, number>>>({});
  const [symptomsLoading, setSymptomsLoading] = useState<number | null>(null);

  const [triagePatients, setTriagePatients] = useState<TriagePatient[]>([]);
  const [triageLoading, setTriageLoading] = useState(false);

  const loadTriageData = async () => {
    setTriageLoading(true);
    try {
      const resp = await listPatientsTriage();
      setTriagePatients(resp.patients);
    } catch { setTriagePatients([]); }
    finally { setTriageLoading(false); }
  };

  useEffect(() => {
    const stored = localStorage.getItem("doctor_id");
    const token = localStorage.getItem("access_token");
    if (stored && token) {
      const id = parseInt(stored, 10);
      const name = localStorage.getItem("doctor_name") || "";
      const spec = localStorage.getItem("doctor_specialty") || "";
      setDoctorInfo({ doctor_id: id, full_name: name, specialty: spec, created_at: "" });
      setView("dashboard");
      loadTriageData();
    }
    listDoctors()
      .then((resp) => setDoctors(resp.doctors))
      .catch(() => setDoctors([]))
      .finally(() => setDoctorsLoading(false));
  }, []);

  const handleLogin = async () => {
    if (!selectedDoctorId) { setLoginError(t("selectDoctor", lang)); return; }
    if (!pin || pin.length < 1) { setLoginError(t("errorEnterPassword", lang)); return; }
    setLoginError("");
    setLoginLoading(true);
    try {
      const resp = await loginDoctor(selectedDoctorId, pin);
      if (resp.login) {
        const doc = resp.doctor || doctors.find((d) => d.doctor_id === selectedDoctorId);
        if (doc) {
          setDoctorInfo(doc);
          localStorage.setItem("doctor_id", String(doc.doctor_id));
          localStorage.setItem("doctor_name", doc.full_name);
          localStorage.setItem("doctor_specialty", doc.specialty);
        }
        setView("dashboard");
        loadTriageData();
      } else {
        setLoginError(t("errorWrongPassword", lang));
      }
    } catch (err: unknown) {
      const status = (err as { response?: { status?: number } })?.response?.status;
      if (status === 401) setLoginError(t("errorWrongPassword", lang));
      else if (status === 429) setLoginError(t("errorTooManyAttempts", lang));
      else setLoginError(t("errorConnection", lang));
    } finally { setLoginLoading(false); }
  };

  const handleLogout = () => {
    apiLogout();
    setDoctorInfo(null); setView("login"); setHistory([]);
    setSearchedPatientId(null); setPatientInfo(null); setPin("");
  };

  const searchPatientById = async (pid: number) => {
    if (pid <= 0) return;
    setHistoryLoading(true); setSearchedPatientId(pid);
    setPatientNotFound(false); setPatientInfo(null);
    setDaySymptoms({}); setExpandedDayId(null);
    try {
      const [infoResp, histResp] = await Promise.all([getPatientInfo(pid), getHistory(pid)]);
      if (infoResp.found && infoResp.patient) setPatientInfo(infoResp.patient);
      else setPatientNotFound(true);
      setHistory(histResp.history);
    } catch { setHistory([]); setPatientNotFound(true); }
    finally { setHistoryLoading(false); }
  };

  const handleSearchPatient = async () => {
    const id = parseIdFromCode(patientIdInput);
    if (!id) return;
    searchPatientById(id);
  };

  const handleExpand = async (dayId: number) => {
    if (expandedDayId === dayId) { setExpandedDayId(null); return; }
    setExpandedDayId(dayId);
    if (!daySymptoms[dayId]) {
      setSymptomsLoading(dayId);
      try {
        const resp = await getDaySymptoms(dayId);
        setDaySymptoms((prev) => ({ ...prev, [dayId]: resp.symptoms }));
      } catch { /* ignore */ }
      finally { setSymptomsLoading(null); }
    }
  };

  const startEdit = (entry: HistoryEntry) => {
    setEditingDayId(entry.day_id);
    setEditRecept(entry.recept || "");
    setEditDiagnosis(entry.disease_setup || "");
  };

  const cancelEdit = () => { setEditingDayId(null); setEditRecept(""); setEditDiagnosis(""); };

  const handleSave = async (entry: HistoryEntry) => {
    if (!doctorInfo || !searchedPatientId) return;
    setSaving(true);
    try {
      await updateByDoctor(searchedPatientId, entry.day_id, doctorInfo.doctor_id, editRecept, editDiagnosis);
      const resp = await getHistory(searchedPatientId);
      setHistory(resp.history);
      setEditingDayId(null);
    } catch { /* silent */ }
    finally { setSaving(false); }
  };

  function getActiveSymptoms(dayId: number): { code: string; label: string; value: number }[] {
    const syms = daySymptoms[dayId];
    if (!syms) return [];
    const result: { code: string; label: string; value: number }[] = [];
    for (const [code, value] of Object.entries(syms)) {
      if (value > 0) result.push({ code, label: getSymptomLabel(code as SymptomCode, lang), value });
    }
    return result.sort((a, b) => b.value - a.value);
  }

  const severityColor = (v: number) => {
    if (v >= 3) return "bg-red-100 text-red-700 border-red-200";
    if (v >= 2) return "bg-orange-100 text-orange-700 border-orange-200";
    return "bg-yellow-100 text-yellow-700 border-yellow-200";
  };

  // Triage zone renderer
  const renderTriageZone = (
    zone: "red" | "yellow" | "green",
    icon: React.ReactNode,
    titleKey: "redZone" | "yellowZone" | "greenZone",
    colorClasses: { border: string; text: string; badge: string; btn: string },
  ) => {
    const patients = triagePatients.filter((p) => p.zone === zone);
    if (patients.length === 0) return null;
    return (
      <div className="space-y-2">
        <div className="flex items-center gap-2">
          {icon}
          <h2 className={`font-semibold ${colorClasses.text}`}>{t(titleKey, lang)}</h2>
          <Badge className={`${colorClasses.badge} border-0 ${zone === "red" ? "animate-pulse" : ""}`}>{patients.length}</Badge>
        </div>
        <div className="space-y-2">
          {patients.map((p) => (
            <Card key={p.patient_id} className={`border-l-4 ${colorClasses.border}`}>
              <CardContent className="py-3 px-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="font-medium text-gray-900">{p.full_name}</p>
                    <p className="text-xs text-gray-500">{makeCode(p.patient_id)} {p.city && `· ${p.city}`}</p>
                    {p.last_disease && (
                      <p className={`text-sm mt-1 ${colorClasses.text}`}>
                        {getDiseaseLabel(p.last_disease.split(" ")[0], lang)}
                        {p.last_score > 0 && ` · ${Math.round(p.last_score * 100)}%`}
                      </p>
                    )}
                  </div>
                  <Button size="sm" variant="outline" className={colorClasses.btn} onClick={() => searchPatientById(p.patient_id)}>
                    {t("details", lang)}
                  </Button>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    );
  };

  // LOGIN VIEW
  if (view === "login") {
    return (
      <div className="min-h-screen bg-gradient-to-br from-indigo-50 via-white to-purple-50 flex items-center justify-center p-4">
        <div className="absolute inset-0 overflow-hidden pointer-events-none">
          <div className="absolute -top-40 -right-40 w-96 h-96 bg-indigo-100 rounded-full opacity-30 blur-3xl" />
          <div className="absolute -bottom-40 -left-40 w-96 h-96 bg-purple-100 rounded-full opacity-30 blur-3xl" />
        </div>
        <div className="relative w-full max-w-md space-y-6">
          <div className="text-center space-y-2">
            <div className="inline-flex items-center justify-center w-16 h-16 bg-gradient-to-br from-indigo-600 to-purple-500 rounded-2xl shadow-lg shadow-indigo-200">
              <Stethoscope className="h-8 w-8 text-white" />
            </div>
            <h1 className="text-2xl font-bold text-gray-900">{t("doctorTitle", lang)}</h1>
            <p className="text-sm text-gray-500">{t("doctorSubtitle", lang)}</p>
          </div>
          <Card className="shadow-xl border-0 bg-white/80 backdrop-blur-sm">
            <CardHeader className="pb-2 pt-6">
              <div className="flex items-center gap-2 justify-center text-sm text-gray-500">
                <Stethoscope className="h-4 w-4 text-indigo-500" />
                <span>{t("loginSystem", lang)}</span>
              </div>
            </CardHeader>
            <CardContent className="pt-4 space-y-4">
              <div className="space-y-2">
                <label className="text-sm font-medium text-gray-700">{t("selectDoctor", lang)}</label>
                {doctorsLoading ? (
                  <div className="flex items-center justify-center py-4"><Loader2 className="h-5 w-5 animate-spin text-gray-400" /></div>
                ) : (
                  <div className="space-y-2">
                    {doctors.map((doc) => (
                      <button key={doc.doctor_id} onClick={() => setSelectedDoctorId(doc.doctor_id)}
                        className={`w-full text-left px-4 py-3 rounded-xl border-2 transition-all ${selectedDoctorId === doc.doctor_id ? "border-indigo-500 bg-indigo-50" : "border-gray-100 bg-gray-50 hover:border-gray-200"}`}>
                        <p className="text-sm font-medium text-gray-900">{doc.full_name}</p>
                        <p className="text-xs text-gray-500">{doc.specialty} (ID: {doc.doctor_id})</p>
                      </button>
                    ))}
                  </div>
                )}
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium text-gray-700">{t("password", lang)}</label>
                <Input type="password" maxLength={64} placeholder={t("passwordPlaceholder", lang)} value={pin}
                  onChange={(e) => setPin(e.target.value.slice(0, 64))} onKeyDown={(e) => e.key === "Enter" && handleLogin()} />
              </div>
              {loginError && <p className="text-sm text-red-500 text-center">{loginError}</p>}
              <Button className="w-full bg-gradient-to-r from-indigo-600 to-purple-500 hover:from-indigo-700 hover:to-purple-600 text-white shadow-lg shadow-indigo-200"
                onClick={handleLogin} disabled={loginLoading}>
                {loginLoading ? <Loader2 className="h-4 w-4 mr-2 animate-spin" /> : <LogIn className="h-4 w-4 mr-2" />}
                {t("loginBtn", lang)}
              </Button>
              <Button variant="ghost" className="w-full text-gray-500" onClick={() => router.push("/")}>
                <ArrowLeft className="h-4 w-4 mr-2" />{t("backToMain", lang)}
              </Button>
            </CardContent>
          </Card>
        </div>
      </div>
    );
  }

  // DASHBOARD VIEW
  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white border-b border-gray-200 px-4 py-3 flex items-center justify-between sticky top-0 z-10">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-gradient-to-br from-indigo-600 to-purple-500 rounded-xl flex items-center justify-center shadow-md">
            <Stethoscope className="h-5 w-5 text-white" />
          </div>
          <div>
            <h1 className="font-semibold text-gray-900 text-sm">{doctorInfo?.full_name}</h1>
            <p className="text-xs text-indigo-500">{doctorInfo?.specialty}</p>
          </div>
        </div>
        <Button variant="ghost" size="icon" onClick={handleLogout} title={t("logout", lang)}>
          <LogOut className="h-5 w-5 text-gray-500" />
        </Button>
      </header>

      <div className="max-w-3xl mx-auto p-4 space-y-6">
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center gap-2 mb-4">
              <Search className="h-5 w-5 text-gray-400" />
              <h2 className="font-semibold text-gray-900">{t("searchPatient", lang)}</h2>
            </div>
            <div className="flex gap-2">
              <Input type="text" placeholder={t("patientCodePlaceholder", lang)} value={patientIdInput}
                onChange={(e) => setPatientIdInput(e.target.value)} onKeyDown={(e) => e.key === "Enter" && handleSearchPatient()} className="flex-1" />
              <Button onClick={handleSearchPatient} disabled={historyLoading} className="bg-indigo-600 hover:bg-indigo-700 text-white">
                {historyLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Search className="h-4 w-4" />}
              </Button>
            </div>
          </CardContent>
        </Card>

        {triagePatients.length > 0 && !searchedPatientId && (
          <div className="space-y-4">
            {renderTriageZone("red", <AlertTriangle className="h-5 w-5 text-red-500" />, "redZone",
              { border: "border-l-red-500", text: "text-red-700", badge: "bg-red-100 text-red-700", btn: "text-red-600 border-red-200 hover:bg-red-50" })}
            {renderTriageZone("yellow", <Clock className="h-5 w-5 text-amber-500" />, "yellowZone",
              { border: "border-l-amber-400", text: "text-amber-700", badge: "bg-amber-100 text-amber-700", btn: "text-amber-600 border-amber-200 hover:bg-amber-50" })}
            {renderTriageZone("green", <CheckCircle className="h-5 w-5 text-emerald-500" />, "greenZone",
              { border: "border-l-emerald-400", text: "text-emerald-700", badge: "bg-emerald-100 text-emerald-700", btn: "text-emerald-600 border-emerald-200 hover:bg-emerald-50" })}
          </div>
        )}

        {triageLoading && !searchedPatientId && (
          <div className="flex justify-center py-8"><Loader2 className="h-6 w-6 animate-spin text-gray-400" /></div>
        )}

        {patientNotFound && !historyLoading && (
          <Card><CardContent className="py-8 text-center text-gray-500 text-sm">{t("patientNotFound", lang)}</CardContent></Card>
        )}

        {patientInfo && (
          <Card className="border-l-4 border-l-indigo-500">
            <CardContent className="pt-5 pb-4">
              <div className="flex items-start gap-4">
                <div className="w-12 h-12 rounded-full bg-indigo-100 flex items-center justify-center shrink-0">
                  <User className="h-6 w-6 text-indigo-600" />
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1">
                    <h3 className="text-lg font-semibold text-gray-900">{patientInfo.full_name}</h3>
                    <Badge variant="secondary" className="text-xs font-mono">{makeCode(patientInfo.patient_id)}</Badge>
                  </div>
                  <div className="flex items-center gap-4 text-sm text-gray-500">
                    {patientInfo.city && <span className="flex items-center gap-1"><MapPin className="h-3.5 w-3.5" />{patientInfo.city}</span>}
                    <span className="flex items-center gap-1"><Calendar className="h-3.5 w-3.5" />{t("registration", lang)}: {patientInfo.created_at?.split(" ")[0] || patientInfo.created_at}</span>
                  </div>
                  <div className="mt-2 flex items-center gap-2">
                    <Badge variant="secondary" className="text-xs">{history.length} {t("diaryEntries", lang)}</Badge>
                    {history.length > 0 && (
                      <Button size="sm" variant="outline" className="text-xs text-indigo-600 border-indigo-200 hover:bg-indigo-50"
                        onClick={() => generatePatientPDF(patientInfo, history, doctorInfo?.full_name || "", lang)}>
                        <Download className="h-3 w-3 mr-1" />{t("downloadPdf", lang)}
                      </Button>
                    )}
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        {searchedPatientId !== null && patientInfo && (
          <Card><CardContent className="pt-5 pb-4"><SymptomTrend patientId={searchedPatientId} lang={lang} /></CardContent></Card>
        )}

        {searchedPatientId !== null && patientInfo && (
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <FileText className="h-5 w-5 text-gray-400" />
              <h2 className="font-semibold text-gray-900">{t("diagnosisHistory", lang)}</h2>
            </div>

            {historyLoading ? (
              <div className="flex justify-center py-8"><Loader2 className="h-6 w-6 animate-spin text-gray-400" /></div>
            ) : history.length === 0 ? (
              <Card><CardContent className="py-8 text-center text-gray-500 text-sm">{t("noEntries", lang)}</CardContent></Card>
            ) : (
              history.map((entry) => {
                const isExpanded = expandedDayId === entry.day_id;
                const isEditing = editingDayId === entry.day_id;
                const diseaseLabel = getDiseaseLabel(entry.disease_predict, lang);
                const activeSymptoms = getActiveSymptoms(entry.day_id);
                const isLoadingSymptoms = symptomsLoading === entry.day_id;

                return (
                  <Card key={entry.day_id} className="overflow-hidden">
                    <button className="w-full text-left px-4 py-3 flex items-center justify-between hover:bg-gray-50 transition-colors"
                      onClick={() => handleExpand(entry.day_id)}>
                      <div className="flex items-center gap-3 min-w-0">
                        <div className="w-8 h-8 rounded-lg bg-indigo-100 flex items-center justify-center shrink-0">
                          <FileText className="h-4 w-4 text-indigo-600" />
                        </div>
                        <div className="min-w-0">
                          <p className="text-sm font-medium text-gray-900 truncate">{diseaseLabel}</p>
                          <div className="flex items-center gap-2 text-xs text-gray-500">
                            <Calendar className="h-3 w-3" />
                            <span>{entry.created_at?.split("T")[0] || entry.created_at}</span>
                            {entry.score !== null && (<><Activity className="h-3 w-3 ml-1" /><span>{Math.round(entry.score * 100)}%</span></>)}
                          </div>
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        {entry.disease_setup && entry.disease_setup !== "Nothing" && (
                          <Badge className="text-xs bg-green-100 text-green-700 border-0">{t("confirmed", lang)}</Badge>
                        )}
                        {entry.doctor_name && <Badge variant="outline" className="text-xs">{entry.doctor_name.split(" ").slice(0, 2).join(" ")}</Badge>}
                        {isExpanded ? <ChevronUp className="h-4 w-4 text-gray-400" /> : <ChevronDown className="h-4 w-4 text-gray-400" />}
                      </div>
                    </button>

                    {isExpanded && (
                      <div className="border-t border-gray-100 px-4 py-4 space-y-4 bg-gray-50/50">
                        <div className="space-y-2">
                          <p className="text-xs font-medium text-gray-500 uppercase tracking-wide flex items-center gap-1">
                            <Thermometer className="h-3.5 w-3.5" />{t("patientSymptoms", lang)}
                          </p>
                          {isLoadingSymptoms ? (
                            <div className="flex items-center gap-2 py-2">
                              <Loader2 className="h-4 w-4 animate-spin text-gray-400" />
                              <span className="text-xs text-gray-400">{t("loadingDots", lang)}</span>
                            </div>
                          ) : activeSymptoms.length > 0 ? (
                            <div className="flex flex-wrap gap-2">
                              {activeSymptoms.map((s) => (
                                <span key={s.code} className={`inline-flex items-center gap-1 px-2.5 py-1 rounded-lg text-xs font-medium border ${severityColor(s.value)}`}>
                                  {s.label}<span className="opacity-70">({s.value}/3)</span>
                                </span>
                              ))}
                            </div>
                          ) : (
                            <p className="text-xs text-gray-400 italic">{t("noSymptomData", lang)}</p>
                          )}
                        </div>

                        <div className="space-y-1">
                          <p className="text-xs font-medium text-gray-500 uppercase tracking-wide">{t("preliminaryDiagnosis", lang)}</p>
                          <p className="text-sm text-gray-900">{diseaseLabel}</p>
                        </div>

                        {entry.doctor_explanation && (
                          <div className="space-y-2 bg-indigo-50 rounded-xl p-4 border border-indigo-200">
                            <p className="text-xs font-medium text-indigo-600 uppercase tracking-wide flex items-center gap-1.5">
                              <Brain className="h-4 w-4" />{t("aiClinicalAnalysis", lang)}
                            </p>
                            <div className="text-sm text-gray-800 leading-relaxed whitespace-pre-wrap prose prose-sm max-w-none prose-strong:text-indigo-700 prose-li:my-0.5">
                              {entry.doctor_explanation}
                            </div>
                            <p className="text-xs text-indigo-400 italic mt-2">{t("aiDisclaimer", lang)}</p>
                          </div>
                        )}

                        <div className="space-y-1">
                          <p className="text-xs font-medium text-gray-500 uppercase tracking-wide">{t("doctorDiagnosis", lang)}</p>
                          {isEditing ? (
                            <Input value={editDiagnosis} onChange={(e) => setEditDiagnosis(e.target.value)} placeholder={t("setDiagnosis", lang)} className="bg-white" />
                          ) : (
                            <p className="text-sm text-gray-900">
                              {entry.disease_setup && entry.disease_setup !== "Nothing" ? entry.disease_setup : <span className="text-gray-400 italic">{t("notSet", lang)}</span>}
                            </p>
                          )}
                        </div>

                        <div className="space-y-1">
                          <p className="text-xs font-medium text-gray-500 uppercase tracking-wide">{t("recept", lang)}</p>
                          {isEditing ? (
                            <Textarea value={editRecept} onChange={(e) => setEditRecept(e.target.value)} placeholder={t("writeRecept", lang)} rows={3} className="bg-white" />
                          ) : (
                            <p className="text-sm text-gray-900 whitespace-pre-wrap">
                              {entry.recept || <span className="text-gray-400 italic">{t("noRecommendations", lang)}</span>}
                            </p>
                          )}
                        </div>

                        {entry.doctor_name && (
                          <div className="space-y-1">
                            <p className="text-xs font-medium text-gray-500 uppercase tracking-wide">{t("responsibleDoctor", lang)}</p>
                            <p className="text-sm text-indigo-600">{entry.doctor_name}</p>
                          </div>
                        )}

                        <div className="flex gap-2 pt-2">
                          {isEditing ? (
                            <>
                              <Button size="sm" onClick={() => handleSave(entry)} disabled={saving} className="bg-green-600 hover:bg-green-700 text-white">
                                {saving ? <Loader2 className="h-3 w-3 mr-1 animate-spin" /> : <Save className="h-3 w-3 mr-1" />}{t("save", lang)}
                              </Button>
                              <Button size="sm" variant="outline" onClick={cancelEdit}><X className="h-3 w-3 mr-1" />{t("cancel", lang)}</Button>
                            </>
                          ) : (
                            <Button size="sm" variant="outline" onClick={() => startEdit(entry)} className="text-indigo-600 border-indigo-200 hover:bg-indigo-50">
                              <Edit3 className="h-3 w-3 mr-1" />{t("edit", lang)}
                            </Button>
                          )}
                        </div>
                      </div>
                    )}
                  </Card>
                );
              })
            )}
          </div>
        )}
      </div>
    </div>
  );
}
