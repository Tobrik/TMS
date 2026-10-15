import axios from "axios";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const api = axios.create({
  baseURL: API_URL,
  headers: { "Content-Type": "application/json" },
});

// Attach JWT token to every request if available
api.interceptors.request.use((config) => {
  if (typeof window !== "undefined") {
    const token = localStorage.getItem("access_token");
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
  }
  return config;
});

// Handle 401 responses — redirect to login
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401 && typeof window !== "undefined") {
      localStorage.removeItem("access_token");
      localStorage.removeItem("patient_id");
      localStorage.removeItem("patient_name");
      localStorage.removeItem("pairing_code");
      localStorage.removeItem("doctor_id");
      localStorage.removeItem("doctor_name");
      localStorage.removeItem("doctor_specialty");
      window.location.href = "/";
    }
    return Promise.reject(error);
  }
);

export interface RegisterResponse {
  patient_id: number;
  access_token: string;
  token_type: string;
}

export interface LoginResponse {
  login: boolean;
  access_token: string;
  token_type: string;
}

export interface DoctorLoginResponse {
  login: boolean;
  access_token: string;
  token_type: string;
  doctor: {
    doctor_id: number;
    full_name: string;
    specialty: string;
    created_at: string;
  } | null;
}

export interface AnalysSlice {
  name: string;
  label: string;
  score: number;
}

export type Zone = "red" | "yellow" | "green";

export interface TriageReason {
  zone: Zone;
  code: "red_flag" | "red_flag_combo" | "llm_emergency" | "out_of_scope" | "red_mass" | "yellow_flag" | "yellow_mass";
  evidence?: string;
  evidences?: string[];
  value?: number;
  verified?: boolean;
}

export interface AnalysResponse {
  day: number;
  diseaseName: string;
  diseaseLabel: string;
  doctor: string;
  recommendation: string;
  slices: AnalysSlice[];
  score: number;
  zone: Zone;
  reasons: TriageReason[];
  out_of_scope: { likely_condition: string | null } | null;
}

export interface HistoryEntry {
  day_id: number;
  created_at: string;
  disease_predict: string;
  score: number;
  disease_setup: string | null;
  recept: string | null;
  doctor_id: number | null;
  doctor_name: string | null;
  patient_explanation: string | null;
  doctor_explanation: string | null;
  zone: Zone | null;
}

export interface Doctor {
  doctor_id: number;
  full_name: string;
  specialty: string;
  created_at: string;
}

export interface SymptomGraphEntry {
  day_id: number;
  created_at: string;
  value: number;
}

export interface PatientInfo {
  patient_id: number;
  full_name: string;
  city: string;
  created_at: string;
}

export async function registerPatient(
  name: string,
  city: string,
  password: string
): Promise<RegisterResponse> {
  const { data } = await api.post<RegisterResponse>("/regist_as_patient", {
    name,
    city,
    password,
  });
  if (data.access_token) {
    localStorage.setItem("access_token", data.access_token);
  }
  return data;
}

export async function loginPatient(
  patientId: number,
  password: string
): Promise<LoginResponse> {
  const { data } = await api.post<LoginResponse>("/login_patient", {
    patient_id: patientId,
    password,
  });
  if (data.access_token) {
    localStorage.setItem("access_token", data.access_token);
  }
  return data;
}

export type LocalizedText = { en: string; ru: string; kk: string };

export interface CriticalAlert {
  level: "red" | "yellow";
  message: string;
}

export interface Finding {
  code: string;
  status: "present" | "absent" | "uncertain";
  quote: string; // the patient's own words (Groq); empty for Clef and regex
  votes: number;
  source: "llm" | "safety_net" | "regex" | "bm25" | "clef";
  confidence: number | null; // Clef: probability of this status
  label: LocalizedText;
}

export interface ExtractSymptomsResponse {
  intake_id: number | null;
  evidences: string[];
  absent: string[];
  uncertain: string[];
  findings: Finding[];
  age: number | null; // null = not stated
  sex: "M" | "F" | null;
  noSymptoms?: boolean;
  critical_alert: CriticalAlert | null;
  mode?: "llm" | "fallback" | "clef";
  error?: string;
}

export async function extractSymptoms(text: string): Promise<ExtractSymptomsResponse> {
  try {
    const { data } = await api.post<ExtractSymptomsResponse>("/extract_symptoms", { text });
    return data;
  } catch (err: unknown) {
    const status = (err as { response?: { status?: number } })?.response?.status;
    return {
      intake_id: null,
      evidences: [],
      absent: [],
      uncertain: [],
      findings: [],
      age: null,
      sex: null,
      critical_alert: null,
      error: status ? `AI Service Error: ${status}` : "Network error",
    };
  }
}

export async function sendAnalysis(
  evidences: string[],
  absent: string[],
  age: number | null,
  sex: string | null,
  intakeId: number | null,
  diagnoseSetup: string = "Nothing"
): Promise<AnalysResponse> {
  const { data } = await api.post<AnalysResponse>("/analys", {
    evidences,
    absent,
    age,
    sex,
    intake_id: intakeId,
    diagnose_setup: diagnoseSetup,
  });
  return data;
}

export interface ClarifyingQuestion {
  evidence_id: string;
  question_en: string;
  question_ru: string;
  question_kk: string;
  score: number;
}

export interface NextQuestionsResponse {
  questions: ClarifyingQuestion[];
  current_top3: { disease: string; label: string; score: number }[];
  zone: Zone;
  critical_alert: CriticalAlert | null;
  should_stop: boolean;
  stop_reason: string;
  confidence: number;
  round_type: string;
}

export async function getNextQuestions(
  evidences: string[],
  absent: string[],
  age: number | null,
  sex: string | null,
  roundNum: number,
  n: number = 3
): Promise<NextQuestionsResponse> {
  const { data } = await api.post<NextQuestionsResponse>("/next_questions", {
    evidences,
    absent,
    age,
    sex,
    round_num: roundNum,
    n,
  });
  return data;
}

export async function explainDiagnosis(
  dayId: number,
  lang: "ru" | "en" | "kk"
): Promise<{ patientExplanation: string; doctorExplanation: string }> {
  const { data } = await api.post<{ patientExplanation: string; doctorExplanation: string }>(
    "/explain",
    { day_id: dayId, lang }
  );
  return data;
}

export async function getHistory(
  patientId: number
): Promise<{ history: HistoryEntry[] }> {
  const { data } = await api.post<{ history: HistoryEntry[] }>("/get_history", {
    patient_id: patientId,
  });
  return data;
}

export async function loginDoctor(
  doctorId: number,
  password: string
): Promise<DoctorLoginResponse> {
  const { data } = await api.post<DoctorLoginResponse>("/login_doctor", {
    doctor_id: doctorId,
    password,
  });
  if (data.access_token) {
    localStorage.setItem("access_token", data.access_token);
  }
  return data;
}

export async function listDoctors(): Promise<{ doctors: Doctor[] }> {
  const { data } = await api.post<{ doctors: Doctor[] }>("/list_doctor", {});
  return data;
}

export async function getSymptomGraph(
  patientId: number,
  symptomCode: string
): Promise<{ symptoms_arr: SymptomGraphEntry[] }> {
  const { data } = await api.post<{ symptoms_arr: SymptomGraphEntry[] }>(
    "/get_symptoms",
    { patient_id: patientId, symptom_str: symptomCode }
  );
  return data;
}

export interface TriagePatient {
  patient_id: number;
  full_name: string;
  city: string;
  created_at: string;
  last_disease: string;
  last_score: number;
  diag_date: string | null;
  zone: Zone;
}

export async function listPatientsTriage(): Promise<{ patients: TriagePatient[] }> {
  const { data } = await api.post<{ patients: TriagePatient[] }>("/list_patients_triage", {});
  return data;
}

export async function getPatientInfo(
  patientId: number
): Promise<{ found: boolean; patient: PatientInfo | null }> {
  const { data } = await api.post<{ found: boolean; patient: PatientInfo | null }>(
    "/get_patient_info",
    { patient_id: patientId }
  );
  return data;
}

export interface DaySymptoms {
  symptoms: Record<string, number>; // 1 reported, 0 denied
  labels?: Record<string, LocalizedText>;
}

export async function getDaySymptoms(dayId: number): Promise<DaySymptoms> {
  const { data } = await api.post<DaySymptoms>("/get_day_symptoms", { day_id: dayId });
  return data;
}

export async function updateByDoctor(
  patientId: number,
  dayId: number,
  doctorId: number,
  recept: string,
  diseaseSetup: string
): Promise<{ ok: boolean }> {
  const { data } = await api.post<{ ok: boolean }>("/update_by_doctor", {
    patient_id: patientId,
    day_id: dayId,
    doctor_id: doctorId,
    recept,
    disease_setup: diseaseSetup,
  });
  return data;
}

export interface LabResultItem {
  name: string;
  value: string;
  unit: string;
  reference_range: string;
  status: string; // "normal" | "high" | "low"
}

export interface LabResult {
  result_id: number;
  test_type: string;
  test_date: string;
  results: LabResultItem[];
  interpretation: string;
  created_at?: string;
}

export async function uploadLabResult(
  _patientId: number,
  imageFile: File
): Promise<LabResult> {
  const formData = new FormData();
  formData.append("image", imageFile);

  const token = localStorage.getItem("access_token");
  const { data } = await axios.post<LabResult>(`${API_URL}/upload_lab_result`, formData, {
    headers: {
      "Content-Type": "multipart/form-data",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
  });
  return data;
}

export async function getLabResults(
  _patientId?: number
): Promise<{ results: LabResult[] }> {
  const { data } = await api.post<{ results: LabResult[] }>("/get_lab_results", {});
  return data;
}

export function logout(): void {
  localStorage.removeItem("access_token");
  localStorage.removeItem("patient_id");
  localStorage.removeItem("patient_name");
  localStorage.removeItem("pairing_code");
  localStorage.removeItem("doctor_id");
  localStorage.removeItem("doctor_name");
  localStorage.removeItem("doctor_specialty");
}
