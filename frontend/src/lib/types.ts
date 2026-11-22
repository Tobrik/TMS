export interface DiseaseSlice {
  name: string;
  label: string;
  score: number;
}

export interface DiagnosisResult {
  diseaseName: string;
  diseaseLabel: string;
  doctor: string;
  recommendation: string;
  slices: DiseaseSlice[];
  patientExplanation?: string;
  doctorExplanation?: string;
  explanationPending?: boolean;
  zone?: "red" | "yellow" | "green";
  reasons?: TriageReason[];
  outOfScope?: { likely_condition: string | null } | null;
}

export type { TriageReason } from "./api";
import type { TriageReason } from "./api";
