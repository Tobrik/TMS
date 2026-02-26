export interface DiseaseSlice {
  name: string;
  label: string;
  score: number;
}

export interface LabInfluence {
  markerName: string;
  status: "high" | "low";
  direction: string;
  effect: "boost" | "suppress";
  diseases: string[];
  delta: number;
}

export interface LabContext {
  adjustments: Record<string, number>;
  influences: LabInfluence[];
  hasData: boolean;
}

export interface DiagnosisResult {
  diseaseName: string;
  diseaseLabel: string;
  doctor: string;
  recommendation: string;
  slices: DiseaseSlice[];
  patientExplanation?: string;
  doctorExplanation?: string;
  labInfluences?: LabInfluence[];
}
