"use client";

import {
  Stethoscope,
  UserRound,
  Pill,
  AlertTriangle,
  ChevronRight,
  Brain,
  Loader2,
  ShieldAlert,
} from "lucide-react";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { PieChart } from "@/components/PieChart";
import type { DiagnosisResult, TriageReason } from "@/lib/types";
import { t, getLang, type Lang, type TranslationKey } from "@/lib/i18n";
import { getDiseaseDoctor, getDiseaseRecommendation } from "@/lib/diseaseWeights";

// The /analys result as the chat keeps it: low_confidence arrives as lowConfidence
export type DiagnosisCardData = DiagnosisResult & { lowConfidence?: boolean };

interface DiagnosisCardProps {
  diagnosis: DiagnosisCardData;
  lang?: Lang;
}

const REASON_KEYS: Record<TriageReason["code"], TranslationKey> = {
  red_flag: "reasonRedFlag",
  red_flag_combo: "reasonRedFlagCombo",
  llm_emergency: "reasonLlmEmergency",
  out_of_scope: "reasonOutOfScope",
  red_mass: "reasonRedMass",
  yellow_flag: "reasonYellowFlag",
  yellow_mass: "reasonYellowMass",
};

/** Distinct reasons behind the final zone, e.g. "probability of a dangerous condition: 22%". */
function urgencyReasons(diagnosis: DiagnosisResult, l: Lang): string[] {
  const texts = (diagnosis.reasons ?? [])
    .filter((r) => r.zone === diagnosis.zone)
    .map((r) => {
      const text = t(REASON_KEYS[r.code], l);
      return r.value !== undefined ? `${text}: ${Math.round(r.value * 100)}%` : text;
    });
  return Array.from(new Set(texts)).slice(0, 3);
}

/**
 * Doctor and recommendation text for the patient's language.
 * The backend writes both in Russian only, so for en/kk they come from the
 * translated table (diseaseWeights.ts); the backend text stays the fallback.
 */
function localizedAdvice(diagnosis: DiagnosisResult, l: Lang): { doctor: string; recommendation: string } {
  if (l === "ru") {
    return { doctor: diagnosis.doctor, recommendation: diagnosis.recommendation };
  }
  const doctor = getDiseaseDoctor(diagnosis.diseaseName, l) || diagnosis.doctor;
  let recommendation = getDiseaseRecommendation(diagnosis.diseaseName, l) || diagnosis.recommendation;
  const notes: string[] = [];
  if (diagnosis.zone === "red") notes.push(t("urgentPrefix", l));
  if (diagnosis.outOfScope) {
    const likely = diagnosis.outOfScope.likely_condition
      ? ` (${t("possibly", l)}: ${diagnosis.outOfScope.likely_condition})`
      : "";
    notes.push(t("outOfScopeNote", l).replace("{likely}", likely));
  }
  if (notes.length > 0) recommendation = `${notes.join(" ")} ${recommendation}`;
  return { doctor, recommendation };
}

// A missing zone is "unknown", never "low": it must not look like a green result
function zoneToUrgency(zone?: "red" | "yellow" | "green"): "low" | "medium" | "high" | "unknown" {
  if (zone === "red") return "high";
  if (zone === "yellow") return "medium";
  if (zone === "green") return "low";
  return "unknown";
}

export function DiagnosisCard({ diagnosis, lang }: DiagnosisCardProps) {
  const l = lang || getLang();
  const urgency = zoneToUrgency(diagnosis.zone);

  const urgencyConfig = {
    low: {
      color: "bg-emerald-50 border-emerald-200",
      badge: "bg-emerald-100 text-emerald-700",
      label: t("lowUrgency", l),
    },
    medium: {
      color: "bg-amber-50 border-amber-200",
      badge: "bg-amber-100 text-amber-700",
      label: t("mediumUrgency", l),
    },
    high: {
      color: "bg-red-50 border-red-200",
      badge: "bg-red-100 text-red-700",
      label: t("highUrgency", l),
    },
    unknown: {
      color: "bg-gray-50 border-gray-300",
      badge: "bg-gray-200 text-gray-700",
      label: t("zoneUnknown", l),
    },
  };

  const config = urgencyConfig[urgency];
  const reasons = urgencyReasons(diagnosis, l);
  const advice = localizedAdvice(diagnosis, l);

  return (
    <Card
      className={`overflow-hidden border-2 ${config.color} shadow-lg max-w-[90%]`}
    >
      {/* Header */}
      <div className="px-4 py-3 border-b border-inherit">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Stethoscope className="h-5 w-5 text-blue-600" />
            <span className="font-semibold text-sm text-gray-700">
              {t("preliminaryDiag", l)}
            </span>
          </div>
          <Badge className={`text-xs ${config.badge} border-0`}>
            {urgency === "high" && (
              <AlertTriangle className="h-3 w-3 mr-1" />
            )}
            {config.label}
          </Badge>
        </div>
      </div>

      {/* Body */}
      <div className="p-4 space-y-4">
        {/* Outside the 49 conditions the model knows: say so before the label */}
        {diagnosis.outOfScope && (
          <div className="flex items-start gap-3 bg-amber-50 rounded-xl p-3 border border-amber-200">
            <ShieldAlert className="h-5 w-5 text-amber-600 mt-0.5 shrink-0" />
            <div>
              <p className="text-sm font-semibold text-amber-800">{t("outOfScopeTitle", l)}</p>
              <p className="text-xs text-amber-700 leading-relaxed">
                {diagnosis.outOfScope.likely_condition && (
                  <>{t("possibly", l)}: {diagnosis.outOfScope.likely_condition}. </>
                )}
                {t("outOfScopeBody", l)}
              </p>
            </div>
          </div>
        )}

        {/* Disease name */}
        <div>
          <h3 className="text-lg font-bold text-gray-900">
            {diagnosis.diseaseLabel}
          </h3>
        </div>

        {/* Model is unsure: the amber notice sits above the top-3 chart; icon + text, not color only */}
        {diagnosis.lowConfidence && (
          <div role="note" className="flex items-start gap-3 bg-amber-50 rounded-xl p-3 border border-amber-300">
            <AlertTriangle className="h-5 w-5 text-amber-600 mt-0.5 shrink-0" aria-hidden="true" />
            <p className="text-sm text-amber-900 leading-relaxed">{t("lowConfidenceNote", l)}</p>
          </div>
        )}

        {/* Pie chart */}
        {diagnosis.slices && diagnosis.slices.length > 0 && (
          <div className="flex justify-center py-2">
            <div className="w-56">
              <PieChart slices={diagnosis.slices} size={130} lang={l} />
            </div>
          </div>
        )}

        {/* Doctor */}
        <div className="flex items-start gap-3 bg-white/60 rounded-xl p-3">
          <UserRound className="h-5 w-5 text-blue-600 mt-0.5 shrink-0" />
          <div>
            <p className="text-xs text-gray-500 font-medium">
              {t("recommendedDoctor", l)}
            </p>
            <p className="text-sm font-semibold text-gray-800">
              {advice.doctor}
            </p>
          </div>
          <ChevronRight className="h-4 w-4 text-gray-400 ml-auto mt-0.5" />
        </div>

        {/* Why this urgency */}
        {reasons.length > 0 && urgency !== "low" && (
          <p className="text-xs text-gray-500">
            <span className="font-medium">{t("urgencyWhy", l)}:</span> {reasons.join(" · ")}
          </p>
        )}

        {/* AI Explanation for patient */}
        {diagnosis.explanationPending && !diagnosis.patientExplanation && (
          <div className="flex items-center gap-2 text-xs text-blue-500 px-1">
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
            {t("explanationPending", l)}
          </div>
        )}
        {diagnosis.patientExplanation && (
          <div className="flex items-start gap-3 bg-blue-50/80 rounded-xl p-3 border border-blue-100">
            <Brain className="h-5 w-5 text-blue-600 mt-0.5 shrink-0" />
            <div>
              <p className="text-xs text-blue-500 font-medium">
                {t("whyThisDiagnosis", l)}
              </p>
              <p className="text-sm text-gray-700 leading-relaxed">
                {diagnosis.patientExplanation}
              </p>
            </div>
          </div>
        )}

        {/* Recommendation */}
        <div className="flex items-start gap-3 bg-white/60 rounded-xl p-3">
          <Pill className="h-5 w-5 text-emerald-600 mt-0.5 shrink-0" />
          <div>
            <p className="text-xs text-gray-500 font-medium">{t("recommendations", l)}</p>
            <p className="text-sm text-gray-700 leading-relaxed">
              {advice.recommendation}
            </p>
          </div>
        </div>
      </div>

      {/* Footer */}
      <div className="px-4 py-2.5 bg-white/40 border-t border-inherit">
        <p className="text-xs text-gray-400 text-center">
          {t("aiDisclaimerShort", l)}
        </p>
      </div>
    </Card>
  );
}
