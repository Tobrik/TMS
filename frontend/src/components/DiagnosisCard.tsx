"use client";

import {
  Stethoscope,
  UserRound,
  Pill,
  AlertTriangle,
  ChevronRight,
  Brain,
  FlaskConical,
} from "lucide-react";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { PieChart } from "@/components/PieChart";
import type { DiagnosisResult } from "@/lib/types";
import { t, getLang, type Lang } from "@/lib/i18n";

interface DiagnosisCardProps {
  diagnosis: DiagnosisResult;
  lang?: Lang;
}

function getUrgencyLevel(diseaseName: string): "low" | "medium" | "high" {
  const high = ["Meningitis", "Appendicitis", "Type 1 Diabetes", "Myocardial Infarction"];
  const medium = ["Pneumonia", "Scarlet Fever", "Influenza"];
  if (high.includes(diseaseName)) return "high";
  if (medium.includes(diseaseName)) return "medium";
  return "low";
}

export function DiagnosisCard({ diagnosis, lang }: DiagnosisCardProps) {
  const l = lang || getLang();
  const urgency = getUrgencyLevel(diagnosis.diseaseName);

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
  };

  const config = urgencyConfig[urgency];

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
        {/* Disease name */}
        <div>
          <h3 className="text-lg font-bold text-gray-900">
            {diagnosis.diseaseLabel}
          </h3>
        </div>

        {/* Pie chart */}
        {diagnosis.slices && diagnosis.slices.length > 0 && (
          <div className="flex justify-center py-2">
            <div className="w-56">
              <PieChart slices={diagnosis.slices} size={130} lang={l} />
            </div>
          </div>
        )}

        {/* Lab influences */}
        {diagnosis.labInfluences && diagnosis.labInfluences.length > 0 && (
          <div className="bg-violet-50/80 rounded-xl p-3 border border-violet-200">
            <div className="flex items-center gap-2 mb-2">
              <FlaskConical className="h-4 w-4 text-violet-600" />
              <span className="text-xs font-semibold text-violet-700">
                {t("labDataConsidered", l)}
              </span>
            </div>
            <div className="space-y-1.5">
              {diagnosis.labInfluences.map((inf, i) => (
                <div key={i} className="flex items-center gap-2 text-sm">
                  <span className="text-gray-700 font-medium">{inf.markerName}</span>
                  <span
                    className={`text-xs px-1.5 py-0.5 rounded font-medium ${
                      inf.status === "high"
                        ? "bg-red-100 text-red-700"
                        : "bg-amber-100 text-amber-700"
                    }`}
                  >
                    {inf.direction}
                  </span>
                  <span className="text-gray-400 text-xs">→</span>
                  <span className="text-xs text-gray-600">
                    {inf.effect === "boost" ? t("boostEffect", l) : t("suppressEffect", l)}{" "}
                    {inf.diseases.map((d) => d).join(", ")}
                  </span>
                </div>
              ))}
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
              {diagnosis.doctor}
            </p>
          </div>
          <ChevronRight className="h-4 w-4 text-gray-400 ml-auto mt-0.5" />
        </div>

        {/* AI Explanation for patient */}
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
              {diagnosis.recommendation}
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
