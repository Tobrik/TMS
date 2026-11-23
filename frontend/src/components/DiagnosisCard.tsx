"use client";

import { UserRound, Pill, Info, Brain, Loader2, ShieldAlert, Phone, CheckCircle2 } from "lucide-react";
import { ZoneBadge, ProbabilityBar, UrgencyBanner } from "@/components/brand";
import { cn } from "@/lib/utils";
import { tc } from "@/lib/strings/chat";
import type { DiagnosisResult, TriageReason } from "@/lib/types";
import { t, getLang, type Lang, type TranslationKey } from "@/lib/i18n";
import { getDiseaseDoctor, getDiseaseRecommendation } from "@/lib/diseaseWeights";

// The /analys result as the chat keeps it: low_confidence arrives as lowConfidence
export type DiagnosisCardData = DiagnosisResult & { lowConfidence?: boolean };

interface DiagnosisCardProps {
  diagnosis: DiagnosisCardData;
  lang?: Lang;
  /** Demo patient: show the "your case is in the doctor's queue" line under the result */
  demoQueued?: boolean;
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
    // The backend text can start with a warning emoji; the result band carries urgency instead
    return { doctor: diagnosis.doctor, recommendation: diagnosis.recommendation.replace(/^[\u26A0\uFE0F\s]+/, "") };
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

const ZONE_STYLE = {
  red: { edge: "border-l-urgent", band: "bg-urgent-bg", text: "text-urgent" },
  yellow: { edge: "border-l-warn", band: "bg-warn-bg", text: "text-warn" },
  green: { edge: "border-l-ok", band: "bg-ok-bg", text: "text-ok" },
  none: { edge: "border-l-input", band: "bg-muted", text: "text-muted-foreground" },
} as const;

function Section({ icon: Icon, label, children }: { icon: typeof UserRound; label: string; children: React.ReactNode }) {
  return (
    <div className="flex items-start gap-3">
      <span className="mt-0.5 inline-flex size-8 shrink-0 items-center justify-center rounded-full bg-accent text-accent-foreground">
        <Icon className="size-4" aria-hidden="true" />
      </span>
      <div className="min-w-0 flex-1">
        <p className="text-[13px] font-medium text-muted-foreground">{label}</p>
        <div className="text-[15px] leading-relaxed text-foreground">{children}</div>
      </div>
    </div>
  );
}

export function DiagnosisCard({ diagnosis, lang, demoQueued }: DiagnosisCardProps) {
  const l = lang || getLang();
  const zone = diagnosis.zone ?? null;
  const style = ZONE_STYLE[zone ?? "none"];
  const reasons = urgencyReasons(diagnosis, l);
  const advice = localizedAdvice(diagnosis, l);
  const band =
    zone === "red" ? tc("resultBandRed", l)
    : zone === "yellow" ? tc("resultBandYellow", l)
    : zone === "green" ? tc("resultBandGreen", l)
    : tc("resultBandUnknown", l);

  // The model's three scores are shown as shares of the three, like the old pie chart
  const slices = diagnosis.slices ?? [];
  const total = slices.reduce((sum, s) => sum + Math.max(s.score, 0), 0);

  return (
    <div className="space-y-2">
      {zone === "red" && (
        <UrgencyBanner
          title={tc("urgentTitle", l)}
          message={tc("urgentMessage", l)}
          action={
            <a
              href="tel:103"
              className="inline-flex min-h-11 items-center gap-2 rounded-[10px] bg-white px-4 text-sm font-semibold text-urgent outline-none focus-visible:ring-[3px] focus-visible:ring-white/70"
            >
              <Phone className="size-4" aria-hidden="true" />
              {tc("callNow", l)}
            </a>
          }
        />
      )}

      <article
        aria-label={tc("resultAria", l)}
        className={cn("overflow-hidden rounded-xl border border-border border-l-4 bg-card shadow-card", style.edge)}
      >
        {/* Band: zone + the triage reason in plain words */}
        <div className={cn("space-y-1.5 px-4 py-3", style.band)}>
          <div className="flex flex-wrap items-center gap-2">
            <ZoneBadge zone={zone} lang={l} />
            <p className={cn("text-[15px] font-semibold", style.text)}>{band}</p>
          </div>
          {reasons.length > 0 && zone !== "green" && (
            <p className="text-[13px] leading-snug text-foreground/80">
              <span className="font-medium">{t("urgencyWhy", l)}:</span> {reasons.join(" · ")}
            </p>
          )}
        </div>

        <div className="space-y-5 p-4">
          {/* Outside the 49 conditions the model knows: say so before the label */}
          {diagnosis.outOfScope && (
            <div role="note" className="flex items-start gap-3 rounded-xl border border-warn-border bg-warn-bg p-3">
              <ShieldAlert className="mt-0.5 size-5 shrink-0 text-warn" aria-hidden="true" />
              <div>
                <p className="text-sm font-semibold text-warn">{t("outOfScopeTitle", l)}</p>
                <p className="text-[13px] leading-relaxed text-foreground/80">
                  {diagnosis.outOfScope.likely_condition && (
                    <>{t("possibly", l)}: {diagnosis.outOfScope.likely_condition}. </>
                  )}
                  {t("outOfScopeBody", l)}
                </p>
              </div>
            </div>
          )}

          <div>
            <p className="text-[13px] font-medium text-muted-foreground">{t("preliminaryDiag", l)}</p>
            <h3 className="mt-0.5 font-display text-2xl font-semibold leading-tight text-foreground">
              {diagnosis.diseaseLabel}
            </h3>
          </div>

          {/* Calm notice, not an error: icon + text, neutral colours */}
          {diagnosis.lowConfidence && (
            <div role="note" className="flex items-start gap-3 rounded-xl bg-muted p-3">
              <Info className="mt-0.5 size-5 shrink-0 text-muted-foreground" aria-hidden="true" />
              <div>
                <p className="text-sm font-semibold text-foreground">{tc("lowConfidenceTitle", l)}</p>
                <p className="text-sm leading-relaxed text-foreground/80">{t("lowConfidenceNote", l)}</p>
              </div>
            </div>
          )}

          {total > 0 && (
            <div className="space-y-3">
              <p className="text-[13px] font-medium text-muted-foreground">{tc("possibleConditions", l)}</p>
              {slices.slice(0, 3).map((s, i) => (
                <ProbabilityBar
                  key={s.name}
                  label={s.label}
                  percent={(Math.max(s.score, 0) / total) * 100}
                  rank={i}
                />
              ))}
            </div>
          )}

          <Section icon={UserRound} label={t("recommendedDoctor", l)}>
            <span className="font-semibold">{advice.doctor}</span>
          </Section>

          {diagnosis.explanationPending && !diagnosis.patientExplanation && (
            <div className="flex items-center gap-2 text-[13px] text-muted-foreground" role="status">
              <Loader2 className="size-4 motion-safe:animate-spin" aria-hidden="true" />
              {t("explanationPending", l)}
            </div>
          )}
          {diagnosis.patientExplanation && (
            <Section icon={Brain} label={t("whyThisDiagnosis", l)}>
              {diagnosis.patientExplanation}
            </Section>
          )}

          <Section icon={Pill} label={t("recommendations", l)}>
            {advice.recommendation}
          </Section>
        </div>

        <div className="border-t border-border bg-muted/50 px-4 py-2.5">
          <p className="text-center text-[13px] leading-snug text-muted-foreground">
            {t("aiDisclaimerShort", l)} {tc("disclaimerResult", l)}
          </p>
        </div>
      </article>

      {demoQueued && (
        <p className="flex items-center gap-2 px-1 text-sm font-medium text-primary" role="status">
          <CheckCircle2 className="size-4 shrink-0" aria-hidden="true" />
          {tc("demoQueued", l)}
        </p>
      )}
    </div>
  );
}
