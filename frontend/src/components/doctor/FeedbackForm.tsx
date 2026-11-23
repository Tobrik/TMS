"use client";

import { useState } from "react";
import { Loader2, Save, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { getDiseaseLabel } from "@/lib/diseaseWeights";
import { t, type Lang } from "@/lib/i18n";
import { postDoctorFeedback, type Condition, type DoctorFeedbackResponse, type HistoryEntry, type Zone } from "@/lib/api";

const SELECT_CLASS =
  "h-11 w-full rounded-md border border-input bg-card px-3 text-base text-foreground outline-none transition-[color,box-shadow] focus-visible:border-ring focus-visible:ring-[3px] focus-visible:ring-ring/60 disabled:opacity-50 md:text-sm";
const LABEL_CLASS = "text-[13px] font-medium text-muted-foreground";

const ZONE_KEYS = { red: "zoneShortRed", yellow: "zoneShortYellow", green: "zoneShortGreen" } as const;

export interface FeedbackFormProps {
  entry: HistoryEntry;
  conditions: Condition[] | null;
  lang: Lang;
  onSaved: (resp: DoctorFeedbackResponse) => void;
  onCancel: () => void;
}

export function FeedbackForm({ entry, conditions, lang, onSaved, onCancel }: FeedbackFormProps) {
  const [diagnosis, setDiagnosis] = useState(entry.disease_predict); // model's top-1 preselected
  const [other, setOther] = useState("");
  const [zone, setZone] = useState<Zone | "">(entry.zone ?? ""); // model's zone preselected
  const [note, setNote] = useState("");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");

  const isOther = diagnosis === "other";
  const canSave = !saving && !!conditions && !!diagnosis && !!zone && (!isOther || other.trim().length > 0);

  const submit = async () => {
    if (!canSave || !zone) return;
    setSaving(true);
    setError("");
    try {
      const resp = await postDoctorFeedback({
        day_id: entry.day_id,
        diagnosis,
        other_text: isOther ? other.trim() : null,
        zone,
        note: note.trim() || null,
      });
      if (resp.ok === false) {
        setError(t("saveFailed", lang));
        return;
      }
      onSaved(resp);
    } catch {
      setError(t("saveFailed", lang));
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="space-y-3 rounded-xl border border-border bg-card p-4">
      <div className="space-y-1.5">
        <label htmlFor="doctor-fb-diagnosis" className={LABEL_CLASS}>{t("fbDiagnosisLabel", lang)}</label>
        <select
          id="doctor-fb-diagnosis"
          value={diagnosis}
          onChange={(e) => setDiagnosis(e.target.value)}
          disabled={saving || !conditions}
          className={SELECT_CLASS}
        >
          {!diagnosis && <option value="" disabled>—</option>}
          {conditions?.map((c) => (
            <option key={c.name} value={c.name}>{getDiseaseLabel(c.name, lang)}</option>
          ))}
          <option value="other">{t("otherDiagnosis", lang)}</option>
        </select>
      </div>
      {isOther && (
        <Input
          maxLength={200}
          aria-label={t("fbDiagnosisLabel", lang)}
          placeholder={t("fbOtherPlaceholder", lang)}
          value={other}
          onChange={(e) => setOther(e.target.value)}
        />
      )}
      <div className="space-y-1.5">
        <label htmlFor="doctor-fb-zone" className={LABEL_CLASS}>{t("zoneLabel", lang)}</label>
        <select
          id="doctor-fb-zone"
          value={zone}
          onChange={(e) => setZone(e.target.value as Zone | "")}
          disabled={saving}
          className={SELECT_CLASS}
        >
          {!zone && <option value="" disabled>—</option>}
          {(["red", "yellow", "green"] as Zone[]).map((z) => (
            <option key={z} value={z}>{t(ZONE_KEYS[z], lang)}</option>
          ))}
        </select>
      </div>
      <div className="space-y-1.5">
        <label htmlFor="doctor-fb-note" className={LABEL_CLASS}>{t("fbNoteLabel", lang)}</label>
        <Textarea
          id="doctor-fb-note"
          maxLength={1000}
          rows={2}
          value={note}
          onChange={(e) => setNote(e.target.value)}
          disabled={saving}
        />
      </div>
      {error && <p role="alert" className="text-sm text-urgent">{error}</p>}
      <div className="flex gap-2">
        <Button size="sm" onClick={submit} disabled={!canSave}>
          {saving ? <Loader2 className="animate-spin" /> : <Save />}{t("save", lang)}
        </Button>
        <Button size="sm" variant="outline" onClick={onCancel} disabled={saving}>
          <X />{t("cancel", lang)}
        </Button>
      </div>
    </div>
  );
}
