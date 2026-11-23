"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { useRouter } from "next/navigation";
import { Upload, ArrowLeft, FlaskConical, Loader2, ImageIcon, ShieldCheck } from "lucide-react";
import { Button } from "@/components/ui/button";
import { AppHeader } from "@/components/brand/AppHeader";
import { Logo } from "@/components/brand/Logo";
import { LabResultCard } from "@/components/LabResultCard";
import { uploadLabResult, getLabResults, type LabResult } from "@/lib/api";
import { t, getLang, type Lang } from "@/lib/i18n";
import { ps } from "@/lib/strings/pages";

async function cropTopOfImage(file: File, cropPercent = 0.15): Promise<File> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement("canvas");
      const cropY = Math.floor(img.height * cropPercent);
      canvas.width = img.width;
      canvas.height = img.height - cropY;
      const ctx = canvas.getContext("2d");
      if (!ctx) { resolve(file); return; }
      ctx.drawImage(img, 0, cropY, img.width, img.height - cropY, 0, 0, canvas.width, canvas.height);
      canvas.toBlob(
        (blob) => {
          if (!blob) { resolve(file); return; }
          resolve(new File([blob], file.name, { type: file.type }));
        },
        file.type,
        0.95,
      );
    };
    img.onerror = () => reject(new Error("Failed to load image"));
    img.src = URL.createObjectURL(file);
  });
}

export default function LabPage() {
  const router = useRouter();
  const fileInputRef = useRef<HTMLInputElement>(null);
  // "ru" on the server and first render; the stored language is read after mount
  const [lang, setLangState] = useState<Lang>("ru");

  const [patientId, setPatientId] = useState<number | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [currentResult, setCurrentResult] = useState<LabResult | null>(null);
  const [history, setHistory] = useState<LabResult[]>([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [historyError, setHistoryError] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [cropEnabled, setCropEnabled] = useState(true);

  useEffect(() => {
    const currentLang = getLang();
    setLangState(currentLang);
    document.documentElement.lang = currentLang;

    const id = localStorage.getItem("patient_id");
    const token = localStorage.getItem("access_token");
    if (!id || !token) {
      router.push("/");
      return;
    }
    setPatientId(parseInt(id, 10));
  }, [router]);

  const loadHistory = useCallback(async () => {
    if (!patientId) return;
    setHistoryLoading(true);
    try {
      const resp = await getLabResults(patientId);
      setHistory(resp.results);
      setHistoryError(false);
    } catch {
      // Shown as "could not load" (not as "no analyses")
      setHistoryError(true);
    } finally {
      setHistoryLoading(false);
    }
  }, [patientId]);

  useEffect(() => {
    if (patientId) loadHistory();
  }, [patientId, loadHistory]);

  async function handleFileSelect(file: File) {
    const allowed = ["image/jpeg", "image/png", "image/webp"];
    if (!allowed.includes(file.type)) {
      setError(t("onlyJpgPngWebp", lang));
      return;
    }
    if (file.size > 10 * 1024 * 1024) {
      setError(t("fileTooLarge", lang));
      return;
    }
    setError(null);
    let displayFile: File;
    try {
      displayFile = cropEnabled ? await cropTopOfImage(file) : file;
    } catch (err) {
      console.error("Lab image processing failed", err);
      setError(t("labImageError", lang));
      return;
    }
    setSelectedFile(file);
    setCurrentResult(null);

    const reader = new FileReader();
    reader.onload = (e) => setPreview(e.target?.result as string);
    reader.readAsDataURL(displayFile);
  }

  function handleDrop(e: React.DragEvent) {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFileSelect(file);
  }

  function handleDragOver(e: React.DragEvent) {
    e.preventDefault();
    setDragOver(true);
  }

  async function handleAnalyze() {
    if (!selectedFile || !patientId) return;
    setIsAnalyzing(true);
    setError(null);

    try {
      const fileToSend = cropEnabled ? await cropTopOfImage(selectedFile) : selectedFile;
      const result = await uploadLabResult(patientId, fileToSend);
      setCurrentResult(result);
      loadHistory();
    } catch (err: unknown) {
      // The backend detail is for developers; the patient sees a translated message
      console.error("Lab upload failed", err);
      setError(t("labUploadFailed", lang));
    } finally {
      setIsAnalyzing(false);
    }
  }

  function resetUpload() {
    setSelectedFile(null);
    setPreview(null);
    setCurrentResult(null);
    setError(null);
  }

  async function toggleCrop() {
    const next = !cropEnabled;
    setCropEnabled(next);
    if (selectedFile) {
      try {
        const displayFile = next ? await cropTopOfImage(selectedFile) : selectedFile;
        const reader = new FileReader();
        reader.onload = (e) => setPreview(e.target?.result as string);
        reader.readAsDataURL(displayFile);
      } catch (err) {
        console.error("Lab image processing failed", err);
        setError(t("labImageError", lang));
      }
    }
  }

  return (
    <div className="min-h-screen bg-background text-foreground">
      <AppHeader
        logoSlot={
          <div className="flex items-center gap-1">
            <Button variant="ghost" size="icon" onClick={() => router.push("/chat")} aria-label={ps("back", lang)}>
              <ArrowLeft />
            </Button>
            <Logo />
          </div>
        }
      />

      <main className="mx-auto max-w-[720px] space-y-6 p-4 sm:py-8">
        <div className="flex items-start gap-3">
          <span className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-accent text-accent-foreground">
            <FlaskConical className="h-5 w-5" aria-hidden="true" />
          </span>
          <div className="min-w-0">
            <h1 className="font-display text-2xl font-semibold leading-tight text-foreground sm:text-[32px]">
              {t("myLabTitle", lang)}
            </h1>
            <p className="mt-1 text-sm text-muted-foreground">{t("uploadPhotoSubtitle", lang)}</p>
          </div>
        </div>

        {!currentResult && (
          <button
            onClick={toggleCrop}
            aria-pressed={cropEnabled}
            className={`flex w-full min-h-11 items-center gap-2 rounded-xl border px-3 py-2 text-left text-[13px] transition-colors duration-150 outline-none focus-visible:ring-[3px] focus-visible:ring-ring ${
              cropEnabled
                ? "border-ok-border bg-ok-bg text-ok"
                : "border-border bg-card text-muted-foreground hover:text-foreground"
            }`}
          >
            <ShieldCheck className="h-4 w-4 shrink-0" />
            <span className="text-left">
              {cropEnabled ? t("privacyOn", lang) : t("privacyOff", lang)}
            </span>
          </button>
        )}

        {!currentResult && (
          <div
            className={`
              rounded-2xl border-2 border-dashed p-8 text-center shadow-card outline-none transition-colors duration-150 focus-visible:ring-[3px] focus-visible:ring-ring
              ${dragOver ? "border-primary bg-accent" : "border-input bg-card hover:border-ring hover:bg-accent/40"}
              ${selectedFile ? "" : "cursor-pointer"}
            `}
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onDragLeave={() => setDragOver(false)}
            onClick={() => !selectedFile && fileInputRef.current?.click()}
            role="button"
            tabIndex={0}
            aria-label={t("dragPhotoHere", lang)}
            onKeyDown={(e) => {
              // Only when the dropzone itself has focus, not a button inside it
              if (e.target !== e.currentTarget) return;
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                if (!selectedFile) fileInputRef.current?.click();
              }
            }}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="image/jpeg,image/png,image/webp"
              className="hidden"
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) handleFileSelect(file);
              }}
            />

            {preview ? (
              <div className="space-y-4">
                {cropEnabled && (
                  <p className="text-xs font-medium text-ok">
                    {t("croppedPreview", lang)}
                  </p>
                )}
                <img
                  src={preview}
                  alt="Preview"
                  className="mx-auto max-h-64 rounded-lg border border-border"
                />
                <p className="text-sm text-muted-foreground">{selectedFile?.name}</p>
                <div className="flex items-center justify-center gap-3">
                  <Button
                    onClick={(e) => { e.stopPropagation(); handleAnalyze(); }}
                    disabled={isAnalyzing}
                  >
                    {isAnalyzing ? (
                      <><Loader2 className="h-4 w-4 mr-2 animate-spin" />{t("recognizing", lang)}</>
                    ) : (
                      <><FlaskConical className="h-4 w-4 mr-2" />{t("recognize", lang)}</>
                    )}
                  </Button>
                  <Button
                    variant="outline"
                    onClick={(e) => { e.stopPropagation(); resetUpload(); }}
                    disabled={isAnalyzing}
                  >
                    {t("cancel", lang)}
                  </Button>
                </div>
              </div>
            ) : (
              <div className="space-y-3">
                <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-2xl bg-accent text-primary">
                  <Upload className="h-8 w-8" aria-hidden="true" />
                </div>
                <div>
                  <p className="text-sm font-medium text-foreground">{t("dragPhotoHere", lang)}</p>
                  <p className="mt-1 text-[13px] text-muted-foreground">{t("orClickToSelect", lang)}</p>
                </div>
              </div>
            )}
          </div>
        )}

        {error && (
          <div role="alert" className="rounded-xl border border-urgent-border border-l-4 border-l-urgent bg-urgent-bg p-3 text-sm text-urgent">
            {error}
          </div>
        )}

        {currentResult && (
          <div className="space-y-3">
            <LabResultCard result={currentResult} lang={lang} />
            <Button variant="outline" onClick={resetUpload} className="w-full">
              <ImageIcon className="h-4 w-4 mr-2" />
              {t("uploadAnother", lang)}
            </Button>
          </div>
        )}

        {history.length > 0 && (
          <div className="space-y-3">
            <h2 className="text-lg font-semibold text-foreground sm:text-xl">{t("previousAnalyses", lang)}</h2>
            {historyLoading ? (
              <div className="flex justify-center py-8">
                <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
              </div>
            ) : (
              history.map((r) => <LabResultCard key={r.result_id} result={r} lang={lang} />)
            )}
          </div>
        )}

        {history.length === 0 && !historyLoading && !currentResult && (
          <p className="py-8 text-center text-sm text-muted-foreground">
            {historyError ? t("labLoadFailed", lang) : t("noAnalysesYet", lang)}
          </p>
        )}
      </main>
    </div>
  );
}
