"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { useRouter } from "next/navigation";
import { Upload, ArrowLeft, FlaskConical, Loader2, ImageIcon, ShieldCheck } from "lucide-react";
import { Button } from "@/components/ui/button";
import { LabResultCard } from "@/components/LabResultCard";
import { uploadLabResult, getLabResults, type LabResult } from "@/lib/api";
import { t, getLang, type Lang } from "@/lib/i18n";

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
  const [lang] = useState<Lang>(getLang());

  const [patientId, setPatientId] = useState<number | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [currentResult, setCurrentResult] = useState<LabResult | null>(null);
  const [history, setHistory] = useState<LabResult[]>([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [cropEnabled, setCropEnabled] = useState(true);

  useEffect(() => {
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
    } catch {
      // silently fail
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
    setSelectedFile(file);
    setCurrentResult(null);

    const displayFile = cropEnabled ? await cropTopOfImage(file) : file;
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
      const msg =
        err && typeof err === "object" && "response" in err
          ? (err as { response?: { data?: { detail?: string } } }).response?.data?.detail
          : undefined;
      setError(msg || t("failedRecognize", lang));
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
      const displayFile = next ? await cropTopOfImage(selectedFile) : selectedFile;
      const reader = new FileReader();
      reader.onload = (e) => setPreview(e.target?.result as string);
      reader.readAsDataURL(displayFile);
    }
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white border-b border-gray-200 px-4 py-3 flex items-center justify-between sticky top-0 z-10">
        <div className="flex items-center gap-3">
          <Button variant="ghost" size="icon" onClick={() => router.push("/chat")}>
            <ArrowLeft className="h-5 w-5 text-gray-500" />
          </Button>
          <div className="w-10 h-10 bg-gradient-to-br from-purple-600 to-pink-500 rounded-xl flex items-center justify-center shadow-md">
            <FlaskConical className="h-5 w-5 text-white" />
          </div>
          <div>
            <h1 className="font-semibold text-gray-900 text-sm">{t("myLabTitle", lang)}</h1>
            <p className="text-xs text-gray-500">{t("uploadPhotoSubtitle", lang)}</p>
          </div>
        </div>
      </header>

      <main className="max-w-2xl mx-auto p-4 space-y-6">
        {!currentResult && (
          <button
            onClick={toggleCrop}
            className={`flex items-center gap-2 text-xs px-3 py-2 rounded-lg border transition-colors w-full ${
              cropEnabled
                ? "bg-emerald-50 border-emerald-200 text-emerald-700"
                : "bg-gray-50 border-gray-200 text-gray-500"
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
              border-2 border-dashed rounded-2xl p-8 text-center transition-colors cursor-pointer
              ${dragOver ? "border-purple-400 bg-purple-50" : "border-gray-300 bg-white hover:border-purple-300 hover:bg-purple-50/50"}
            `}
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onDragLeave={() => setDragOver(false)}
            onClick={() => !selectedFile && fileInputRef.current?.click()}
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
                  <p className="text-xs text-emerald-600 font-medium">
                    {t("croppedPreview", lang)}
                  </p>
                )}
                <img
                  src={preview}
                  alt="Preview"
                  className="max-h-64 mx-auto rounded-lg shadow-md"
                />
                <p className="text-sm text-gray-600">{selectedFile?.name}</p>
                <div className="flex items-center justify-center gap-3">
                  <Button
                    onClick={(e) => { e.stopPropagation(); handleAnalyze(); }}
                    disabled={isAnalyzing}
                    className="bg-gradient-to-r from-purple-600 to-pink-500 hover:from-purple-700 hover:to-pink-600 text-white"
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
                <div className="w-16 h-16 mx-auto bg-purple-100 rounded-2xl flex items-center justify-center">
                  <Upload className="h-8 w-8 text-purple-500" />
                </div>
                <div>
                  <p className="text-sm font-medium text-gray-700">{t("dragPhotoHere", lang)}</p>
                  <p className="text-xs text-gray-500 mt-1">{t("orClickToSelect", lang)}</p>
                </div>
              </div>
            )}
          </div>
        )}

        {error && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-3 text-sm text-red-700">
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
            <h2 className="text-sm font-semibold text-gray-700">{t("previousAnalyses", lang)}</h2>
            {historyLoading ? (
              <div className="flex justify-center py-8">
                <Loader2 className="h-6 w-6 animate-spin text-gray-400" />
              </div>
            ) : (
              history.map((r) => <LabResultCard key={r.result_id} result={r} lang={lang} />)
            )}
          </div>
        )}

        {history.length === 0 && !historyLoading && !currentResult && (
          <p className="text-center text-sm text-gray-400 py-8">
            {t("noAnalysesYet", lang)}
          </p>
        )}
      </main>
    </div>
  );
}
