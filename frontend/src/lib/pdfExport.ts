import jsPDF from "jspdf";
import type { PatientInfo, HistoryEntry } from "./api";
import { getDiseaseLabel } from "./diseaseWeights";
import { t, type Lang } from "./i18n";

// jsPDF's built-in fonts have no Cyrillic, so the report embeds Noto Sans (public/fonts).
// Only the regular face is bundled: there is no bold, so headings use a larger size instead.
const FONT_FILE = "NotoSans-Regular.ttf";
const FONT_NAME = "NotoSans";
let fontBase64: string | null = null;

function arrayBufferToBase64(buffer: ArrayBuffer): string {
  const bytes = new Uint8Array(buffer);
  const CHUNK = 0x8000; // String.fromCharCode with a spread needs bounded argument lists
  let binary = "";
  for (let i = 0; i < bytes.length; i += CHUNK) {
    binary += String.fromCharCode(...bytes.subarray(i, i + CHUNK));
  }
  return btoa(binary);
}

async function loadFontBase64(): Promise<string> {
  if (fontBase64 === null) {
    const res = await fetch(`/fonts/${FONT_FILE}`);
    if (!res.ok) throw new Error(`Font request failed: ${res.status}`);
    fontBase64 = arrayBufferToBase64(await res.arrayBuffer());
  }
  return fontBase64;
}

export async function generatePatientPDF(
  patientInfo: PatientInfo,
  history: HistoryEntry[],
  doctorName: string,
  lang: Lang = "ru"
): Promise<void> {
  const font = await loadFontBase64();
  const doc = new jsPDF();
  doc.addFileToVFS(FONT_FILE, font);
  doc.addFont(FONT_FILE, FONT_NAME, "normal");
  doc.setFont(FONT_NAME, "normal");
  const pageWidth = doc.internal.pageSize.getWidth();
  let y = 20;

  const locale = lang === "en" ? "en-US" : lang === "kk" ? "kk-KZ" : "ru-RU";

  // Title
  doc.setFontSize(18);
  doc.text(t("pdfTitle", lang), pageWidth / 2, y, { align: "center" });
  y += 12;

  // Patient info
  doc.setFontSize(12);
  doc.text(`${t("pdfPatient", lang)}: ${patientInfo.full_name}`, 20, y);
  y += 7;
  doc.text(`ID: ${patientInfo.patient_id}`, 20, y);
  y += 7;
  if (patientInfo.city) {
    doc.text(`${t("pdfCity", lang)}: ${patientInfo.city}`, 20, y);
    y += 7;
  }
  doc.text(`${t("pdfRegistration", lang)}: ${patientInfo.created_at?.split(" ")[0] || patientInfo.created_at}`, 20, y);
  y += 7;
  doc.text(`${t("pdfReportDate", lang)}: ${new Date().toLocaleDateString(locale)}`, 20, y);
  y += 7;
  doc.text(`${t("pdfDoctor", lang)}: ${doctorName}`, 20, y);
  y += 12;

  // Separator
  doc.setLineWidth(0.5);
  doc.line(20, y, pageWidth - 20, y);
  y += 10;

  // History header
  doc.setFontSize(14);
  doc.text(t("pdfDiagnosisHistory", lang), 20, y);
  y += 10;

  doc.setFontSize(10);

  if (history.length === 0) {
    doc.text(t("pdfNoRecords", lang), 20, y);
  } else {
    for (const entry of history) {
      // Check if we need a new page
      if (y > 260) {
        doc.addPage();
        y = 20;
      }

      const diseaseLabel = getDiseaseLabel(entry.disease_predict, lang);
      const date = entry.created_at?.split("T")[0] || entry.created_at?.split(" ")[0] || "";
      const score = entry.score !== null ? `${Math.round(entry.score * 100)}%` : "N/A";

      doc.setFontSize(11);
      doc.text(`${date} - ${diseaseLabel} (${score})`, 20, y);
      y += 6;

      doc.setFontSize(9);
      if (entry.disease_setup && entry.disease_setup !== "Nothing") {
        doc.text(`  ${t("pdfDoctorDiagnosis", lang)}: ${entry.disease_setup}`, 25, y);
        y += 5;
      }
      if (entry.recept) {
        const receptLines = doc.splitTextToSize(`  ${t("pdfRecommendations", lang)}: ${entry.recept}`, pageWidth - 50);
        doc.text(receptLines, 25, y);
        y += receptLines.length * 4.5;
      }
      if (entry.doctor_name) {
        doc.text(`  ${t("pdfVerifiedBy", lang)}: ${entry.doctor_name}`, 25, y);
        y += 5;
      }

      y += 5;
    }
  }

  // Footer
  y = Math.max(y + 10, 250);
  if (y > 270) {
    doc.addPage();
    y = 250;
  }
  doc.setFontSize(8);
  doc.text(
    t("pdfFooter", lang),
    pageWidth / 2,
    285,
    { align: "center" }
  );

  doc.save(`patient_${patientInfo.patient_id}_report.pdf`);
}
