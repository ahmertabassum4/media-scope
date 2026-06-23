import { useState } from "react";
import Dropzone from "./components/Dropzone";
import FactualityCard from "./components/FactualityCard";
import ReactionClip from "./components/ReactionClip";
import BiasCard from "./components/BiasCard";
import ExplanationCard from "./components/ExplanationCard";
import EvidencePreview from "./components/EvidencePreview";
import EvidenceList from "./components/EvidenceList";
import { analyzeImage, analyzeUrl, analyzeBias, explain, fetchEvidence } from "./api";
import { AnalyzeResult, BiasResult, EvidenceResult } from "./types";

export default function App() {
  const [preview, setPreview] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);
  const [pendingFile, setPendingFile] = useState<File | null>(null);
  const [pendingUrl, setPendingUrl] = useState<string | null>(null);

  const [result, setResult] = useState<AnalyzeResult | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [bias, setBias] = useState<BiasResult | null>(null);
  const [biasLoading, setBiasLoading] = useState(false);
  const [explanation, setExplanation] = useState<string | null>(null);
  const [explaining, setExplaining] = useState(false);
  const [evidence, setEvidence] = useState<EvidenceResult | null>(null);
  const [evidenceLoading, setEvidenceLoading] = useState(false);
  const [evidenceError, setEvidenceError] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const status = analyzing ? "Analyzing" : "Ready";

  const onFile = (f: File) => {
    setPendingFile(f);
    setPendingUrl(null);
    setFileName(f.name);
    setPreview(URL.createObjectURL(f));
  };

  const onUrl = (u: string) => {
    setPendingUrl(u);
    setPendingFile(null);
    setFileName(u);
    setPreview(null);
    run(undefined, u);
  };

  async function run(file?: File, url?: string) {
    const f = file ?? pendingFile;
    const u = url ?? pendingUrl;
    if (!f && !u) return;
    setError(null);
    setResult(null);
    setBias(null);
    setExplanation(null);
    setEvidence(null);
    setEvidenceError(null);
    setAnalyzing(true);
    try {
      const res = f ? await analyzeImage(f) : await analyzeUrl(u!);
      setResult(res);
      setPreview(`data:image/png;base64,${res.screenshot_b64}`);
      setAnalyzing(false);

      // bias prediction runs after factuality renders; the explanation waits for it
      // so the paragraph can reference both ratings (falls back to factuality-only).
      setBiasLoading(true);
      setExplaining(true);
      let biasLabel: string | undefined;
      try {
        const b = await analyzeBias(res.screenshot_b64);
        setBias(b);
        biasLabel = b.label;
      } catch (e) {
        console.error("bias prediction failed:", e);
        setBias(null);
      } finally {
        setBiasLoading(false);
      }

      try {
        const text = await explain(res.screenshot_b64, res.factuality.label, biasLabel);
        setExplanation(text);
      } catch (e) {
        setExplanation("Explanation unavailable: " + (e as Error).message);
      } finally {
        setExplaining(false);
      }

      // evidence overlay is the slowest step (OCR + a vision call); it runs last
      // so the factuality/bias cards and explanation are already on screen.
      setEvidenceLoading(true);
      try {
        const ev = await fetchEvidence(res.screenshot_b64);
        setEvidence(ev);
      } catch (e) {
        setEvidence(null);
        setEvidenceError("Evidence unavailable: " + (e as Error).message);
      } finally {
        setEvidenceLoading(false);
      }
    } catch (e) {
      setError((e as Error).message);
      setAnalyzing(false);
    }
  }

  return (
    <div className="min-h-screen px-6 py-10">
      <div className="mx-auto max-w-6xl">
        <header className="mb-8 flex items-center justify-between">
          <div>
            <h1 className="text-[32px] font-extrabold leading-[1.08] tracking-tight">
              Media bias & Factuality
            </h1>
            <p className="mt-1 text-sm text-muted">
              Visual audit of a news homepage — factuality and political bias from the trained classifiers.
            </p>
          </div>
          <span
            className={`rounded-full px-4 py-1.5 text-sm font-bold ring-1 ${
              analyzing
                ? "bg-amber-50 text-amber-700 ring-amber-200"
                : "bg-bias/10 text-bias ring-bias/30"
            }`}
          >
            {status}
          </span>
        </header>

        <div className="grid gap-[18px] lg:grid-cols-[minmax(0,1fr)_minmax(0,1.1fr)]">
          {/* left: input */}
          <div className="space-y-[18px]">
            <div className="rounded-[10px] border border-line bg-white p-[18px] shadow-soft">
              <Dropzone
                onFile={onFile}
                onUrl={onUrl}
                preview={preview}
                fileName={fileName}
                busy={analyzing}
              />
              <button
                onClick={() => run()}
                disabled={analyzing || (!pendingFile && !pendingUrl)}
                className="mt-4 w-full rounded-[9px] bg-ink py-3 text-sm font-bold text-white shadow-soft transition hover:bg-[#273142] disabled:cursor-not-allowed disabled:opacity-40 disabled:shadow-none"
              >
                {analyzing ? "Analyzing…" : "Analyze"}
              </button>
              {error && (
                <p className="mt-3 rounded-lg bg-red-50 px-3 py-2 text-xs text-red-600">
                  {error}
                </p>
              )}
            </div>

            {/* contain-fitted preview + factuality evidence overlay */}
            <EvidencePreview preview={preview} evidence={evidence?.evidence ?? null} />
          </div>

          {/* right: results */}
          <div className="space-y-[18px]">
            <FactualityCard result={result} loading={analyzing} />
            <EvidenceList
              evidence={evidence?.evidence ?? null}
              loading={evidenceLoading}
              error={evidenceError}
            />
            <ReactionClip label={result?.factuality.label ?? null} />
            <BiasCard bias={bias} loading={biasLoading} />
            <ExplanationCard
              text={explanation}
              loading={explaining}
              show={!!result || explaining}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
