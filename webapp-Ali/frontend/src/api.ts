import { AnalyzeResult, BiasResult, EvidenceResult } from "./types";

export async function analyzeImage(file: File): Promise<AnalyzeResult> {
  const fd = new FormData();
  fd.append("image", file);
  return post("/analyze", fd);
}

export async function analyzeUrl(url: string): Promise<AnalyzeResult> {
  const fd = new FormData();
  fd.append("url", url);
  return post("/analyze", fd);
}

export async function explain(
  screenshot_b64: string,
  factuality_label: string,
  bias_label?: string
): Promise<string> {
  const res = await fetch("/explain", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ screenshot_b64, factuality_label, bias_label }),
  });
  if (!res.ok) throw new Error((await res.json()).detail || "explain failed");
  return (await res.json()).explanation;
}

export async function analyzeBias(screenshot_b64: string): Promise<BiasResult> {
  const res = await fetch("/bias", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ screenshot_b64 }),
  });
  if (!res.ok) throw new Error((await res.json()).detail || "bias failed");
  return (await res.json()).bias;
}

// Modeled on analyzeBias: same screenshot sent as a JSON body. A full-page
// screenshot's base64 exceeds Starlette's 1 MB multipart-part limit, so it must
// ride in JSON like /bias rather than a FormData field. The /evidence response
// is {label, evidence} at top level — no wrapper key to unwrap.
export async function fetchEvidence(screenshot_b64: string): Promise<EvidenceResult> {
  const res = await fetch("/evidence", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ screenshot_b64 }),
  });
  if (!res.ok) throw new Error((await res.json()).detail || "evidence failed");
  return res.json();
}

async function post(url: string, fd: FormData): Promise<AnalyzeResult> {
  const res = await fetch(url, { method: "POST", body: fd });
  if (!res.ok) throw new Error((await res.json()).detail || "analyze failed");
  return res.json();
}
