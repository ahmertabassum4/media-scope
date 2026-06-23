export type Confidences = Record<string, number>;

export interface AnalyzeResult {
  factuality: { label: string; confidences: Confidences };
  screenshot_b64: string;
}

export const FACT_SCALE = ["VERY LOW", "LOW", "MIXED", "HIGH", "VERY HIGH"];

// aligned to Oleg's factuality spectrum: red -> amber -> teal -> green
export const FACT_COLORS: Record<string, string> = {
  "VERY LOW": "#b91c1c",
  LOW: "#f59e0b",
  MIXED: "#ca8a04",
  HIGH: "#14b8a6",
  "VERY HIGH": "#15803d",
};

// ---- bias (separate /bias call, keyed to the same screenshot) ----
export interface BiasResult {
  label: string;
  confidences: Confidences;
  llm_label?: string | null;
}

// ---- factuality evidence overlay (separate /evidence call, same screenshot) ----
// One box per grounded region; bbox is [x1,y1,x2,y2] in the screenshot's
// natural pixel space. Matches the backend /evidence per-box schema.
export interface EvidenceBox {
  id: string;
  kind: string;
  role: string;
  bbox: [number, number, number, number];
  text: string | null;
  importance: number;
  reason: string;
}

export interface EvidenceResult {
  label: string;
  evidence: EvidenceBox[];
}

export const BIAS_SCALE = [
  "LEFT",
  "LEFT-CENTER",
  "LEAST BIASED",
  "RIGHT-CENTER",
  "RIGHT",
];

// aligned to Oleg's bias spectrum: blue -> light blue -> neutral -> orange -> red
export const BIAS_COLORS: Record<string, string> = {
  LEFT: "#2563eb",
  "LEFT-CENTER": "#60a5fa",
  "LEAST BIASED": "#64748b",
  "RIGHT-CENTER": "#fb923c",
  RIGHT: "#dc2626",
};

// ---- reaction clips: display-only output keyed to the factuality label ----
// Drop the matching files into frontend/public/reactions/ (a missing file
// degrades gracefully — the clip simply doesn't render).
export const REACTION_CLIPS: Record<string, string> = {
  "VERY LOW": "/reactions/very_low.mp4",
  LOW: "/reactions/low.mp4",
  MIXED: "/reactions/mixed.mp4",
  HIGH: "/reactions/high.mp4",
  "VERY HIGH": "/reactions/very_high.mp4",
};
