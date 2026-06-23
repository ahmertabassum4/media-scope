import { AnalyzeResult, FACT_SCALE, FACT_COLORS } from "../types";

interface Props {
  result: AnalyzeResult | null;
  loading: boolean;
}

export default function FactualityCard({ result, loading }: Props) {
  const fact = result?.factuality;
  const conf = fact ? fact.confidences[fact.label] ?? 0 : 0;

  return (
    <section className="relative overflow-hidden rounded-[10px] border border-line bg-white p-[18px] shadow-soft">
      <span className="absolute inset-y-0 left-0 w-[5px] bg-fact" />
      <div className="mb-4 flex items-center justify-between">
        <h2 className="text-sm font-extrabold uppercase tracking-wide">Factuality</h2>
        {fact && (
          <span className="rounded-full bg-fact/10 px-3 py-1 text-xs font-semibold text-fact">
            {(conf * 100).toFixed(1)}% confidence
          </span>
        )}
      </div>

      {loading ? (
        <div className="space-y-3">
          <div className="h-8 w-40 animate-pulse rounded-lg bg-slate-100" />
          <div className="h-3 w-full animate-pulse rounded bg-slate-100" />
        </div>
      ) : !fact ? (
        <p className="py-6 text-center text-sm text-slate-400">
          Upload a screenshot or capture a URL to see the factuality rating.
        </p>
      ) : (
        <>
          <div className="mb-5 flex items-baseline gap-3">
            <span
              className="text-[38px] font-extrabold leading-[1.04] tracking-tight"
              style={{ color: FACT_COLORS[fact.label] }}
            >
              {fact.label}
            </span>
          </div>

          <div className="flex h-3 overflow-hidden rounded-full ring-1 ring-slate-200">
            {FACT_SCALE.map((c) => (
              <div
                key={c}
                className="flex-1 transition"
                style={{
                  backgroundColor: FACT_COLORS[c],
                  opacity: c === fact.label ? 1 : 0.25,
                }}
              />
            ))}
          </div>
          <div className="mt-2 flex justify-between text-[10px] font-medium uppercase tracking-wide text-slate-400">
            {FACT_SCALE.map((c) => (
              <span
                key={c}
                className="flex-1 text-center"
                style={c === fact.label ? { color: FACT_COLORS[c], fontWeight: 700 } : {}}
              >
                {c}
              </span>
            ))}
          </div>

          <div className="mt-5 space-y-1.5">
            {FACT_SCALE.map((c) => {
              const p = fact.confidences[c] ?? 0;
              return (
                <div key={c} className="flex items-center gap-2 text-xs">
                  <span className="w-20 shrink-0 text-slate-500">{c}</span>
                  <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-slate-100">
                    <div
                      className="h-full rounded-full"
                      style={{ width: `${p * 100}%`, backgroundColor: FACT_COLORS[c] }}
                    />
                  </div>
                  <span className="w-10 shrink-0 text-right tabular-nums text-slate-400">
                    {(p * 100).toFixed(0)}%
                  </span>
                </div>
              );
            })}
          </div>
        </>
      )}
    </section>
  );
}
