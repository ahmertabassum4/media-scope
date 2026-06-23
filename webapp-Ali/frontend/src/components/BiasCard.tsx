import { BiasResult, BIAS_SCALE, BIAS_COLORS } from "../types";

interface Props {
  bias: BiasResult | null;
  loading: boolean;
}

export default function BiasCard({ bias, loading }: Props) {
  const conf = bias ? bias.confidences[bias.label] ?? 0 : 0;

  return (
    <section className="relative overflow-hidden rounded-[10px] border border-line bg-white p-[18px] shadow-soft">
      <span className="absolute inset-y-0 left-0 w-[5px] bg-bias" />
      <div className="mb-4 flex items-center justify-between">
        <h2 className="text-sm font-extrabold uppercase tracking-wide">Bias</h2>
        {bias && (
          <span className="rounded-full bg-bias/10 px-3 py-1 text-xs font-semibold text-bias">
            {(conf * 100).toFixed(1)}% confidence
          </span>
        )}
      </div>

      {loading ? (
        <div className="space-y-3">
          <div className="h-8 w-40 animate-pulse rounded-lg bg-slate-100" />
          <div className="h-3 w-full animate-pulse rounded bg-slate-100" />
        </div>
      ) : !bias ? (
        <p className="py-6 text-center text-sm text-slate-400">
          Upload a screenshot or capture a URL to see the political-bias rating.
        </p>
      ) : (
        <>
          <div className="mb-5 flex items-baseline gap-3">
            <span
              className="text-[38px] font-extrabold leading-[1.04] tracking-tight"
              style={{ color: BIAS_COLORS[bias.label] }}
            >
              {bias.label}
            </span>
          </div>

          <div className="flex h-3 overflow-hidden rounded-full ring-1 ring-slate-200">
            {BIAS_SCALE.map((c) => (
              <div
                key={c}
                className="flex-1 transition"
                style={{
                  backgroundColor: BIAS_COLORS[c],
                  opacity: c === bias.label ? 1 : 0.25,
                }}
              />
            ))}
          </div>
          <div className="mt-2 flex justify-between text-[10px] font-medium uppercase tracking-wide text-slate-400">
            {BIAS_SCALE.map((c) => (
              <span
                key={c}
                className="flex-1 text-center"
                style={c === bias.label ? { color: BIAS_COLORS[c], fontWeight: 700 } : {}}
              >
                {c}
              </span>
            ))}
          </div>

          <div className="mt-5 space-y-1.5">
            {BIAS_SCALE.map((c) => {
              const p = bias.confidences[c] ?? 0;
              return (
                <div key={c} className="flex items-center gap-2 text-xs">
                  <span className="w-24 shrink-0 text-slate-500">{c}</span>
                  <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-slate-100">
                    <div
                      className="h-full rounded-full"
                      style={{ width: `${p * 100}%`, backgroundColor: BIAS_COLORS[c] }}
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
