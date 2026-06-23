interface Props {
  text: string | null;
  loading: boolean;
  show: boolean;
}

export default function ExplanationCard({ text, loading, show }: Props) {
  if (!show) return null;
  return (
    <section className="relative overflow-hidden rounded-[10px] border border-line bg-white p-[18px] shadow-soft">
      <span className="absolute inset-y-0 left-0 w-[5px] bg-muted" />
      <div className="mb-3 flex items-center gap-2">
        <h2 className="text-sm font-extrabold uppercase tracking-wide">Why these ratings?</h2>
        <span className="rounded-full bg-slate-100 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-muted">
          GPT-5.5
        </span>
      </div>
      {loading ? (
        <div className="space-y-2">
          <div className="h-3 w-full animate-pulse rounded bg-slate-100" />
          <div className="h-3 w-11/12 animate-pulse rounded bg-slate-100" />
          <div className="h-3 w-4/5 animate-pulse rounded bg-slate-100" />
        </div>
      ) : (
        <p className="text-sm leading-relaxed text-slate-600">{text}</p>
      )}
    </section>
  );
}
