import { EvidenceBox } from "../types";

interface Props {
  evidence: EvidenceBox[] | null;
  loading: boolean;
  error: string | null;
}

// Oleg's renderEvidenceList, beneath the factuality card: a titled list of up to
// five rows, each a re-indexed "F" badge + the region's reason (falling back to
// text, then id). Rendered as its own panel so the FactualityCard stays unchanged.
// A failed /evidence call surfaces an "Evidence unavailable: …" message here
// rather than disappearing silently.
export default function EvidenceList({ evidence, loading, error }: Props) {
  const items = evidence ?? [];
  if (!loading && !items.length && !error) return null;

  return (
    <section className="rounded-[10px] border border-line bg-white p-[18px] shadow-soft">
      <div className="evidence-list visible">
        <div className="evidence-title">Evidence</div>
        {items.length ? (
          items.slice(0, 5).map((item, index) => (
            <div key={`${item.id}-${index}`} className="evidence-row factuality">
              <span className="evidence-id">{`F${index + 1}`}</span>
              <span>{item.reason || item.text || item.id}</span>
            </div>
          ))
        ) : error ? (
          <p className="text-xs text-red-600">{error}</p>
        ) : (
          <p className="text-xs text-slate-400">Locating evidence…</p>
        )}
      </div>
    </section>
  );
}
