import { useState, useEffect } from "react";
import { REACTION_CLIPS } from "../types";

interface Props {
  // factuality label the clip reacts to; clip is OUTPUT-only, never a model input
  label: string | null;
}

export default function ReactionClip({ label }: Props) {
  const src = label ? REACTION_CLIPS[label] : undefined;
  const [failed, setFailed] = useState(false);

  // reset the error state whenever the predicted label (and thus the source) changes
  useEffect(() => {
    setFailed(false);
  }, [src]);

  if (!src || failed) return null;

  return (
    <div className="overflow-hidden rounded-[10px] border border-line bg-white p-2 shadow-soft">
      <video
        key={src}
        src={src}
        autoPlay
        muted
        loop
        playsInline
        controls
        onError={() => setFailed(true)}
        className="w-full rounded-[8px]"
      />
    </div>
  );
}
