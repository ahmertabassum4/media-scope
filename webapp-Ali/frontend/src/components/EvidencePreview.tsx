import { useEffect, useReducer, useRef } from "react";
import { EvidenceBox } from "../types";

interface Props {
  preview: string | null;
  evidence: EvidenceBox[] | null;
}

interface ImageBox {
  left: number;
  top: number;
  width: number;
  height: number;
  imageWidth: number;
  imageHeight: number;
}

// Faithful React port of Oleg's preview substrate + renderEvidence overlay
// (origin/Oleg:application/frontend/app.js). The screenshot is shown at
// object-fit: contain inside a fixed 16:9 frame; boxes are drawn over it using
// his displayedImageBox letterbox math (centering offset, not a naive scale).
export default function EvidencePreview({ preview, evidence }: Props) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const imgRef = useRef<HTMLImageElement>(null);
  // Refs don't trigger renders; bump this to recompute geometry after the image
  // loads (natural size becomes known) and on window resize (frame size changes).
  const [, recompute] = useReducer((n: number) => n + 1, 0);

  useEffect(() => {
    const onResize = () => recompute();
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);

  // Oleg's displayedImageBox: the contained-image rectangle inside the frame,
  // with centering offset for object-fit: contain. Natural dimensions come from
  // the loaded <img> (naturalWidth/Height), since the /evidence payload has no
  // original_size.
  function displayedImageBox(): ImageBox | null {
    const wrap = wrapRef.current;
    const img = imgRef.current;
    if (!wrap || !img) return null;
    const frameWidth = wrap.clientWidth;
    const frameHeight = wrap.clientHeight;
    const imageWidth = img.naturalWidth;
    const imageHeight = img.naturalHeight;
    if (!frameWidth || !frameHeight || !imageWidth || !imageHeight) return null;
    const frameRatio = frameWidth / frameHeight;
    const imageRatio = imageWidth / imageHeight;
    if (frameRatio > imageRatio) {
      const height = frameHeight;
      const width = height * imageRatio;
      return { left: (frameWidth - width) / 2, top: 0, width, height, imageWidth, imageHeight };
    }
    const width = frameWidth;
    const height = width / imageRatio;
    return { left: 0, top: (frameHeight - height) / 2, width, height, imageWidth, imageHeight };
  }

  const imageBox = preview && evidence && evidence.length ? displayedImageBox() : null;
  const boxes =
    imageBox && evidence
      ? evidence.map((item, index) => {
          const [x1, y1, x2, y2] = item.bbox;
          const left = imageBox.left + (x1 / imageBox.imageWidth) * imageBox.width;
          const top = imageBox.top + (y1 / imageBox.imageHeight) * imageBox.height;
          const width = ((x2 - x1) / imageBox.imageWidth) * imageBox.width;
          const height = ((y2 - y1) / imageBox.imageHeight) * imageBox.height;
          // importance -> fill opacity (0.12..0.28); border stays solid.
          const alpha = 0.12 + Math.min(1, item.importance || 0.5) * 0.16;
          return {
            key: `${item.id}-${index}`,
            badge: `F${index + 1}`,
            title: item.reason || item.text || item.id,
            left,
            top,
            width: Math.max(8, width),
            height: Math.max(8, height),
            alpha,
          };
        })
      : [];

  return (
    <div ref={wrapRef} className={`image-frame preview-wrap ${preview ? "has-image" : ""}`}>
      <img ref={imgRef} src={preview ?? undefined} alt="" onLoad={() => recompute()} />
      <div className={`evidence-overlay ${boxes.length ? "visible" : ""}`}>
        {boxes.map((b) => (
          <div
            key={b.key}
            className="evidence-box factuality"
            title={b.title}
            style={{
              left: `${b.left}px`,
              top: `${b.top}px`,
              width: `${b.width}px`,
              height: `${b.height}px`,
              backgroundColor: `rgba(15, 118, 110, ${b.alpha})`,
            }}
          >
            <span>{b.badge}</span>
          </div>
        ))}
      </div>
      <div className="empty-preview">No image selected</div>
    </div>
  );
}
