import { useRef, useState } from "react";

interface Props {
  onFile: (file: File) => void;
  onUrl: (url: string) => void;
  preview: string | null;
  fileName: string | null;
  busy: boolean;
}

export default function Dropzone({ onFile, onUrl, preview, fileName, busy }: Props) {
  const [drag, setDrag] = useState(false);
  const [url, setUrl] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  const pick = (files: FileList | null) => {
    const f = files?.[0];
    if (f && f.type.startsWith("image/")) onFile(f);
  };

  return (
    <div className="space-y-3">
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDrag(true);
        }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDrag(false);
          pick(e.dataTransfer.files);
        }}
        onClick={() => inputRef.current?.click()}
        className={`group cursor-pointer rounded-[10px] border-2 border-dashed p-6 transition
          ${
            drag
              ? "border-bias bg-teal-50 scale-[1.01]"
              : "border-slate-300 bg-white hover:border-bias/60 hover:bg-slate-50"
          }`}
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={(e) => pick(e.target.files)}
        />
        {preview ? (
          <div className="flex items-center gap-4">
            <img
              src={preview}
              alt="preview"
              className="h-20 w-28 rounded-lg object-cover object-top ring-1 ring-slate-200"
            />
            <div className="min-w-0">
              <p className="truncate text-sm font-semibold">{fileName || "Screenshot"}</p>
              <p className="text-xs text-slate-500">Click to replace, or drop a new image</p>
            </div>
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center py-4 text-center">
            <div className="mb-2 flex h-11 w-11 items-center justify-center rounded-full bg-bias/10 text-bias transition group-hover:scale-110">
              <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M12 16V4M12 4l-4 4M12 4l4 4" strokeLinecap="round" strokeLinejoin="round" />
                <path d="M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2" strokeLinecap="round" />
              </svg>
            </div>
            <p className="text-sm font-semibold">Drop a homepage screenshot or browse</p>
            <p className="text-xs text-slate-500">PNG or JPG</p>
          </div>
        )}
      </div>

      <div className="flex items-center gap-2 text-xs text-slate-400">
        <span className="h-px flex-1 bg-slate-200" />
        OR PASTE A URL
        <span className="h-px flex-1 bg-slate-200" />
      </div>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          if (url.trim()) onUrl(url.trim());
        }}
        className="flex gap-2"
      >
        <input
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          placeholder="https://example.com"
          className="flex-1 rounded-[9px] border border-line-strong bg-white px-3 py-2 text-sm outline-none focus:border-bias focus:ring-2 focus:ring-bias/20"
        />
        <button
          type="submit"
          disabled={busy || !url.trim()}
          className="rounded-[9px] bg-bias px-4 py-2 text-sm font-bold text-white transition hover:bg-[#0b5c55] disabled:cursor-not-allowed disabled:opacity-40"
        >
          Capture
        </button>
      </form>
    </div>
  );
}
