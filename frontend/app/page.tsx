"use client";

import { useState, useRef, useCallback, useEffect } from "react";

// ─── Types ────────────────────────────────────────────────────────────────────
interface Prediction {
  label: "Real" | "AI-Generated";
  prob_ai: number;
  prob_real: number;
  confidence: number;
}

type AppState = "idle" | "preview" | "scanning" | "result" | "error";

// ─── Icons ────────────────────────────────────────────────────────────────────
const UploadIcon = () => (
  <svg className="w-12 h-12" fill="none" viewBox="0 0 24 24" stroke="currentColor">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
      d="M3 16.5v2.25A2.25 2.25 0 0 0 5.25 21h13.5A2.25 2.25 0 0 0 21 18.75V16.5m-13.5-9L12 3m0 0 4.5 4.5M12 3v13.5" />
  </svg>
);

const ScanIcon = () => (
  <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
      d="m21 21-5.197-5.197m0 0A7.5 7.5 0 1 0 5.196 5.196a7.5 7.5 0 0 0 10.607 10.607Z" />
  </svg>
);

const ShieldCheckIcon = () => (
  <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
      d="M9 12.75 11.25 15 15 9.75m-3-7.036A11.959 11.959 0 0 1 3.598 6 11.99 11.99 0 0 0 3 9.749c0 5.592 3.824 10.29 9 11.623 5.176-1.332 9-6.03 9-11.622 0-1.31-.21-2.571-.598-3.751h-.152c-3.196 0-6.1-1.248-8.25-3.285Z" />
  </svg>
);

const BrainIcon = () => (
  <svg className="w-7 h-7" viewBox="0 0 24 24" fill="none" stroke="currentColor">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.8}
      d="M9.75 3.104v5.714a2.25 2.25 0 0 1-.659 1.591L5 14.5M9.75 3.104c-.251.023-.501.05-.75.082m.75-.082a24.301 24.301 0 0 1 4.5 0m0 0v5.714c0 .597.237 1.17.659 1.591L19.8 15.3M14.25 3.104c.251.023.501.05.75.082M19.8 15.3l-1.57.393A9.065 9.065 0 0 1 12 15a9.065 9.065 0 0 1-6.23-.693L4.2 13.914m15.6 1.386-.318 1.487a3.75 3.75 0 0 1-3.658 3.013H8.175a3.75 3.75 0 0 1-3.659-3.013L4.2 15.3" />
  </svg>
);

const XIcon = () => (
  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18 18 6M6 6l12 12" />
  </svg>
);

const InfoIcon = () => (
  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
      d="m11.25 11.25.041-.02a.75.75 0 0 1 1.063.852l-.708 2.836a.75.75 0 0 0 1.063.853l.041-.021M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Zm-9-3.75h.008v.008H12V8.25Z" />
  </svg>
);

// ─── Probability Bar ──────────────────────────────────────────────────────────
function ProbBar({
  label, value, type, delay,
}: {
  label: string; value: number; type: "real" | "ai"; delay?: number;
}) {
  const [width, setWidth] = useState(0);

  useEffect(() => {
    const timer = setTimeout(() => {
      setWidth(value * 100);
    }, delay ?? 200);
    return () => clearTimeout(timer);
  }, [value, delay]);

  return (
    <div className="space-y-2">
      <div className="flex justify-between items-center">
        <span className="text-sm font-medium text-slate-300">{label}</span>
        <span className={`text-sm font-bold tabular-nums ${type === "real" ? "text-green-400" : "text-red-400"}`}>
          {(value * 100).toFixed(1)}%
        </span>
      </div>
      <div className="h-2.5 rounded-full" style={{ background: "rgba(255,255,255,0.06)" }}>
        <div
          className={`progress-bar-fill ${type === "real" ? "progress-bar-real" : "progress-bar-ai"}`}
          style={{ width: `${width}%` }}
        />
      </div>
    </div>
  );
}

// ─── Main Page ────────────────────────────────────────────────────────────────
export default function Home() {
  const [state, setState] = useState<AppState>("idle");
  const [preview, setPreview] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [errorMsg, setErrorMsg] = useState<string>("");
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback((f: File) => {
    if (!f.type.startsWith("image/")) {
      setErrorMsg("Please upload an image file (JPG, PNG, WEBP, BMP).");
      setState("error");
      return;
    }
    if (f.size > 10 * 1024 * 1024) {
      setErrorMsg("Image is too large. Please upload a file smaller than 10 MB.");
      setState("error");
      return;
    }
    const url = URL.createObjectURL(f);
    setPreview(url);
    setFile(f);
    setPrediction(null);
    setErrorMsg("");
    setState("preview");
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      setIsDragOver(false);
      const dropped = e.dataTransfer.files[0];
      if (dropped) handleFile(dropped);
    },
    [handleFile]
  );

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const picked = e.target.files?.[0];
    if (picked) handleFile(picked);
  };

  const runDetection = async () => {
    if (!file) return;
    setState("scanning");
    setPrediction(null);

    try {
      const form = new FormData();
      form.append("file", file);

      const res = await fetch("http://localhost:8000/predict", {
        method: "POST",
        body: form,
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: "Server error." }));
        throw new Error(err.detail || "Prediction failed.");
      }

      const data: Prediction = await res.json();
      setPrediction(data);
      setState("result");
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Could not connect to the backend. Make sure the server is running.";
      setErrorMsg(msg);
      setState("error");
    }
  };

  const reset = () => {
    if (preview) URL.revokeObjectURL(preview);
    setPreview(null);
    setFile(null);
    setPrediction(null);
    setErrorMsg("");
    setState("idle");
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const isAI = prediction?.label === "AI-Generated";

  return (
    <div className="relative min-h-screen overflow-hidden">
      {/* Background orbs */}
      <div className="bg-orb bg-orb-1" />
      <div className="bg-orb bg-orb-2" />

      {/* Grid pattern overlay */}
      <div
        className="fixed inset-0 pointer-events-none z-0"
        style={{
          backgroundImage: `linear-gradient(rgba(139,92,246,0.03) 1px, transparent 1px),
                            linear-gradient(90deg, rgba(139,92,246,0.03) 1px, transparent 1px)`,
          backgroundSize: "60px 60px",
        }}
      />

      {/* Content */}
      <div className="relative z-10 min-h-screen flex flex-col">

        {/* ── Nav ──────────────────────────────────────────────────────────── */}
        <nav className="flex items-center justify-between px-8 py-5 border-b border-white/5">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-xl" style={{ background: "rgba(139,92,246,0.15)", border: "1px solid rgba(139,92,246,0.3)" }}>
              <BrainIcon />
            </div>
            <div>
              <h1 className="text-lg font-bold tracking-tight">
                <span style={{ color: "#a78bfa" }}>CIFAKE</span>
                <span className="text-white"> Detector</span>
              </h1>
              <p className="text-xs text-slate-500 leading-none">Real vs AI-Generated</p>
            </div>
          </div>

          <div className="hidden sm:flex items-center gap-2 px-3 py-1.5 rounded-full text-xs text-slate-400"
            style={{ background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.08)" }}>
            <div className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse" />
            Model Ready · 93,377 params
          </div>
        </nav>

        {/* ── Hero ─────────────────────────────────────────────────────────── */}
        <main className="flex-1 flex flex-col items-center justify-center px-4 py-12 gap-8 max-w-2xl mx-auto w-full">

          {/* Title */}
          <div className="text-center space-y-3 animate-fade-in-up">
            <h2 className="text-4xl sm:text-5xl font-extrabold tracking-tight leading-tight">
              <span className="text-white">Is this image </span>
              <span style={{
                background: "linear-gradient(135deg, #a78bfa, #60a5fa)",
                WebkitBackgroundClip: "text",
                WebkitTextFillColor: "transparent",
              }}>
                real or AI?
              </span>
            </h2>
            <p className="text-slate-400 text-base sm:text-lg max-w-md mx-auto">
              Upload any image and our CNN will instantly classify it as a real photo or AI-generated.
            </p>
          </div>

          {/* ── Upload / Preview Card ────────────────────────────────────── */}
          <div className="glass-card w-full p-1 animate-fade-in-up" style={{ animationDelay: "0.1s" }}>

            {/* Upload zone or preview */}
            {state === "idle" || state === "error" ? (
              <div
                id="upload-zone"
                className={`upload-zone flex flex-col items-center justify-center gap-4 py-16 px-8 text-center cursor-pointer ${isDragOver ? "drag-over" : ""}`}
                onClick={() => fileInputRef.current?.click()}
                onDragOver={(e) => { e.preventDefault(); setIsDragOver(true); }}
                onDragLeave={() => setIsDragOver(false)}
                onDrop={handleDrop}
              >
                <div className="text-purple-400 opacity-70">
                  <UploadIcon />
                </div>
                <div>
                  <p className="text-white font-semibold text-lg">Drop your image here</p>
                  <p className="text-slate-500 text-sm mt-1">or <span className="text-purple-400 underline underline-offset-2 cursor-pointer">click to browse</span></p>
                </div>
                <p className="text-xs text-slate-600">JPG, PNG, WEBP, BMP · Max 10 MB</p>

                {state === "error" && (
                  <div className="mt-2 flex items-center gap-2 px-4 py-2.5 rounded-xl text-sm text-red-300"
                    style={{ background: "rgba(239,68,68,0.1)", border: "1px solid rgba(239,68,68,0.25)" }}>
                    <InfoIcon />
                    {errorMsg}
                  </div>
                )}
              </div>
            ) : (
              <div className="relative">
                {/* Image preview */}
                <div className="relative w-full overflow-hidden rounded-[15px]" style={{ maxHeight: "340px" }}>
                  {preview && (
                    <img
                      src={preview}
                      alt="Preview"
                      className="w-full object-cover"
                      style={{ maxHeight: "340px" }}
                    />
                  )}

                  {/* Scanning overlay */}
                  {state === "scanning" && (
                    <div className="absolute inset-0 flex flex-col items-center justify-center gap-3"
                      style={{ background: "rgba(7,7,15,0.8)", backdropFilter: "blur(4px)" }}>
                      {/* Spinner */}
                      <div className="relative w-14 h-14">
                        <div className="absolute inset-0 rounded-full border-2 border-purple-500/20" />
                        <div className="absolute inset-0 rounded-full border-2 border-transparent border-t-purple-500 animate-spin" />
                      </div>
                      <p className="text-white font-semibold scanning">Analysing image…</p>
                      <p className="text-slate-400 text-sm">Running CNN inference</p>
                    </div>
                  )}

                  {/* Clear button */}
                  {state !== "scanning" && (
                    <button
                      onClick={reset}
                      className="absolute top-3 right-3 p-1.5 rounded-lg text-slate-300 hover:text-white transition-colors"
                      style={{ background: "rgba(0,0,0,0.6)", border: "1px solid rgba(255,255,255,0.1)" }}
                      aria-label="Remove image"
                    >
                      <XIcon />
                    </button>
                  )}
                </div>
              </div>
            )}

            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              className="hidden"
              onChange={handleInputChange}
              id="file-input"
            />
          </div>

          {/* ── Action Button ─────────────────────────────────────────────── */}
          {(state === "preview" || state === "result") && (
            <div className="flex gap-3 w-full animate-fade-in-up">
              <button
                id="detect-btn"
                onClick={runDetection}
                className="flex-1 flex items-center justify-center gap-2.5 py-4 rounded-2xl font-semibold text-white text-base transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                style={{
                  background: "linear-gradient(135deg, #7c3aed, #4f46e5)",
                  boxShadow: "0 4px 24px rgba(124,58,237,0.4)",
                }}
                onMouseEnter={e => { (e.currentTarget as HTMLButtonElement).style.transform = "translateY(-1px)"; }}
                onMouseLeave={e => { (e.currentTarget as HTMLButtonElement).style.transform = "translateY(0)"; }}
              >
                <ScanIcon />
                {state === "result" ? "Scan Again" : "Detect Now"}
              </button>

              <button
                id="reset-btn"
                onClick={reset}
                className="px-5 py-4 rounded-2xl font-medium text-slate-400 hover:text-white transition-all"
                style={{ background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.08)" }}
              >
                New Image
              </button>
            </div>
          )}

          {/* ── Results Panel ─────────────────────────────────────────────── */}
          {state === "result" && prediction && (
            <div className="glass-card w-full p-6 space-y-6 animate-fade-in-up">

              {/* Verdict */}
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-xs uppercase tracking-widest text-slate-500 font-medium mb-1">Verdict</p>
                  <div className={`inline-flex items-center gap-2 px-4 py-2 rounded-xl font-bold text-lg ${isAI ? "verdict-ai" : "verdict-real"}`}>
                    <ShieldCheckIcon />
                    {prediction.label}
                  </div>
                </div>
                <div className="text-right">
                  <p className="text-xs uppercase tracking-widest text-slate-500 font-medium mb-1">Confidence</p>
                  <p className={`text-3xl font-extrabold tabular-nums ${isAI ? "text-red-400" : "text-green-400"}`}>
                    {(prediction.confidence * 100).toFixed(1)}
                    <span className="text-lg font-semibold">%</span>
                  </p>
                </div>
              </div>

              {/* Divider */}
              <div style={{ borderTop: "1px solid rgba(255,255,255,0.06)" }} />

              {/* Probability bars */}
              <div className="space-y-5">
                <ProbBar label="📷  Real Photo" value={prediction.prob_real} type="real" delay={100} />
                <ProbBar label="🤖  AI-Generated" value={prediction.prob_ai} type="ai" delay={300} />
              </div>

              {/* Interpretation note */}
              <div className="flex gap-2 px-4 py-3 rounded-xl text-xs text-slate-400"
                style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.06)" }}>
                <InfoIcon />
                <p>
                  {isAI
                    ? "This image shows strong characteristics of AI-generated synthetic imagery. The model is calibrated to favour catching AI images over missing them."
                    : "This image appears to be a genuine photograph. The model found no strong AI-generation artifacts."}
                </p>
              </div>
            </div>
          )}

          {/* ── How It Works ─────────────────────────────────────────────── */}
          {state === "idle" && (
            <div className="grid grid-cols-3 gap-3 w-full animate-fade-in-up" style={{ animationDelay: "0.2s" }}>
              {[
                { icon: "⬆️", title: "Upload", desc: "Drop any image — photo or AI art" },
                { icon: "🧠", title: "Analyse", desc: "CNN scans 32×32 pixel patterns" },
                { icon: "✅", title: "Result", desc: "Get probability score instantly" },
              ].map(({ icon, title, desc }) => (
                <div key={title} className="glass-card p-4 text-center space-y-2">
                  <div className="text-2xl">{icon}</div>
                  <p className="text-sm font-semibold text-white">{title}</p>
                  <p className="text-xs text-slate-500">{desc}</p>
                </div>
              ))}
            </div>
          )}
        </main>

        {/* ── Footer ───────────────────────────────────────────────────────── */}
        <footer className="text-center py-5 border-t border-white/5">
          <p className="text-xs text-slate-600">
            Trained on CIFAKE · 120,000 images · 93,377 params · AUC 0.97
          </p>
        </footer>
      </div>
    </div>
  );
}
