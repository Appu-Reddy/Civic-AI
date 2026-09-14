"use client";

import { useState } from "react";
import {
  Eraser,
  History,
  Loader2,
  MessageSquare,
  ShieldCheck,
  SlidersHorizontal,
} from "lucide-react";
import QueryResultModal from "@/components/QueryResultModal";
import { submitQuery } from "@/lib/api";
import type { QueryResponse } from "@/lib/types";

export default function QueryPage() {
  const [query, setQuery] = useState("");
  const [topK, setTopK] = useState(5);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState<QueryResponse | null>(null);

  const handleSubmit = async () => {
    if (!query.trim()) {
      setError("Please enter a question.");
      return;
    }
    setError("");
    setLoading(true);
    setResult(null);

    try {
      const response = await submitQuery(query.trim(), topK);
      setResult(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Query failed");
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setQuery("");
    setError("");
    setResult(null);
  };

  return (
    <div className="mx-auto max-w-3xl px-6 py-10">
      <div className="mb-8 text-center">
        <h1 className="text-3xl font-bold text-slate-900">Document Query Analysis</h1>
        <p className="mt-2 text-slate-600">
          Ask a question about your indexed documents for AI-powered, cited answers.
        </p>
      </div>

      {/* Action bar */}
      <div className="mb-6 flex flex-wrap justify-center gap-3">
        <button
          type="button"
          onClick={handleClear}
          className="flex items-center gap-2 rounded-lg bg-slate-900 px-4 py-2 text-sm font-medium text-white hover:bg-slate-800"
        >
          <Eraser className="h-4 w-4" />
          Clear Form
        </button>
        <a
          href="/dashboard"
          className="flex items-center gap-2 rounded-lg bg-slate-900 px-4 py-2 text-sm font-medium text-white hover:bg-slate-800"
        >
          <History className="h-4 w-4" />
          View Query History
        </a>
      </div>

      {/* Retrieval settings */}
      <div className="mb-4 rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
        <div className="mb-3 flex items-center gap-2">
          <SlidersHorizontal className="h-5 w-5 text-slate-600" />
          <h2 className="font-semibold text-slate-900">Retrieval Settings</h2>
        </div>
        <label className="block text-sm text-slate-600">
          Top-K evidence chunks
          <input
            type="number"
            min={1}
            max={20}
            value={topK}
            onChange={(e) => setTopK(Number(e.target.value))}
            className="mt-1 w-full rounded-lg border border-slate-200 px-3 py-2 text-sm focus:border-slate-400 focus:outline-none"
          />
        </label>
      </div>

      {/* Query input */}
      <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
        <div className="mb-3 flex items-center gap-2">
          <MessageSquare className="h-5 w-5 text-slate-600" />
          <h2 className="font-semibold text-slate-900">Your Question</h2>
        </div>
        <p className="mb-3 text-sm text-slate-500">
          Describe your question in detail. Multi-part questions (e.g. explain X, explain Y,
          then compare) will automatically use multi-hop reasoning.
        </p>
        <textarea
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          rows={6}
          placeholder="What are the main goals of the National Education Policy 2020?&#10;&#10;Or: Explain Article 14, explain Article 15, and compare them."
          className="w-full resize-none rounded-lg border border-slate-200 px-4 py-3 text-sm leading-relaxed focus:border-slate-400 focus:outline-none"
        />

        {error && (
          <div className="mt-4 rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
            {error}
          </div>
        )}

        <div className="mt-5 flex justify-end">
          <button
            type="button"
            onClick={handleSubmit}
            disabled={loading}
            className="flex items-center gap-2 rounded-lg bg-black px-6 py-3 text-sm font-medium text-white hover:bg-slate-800 disabled:opacity-50"
          >
            {loading ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" />
                Analyzing…
              </>
            ) : (
              <>
                <ShieldCheck className="h-4 w-4" />
                Analyze Query
              </>
            )}
          </button>
        </div>
      </div>

      {result && <QueryResultModal result={result} onClose={() => setResult(null)} />}
    </div>
  );
}
