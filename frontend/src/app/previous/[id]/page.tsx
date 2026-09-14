"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useParams } from "next/navigation";
import { ArrowLeft, CheckCircle2, Clock, FileText, GitBranch, AlertTriangle } from "lucide-react";
import { getQueryById } from "@/lib/api";
import type { QueryRecord } from "@/lib/types";

function formatDate(iso?: string): string {
  if (!iso) return "Unknown time";
  const date = new Date(iso);
  return Number.isNaN(date.getTime()) ? "Unknown time" : date.toLocaleString();
}

export default function PreviousQueryPage() {
  const { id } = useParams<{ id: string }>();
  const [query, setQuery] = useState<QueryRecord | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    getQueryById(id)
      .then(setQuery)
      .catch((err) => {
        setError(err instanceof Error ? err.message : "Failed to load query");
      })
      .finally(() => setLoading(false));
  }, [id]);

  return (
    <div className="mx-auto max-w-4xl px-6 py-8">
      <Link
        href="/dashboard"
        className="mb-6 inline-flex items-center gap-2 text-sm font-medium text-slate-600 hover:text-slate-900"
      >
        <ArrowLeft className="h-4 w-4" />
        Back to dashboard
      </Link>

      {loading && (
        <div className="rounded-xl border border-slate-200 bg-white px-6 py-12 text-center text-sm text-slate-500 shadow-sm">
          Loading query...
        </div>
      )}

      {error && (
        <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
          {error}
        </div>
      )}

      {query && (
        <article className="space-y-6">
          <header>
            <div className="flex flex-wrap items-center gap-2">
              <span className="rounded-full bg-slate-100 px-2.5 py-1 text-xs font-medium text-slate-700">
                {query.is_multi_hop ? "Multi-hop" : "Single-hop"}
              </span>
              {query.cached && (
                <span className="rounded-full bg-blue-100 px-2.5 py-1 text-xs font-medium text-blue-700">
                  Cached
                </span>
              )}
              <span className="font-mono text-xs text-slate-400">{query.id}</span>
            </div>
            <h1 className="mt-4 text-3xl font-bold text-slate-900">Previous Query</h1>
            <p className="mt-2 text-lg leading-relaxed text-slate-700">{query.query}</p>
          </header>

          <section className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
            <h2 className="mb-3 font-semibold text-slate-900">Answer</h2>
            <p className="whitespace-pre-wrap text-sm leading-7 text-slate-700">{query.answer}</p>
          </section>

          <section className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
              <div className="flex items-center gap-2 text-sm text-slate-500">
                <GitBranch className="h-4 w-4" />
                Query type
              </div>
              <p className="mt-2 font-semibold text-slate-900">
                {query.is_multi_hop ? "Multi-hop" : "Single-hop"}
              </p>
            </div>
            <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
              <div className="flex items-center gap-2 text-sm text-slate-500">
                <CheckCircle2 className="h-4 w-4" />
                Grounding
              </div>
              <p className="mt-2 font-semibold text-slate-900">
                {Math.round(query.grounding_score * 100)}%
              </p>
            </div>
            <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
              <div className="flex items-center gap-2 text-sm text-slate-500">
                {query.is_valid ? (
                  <CheckCircle2 className="h-4 w-4 text-emerald-600" />
                ) : (
                  <AlertTriangle className="h-4 w-4 text-amber-600" />
                )}
                Validation
              </div>
              <p className="mt-2 font-semibold text-slate-900">
                {query.is_valid ? "Validated" : "Review"}
              </p>
            </div>
            <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
              <div className="flex items-center gap-2 text-sm text-slate-500">
                <Clock className="h-4 w-4" />
                Processed
              </div>
              <p className="mt-2 font-semibold text-slate-900">{query.elapsed_seconds}s</p>
            </div>
          </section>

          <section className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
            <div className="flex items-center gap-2">
              <FileText className="h-5 w-5 text-slate-600" />
              <h2 className="font-semibold text-slate-900">Citations</h2>
            </div>
            {query.citations.length > 0 ? (
              <ul className="mt-4 space-y-2">
                {query.citations.map((citation) => (
                  <li key={citation} className="border-l-2 border-slate-300 pl-3 text-sm text-slate-600">
                    {citation}
                  </li>
                ))}
              </ul>
            ) : (
              <p className="mt-4 text-sm text-slate-500">No citations recorded.</p>
            )}
          </section>

          <p className="text-sm text-slate-500">Created {formatDate(query.created_at)}</p>
        </article>
      )}
    </div>
  );
}
