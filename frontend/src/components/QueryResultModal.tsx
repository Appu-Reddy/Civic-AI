"use client";

import {
  CheckCircle2,
  Download,
  MessageSquare,
  ShieldCheck,
  X,
} from "lucide-react";
import { useState } from "react";
import type { QueryResponse } from "@/lib/types";

interface QueryResultModalProps {
  result: QueryResponse;
  onClose: () => void;
}

function MetricCard({
  title,
  score,
  passed,
}: {
  title: string;
  score: string;
  passed: boolean;
}) {
  return (
    <div
      className={`rounded-lg border p-4 ${
        passed ? "border-emerald-200 bg-emerald-50" : "border-amber-200 bg-amber-50"
      }`}
    >
      <p className="text-sm font-medium text-slate-700">{title}</p>
      <p className="mt-1 text-lg font-semibold text-slate-900">{score}</p>
      <div className="mt-2 flex items-center gap-1.5 text-sm">
        {passed ? (
          <>
            <CheckCircle2 className="h-4 w-4 text-emerald-600" />
            <span className="text-emerald-700">Passed</span>
          </>
        ) : (
          <>
            <ShieldCheck className="h-4 w-4 text-amber-600" />
            <span className="text-amber-700">Review</span>
          </>
        )}
      </div>
    </div>
  );
}

export default function QueryResultModal({ result, onClose }: QueryResultModalProps) {
  const [expandedSteps, setExpandedSteps] = useState<Record<number, boolean>>({});
  const groundingPct = Math.round(result.grounding_score * 100);
  const qualityPct = Math.round((result.quality_score ?? result.grounding_score) * 100);

  const handleDownload = () => {
    const blob = new Blob(
      [
        `Query: ${result.query}\n\nAnswer:\n${result.answer}\n\nCitations:\n${result.citations.join("\n")}`,
      ],
      { type: "text/plain" },
    );
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "civic-ai-response.txt";
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4">
      <div className="flex max-h-[90vh] w-full max-w-3xl flex-col overflow-hidden rounded-xl bg-white shadow-2xl">
        {/* Header */}
        <div className="flex items-start justify-between bg-slate-900 px-6 py-4 text-white">
          <div>
            <h2 className="text-lg font-semibold">Query Analysis Results</h2>
            <p className="mt-0.5 text-sm text-slate-300">
              Analysis for: {result.query.slice(0, 80)}
              {result.query.length > 80 ? "…" : ""}
            </p>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="rounded-lg p-1 hover:bg-slate-800"
            aria-label="Close"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Body */}
        <div className="flex-1 overflow-y-auto px-6 py-5">
          <div className="rounded-lg border border-slate-200 bg-slate-50 p-5">
            <p className="whitespace-pre-wrap text-sm leading-relaxed text-slate-800">
              {result.answer}
            </p>
          </div>

          {result.citations.length > 0 && (
            <div className="mt-5">
              <h3 className="mb-2 text-sm font-semibold text-slate-900">Citations</h3>
              <ul className="space-y-1">
                {result.citations.map((cite) => (
                  <li key={cite} className="text-sm text-slate-600">
                    • {cite}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {result.is_multi_hop && result.steps.length > 1 && (
            <div className="mt-5">
              <h3 className="mb-2 text-sm font-semibold text-slate-900">Multi-step Breakdown</h3>
              <div className="space-y-3">
                {result.steps.map((step) => {
                  const isExpanded = !!expandedSteps[step.step_number];

                  return (
                    <div
                      key={step.step_number}
                      className="rounded-lg border border-slate-200 p-3"
                    >
                      <p className="text-xs font-medium text-slate-500">
                        Step {step.step_number}
                      </p>
                      <p className="mt-1 text-sm font-medium text-slate-800">{step.sub_query}</p>
                      <p className={`mt-1 text-sm text-slate-600 ${isExpanded ? "" : "line-clamp-3"}`}>
                        {step.answer}
                      </p>
                      <button
                        type="button"
                        onClick={() =>
                          setExpandedSteps((current) => ({
                            ...current,
                            [step.step_number]: !isExpanded,
                          }))
                        }
                        className="mt-1 text-sm font-medium text-slate-900 underline underline-offset-2 hover:text-slate-600"
                      >
                        {isExpanded ? "Less" : "More"}
                      </button>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          <div className="mt-6">
            <h3 className="mb-3 text-sm font-semibold text-slate-900">
              Validation Quality Evaluation
            </h3>
            <div className="grid gap-3 sm:grid-cols-3">
              <MetricCard
                title="Grounding Check"
                score={`${groundingPct}%`}
                passed={result.grounding_score >= 0.6}
              />
              <MetricCard
                title="Quality Score"
                score={`${qualityPct}%`}
                passed={(result.quality_score ?? 0) >= 0.6}
              />
              <MetricCard
                title="Evidence Check"
                score={result.is_sufficient ? "Sufficient" : "Insufficient"}
                passed={result.is_sufficient}
              />
            </div>
            {result.validation_note && (
              <p className="mt-3 text-xs text-slate-500">{result.validation_note}</p>
            )}
          </div>
        </div>

        {/* Footer */}
        <div className="flex flex-wrap items-center justify-between gap-3 border-t border-slate-200 px-6 py-4">
          <button
            type="button"
            onClick={onClose}
            className="rounded-lg border border-slate-300 px-4 py-2 text-sm font-medium text-slate-700 hover:bg-slate-50"
          >
            Close
          </button>
          <div className="flex flex-wrap gap-2">
            <button
              type="button"
              onClick={handleDownload}
              className="flex items-center gap-2 rounded-lg bg-slate-900 px-4 py-2 text-sm font-medium text-white hover:bg-slate-800"
            >
              <Download className="h-4 w-4" />
              Download Report
            </button>
            <button
              type="button"
              onClick={onClose}
              className="flex items-center gap-2 rounded-lg bg-black px-4 py-2 text-sm font-medium text-white hover:bg-slate-800"
            >
              <MessageSquare className="h-4 w-4" />
              Done
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
