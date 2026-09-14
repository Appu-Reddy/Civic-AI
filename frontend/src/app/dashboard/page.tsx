"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  Clock,
  FileText,
  GitBranch,
  MessageSquare,
  Plus,
  TrendingUp,
} from "lucide-react";
import StatCard from "@/components/StatCard";
import { listQueries, getHealth } from "@/lib/api";
import type { HealthResponse, QueryRecord, QueryStats } from "@/lib/types";

const HEALTH_CACHE_KEY = "civic-ai-health";

function readCachedHealth(): HealthResponse | null {
  try {
    const cached = sessionStorage.getItem(HEALTH_CACHE_KEY);
    return cached ? (JSON.parse(cached) as HealthResponse) : null;
  } catch {
    return null;
  }
}

function cacheHealth(health: HealthResponse) {
  try {
    sessionStorage.setItem(HEALTH_CACHE_KEY, JSON.stringify(health));
  } catch {
  }
}

function timeAgo(iso?: string): string {
  if (!iso) return "—";
  const diff = Date.now() - new Date(iso).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  return `${Math.floor(hrs / 24)}d ago`;
}

export default function DashboardPage() {
  const [queries, setQueries] = useState<QueryRecord[]>([]);
  const [stats, setStats] = useState<QueryStats>({ total: 0, multi_hop: 0, avg_grounding: 0 });
  const [services, setServices] = useState<Record<string, boolean>>({});
  const [indexes, setIndexes] = useState<Record<string, boolean>>({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    async function load() {
      try {
        const cachedHealth = readCachedHealth();
        const isBrowserRefresh =
          performance.getEntriesByType("navigation")[0]?.type === "reload";
        const healthPromise =
          cachedHealth && !isBrowserRefresh ? Promise.resolve(cachedHealth) : getHealth();
        const [queryData, health] = await Promise.all([listQueries(20), healthPromise]);
        setQueries(queryData.queries);
        setStats(queryData.stats);
        setServices(health.services ?? {});
        setIndexes(health.indexes ?? {});
        cacheHealth(health);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load dashboard");
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  const singleHop = stats.total - stats.multi_hop;

  return (
    <div className="mx-auto max-w-7xl px-6 py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-slate-900">Dashboard</h1>
        <p className="mt-1 text-slate-600">
          Welcome back! Track your queries, document activity, and system status.
        </p>
      </div>

      {error && (
        <div className="mb-6 rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
          {error}
        </div>
      )}

      {/* Stats */}
      <div className="mb-8 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard
          label="Total Queries"
          value={loading ? "—" : stats.total}
          sublabel={`${singleHop} single-hop`}
          icon={MessageSquare}
        />
        <StatCard
          label="Multi-hop Queries"
          value={loading ? "—" : stats.multi_hop}
          sublabel="Complex reasoning"
          icon={GitBranch}
        />
        <StatCard
          label="Avg Grounding"
          value={loading ? "—" : `${Math.round(stats.avg_grounding * 100)}%`}
          sublabel="Validation score"
          icon={TrendingUp}
        />
        <StatCard
          label="Recent Queries"
          value={loading ? "—" : queries.length}
          sublabel="Last 20 shown"
          icon={FileText}
        />
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        {/* Recent Queries */}
        <div className="lg:col-span-2">
          <div className="rounded-xl border border-slate-200 bg-white shadow-sm">
            <div className="border-b border-slate-100 px-5 py-4">
              <h2 className="font-semibold text-slate-900">Recent Queries</h2>
              <p className="text-sm text-slate-500">
                Latest questions and their status ({stats.total} total)
              </p>
            </div>

            {loading ? (
              <div className="px-5 py-10 text-center text-sm text-slate-500">Loading…</div>
            ) : queries.length === 0 ? (
              <div className="px-5 py-10 text-center text-sm text-slate-500">
                No queries yet.{" "}
                <Link href="/query" className="text-slate-900 underline">
                  Ask your first question
                </Link>
              </div>
            ) : (
              <div className="divide-y divide-slate-100">
                {queries.map((q) => (
                  <Link
                    key={q.id}
                    href={`/previous/${q.id}`}
                    className="block px-5 py-4 hover:bg-slate-50 focus:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-slate-500"
                  >
                    <div className="flex items-start justify-between gap-3">
                      <div className="min-w-0 flex-1">
                        <div className="flex flex-wrap items-center gap-2">
                          <span className="truncate font-mono text-xs text-slate-400">
                            {q.id.slice(0, 12)}…
                          </span>
                          <span
                            className={`rounded-full px-2 py-0.5 text-xs font-medium ${
                              q.is_multi_hop
                                ? "bg-violet-100 text-violet-700"
                                : "bg-slate-100 text-slate-700"
                            }`}
                          >
                            {q.is_multi_hop ? "Multi-hop" : "Single-hop"}
                          </span>
                          {q.cached && (
                            <span className="rounded-full bg-blue-100 px-2 py-0.5 text-xs text-blue-700">
                              Cached
                            </span>
                          )}
                        </div>
                        <p className="mt-1 font-medium text-slate-900 line-clamp-1">{q.query}</p>
                        <p className="mt-1 text-sm text-slate-500 line-clamp-2">{q.answer}</p>
                        <div className="mt-2 flex flex-wrap gap-3 text-xs text-slate-400">
                          <span>{timeAgo(q.created_at)}</span>
                          <span>Grounding: {Math.round(q.grounding_score * 100)}%</span>
                          <span>{q.citations?.length ?? 0} citations</span>
                          <span>{q.elapsed_seconds}s</span>
                        </div>
                      </div>
                      <div className="flex shrink-0 flex-col items-end gap-1">
                        {q.is_valid ? (
                          <CheckCircle2 className="h-5 w-5 text-emerald-500" />
                        ) : (
                          <AlertTriangle className="h-5 w-5 text-amber-500" />
                        )}
                        <span className="text-xs text-slate-500">
                          {q.is_valid ? "Validated" : "Review"}
                        </span>
                      </div>
                    </div>
                  </Link>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Sidebar */}
        <div className="space-y-6">
          {/* Services */}
          <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
            <h2 className="mb-4 font-semibold text-slate-900">System Status</h2>
            <ul className="space-y-3">
              {[
                { name: "MongoDB", key: "mongodb", source: "services" as const },
                { name: "Redis Cache", key: "redis", source: "services" as const },
                { name: "RabbitMQ", key: "rabbitmq", source: "services" as const },
                { name: "FAISS Index", key: "faiss", source: "indexes" as const },
                { name: "Graph Index", key: "graph", source: "indexes" as const },
              ].map(({ name, key, source }) => {
                const up = source === "indexes" ? !!indexes[key] : !!services[key];
                return (
                  <li key={key} className="flex items-center justify-between text-sm">
                    <span className="flex items-center gap-2 text-slate-700">
                      <Activity className="h-4 w-4 text-slate-400" />
                      {name}
                    </span>
                    <span className="flex items-center gap-1.5">
                      <span
                        className={`h-2 w-2 rounded-full ${up ? "bg-emerald-500" : "bg-slate-300"}`}
                      />
                      <span className={up ? "text-emerald-700" : "text-slate-400"}>
                        {up ? "Active" : "Offline"}
                      </span>
                    </span>
                  </li>
                );
              })}
            </ul>
          </div>

          {/* Quick Actions */}
          <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
            <h2 className="mb-4 font-semibold text-slate-900">Quick Actions</h2>
            <div className="space-y-2">
              <Link
                href="/query"
                className="flex items-center gap-3 rounded-lg border border-slate-200 px-4 py-3 text-sm font-medium text-slate-700 hover:bg-slate-50"
              >
                <Plus className="h-4 w-4" />
                New Query
              </Link>
              <Link
                href="/upload"
                className="flex items-center gap-3 rounded-lg border border-slate-200 px-4 py-3 text-sm font-medium text-slate-700 hover:bg-slate-50"
              >
                <FileText className="h-4 w-4" />
                Upload Document
              </Link>
              <Link
                href="/dashboard"
                className="flex items-center gap-3 rounded-lg border border-slate-200 px-4 py-3 text-sm font-medium text-slate-700 hover:bg-slate-50"
              >
                <Clock className="h-4 w-4" />
                Query History
              </Link>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
