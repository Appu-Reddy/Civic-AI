import type {
  HealthResponse,
  JobStatus,
  QueryResponse,
  QueryRecord,
  QueryStats,
  UploadRecord,
  UploadStats,
} from "./types";

const API_BASE =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") ?? "http://localhost:3001";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, init);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(
      typeof data.error === "string" ? data.error : `Request failed (${res.status})`,
    );
  }
  return data as T;
}

export async function getHealth(): Promise<HealthResponse> {
  return request<HealthResponse>("/api/v1/health");
}

export async function submitQuery(
  query: string,
  finalTopK = 5,
): Promise<QueryResponse> {
  return request<QueryResponse>("/api/v1/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, final_top_k: finalTopK }),
  });
}

export async function listQueries(limit = 50): Promise<{
  queries: QueryRecord[];
  stats: QueryStats;
}> {
  return request(`/api/v1/queries?limit=${limit}`);
}

export async function getQueryById(id: string): Promise<QueryRecord> {
  return request<QueryRecord>(`/api/v1/queries/${id}`);
}

export async function uploadDocument(
  file: File,
  rebuild = false,
): Promise<{
  status: string;
  async: boolean;
  job_id?: string;
  filename: string;
  message?: string;
  summary?: Record<string, unknown>;
}> {
  const form = new FormData();
  form.append("file", file);
  form.append("rebuild", rebuild ? "true" : "false");

  const res = await fetch(`${API_BASE}/api/v1/upload`, {
    method: "POST",
    body: form,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(
      typeof data.error === "string" ? data.error : `Upload failed (${res.status})`,
    );
  }
  return data;
}

export async function getJobStatus(jobId: string): Promise<JobStatus> {
  return request<JobStatus>(`/api/v1/status/${jobId}`);
}

export async function listUploads(limit = 50): Promise<{
  uploads: UploadRecord[];
  stats: UploadStats;
}> {
  return request(`/api/v1/uploads?limit=${limit}`);
}

export { API_BASE };
