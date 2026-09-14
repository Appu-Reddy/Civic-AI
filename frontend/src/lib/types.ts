export interface QueryStep {
  step_number: number;
  sub_query: string;
  answer: string;
  evidence: string[];
  score: number;
  is_sufficient: boolean;
  is_final: boolean;
}

export interface QueryResponse {
  query: string;
  is_multi_hop: boolean;
  answer: string;
  is_sufficient: boolean;
  is_valid: boolean;
  grounding_score: number;
  quality_score?: number;
  offensive_score?: number;
  citations: string[];
  steps: QueryStep[];
  flagged_sentences: string[];
  validation_note: string;
  elapsed_seconds: number;
  cached: boolean;
  record_id?: string;
}

export interface QueryRecord {
  id: string;
  query: string;
  answer: string;
  is_multi_hop: boolean;
  grounding_score: number;
  is_valid: boolean;
  citations: string[];
  elapsed_seconds: number;
  cached: boolean;
  created_at?: string;
}

export interface QueryStats {
  total: number;
  multi_hop: number;
  avg_grounding: number;
}

export interface UploadRecord {
  id?: string;
  job_id: string;
  filename: string;
  status: string;
  async: boolean;
  rebuild?: boolean;
  summary?: Record<string, unknown> | null;
  error?: string | null;
  created_at?: string;
  updated_at?: string;
}

export interface UploadStats {
  total: number;
  completed: number;
  processing: number;
  failed: number;
}

export interface JobStatus {
  job_id: string;
  filename: string;
  status: "queued" | "processing" | "completed" | "failed";
  rebuild?: boolean;
  summary?: Record<string, unknown> | null;
  error?: string | null;
}

export interface HealthResponse {
  status: string;
  indexes: Record<string, boolean>;
  services: Record<string, boolean>;
}
