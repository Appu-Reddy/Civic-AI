"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  Clock,
  FileUp,
  Loader2,
  RefreshCw,
  Upload,
} from "lucide-react";
import { getJobStatus, listUploads, uploadDocument } from "@/lib/api";
import type { JobStatus, UploadRecord } from "@/lib/types";

function StatusBadge({ status }: { status: string }) {
  const styles: Record<string, string> = {
    completed: "bg-emerald-100 text-emerald-700",
    processing: "bg-amber-100 text-amber-700",
    queued: "bg-blue-100 text-blue-700",
    failed: "bg-red-100 text-red-700",
  };
  return (
    <span
      className={`rounded-full px-2.5 py-0.5 text-xs font-medium ${styles[status] ?? "bg-slate-100 text-slate-700"}`}
    >
      {status}
    </span>
  );
}

export default function UploadPage() {
  const [file, setFile] = useState<File | null>(null);
  const [rebuild, setRebuild] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState("");
  const [uploads, setUploads] = useState<UploadRecord[]>([]);
  const [loadingList, setLoadingList] = useState(true);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const loadUploads = useCallback(async () => {
    try {
      const data = await listUploads(30);
      setUploads(data.uploads);
    } catch {
      /* list may fail if MongoDB is down */
    } finally {
      setLoadingList(false);
    }
  }, []);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      void loadUploads();
    }, 0);
    return () => {
      window.clearTimeout(timer);
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, [loadUploads]);

  const pollJob = (jobId: string) => {
    if (pollRef.current) clearInterval(pollRef.current);

    pollRef.current = setInterval(async () => {
      try {
        const status = await getJobStatus(jobId);
        if (status.status === "completed" || status.status === "failed") {
          if (pollRef.current) clearInterval(pollRef.current);
          loadUploads();
        }
      } catch {
        if (pollRef.current) clearInterval(pollRef.current);
      }
    }, 60000);
  };

  const handleUpload = async () => {
    if (!file) {
      setError("Please select a PDF file.");
      return;
    }
    setError("");
    setUploading(true);

    try {
      const result = await uploadDocument(file, rebuild);
      if (result.job_id) {
        if (result.async && result.status === "queued") {
          pollJob(result.job_id);
        }
      }
      setFile(null);
      loadUploads();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setUploading(false);
    }
  };

  const checkStatus = async (jobId: string) => {
    try {
      const status = await getJobStatus(jobId);
      if (status.status === "queued" || status.status === "processing") {
        pollJob(jobId);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Status check failed");
    }
  };

  return (
    <div className="mx-auto max-w-4xl px-6 py-10">
      <div className="mb-8 text-center">
        <h1 className="text-3xl font-bold text-slate-900">Document Upload</h1>
        <p className="mt-2 text-slate-600">
          Upload PDF documents for indexing. Async uploads receive a RabbitMQ job ID for
          status tracking.
        </p>
      </div>

      {/* Upload card */}
      <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
        <div className="mb-4 flex items-center gap-2">
          <FileUp className="h-5 w-5 text-slate-600" />
          <h2 className="font-semibold text-slate-900">Upload PDF</h2>
        </div>

        <div
          className="flex flex-col items-center justify-center rounded-lg border-2 border-dashed border-slate-200 bg-slate-50 px-6 py-10"
          onDragOver={(e) => e.preventDefault()}
          onDrop={(e) => {
            e.preventDefault();
            const dropped = e.dataTransfer.files[0];
            if (dropped?.type === "application/pdf") setFile(dropped);
          }}
        >
          <Upload className="mb-3 h-10 w-10 text-slate-400" />
          <p className="text-sm text-slate-600">
            Drag & drop a PDF here, or click to browse
          </p>
          <label className="mt-4 cursor-pointer rounded-lg bg-slate-900 px-4 py-2 text-sm font-medium text-white hover:bg-slate-800">
            Choose File
            <input
              type="file"
              accept=".pdf,application/pdf"
              className="hidden"
              onChange={(e) => setFile(e.target.files?.[0] ?? null)}
            />
          </label>
          {file && (
            <p className="mt-3 text-sm font-medium text-slate-700">{file.name}</p>
          )}
        </div>

        <label className="mt-4 flex items-center gap-2 text-sm text-slate-600">
          <input
            type="checkbox"
            checked={rebuild}
            onChange={(e) => setRebuild(e.target.checked)}
            className="rounded border-slate-300"
          />
          Rebuild index if file already exists
        </label>

        {error && (
          <div className="mt-4 rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
            {error}
          </div>
        )}

        <button
          type="button"
          onClick={handleUpload}
          disabled={uploading || !file}
          className="mt-5 flex w-full items-center justify-center gap-2 rounded-lg bg-black py-3 text-sm font-medium text-white hover:bg-slate-800 disabled:opacity-50"
        >
          {uploading ? (
            <>
              <Loader2 className="h-4 w-4 animate-spin" />
              Uploading…
            </>
          ) : (
            <>
              <Upload className="h-4 w-4" />
              Upload & Index Document
            </>
          )}
        </button>
      </div>

      {/* Previous uploads */}
      <div className="mt-8 rounded-xl border border-slate-200 bg-white shadow-sm">
        <div className="flex items-center justify-between border-b border-slate-100 px-5 py-4">
          <div>
            <h2 className="font-semibold text-slate-900">Previous Uploads</h2>
            <p className="text-sm text-slate-500">Job IDs and processing status</p>
          </div>
          <button
            type="button"
            onClick={loadUploads}
            className="flex items-center gap-1.5 rounded-lg border border-slate-200 px-3 py-1.5 text-xs font-medium text-slate-600 hover:bg-slate-50"
          >
            <RefreshCw className="h-3.5 w-3.5" />
            Refresh
          </button>
        </div>

        {loadingList ? (
          <div className="px-5 py-8 text-center text-sm text-slate-500">Loading…</div>
        ) : uploads.length === 0 ? (
          <div className="px-5 py-8 text-center text-sm text-slate-500">
            No uploads recorded yet.
          </div>
        ) : (
          <div className="divide-y divide-slate-100">
            {uploads.map((u) => (
              <div key={u.job_id} className="flex items-center justify-between gap-4 px-5 py-4">
                <div className="min-w-0 flex-1">
                  <p className="font-medium text-slate-900">{u.filename}</p>
                  <p className="mt-0.5 truncate font-mono text-xs text-slate-400">
                    {u.job_id}
                  </p>
                  {u.created_at && (
                    <p className="mt-1 text-xs text-slate-400">
                      <Clock className="mr-1 inline h-3 w-3" />
                      {new Date(u.created_at).toLocaleString()}
                    </p>
                  )}
                </div>
                <div className="flex shrink-0 items-center gap-3">
                  <StatusBadge status={u.status} />
                  <button
                    type="button"
                    onClick={() => checkStatus(u.job_id)}
                    className="rounded-lg border border-slate-200 px-3 py-1.5 text-xs font-medium text-slate-600 hover:bg-slate-50"
                  >
                    Check Status
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
