import Link from "next/link";
import {
  ArrowRight,
  BookOpen,
  FileText,
  GitBranch,
  Search,
  Shield,
  Users,
  Zap,
} from "lucide-react";

const features = [
  {
    icon: FileText,
    title: "PDF Ingestion",
    description: "Upload documents and index them automatically with FAISS and knowledge graphs.",
  },
  {
    icon: Search,
    title: "Hybrid Retrieval",
    description: "Semantic search plus graph-based relational retrieval for accurate evidence.",
  },
  {
    icon: GitBranch,
    title: "Multi-hop Reasoning",
    description: "Complex questions are broken into steps with context carried forward.",
  },
  {
    icon: Shield,
    title: "Grounded Answers",
    description: "Every answer cites source pages and flags insufficient evidence clearly.",
  },
  {
    icon: Zap,
    title: "Async Processing",
    description: "Large PDFs are queued via RabbitMQ with real-time job status tracking.",
  },
  {
    icon: BookOpen,
    title: "Domain Independent",
    description: "Works for education, health, law, finance, policy, or any PDF content.",
  },
];

const users = [
  { role: "Researchers", desc: "Extract findings and compare sections across policy documents." },
  { role: "Students", desc: "Ask questions about textbooks and get cited, plain-language answers." },
  { role: "Analysts", desc: "Query large report libraries without reading every page manually." },
  { role: "Teams", desc: "Share a document corpus and track query history on the dashboard." },
];

export default function HomePage() {
  return (
    <div>
      {/* Hero */}
      <section className="border-b border-slate-200 bg-white">
        <div className="mx-auto max-w-7xl px-6 py-20 text-center">
          <div className="mb-4 inline-flex items-center gap-2 rounded-full border border-slate-200 bg-slate-50 px-4 py-1.5 text-sm text-slate-600">
            <Search className="h-4 w-4" />
            Document-grounded question answering
          </div>
          <h1 className="text-5xl font-bold tracking-tight text-slate-900 sm:text-6xl">
            Civic-AI
          </h1>
          <p className="mx-auto mt-6 max-w-2xl text-lg leading-relaxed text-slate-600">
            Upload PDFs, ask questions in plain English, and get answers backed by citations
            to the exact source pages — with a clear response when documents don&apos;t cover it.
          </p>
          <div className="mt-10 flex flex-wrap items-center justify-center gap-4">
            <Link
              href="/query"
              className="inline-flex items-center gap-2 rounded-lg bg-slate-900 px-6 py-3 text-sm font-medium text-white hover:bg-slate-800"
            >
              Ask a Question
              <ArrowRight className="h-4 w-4" />
            </Link>
            <Link
              href="/upload"
              className="inline-flex items-center gap-2 rounded-lg border border-slate-300 bg-white px-6 py-3 text-sm font-medium text-slate-700 hover:bg-slate-50"
            >
              Upload Documents
            </Link>
            <Link
              href="/dashboard"
              className="inline-flex items-center gap-2 rounded-lg border border-slate-300 bg-white px-6 py-3 text-sm font-medium text-slate-700 hover:bg-slate-50"
            >
              View Dashboard
            </Link>
          </div>
        </div>
      </section>

      {/* Features */}
      <section className="mx-auto max-w-7xl px-6 py-16">
        <h2 className="text-center text-2xl font-semibold text-slate-900">Features</h2>
        <p className="mx-auto mt-2 max-w-xl text-center text-slate-600">
          A complete pipeline from document ingestion to validated, cited answers.
        </p>
        <div className="mt-10 grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
          {features.map(({ icon: Icon, title, description }) => (
            <div
              key={title}
              className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm"
            >
              <div className="mb-4 flex h-10 w-10 items-center justify-center rounded-lg bg-slate-100">
                <Icon className="h-5 w-5 text-slate-700" />
              </div>
              <h3 className="font-semibold text-slate-900">{title}</h3>
              <p className="mt-2 text-sm leading-relaxed text-slate-600">{description}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Users */}
      <section className="border-t border-slate-200 bg-white">
        <div className="mx-auto max-w-7xl px-6 py-16">
          <div className="flex items-center justify-center gap-2">
            <Users className="h-5 w-5 text-slate-500" />
            <h2 className="text-2xl font-semibold text-slate-900">Built For</h2>
          </div>
          <div className="mt-10 grid gap-6 sm:grid-cols-2 lg:grid-cols-4">
            {users.map(({ role, desc }) => (
              <div
                key={role}
                className="rounded-xl border border-slate-200 p-5 text-center"
              >
                <p className="font-semibold text-slate-900">{role}</p>
                <p className="mt-2 text-sm text-slate-600">{desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="border-t border-slate-200">
        <div className="mx-auto max-w-7xl px-6 py-12 text-center">
          <p className="text-slate-600">Ready to explore your documents?</p>
          <Link
            href="/query"
            className="mt-4 inline-flex items-center gap-2 rounded-lg bg-black px-6 py-3 text-sm font-medium text-white hover:bg-slate-800"
          >
            Get Started
            <ArrowRight className="h-4 w-4" />
          </Link>
        </div>
      </section>
    </div>
  );
}
