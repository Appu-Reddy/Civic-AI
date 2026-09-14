"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { FileSearch, LayoutDashboard, MessageSquare, Upload } from "lucide-react";

const links = [
  { href: "/dashboard", label: "Dashboard", icon: LayoutDashboard },
  { href: "/upload", label: "Upload", icon: Upload },
  { href: "/query", label: "Query", icon: MessageSquare },
];

export default function Navbar() {
  const pathname = usePathname();
  const isHome = pathname === "/";

  return (
    <header className="border-b border-slate-200 bg-white">
      <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-4">
        <Link href="/" className="flex items-center gap-2 text-slate-900">
          <FileSearch className="h-6 w-6" />
          <span className="text-xl font-semibold tracking-tight">Civic-AI</span>
        </Link>

        {!isHome && (
          <nav className="flex items-center gap-6 text-sm font-medium text-slate-600">
            {links.map(({ href, label, icon: Icon }) => {
              const active = pathname === href;
              return (
                <Link
                  key={href}
                  href={href}
                  className={`flex items-center gap-1.5 transition-colors hover:text-slate-900 ${
                    active ? "text-slate-900" : ""
                  }`}
                >
                  <Icon className="h-4 w-4" />
                  {label}
                </Link>
              );
            })}
          </nav>
        )}
      </div>
    </header>
  );
}
