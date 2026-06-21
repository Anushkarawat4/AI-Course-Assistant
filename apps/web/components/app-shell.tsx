"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import {
  BookOpen,
  GraduationCap,
  LayoutDashboard,
  LogOut,
  Moon,
  Upload,
  BarChart3,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { clearSession, getStoredUser } from "@/lib/api";
import { cn } from "@/lib/utils";
import { useEffect, useState } from "react";

const teacherLinks = [
  { href: "/teacher/dashboard", label: "Dashboard", icon: LayoutDashboard },
  { href: "/teacher/courses", label: "Courses", icon: BookOpen },
  { href: "/teacher/upload", label: "Materials", icon: Upload },
  { href: "/teacher/analytics", label: "Analytics", icon: BarChart3 },
  { href: "/teacher/exams", label: "Exams", icon: GraduationCap },
];

const studentLinks = [
  { href: "/student/dashboard", label: "Dashboard", icon: LayoutDashboard },
  { href: "/student/courses", label: "My Courses", icon: BookOpen },
  { href: "/student/chat", label: "Course Chat", icon: GraduationCap },
  { href: "/student/quizzes", label: "Quizzes", icon: Upload },
  { href: "/student/progress", label: "Progress", icon: BarChart3 },
];

export function AppShell({
  children,
  role,
}: {
  children: React.ReactNode;
  role: "teacher" | "student";
}) {
  const pathname = usePathname();
  const router = useRouter();

  const [user, setUser] = useState<any>(null);

  useEffect(() => {
    setUser(getStoredUser());
  }, []);

  const links = role === "teacher" ? teacherLinks : studentLinks;

  const dashboardLink =
    role === "teacher"
      ? "/teacher/dashboard"
      : "/student/dashboard";

  return (
    <div className="min-h-screen md:grid md:grid-cols-[260px_1fr]">
      {/* Sidebar */}
      <aside className="border-b border-border bg-card md:min-h-screen md:border-b-0 md:border-r">
        {/* Logo */}
        <div className="flex h-20 items-center px-5">
          <Link
            href={dashboardLink}
            className="flex items-center gap-3"
          >
            <div className="grid h-10 w-10 place-items-center rounded-lg bg-primary font-bold text-primary-foreground">
              CG
            </div>

            <div>
              <h2 className="font-bold text-lg">CourseGPT</h2>
              <p className="text-xs text-muted-foreground">
                AI Learning Platform
              </p>
            </div>
          </Link>

          <Moon className="ml-auto size-5 text-muted md:hidden" />
        </div>

        {/* Navigation */}
        <nav className="flex gap-1 overflow-x-auto px-3 pb-3 md:flex-col md:overflow-visible">
          {links.map((item) => (
            <Link
              key={item.href}
              href={item.href}
              className={cn(
                "flex h-11 shrink-0 items-center gap-3 rounded-lg px-3 text-sm transition",
                "text-muted-foreground hover:bg-primary/10 hover:text-foreground",
                pathname === item.href &&
                  "bg-primary/10 text-foreground font-medium"
              )}
            >
              <item.icon className="size-4" />
              {item.label}
            </Link>
          ))}
        </nav>
      </aside>

      {/* Main Content */}
      <main>
        <header className="flex min-h-16 items-center justify-between border-b border-border px-5">
          <div>
            <p className="text-xs uppercase tracking-[0.18em] text-muted-foreground">
              {role}
            </p>

            <h1 className="text-lg font-semibold">
              {user?.name ?? "CourseGPT"}
            </h1>
          </div>

          <Button
            variant="secondary"
            onClick={() => {
              clearSession();
              router.push("/login");
            }}
          >
            <LogOut className="mr-2 size-4" />
            Logout
          </Button>
        </header>

        <div className="mx-auto w-full max-w-7xl p-5 md:p-8">
          {children}
        </div>
      </main>
    </div>
  );
}