"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import {
ArrowRight,
BookOpen,
FileText,
MessageSquare,
Brain,
GraduationCap,
} from "lucide-react";

import { getStoredUser } from "@/lib/api";

const features = [
{
title: "Multi-Format Uploads",
body: "Upload PDFs, PPTs, DOCX files, lecture notes, videos and YouTube links.",
Icon: FileText,
},
{
title: "AI Course Chat",
body: "Get instant answers grounded in your course materials with citations.",
Icon: MessageSquare,
},
{
title: "Smart Quiz Generation",
body: "Generate MCQs, quizzes and practice assessments automatically.",
Icon: BookOpen,
},
{
title: "Personalized Learning",
body: "Receive summaries, explanations and study assistance tailored to your course.",
Icon: Brain,
},
];

export default function Home() {
const [user, setUser] = useState<any>(null);

useEffect(() => {
setUser(getStoredUser());
}, []);

const dashboardHref =
user?.role === "teacher"
? "/teacher/dashboard"
: "/student/dashboard";

return ( <main className="min-h-screen bg-background">
{/* Navbar */} <nav className="sticky top-0 z-50 border-b border-border bg-background/80 backdrop-blur"> <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-4">
<Link
href={user ? dashboardHref : "/"}
className="flex items-center gap-3 transition-opacity hover:opacity-90"
> <div className="grid h-10 w-10 place-items-center rounded-lg bg-primary text-primary-foreground font-bold">
CG </div>

```
        <div>
          <span className="text-xl font-bold">
            CourseGPT
          </span>

          <p className="text-xs text-muted-foreground">
            AI Learning Platform
          </p>
        </div>
      </Link>

      <div className="flex items-center gap-3">
        {user ? (
          <>
            <div className="hidden text-right md:block">
              <p className="text-sm font-medium">
                {user.name}
              </p>

              <p className="text-xs uppercase tracking-wider text-muted-foreground">
                {user.role}
              </p>
            </div>

            <Link
              href={dashboardHref}
              className="rounded-lg bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition hover:opacity-90"
            >
              Dashboard
            </Link>
          </>
        ) : (
          <>
            <Link
              href="/login"
              className="rounded-lg border px-4 py-2 text-sm font-medium transition hover:bg-muted"
            >
              Login
            </Link>

            <Link
              href="/register"
              className="rounded-lg bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition hover:opacity-90"
            >
              Get Started
            </Link>
          </>
        )}
      </div>
    </div>
  </nav>

  {/* Hero */}
  <section className="relative overflow-hidden">
    <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_left,rgba(20,184,166,.15),transparent_35%),radial-gradient(circle_at_bottom_right,rgba(249,115,22,.12),transparent_35%)]" />

    <div className="relative mx-auto grid max-w-7xl gap-16 px-6 py-24 lg:grid-cols-2 lg:items-center">
      <div>
        <div className="mb-4 inline-flex items-center rounded-full border px-4 py-1 text-sm text-primary">
          AI Powered Learning Platform
        </div>

        <h1 className="text-5xl font-bold leading-tight md:text-7xl">
          Learn Smarter with
          <span className="block text-primary">
            CourseGPT
          </span>
        </h1>

        <p className="mt-6 max-w-xl text-lg text-muted-foreground">
          Upload lecture notes, PDFs, videos and
          course materials. Ask questions,
          generate quizzes and get grounded
          answers directly from your content.
        </p>

        <div className="mt-8 flex flex-wrap gap-4">
          <Link
            href={
              user
                ? dashboardHref
                : "/register"
            }
            className="inline-flex items-center gap-2 rounded-xl bg-primary px-6 py-3 font-medium text-primary-foreground"
          >
            {user
              ? "Go To Dashboard"
              : "Get Started"}

            <ArrowRight className="h-4 w-4" />
          </Link>

          {!user && (
            <Link
              href="/login"
              className="inline-flex items-center rounded-xl border px-6 py-3 font-medium"
            >
              Login
            </Link>
          )}
        </div>
      </div>

      <div className="relative">
        <div className="rounded-3xl border border-border bg-card p-8 shadow-xl">
          <div className="flex items-center gap-3">
            <GraduationCap className="h-8 w-8 text-primary" />

            <div>
              <h3 className="font-semibold">
                AI Teaching Assistant
              </h3>

              <p className="text-sm text-muted-foreground">
                Course-specific answers
              </p>
            </div>
          </div>

          <div className="mt-8 rounded-2xl border p-4">
            <p className="text-xs text-muted-foreground">
              Student asks
            </p>

            <p className="mt-2 text-sm">
              Summarize Rachel Carson's
              contribution to environmental
              science.
            </p>
          </div>

          <div className="mt-4 rounded-2xl border border-primary/20 bg-primary/5 p-4">
            <p className="text-xs text-primary">
              CourseGPT
            </p>

            <p className="mt-2 text-sm leading-6">
              Rachel Carson was a marine
              biologist and author whose
              influential book <b>Silent Spring</b>
              raised awareness about pesticides
              and inspired the modern
              environmental movement...
            </p>
          </div>

          <div className="mt-6 flex gap-2">
            <span className="rounded-full bg-primary/10 px-3 py-1 text-xs text-primary">
              PDF Sources
            </span>

            <span className="rounded-full bg-primary/10 px-3 py-1 text-xs text-primary">
              Citations
            </span>

            <span className="rounded-full bg-primary/10 px-3 py-1 text-xs text-primary">
              Quiz Ready
            </span>
          </div>
        </div>
      </div>
    </div>
  </section>

  {/* Features */}
  <section className="border-t py-24">
    <div className="mx-auto max-w-7xl px-6">
      <div className="mb-12 text-center">
        <h2 className="text-4xl font-bold">
          Everything You Need to Learn Better
        </h2>

        <p className="mt-4 text-muted-foreground">
          Designed for students and educators.
        </p>
      </div>

      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
        {features.map(
          ({ title, body, Icon }) => (
            <div
              key={title}
              className="rounded-2xl border bg-card p-6 transition hover:-translate-y-1 hover:shadow-lg"
            >
              <Icon className="mb-4 h-8 w-8 text-primary" />

              <h3 className="font-semibold">
                {title}
              </h3>

              <p className="mt-2 text-sm text-muted-foreground">
                {body}
              </p>
            </div>
          )
        )}
      </div>
    </div>
  </section>

  {/* Footer */}
  <footer className="border-t py-8">
    <div className="mx-auto flex max-w-7xl flex-col items-center justify-between gap-4 px-6 text-sm text-muted-foreground md:flex-row">
      <p>
        © 2026 CourseGPT. AI-powered learning
        platform.
      </p>

      <div className="flex gap-6">
        <span>Course Chat</span>
        <span>Grounded Answers</span>
        <span>Quiz Generation</span>
        <span>Personalized Learning</span>
      </div>
    </div>
  </footer>
</main>


);
}
      