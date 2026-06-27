
"use client";

import { useState } from "react";
import { Send } from "lucide-react";
import { AppShell } from "@/components/app-shell";
import { CourseSelect } from "@/components/course-select";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/input";
import { api, type Source } from "@/lib/api";

type Message = {
  role: "student" | "assistant";
  text: string;
  sources?: Source[];
};

export default function ChatPage() {
  const [courseId, setCourseId] = useState("");
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);

  async function ask(mode: "ask" | "summary" | "flashcards" = "ask") {
    if (!input.trim()) return;

    setMessages((items) => [
      ...items,
      { role: "student", text: input },
    ]);

    const result = await api<{
      response: string;
      sources: Source[];
    }>(`/courses/${courseId}/chat/${mode}`, {
      method: "POST",
      body: JSON.stringify({ message: input }),
    });

    setMessages((items) => [
      ...items,
      {
        role: "assistant",
        text: result.response,
        sources: result.sources,
      },
    ]);

    setInput("");
  }

 return (
  <AppShell role="student">
    <div className="flex h-[calc(100vh-80px)] flex-col overflow-hidden">
      {/* Header */}
      <div className="border-b bg-background px-6 py-4">
        <div className="mx-auto flex max-w-5xl items-center justify-between">

          <CourseSelect
            value={courseId}
            onChange={setCourseId}
          />
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-4 py-2">
        {messages.length === 0 ? (
          <div className="flex h-full items-center justify-center px-4">
            <div className="max-w-xl text-center">
              <h1 className="text-4xl font-semibold tracking-tight">
                What would you like to learn?
              </h1>

              <p className="mt-4 text-muted-foreground">
                Select a course and ask questions based on
                your uploaded materials.
              </p>
            </div>
          </div>
        ) : (
          <div className="mx-auto max-w-4xl px-4 py-8">
            <div className="space-y-8">
              {messages.map((message, index) => (
                <div
                  key={index}
                  className={`flex ${
                    message.role === "student"
                      ? "justify-end"
                      : "justify-start"
                  }`}
                >
                  {message.role === "student" ? (
                    <div className="max-w-[80%] rounded-3xl bg-primary px-5 py-3 text-primary-foreground shadow-sm">
                      <p className="whitespace-pre-wrap text-sm leading-7">
                        {message.text}
                      </p>
                    </div>
                  ) : (
                    <div className="max-w-full">
                      <div className="whitespace-pre-wrap text-sm leading-7">
                        {message.text}
                      </div>

                      {!!message.sources?.length && (
                        <div className="mt-4 flex flex-wrap gap-2">
                          {message.sources.map((source) => (
                            <div
                              key={source.citation}
                              className="rounded-xl border bg-card px-3 py-2 text-xs"
                            >
                              <span className="font-medium">
                                [{source.citation}]
                              </span>{" "}
                              {source.document}
                              {source.page
                                ? ` • Page ${source.page}`
                                : ""}
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Input */}
      <div className="border-t bg-background p-4">
        <div className="mx-auto max-w-4xl">
          <div className="rounded-3xl border bg-card p-3 shadow-sm">
            <Textarea
              placeholder="Message Course Assistant..."
              value={input}
              onChange={(event) =>
                setInput(event.target.value)
              }
              className="min-h-[70px] resize-none border-0 bg-transparent shadow-none focus-visible:ring-0"
            />

            <div className="mt-3 flex justify-end">
              <Button
                onClick={() => ask("ask")}
                disabled={!input.trim() || !courseId}
                className="rounded-full"
              >
                <Send className="size-4" />
                Ask
              </Button>
            </div>
          </div>

          <p className="mt-2 text-center text-xs text-muted-foreground">
            Responses are generated from your uploaded
            course materials.
          </p>
        </div>
      </div>
    </div>
  </AppShell>
);
}