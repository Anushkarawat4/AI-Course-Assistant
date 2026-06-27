"use client";

import { useEffect, useState } from "react";
import {
  MessageSquare,
  Clock,
  Search,
} from "lucide-react";

import { AppShell } from "@/components/app-shell";
import { CourseSelect } from "@/components/course-select";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { api, type Source } from "@/lib/api";

type HistoryItem = {
  id: string;
  message: string;
  response: string;
  sources?: Source[];
};

export default function HistoryPage() {
  const [courseId, setCourseId] = useState("");
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [search, setSearch] = useState("");

  useEffect(() => {
    if (!courseId) return;

    async function loadHistory() {
      try {
        setLoading(true);

        const data = await api<HistoryItem[]>(
          `/courses/${courseId}/chat/history`
        );

        setHistory(data);
      } catch (error) {
        console.error(error);
      } finally {
        setLoading(false);
      }
    }

    loadHistory();
  }, [courseId]);

  const filteredHistory = history.filter((chat) =>
    `${chat.message} ${chat.response}`
      .toLowerCase()
      .includes(search.toLowerCase())
  );

  return (
    <AppShell role="student">
      <div className="mx-auto w-full max-w-6xl space-y-6">
        {/* Header */}
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <h1 className="text-3xl font-bold">
              Chat History
            </h1>

            <p className="mt-1 text-muted-foreground">
              Review your previous conversations.
            </p>
          </div>

          <CourseSelect
            value={courseId}
            onChange={setCourseId}
          />
        </div>

        {!courseId ? (
          <Card className="p-10 text-center">
            <h2 className="text-lg font-semibold">
              Select a course
            </h2>

            <p className="mt-2 text-muted-foreground">
              Choose a course to view chat history.
            </p>
          </Card>
        ) : loading ? (
          <Card className="p-10 text-center">
            Loading history...
          </Card>
        ) : (
          <>
            {/* Search */}
            <Card className="p-4">
              <div className="relative">
                <Search className="absolute left-3 top-3 h-4 w-4 text-muted-foreground" />

                <Input
                  placeholder="Search previous chats..."
                  value={search}
                  onChange={(e) =>
                    setSearch(e.target.value)
                  }
                  className="pl-10"
                />
              </div>
            </Card>

            {filteredHistory.length === 0 ? (
              <Card className="p-10 text-center">
                {history.length === 0 ? (
                  <>
                    <MessageSquare className="mx-auto mb-4 h-12 w-12 text-muted-foreground" />

                    <h2 className="text-lg font-semibold">
                      No chats yet
                    </h2>

                    <p className="mt-2 text-muted-foreground">
                      Ask questions in Course Chat and
                      they will appear here.
                    </p>
                  </>
                ) : (
                  <>
                    <Search className="mx-auto mb-4 h-12 w-12 text-muted-foreground" />

                    <h2 className="text-lg font-semibold">
                      No matching chats found
                    </h2>

                    <p className="mt-2 text-muted-foreground">
                      Try a different search term.
                    </p>
                  </>
                )}
              </Card>
            ) : (
              <div className="space-y-4">
                {filteredHistory.map((chat) => (
                  <Card
                    key={chat.id}
                    className="p-5 transition-all hover:border-primary"
                  >
                    <div className="space-y-4">
                      {/* Question */}
                      <div>
                        <p className="text-xs uppercase tracking-wide text-muted-foreground">
                          Question
                        </p>

                        <p className="mt-2 font-medium">
                          {chat.message}
                        </p>
                      </div>

                      {/* Response */}
                      <div>
                        <p className="text-xs uppercase tracking-wide text-muted-foreground">
                          Response
                        </p>

                        <p className="mt-2 whitespace-pre-wrap text-sm leading-7">
                          {chat.response}
                        </p>
                      </div>

                      {/* Sources */}
                      {Array.isArray(chat.sources) && chat.sources.length > 0 && (
                        <div>
                          <p className="mb-2 text-xs uppercase tracking-wide text-muted-foreground">
                            Sources
                          </p>

                          <div className="flex flex-wrap gap-2">
                            {chat.sources.map(
                              (
                                source: Source,
                                index: number
                              ) => (
                                <div
                                  key={index}
                                  className="rounded-lg border px-3 py-2 text-xs"
                                >
                                  {source.document ||
                                    source.citation ||
                                    `Source ${
                                      index + 1
                                    }`}
                                </div>
                              )
                            )}
                          </div>
                        </div>
                      )}

                      <div className="flex items-center gap-2 text-xs text-muted-foreground">
                        <Clock className="h-3 w-3" />
                        Saved conversation
                      </div>
                    </div>
                  </Card>
                ))}
              </div>
            )}
          </>
        )}
      </div>
    </AppShell>
  );
}