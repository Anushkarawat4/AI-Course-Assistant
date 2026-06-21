export type Role = "teacher" | "student";

export type User = {
  id: string;
  name: string;
  email: string;
  role: Role;
};

export type Course = {
  id: string;
  title: string;
  description: string;
  teacher_id: string;
  created_at: string;
};

export type Material = {
  id: string;
  course_id: string;
  file_name: string;
  file_type: string;
  storage_path: string;
  ingestion_job_id?: string;
  created_at: string;
};

export type Quiz = {
  id: string;
  course_id: string;
  generated_content: { content: string; sources?: Source[] };
  created_at: string;
};

export type Source = {
  citation: number;
  document: string;
  page?: string | number;
  preview?: string;
};

export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

export function getToken() {
  if (typeof window === "undefined") return "";
  return window.localStorage.getItem("coursegpt_token") ?? "";
}

export function setSession(token: string, user: User) {
  window.localStorage.setItem("coursegpt_token", token);
  window.localStorage.setItem("coursegpt_user", JSON.stringify(user));
}

export function getStoredUser(): User | null {
  if (typeof window === "undefined") return null;
  const value = window.localStorage.getItem("coursegpt_user");
  return value ? (JSON.parse(value) as User) : null;
}

export function clearSession() {
  window.localStorage.removeItem("coursegpt_token");
  window.localStorage.removeItem("coursegpt_user");
}

export async function api<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  if (!(init.body instanceof FormData)) headers.set("Content-Type", "application/json");
  const token = getToken();
  if (token) headers.set("Authorization", `Bearer ${token}`);
  const response = await fetch(`${API_URL}${path}`, { ...init, headers });
  if (!response.ok) {
  const payload = await response.json().catch(() => ({}));

  let message = "Request failed";

  if (typeof payload.detail === "string") {
    message = payload.detail;
  } else if (
    Array.isArray(payload.detail) &&
    payload.detail.length > 0
  ) {
    const msg = payload.detail[0].msg;

    if (msg.includes("at least 8 characters")) {
      message =
        "Password must be at least 8 characters long";
    } else {
      message = msg;
    }
  }

  throw new Error(message);
}
  return response.json() as Promise<T>;
}
