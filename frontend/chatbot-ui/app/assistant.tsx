"use client";

import { AssistantRuntimeProvider, useLocalRuntime, type ChatModelAdapter } from "@assistant-ui/react";
import { Thread } from "@/components/assistant-ui/thread";

const RAGAdapter: ChatModelAdapter = {
  async run({ messages, abortSignal }) {
    const lastMessage = messages.findLast((m) => m.role === "user");
    const question =
      lastMessage?.content
        ?.filter((p) => p.type === "text")
        .map((p) => p.type === "text" ? p.text : "")
        .join("") ?? "";

    const res = await fetch("http://localhost:8000/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
      signal: abortSignal,
    });

    if (!res.ok) throw new Error("Backend error");

    const data = await res.json();
    const sources = (data.sources as string[]) ?? [];
    const sourcesText = sources.length > 0 ? `\n\n---\n*Sources: ${sources.join(", ")}*` : "";

    return {
      content: [{ type: "text", text: data.answer }],
    };
  },
};

export const Assistant = () => {
  const runtime = useLocalRuntime(RAGAdapter);
  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <div className="h-dvh">
        <Thread />
      </div>
    </AssistantRuntimeProvider>
  );
};