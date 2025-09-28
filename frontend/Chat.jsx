import React, { useEffect, useMemo, useRef, useState } from "react";

const STORAGE_KEY_PREFIX = "chat_history:";
const newId = () => crypto.randomUUID?.() || String(Date.now());

export default function Chat({ apiUrl = "http://127.0.0.1:8000", conversationId: propId }) {
  const [conversationId, setConversationId] = useState(propId || "");
  const [messages, setMessages] = useState([]); // [{role:'user'|'assistant'|'system', content:string, ts:number}]
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const listRef = useRef(null);

  // Ensure conversation id exists
  useEffect(() => {
    if (!conversationId) {
      const saved = localStorage.getItem("active_conversation_id");
      const cid = saved || newId();
      setConversationId(cid);
      localStorage.setItem("active_conversation_id", cid);
    }
  }, [conversationId]);

  const storageKey = useMemo(
    () => (conversationId ? `${STORAGE_KEY_PREFIX}${conversationId}` : null),
    [conversationId]
  );

  // Load history from localStorage
  useEffect(() => {
    if (!storageKey) return;
    const raw = localStorage.getItem(storageKey);
    if (raw) {
      try {
        setMessages(JSON.parse(raw));
      } catch {
        setMessages([]);
      }
    }
  }, [storageKey]);

  // Persist to localStorage whenever messages change
  useEffect(() => {
    if (!storageKey) return;
    localStorage.setItem(storageKey, JSON.stringify(messages));
  }, [messages, storageKey]);

  // Auto scroll to bottom
  useEffect(() => {
    listRef.current?.scrollTo({ top: listRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, sending]);

  async function sendMessage(e) {
    e?.preventDefault?.();
    const text = input.trim();
    if (!text || sending) return;

    const userMsg = { role: "user", content: text, ts: Date.now() };
    setMessages((m) => [...m, userMsg]);
    setInput("");
    setSending(true);

    try {
      // For a demo without backend, fake an assistant response:
      // Replace this block with a real fetch to your backend if you already have one.
      // Example backend call:
      // const res = await fetch(`${apiUrl}/predict?query=${encodeURIComponent(text)}`);
      // const data = await res.json();
      // const assistantText = data.answer ?? "No answer.";
      const assistantText = `Echo: ${text}`;

      const assistantMsg = { role: "assistant", content: assistantText, ts: Date.now() };
      setMessages((m) => [...m, assistantMsg]);
    } catch (err) {
      const errMsg = { role: "assistant", content: "Sorry—request failed.", ts: Date.now() };
      setMessages((m) => [...m, errMsg]);
    } finally {
      setSending(false);
    }
  }

  function newConversation() {
    const cid = newId();
    setConversationId(cid);
    setMessages([]);
    localStorage.setItem("active_conversation_id", cid);
  }

  return (
    <div className="w-full h-full max-w-3xl mx-auto flex flex-col rounded-2xl border shadow-sm overflow-hidden">
      <header className="px-4 py-3 border-b bg-white flex items-center justify-between">
        <div className="font-semibold">
          Chat {conversationId ? `· ${conversationId.slice(0, 8)}` : ""}
        </div>
        <div className="flex gap-2">
          <button
            onClick={newConversation}
            className="px-3 py-1.5 rounded-lg border hover:bg-gray-50 text-sm"
          >
            New chat
          </button>
        </div>
      </header>

      <div ref={listRef} className="flex-1 overflow-auto bg-gray-50 p-4 space-y-3">
        {messages.length === 0 && (
          <div className="text-gray-500 text-sm text-center py-8">
            No messages yet. Say hi! ✨
          </div>
        )}
        {messages.map((m, idx) => (
          <div
            key={idx}
            className={`max-w-[85%] rounded-2xl px-3 py-2 text-sm leading-relaxed ${
              m.role === "user"
                ? "ml-auto bg-black text-white"
                : "mr-auto bg-white border"
            }`}
            title={new Date(m.ts).toLocaleString()}
          >
            {m.content}
          </div>
        ))}
        {sending && (
          <div className="mr-auto bg-white border max-w-[60%] rounded-2xl px-3 py-2 text-sm text-gray-500">
            Thinking…
          </div>
        )}
      </div>

      <form onSubmit={sendMessage} className="p-3 border-t bg-white flex gap-2">
        <input
          className="flex-1 border rounded-xl px-3 py-2 focus:outline-none focus:ring"
          placeholder="Type a message…"
          value={input}
          onChange={(e) => setInput(e.target.value)}
        />
        <button
          disabled={sending || !input.trim()}
          className="px-4 py-2 rounded-xl bg-black text-white disabled:opacity-50"
        >
          Send
        </button>
      </form>
    </div>
  );
}
