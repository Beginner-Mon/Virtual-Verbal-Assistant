import React, { useState } from "react";
import MotionViewer from "./MotionViewer";
import Bubble from "./Bubble";
import { INIT, RESPONSES } from "../data/constants";

export default function App() {
  const [messages, setMessages] = useState(INIT);
  const [motionKey, setMotionKey] = useState("idle");
  const [thinking, setThinking] = useState(false);

  const detectMotion = (text) => {
    const l = text.toLowerCase();
    if (l.includes("walk")) return "walk";
    if (l.includes("raise")) return "raise";
    if (l.includes("squat")) return "squat";
    if (l.includes("stretch")) return "stretch";
    return null;
  };

  const send = async (text) => {
    if (!text.trim()) return;

    setMessages(m => [...m, { id: Date.now(), role: "user", text }]);

    setThinking(true);
    await new Promise(r => setTimeout(r, 800));

    const motion = detectMotion(text);
    if (motion) setMotionKey(motion);

    setThinking(false);

    setMessages(m => [...m, {
      id: Date.now() + 1,
      role: "assistant",
      text: motion ? RESPONSES[motion] : "Try asking about a movement.",
      motion,
    }]);
  };

  return (
    <div className="app">
      <div className="chat">
        {messages.map(m => <Bubble key={m.id} msg={m} />)}
      </div>
      <MotionViewer motionKey={motionKey} isGenerating={thinking} />
    </div>
  );
}
