import React from "react";
import { MOTIONS } from "../data/constants";

export default function Bubble({ msg }) {
  const isUser = msg.role === "user";

  return (
    <div className={`bubble ${isUser ? "user" : "assistant"}`}>
      <div>{msg.text}</div>

      {msg.motion && (
        <div className="motion-tag">
          Motion: {MOTIONS[msg.motion]?.label}
        </div>
      )}

      <span className="time">{msg.time}</span>
    </div>
  );
}
