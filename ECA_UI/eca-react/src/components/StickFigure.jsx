import React from "react";
import { MOTIONS } from "../data/constants";

export default function StickFigure({ motionKey, tick }) {
  const m = MOTIONS[motionKey] || MOTIONS.idle;
  const t = (tick % m.frames) / m.frames;
  const PI = Math.PI;
  const cx = 100;

  let lA = 18 * Math.sin(t * 2 * PI);
  let rA = -18 * Math.sin(t * 2 * PI);

  return (
    <svg width="200" height="185" viewBox="0 0 200 185">
      <circle cx={cx} cy={40} r={12} stroke={m.color} fill="none" />
      <line x1={cx} y1={52} x2={cx} y2={100} stroke={m.color} />
      <line x1={cx} y1={60} x2={cx - 30} y2={60 + lA} stroke={m.color} />
      <line x1={cx} y1={60} x2={cx + 30} y2={60 + rA} stroke={m.color} />
    </svg>
  );
}
