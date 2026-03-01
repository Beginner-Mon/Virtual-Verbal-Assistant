import React, { useState, useEffect } from "react";
import StickFigure from "./StickFigure";
import { MOTIONS } from "../data/constants";

export default function MotionViewer({ motionKey, isGenerating }) {
  const [tick, setTick] = useState(0);

  useEffect(() => {
    const iv = setInterval(() => setTick(t => t + 1), 40);
    return () => clearInterval(iv);
  }, []);

  const m = MOTIONS[motionKey] || MOTIONS.idle;

  return (
    <div className="motion-viewer">
      {isGenerating ? (
        <div>Generating motion…</div>
      ) : (
        <>
          <StickFigure motionKey={motionKey} tick={tick} />
          <div>{m.label}</div>
        </>
      )}
    </div>
  );
}
