export const INIT = [
  {
    id: 1,
    role: "assistant",
    text: "Hi! I'm ECA. Ask me about any physical movement or exercise and I'll demonstrate it in 3D.",
    motion: null,
    time: "Now",
  },
];

export const MOTIONS = {
  idle:    { label: "Standing Idle",    color: "#34C759", frames: 60 },
  walk:    { label: "Walking",          color: "#007AFF", frames: 45 },
  raise:   { label: "Raising Arms",     color: "#FF9500", frames: 30 },
  squat:   { label: "Squat",            color: "#AF52DE", frames: 50 },
  stretch: { label: "Shoulder Stretch", color: "#FF3B30", frames: 40 },
};

export const SUGGESTIONS = [
  "Show me a shoulder stretch",
  "Demonstrate a squat",
  "Walking gait pattern",
  "Raise both arms slowly",
];

export const RESPONSES = {
  walk:    "Here's a walking gait pattern...",
  raise:   "Demonstrating a bilateral arm raise...",
  squat:   "Showing a squat pattern...",
  stretch: "This shoulder stretch targets the posterior capsule...",
};
