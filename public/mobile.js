const socket = io();

const mobileFeed = document.getElementById("mobile-feed");
const mobileStatus = document.getElementById("mobile-status");
const mobileClarity = document.getElementById("mobile-clarity");

let lastFrameTime = 0;
let frameCount = 0;

socket.on("connect", () => {
  mobileStatus.textContent = "Connected. Waiting for desktop feed...";
});

socket.on("cameraFrame", (payload) => {
  if (!payload || !payload.imageData) return;

  mobileFeed.src = payload.imageData;
  lastFrameTime = Date.now();
  frameCount += 1;
  mobileStatus.textContent = `Receiving clear feed (${frameCount})`;
});

socket.on("visionState", (payload) => {
  if (!payload) return;

  if (typeof payload.clarity === "number") {
    mobileClarity.textContent = `Clarity: ${payload.clarity.toFixed(2)}`;
  }
});

setInterval(() => {
  const now = Date.now();
  if (now - lastFrameTime > 2000) {
    mobileStatus.textContent = "Waiting for desktop feed...";
  }
}, 500);