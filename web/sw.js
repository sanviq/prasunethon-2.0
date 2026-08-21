// Caches the shell only. Answers are never cached: a stale eligibility verdict
// shown to someone who has since opened a bank account is worse than an error.
const SHELL = "setu-shell-v1";
const FILES = ["./", "./index.html", "./manifest.webmanifest", "./icon.svg"];

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(SHELL).then((c) => c.addAll(FILES)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", (e) => {
  e.waitUntil(caches.keys()
    .then((keys) => Promise.all(keys.filter((k) => k !== SHELL).map((k) => caches.delete(k))))
    .then(() => self.clients.claim()));
});

self.addEventListener("fetch", (e) => {
  const url = new URL(e.request.url);
  if (e.request.method !== "GET") return;
  if (["/ask", "/audio", "/schemes", "/health"].some((p) => url.pathname.startsWith(p))) return;
  e.respondWith(caches.match(e.request).then((hit) => hit || fetch(e.request)));
});
