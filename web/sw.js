// Caches the shell only. Answers are never cached: a stale eligibility verdict
// shown to someone who has since opened a bank account is worse than an error.
//
// The page itself is network-first, and that is the whole point of this file.
// It used to be cache-first for everything, which meant a deployed fix was
// invisible -- the browser held the first index.html it ever saw and only
// reconsidered if THIS script's bytes changed, which editing the page does not
// do. A demo where the fix you just shipped does not appear, and no error says
// why, is a bad hour to have twenty minutes before judging.
//
// So: the page comes from the network when there is one and from the cache when
// there is not, which is the behaviour offline support was supposed to buy
// without the behaviour it accidentally bought as well.
const SHELL = "setu-shell-v2";
const FILES = ["./", "./index.html", "./manifest.webmanifest", "./icon.svg"];

// Answers, audio and the catalogue always come from the server.
const LIVE = ["/ask", "/audio", "/schemes", "/health", "/sessions", "/eval", "/session"];

self.addEventListener("install", (e) => {
  e.waitUntil(
    caches.open(SHELL).then((c) => c.addAll(FILES)).then(() => self.skipWaiting())
  );
});

self.addEventListener("activate", (e) => {
  e.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== SHELL).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (e) => {
  if (e.request.method !== "GET") return;

  const url = new URL(e.request.url);
  if (url.origin !== self.location.origin) return;
  if (LIVE.some((p) => url.pathname.startsWith(p))) return;

  const isPage =
    e.request.mode === "navigate" ||
    url.pathname === "/" ||
    url.pathname.endsWith(".html");

  if (isPage) {
    // Network first: take the fresh page, keep a copy for the next tunnel drop.
    e.respondWith(
      fetch(e.request)
        .then((res) => {
          const copy = res.clone();
          caches.open(SHELL).then((c) => c.put(e.request, copy)).catch(() => {});
          return res;
        })
        .catch(() => caches.match(e.request).then((hit) => hit || caches.match("./index.html")))
    );
    return;
  }

  // Everything else is the icon and the manifest: cache first is right, and a
  // background refresh keeps them from going stale for the life of the install.
  e.respondWith(
    caches.match(e.request).then((hit) => {
      const live = fetch(e.request)
        .then((res) => {
          const copy = res.clone();
          caches.open(SHELL).then((c) => c.put(e.request, copy)).catch(() => {});
          return res;
        })
        .catch(() => hit);
      return hit || live;
    })
  );
});
