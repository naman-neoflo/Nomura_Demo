// Local development server. Mirrors how Vercel serves this project: static
// files from the project root, and each api/*.js module mounted at /api/<name>
// with the same (req, res) shape the handlers get in production.
//
//   node dev-server.mjs          -> http://localhost:8000
//
// Running the real handlers locally means the chat, ticket and callback routes
// behave here exactly as they do once deployed.

import { createServer } from "node:http";
import { readFile, readdir } from "node:fs/promises";
import { extname, join, normalize } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = fileURLToPath(new URL(".", import.meta.url));
const PORT = Number(process.argv[2]) || 8000;

const TYPES = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".pdf": "application/pdf",
};

const routes = new Map();
for (const file of await readdir(join(ROOT, "api"))) {
  if (!file.endsWith(".js") || file.startsWith("_")) continue;
  const name = file.replace(/\.js$/, "");
  routes.set(`/api/${name}`, (await import(new URL(`./api/${file}`, import.meta.url))).default);
}

/** Give the handler the subset of the Vercel response API these routes use. */
function wrapResponse(res) {
  let code = 200;
  const api = {
    status(n) { code = n; return api; },
    setHeader(k, v) { res.setHeader(k, v); return api; },
    json(body) {
      res.writeHead(code, { "Content-Type": "application/json; charset=utf-8" });
      res.end(JSON.stringify(body));
    },
    send(body) {
      if (!res.hasHeader("Content-Type")) res.setHeader("Content-Type", "text/plain; charset=utf-8");
      res.writeHead(code);
      res.end(body);
    },
  };
  return api;
}

function readBody(req) {
  return new Promise((resolve) => {
    let raw = "";
    req.on("data", (c) => { raw += c; });
    req.on("end", () => {
      if (!raw) return resolve({});
      try { resolve(JSON.parse(raw)); } catch { resolve({}); }
    });
  });
}

createServer(async (req, res) => {
  const url = new URL(req.url, `http://localhost:${PORT}`);
  const handler = routes.get(url.pathname);

  if (handler) {
    req.body = req.method === "POST" ? await readBody(req) : {};
    const started = Date.now();
    const wrapped = wrapResponse(res);
    try {
      await handler(req, wrapped);
    } catch (err) {
      console.error(`  ${url.pathname} threw:`, err);
      if (!res.headersSent) wrapped.status(500).json({ message: "Handler error" });
    }
    console.log(`${req.method} ${url.pathname} -> ${res.statusCode} (${Date.now() - started}ms)`);
    return;
  }

  const rel = url.pathname === "/" ? "index.html" : normalize(url.pathname).replace(/^(\.\.[/\\])+/, "").slice(1);
  try {
    const body = await readFile(join(ROOT, rel));
    res.writeHead(200, { "Content-Type": TYPES[extname(rel)] || "application/octet-stream" });
    res.end(body);
  } catch {
    res.writeHead(404, { "Content-Type": "text/plain" });
    res.end("Not found");
  }
  console.log(`${req.method} ${url.pathname} -> ${res.statusCode}`);
}).listen(PORT, "127.0.0.1", () => {
  console.log(`http://localhost:${PORT}`);
  console.log(`routes: ${[...routes.keys()].join(", ")}`);
  console.log(`ANTHROPIC_API_KEY ${process.env.ANTHROPIC_API_KEY ? "set - Claude will answer" : "not set - knowledge base steps will be served directly"}`);
});
