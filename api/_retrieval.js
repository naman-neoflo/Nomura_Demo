// Retrieval over the IT knowledge base.
//
// Scoring is BM25 over 28 short articles, with the title and section weighted
// above the body. A vector store is the usual reach here, but it would need an
// embedding API key this project does not have yet, and at this corpus size
// lexical matching on distinctive terms ("Outlook", "VPN", "spooler") is
// accurate and has no cold start. `retrieve()` is the seam: swapping in
// embeddings later means changing this file only.

import { ARTICLES } from "./_kb-data.js";

const K1 = 1.4;
const B = 0.72;

const STOPWORDS = new Set((
  "a an the and or but if then than that this these those is are was were be been being " +
  "do does did doing have has had having i me my we our you your it its they them to of " +
  "in on at by for with about into over after before from up down out so no not can cant " +
  "cannot will would should could may might just now get getting got im ive am"
).split(" "));

// Helpdesk vocabulary the user's words rarely match on their own.
const SYNONYMS = {
  wifi: ["wireless", "network"],
  wireless: ["wifi", "network"],
  email: ["outlook", "mail", "messaging"],
  mail: ["outlook", "email"],
  mails: ["outlook", "email"],
  internet: ["network", "connection"],
  web: ["internet", "browser"],
  pw: ["password"],
  pwd: ["password"],
  passcode: ["password"],
  login: ["password", "sign"],
  signin: ["password", "sign"],
  mfa: ["authentication", "authenticator", "factor"],
  otp: ["authentication", "authenticator", "code"],
  "2fa": ["authentication", "authenticator", "factor"],
  vpn: ["vpn", "remote"],
  printer: ["print", "spooler", "queue"],
  printing: ["print", "printer", "spooler"],
  laptop: ["laptop", "device", "computer"],
  pc: ["laptop", "device", "computer"],
  machine: ["laptop", "device"],
  screen: ["monitor", "display"],
  display: ["monitor", "screen"],
  slow: ["slow", "performance"],
  frozen: ["stuck", "loading"],
  hang: ["stuck", "loading"],
  hanging: ["stuck", "loading"],
  crash: ["stuck", "opening"],
  crashing: ["stuck", "opening"],
  charge: ["charging", "battery", "power"],
  charging: ["charging", "battery", "power"],
  battery: ["charging", "battery", "power"],
  locked: ["locked", "lock", "access"],
  teams: ["teams", "meeting", "call"],
  meeting: ["teams", "meeting"],
  drive: ["shared", "folder", "drive"],
  folder: ["shared", "folder", "drive"],
  spam: ["junk", "spam"],
  junk: ["junk", "spam"],
};

function stem(word) {
  if (word.length > 4 && word.endsWith("ing")) return word.slice(0, -3);
  if (word.length > 4 && word.endsWith("ed")) return word.slice(0, -2);
  if (word.length > 3 && word.endsWith("s") && !word.endsWith("ss")) return word.slice(0, -1);
  return word;
}

function tokenize(text) {
  return (text || "")
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .filter((w) => w && w.length > 1 && !STOPWORDS.has(w))
    .map(stem);
}

function expand(tokens, raw) {
  const out = [...tokens];
  for (const word of (raw || "").toLowerCase().split(/[^a-z0-9]+/)) {
    const extra = SYNONYMS[word];
    if (extra) out.push(...extra.map(stem));
  }
  return out;
}

// Each article is indexed once per process, then reused across invocations.
const INDEX = (() => {
  const docs = ARTICLES.map((article) => {
    // Repeating the title and section is how they get their weight.
    const weighted = [
      article.title, article.title, article.title,
      article.section || "", article.section || "",
      ...article.steps,
      ...(article.notes || []),
    ].join(" ");
    const terms = tokenize(weighted);
    const freq = new Map();
    for (const t of terms) freq.set(t, (freq.get(t) || 0) + 1);
    return { article, freq, length: terms.length };
  });

  const df = new Map();
  for (const d of docs) for (const t of d.freq.keys()) df.set(t, (df.get(t) || 0) + 1);

  const avgLength = docs.reduce((sum, d) => sum + d.length, 0) / docs.length;
  return { docs, df, avgLength, total: docs.length };
})();

function score(doc, queryTerms) {
  let total = 0;
  for (const term of queryTerms) {
    const tf = doc.freq.get(term);
    if (!tf) continue;
    const df = INDEX.df.get(term) || 0;
    const idf = Math.log(1 + (INDEX.total - df + 0.5) / (df + 0.5));
    const norm = tf * (K1 + 1) / (tf + K1 * (1 - B + B * (doc.length / INDEX.avgLength)));
    total += idf * norm;
  }
  return total;
}

/**
 * Find the knowledge base articles most likely to answer a question.
 * Returns [{article, score}] sorted best-first, empty when nothing matches.
 */
export function retrieve(query, limit = 3) {
  const terms = expand(tokenize(query), query);
  if (!terms.length) return [];

  return INDEX.docs
    .map((doc) => ({ article: doc.article, score: score(doc, terms) }))
    .filter((hit) => hit.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, limit);
}

// Below this the top hit is a coincidence rather than an answer. Calibrated in
// scripts/test_retrieval.mjs: the weakest real question scores 6.2, while the
// best-scoring out-of-scope question ("what is the share price of nomura",
// which matches only on "nomura") scores 3.0.
export const RELEVANCE_FLOOR = 4.5;

export function isConfident(hits) {
  return hits.length > 0 && hits[0].score >= RELEVANCE_FLOOR;
}

/** Render an article as the plain troubleshooting answer, used when no LLM key is set. */
export function renderArticle(article) {
  const lines = [`**${article.title}**`, ""];
  article.steps.forEach((step, i) => lines.push(`${i + 1}. ${step}`));
  if (article.notes?.length) {
    lines.push("", ...article.notes.map((n) => `Note: ${n}`));
  }
  return lines.join("\n");
}
