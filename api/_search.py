"""Retrieval over the IT knowledge base.

Two retrievers, fused:

* **FAISS** over OpenRouter embeddings. `scripts/build_index.py` embeds every
  article once and writes a flat inner-product index; at query time the question
  is embedded and searched against it. Vectors are L2-normalised, so inner
  product is cosine similarity.
* **BM25**, lexical. It needs no key and no index, so the chat still works
  before the index is built, and it catches exact tokens that embeddings blur
  together -- "P1", "spooler", "ipconfig".

Results are combined with reciprocal rank fusion. Each retriever keeps its own
confidence floor, because their scores are on unrelated scales.
"""

import json
import math
import os
import re

from _net import post_json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_dotenv():
    """Read .env for local development. On Vercel the environment is already
    set, and no .env file is deployed, so this is a no-op there. Existing
    variables always win."""
    path = os.path.join(ROOT, ".env")
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key, value = key.strip(), value.strip().strip('"').strip("'")
                if key and value and key not in os.environ:
                    os.environ[key] = value
    except OSError:
        pass


_load_dotenv()
KB_JSON = os.path.join(ROOT, "kb", "kb.json")
INDEX_PATH = os.path.join(ROOT, "kb", "kb.faiss")
INDEX_META = os.path.join(ROOT, "kb", "kb_index.json")

EMBED_URL = "https://openrouter.ai/api/v1/embeddings"
DEFAULT_EMBED_MODEL = "openai/text-embedding-3-small"

# Both floors are measured, not guessed: scripts/test_retrieval.py prints the
# gap between the weakest genuine question and the strongest out-of-scope one on
# each signal, and each floor sits in the middle of its own gap. Re-run it after
# changing the knowledge base -- adding articles moves both.
#
#   BM25:   in-scope from 5.92, out-of-scope to 5.09
#   Cosine: in-scope from 0.315, out-of-scope to 0.255
BM25_FLOOR = float(os.environ.get("BM25_FLOOR", "5.5"))
VECTOR_FLOOR = float(os.environ.get("VECTOR_FLOOR", "0.285"))

RRF_K = 60  # reciprocal rank fusion constant

with open(KB_JSON, encoding="utf-8") as fh:
    ALL_ARTICLES = json.load(fh)["articles"]

# Internal articles (the routing and SLA tables) are operational metadata, not
# answers. They stay in kb.json as the record of how escalation is routed, but
# they are never retrieved -- a question containing "ticket" would otherwise
# surface the routing table and the model would read it out as steps.
ARTICLES = [a for a in ALL_ARTICLES if not a.get("internal")]

# --------------------------------------------------------------------------
# BM25
# --------------------------------------------------------------------------

K1, B = 1.4, 0.72

STOPWORDS = set((
    "a an the and or but if then than that this these those is are was were be been being "
    "do does did doing have has had having i me my we our you your it its they them to of "
    "in on at by for with about into over after before from up down out so no not can cant "
    "cannot will would should could may might just now get getting got im ive am "
    # The company name is in titles across the corpus, so it discriminates
    # nothing -- but it will match any question that merely mentions Nomura.
    "nomura"
).split())

# Helpdesk vocabulary the user's own words rarely match on.
SYNONYMS = {
    "wifi": ["wireless", "network"], "wireless": ["wifi", "network"],
    "email": ["outlook", "mail", "messaging"], "mail": ["outlook", "email"],
    "mails": ["outlook", "email"], "internet": ["network", "connection"],
    "web": ["internet", "browser"], "pw": ["password"], "pwd": ["password"],
    "passcode": ["password"], "login": ["password", "sign"], "signin": ["password", "sign"],
    "mfa": ["authentication", "authenticator", "factor"],
    "otp": ["authentication", "authenticator", "code"],
    "2fa": ["authentication", "authenticator", "factor"],
    "vpn": ["vpn", "remote"], "printer": ["print", "spooler", "queue"],
    "printing": ["print", "printer", "spooler"],
    "laptop": ["laptop", "device", "computer"], "pc": ["laptop", "device", "computer"],
    "machine": ["laptop", "device"], "screen": ["monitor", "display"],
    "display": ["monitor", "screen"], "slow": ["slow", "performance"],
    "frozen": ["stuck", "loading"], "hang": ["stuck", "loading"],
    "hanging": ["stuck", "loading"], "crash": ["stuck", "opening"],
    "crashing": ["stuck", "opening"], "charge": ["charging", "battery", "power"],
    "charging": ["charging", "battery", "power"], "battery": ["charging", "battery", "power"],
    "locked": ["locked", "lock", "access"], "teams": ["teams", "meeting", "call"],
    "meeting": ["teams", "meeting"], "drive": ["shared", "folder", "drive"],
    "folder": ["shared", "folder", "drive"], "spam": ["junk", "spam"],
    "junk": ["junk", "spam"],
}


def _stem(word):
    if len(word) > 4 and word.endswith("ing"):
        return word[:-3]
    if len(word) > 4 and word.endswith("ed"):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _tokenize(text):
    words = re.sub(r"[^a-z0-9\s]", " ", (text or "").lower()).split()
    return [_stem(w) for w in words if len(w) > 1 and w not in STOPWORDS]


def _expand(tokens, raw):
    out = list(tokens)
    for word in re.split(r"[^a-z0-9]+", (raw or "").lower()):
        for extra in SYNONYMS.get(word, ()):
            out.append(_stem(extra))
    return out


def _build_bm25():
    docs = []
    for article in ARTICLES:
        # Repeating title and section is how they get their weight.
        # Keywords carry title-level weight: they exist so a two-word query
        # ("install zoom") has something strong to match against.
        weighted = " ".join(
            [article["title"]] * 3
            + article.get("keywords", []) * 3
            + [article.get("section") or ""] * 2
            + article["steps"]
            + article.get("notes", [])
        )
        terms = _tokenize(weighted)
        freq = {}
        for t in terms:
            freq[t] = freq.get(t, 0) + 1
        docs.append({"article": article, "freq": freq, "length": len(terms)})

    df = {}
    for d in docs:
        for t in d["freq"]:
            df[t] = df.get(t, 0) + 1

    avg = sum(d["length"] for d in docs) / len(docs)
    return docs, df, avg


_BM25_DOCS, _BM25_DF, _BM25_AVG = _build_bm25()
_TOTAL = len(_BM25_DOCS)


def bm25_search(query, limit=5):
    terms = _expand(_tokenize(query), query)
    if not terms:
        return []
    scored = []
    for doc in _BM25_DOCS:
        total = 0.0
        for term in terms:
            tf = doc["freq"].get(term)
            if not tf:
                continue
            df = _BM25_DF.get(term, 0)
            idf = math.log(1 + (_TOTAL - df + 0.5) / (df + 0.5))
            norm = tf * (K1 + 1) / (tf + K1 * (1 - B + B * (doc["length"] / _BM25_AVG)))
            total += idf * norm
        if total > 0:
            scored.append((total, doc["article"]))
    scored.sort(key=lambda p: p[0], reverse=True)
    return [{"article": a, "score": s} for s, a in scored[:limit]]


# --------------------------------------------------------------------------
# FAISS
# --------------------------------------------------------------------------

_index = None
_index_meta = None
_index_error = None


def _load_index():
    """Load the FAISS index once per process. Missing index is not an error."""
    global _index, _index_meta, _index_error
    if _index is not None or _index_error is not None:
        return _index

    if not (os.path.exists(INDEX_PATH) and os.path.exists(INDEX_META)):
        _index_error = "no index built"
        return None
    try:
        import faiss
        with open(INDEX_META, encoding="utf-8") as fh:
            meta = json.load(fh)
        index = faiss.read_index(INDEX_PATH)
        if index.ntotal != len(meta["ids"]):
            raise ValueError(
                "index has %d vectors but metadata lists %d ids -- rebuild it"
                % (index.ntotal, len(meta["ids"]))
            )
        _index, _index_meta = index, meta
        return _index
    except Exception as exc:  # a broken index must not take the chat down
        _index_error = str(exc)
        return None


def index_status():
    _load_index()
    if _index is None:
        return {"available": False, "reason": _index_error}
    return {
        "available": True,
        "vectors": _index.ntotal,
        "model": _index_meta.get("model"),
        "dimensions": _index_meta.get("dimensions"),
        "built_at": _index_meta.get("built_at"),
    }


def embed(texts, model=None, api_key=None, timeout=30):
    """Embed texts through OpenRouter. Returns a list of float lists."""
    key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    status, raw = post_json(
        EMBED_URL,
        {
            "model": model or os.environ.get("EMBEDDING_MODEL", DEFAULT_EMBED_MODEL),
            "input": texts,
        },
        headers={"Authorization": "Bearer " + key},
        timeout=timeout,
    )
    if status != 200:
        raise RuntimeError("embeddings returned %d: %s" % (status, raw[:300].decode("utf-8", "replace")))
    body = json.loads(raw)

    # Keep OpenRouter's order explicit rather than trusting the response order.
    rows = sorted(body["data"], key=lambda d: d.get("index", 0))
    return [r["embedding"] for r in rows]


def _normalise(vectors):
    import numpy as np
    arr = np.asarray(vectors, dtype="float32")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def vector_search(query, limit=5):
    """FAISS search. Returns [] when no index or no key, rather than raising."""
    index = _load_index()
    if index is None:
        return []
    try:
        vec = _normalise(embed([query], model=_index_meta.get("model")))
        scores, positions = index.search(vec, min(limit, index.ntotal))
    except Exception as exc:
        print("vector search unavailable:", exc)
        return []

    by_id = {a["id"]: a for a in ARTICLES}
    hits = []
    for score, pos in zip(scores[0], positions[0]):
        if pos < 0:
            continue
        article = by_id.get(_index_meta["ids"][pos])
        if article:
            hits.append({"article": article, "score": float(score)})
    return hits


# --------------------------------------------------------------------------
# Fusion
# --------------------------------------------------------------------------

def search(query, limit=3):
    """Retrieve the articles most likely to answer a question.

    Returns (hits, info). `hits` is [{article, score, vector, lexical}] best
    first; `info` records which retrievers ran and whether the top hit clears
    its floor.
    """
    lexical = bm25_search(query, limit=5)
    vector = vector_search(query, limit=5)

    ranks = {}
    for rank, hit in enumerate(vector):
        ranks.setdefault(hit["article"]["id"], {})["vector"] = (rank, hit["score"])
    for rank, hit in enumerate(lexical):
        ranks.setdefault(hit["article"]["id"], {})["lexical"] = (rank, hit["score"])

    by_id = {a["id"]: a for a in ARTICLES}
    fused = []
    for article_id, signals in ranks.items():
        score = sum(1.0 / (RRF_K + rank + 1) for rank, _ in signals.values())
        fused.append({
            "article": by_id[article_id],
            "score": score,
            "vector": signals.get("vector", (None, None))[1],
            "lexical": signals.get("lexical", (None, None))[1],
        })
    fused.sort(key=lambda h: h["score"], reverse=True)
    fused = fused[:limit]

    top_vector = vector[0]["score"] if vector else None
    top_lexical = lexical[0]["score"] if lexical else None

    # Either retriever being sure is enough; they fail on different questions.
    confident = bool(
        (top_vector is not None and top_vector >= VECTOR_FLOOR)
        or (top_lexical is not None and top_lexical >= BM25_FLOOR)
    )

    return fused, {
        "retrievers": [n for n, hs in (("vector", vector), ("bm25", lexical)) if hs],
        "top_vector": top_vector,
        "top_lexical": top_lexical,
        "confident": confident,
    }


# The distinctive words the knowledge base is actually about, taken from the
# article titles and sections. Used to tell "email is not working" (names a
# topic) from "no same issue" (names nothing) -- the second is a follow-up, and
# any article it happens to score against is noise.
TOPIC_TERMS = frozenset(
    t for article in ARTICLES
    for t in _tokenize("%s %s %s" % (article["title"], article.get("section") or "",
                                     " ".join(article.get("keywords", []))))
    if len(t) > 2
) - {"not", "work", "cannot", "issue", "problem", "your", "keep", "stuck", "access"}


def mentions_topic(text):
    """Does this message name something the knowledge base covers?"""
    return any(t in TOPIC_TERMS for t in _tokenize(text))


# A request to install something is a category the knowledge base answers even
# when it cannot match the product -- an unapproved title should be told it is
# unapproved, not met with "I have no steps for that". Deliberately narrow: it
# requires an install word, so "I need help" cannot trigger it.
INSTALL_RE = re.compile(
    r"\b(install|installs|installing|installed|installation|download|downloaded"
    r"|licence|license|software|application)\b", re.I)


def install_intent(text):
    return bool(INSTALL_RE.search(text or ""))


def software_articles():
    """The software installation articles, in catalogue-first order."""
    found = [a for a in ARTICLES if a.get("section") == "Software Installation"]
    return sorted(found, key=lambda a: "not on the approved" in a["title"])


def render_article(article):
    """The plain troubleshooting answer, used when no LLM key is set."""
    lines = ["**%s**" % article["title"], ""]
    for i, step in enumerate(article["steps"], 1):
        lines.append("%d. %s" % (i, step))
    for note in article.get("notes", []):
        lines.append("")
        lines.append("Note: " + note)
    return "\n".join(lines)
