#!/usr/bin/env python3
"""Embed the knowledge base and build the FAISS index.

    export OPENROUTER_API_KEY=sk-or-...
    python3 scripts/build_index.py

Writes kb/kb.faiss (the vectors) and kb/kb_index.json (which row is which
article, plus the model it was built with). Re-run after changing the PDF or
the embedding model -- the index and the query must come from the same model,
so kb_index.json records it and api/_search.py embeds queries with it.

Vectors are L2-normalised and stored in an IndexFlatIP, which makes inner
product equal cosine similarity. Flat means exhaustive and exact: at 28
articles there is nothing to approximate, and an exact index has no recall
loss to tune away.
"""

import json
import os
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "api"))

import _search  # noqa: E402  (needs the path set above)

KB_JSON = os.path.join(HERE, "kb", "kb.json")
INDEX_PATH = os.path.join(HERE, "kb", "kb.faiss")
INDEX_META = os.path.join(HERE, "kb", "kb_index.json")

BATCH = 32


def passage(article):
    """What gets embedded for an article.

    The title carries most of the signal, so it leads; the steps follow for the
    vocabulary a user is likely to echo ("spooler", "Credential Manager").
    """
    parts = [article["title"]]
    if article.get("section"):
        parts.append("Category: " + article["section"])
    parts.extend(article["steps"])
    parts.extend(article.get("notes", []))
    return "\n".join(parts)


def main():
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        sys.exit(
            "OPENROUTER_API_KEY is not set.\n"
            "  export OPENROUTER_API_KEY=sk-or-...\n"
            "Until the index exists the chat falls back to BM25, which works."
        )

    model = os.environ.get("EMBEDDING_MODEL", _search.DEFAULT_EMBED_MODEL)

    with open(KB_JSON, encoding="utf-8") as fh:
        articles = json.load(fh)["articles"]

    print("embedding %d articles with %s" % (len(articles), model))
    texts = [passage(a) for a in articles]

    vectors = []
    for start in range(0, len(texts), BATCH):
        chunk = texts[start:start + BATCH]
        vectors.extend(_search.embed(chunk, model=model, api_key=key))
        print("  %d/%d" % (min(start + BATCH, len(texts)), len(texts)))

    if len(vectors) != len(articles):
        sys.exit("expected %d vectors, got %d" % (len(articles), len(vectors)))

    import faiss
    matrix = _search._normalise(vectors)
    dimensions = matrix.shape[1]

    index = faiss.IndexFlatIP(dimensions)
    index.add(matrix)
    faiss.write_index(index, INDEX_PATH)

    with open(INDEX_META, "w", encoding="utf-8") as fh:
        json.dump({
            "model": model,
            "dimensions": dimensions,
            "ids": [a["id"] for a in articles],
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }, fh, indent=1)

    size = os.path.getsize(INDEX_PATH)
    print("\nwrote kb/kb.faiss  (%d vectors x %d dims, %.1f KB)"
          % (index.ntotal, dimensions, size / 1024))
    print("wrote kb/kb_index.json")
    print("\nNow calibrate the cosine floor:  python3 scripts/test_retrieval.py")


if __name__ == "__main__":
    main()
