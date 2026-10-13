# Nomura IT Service Desk

A single-page IT service desk, styled to Nomura's visual identity. Two ways in:

- **Request a callback** — a form that posts to an n8n webhook, which places an
  outbound AI voice call via Vapi.
- **IT Assistant** — a chat widget that answers from the IT knowledge base, and
  escalates to a ticket or a callback when it cannot help.

## Files

| Path | Purpose |
| --- | --- |
| `index.html` | The page and the chat widget. No build step. |
| `api/chat.py` | Answers a question from the knowledge base. |
| `api/callback.py` | Forwards callback requests to n8n. |
| `api/ticket.py` | Raises a ticket, when a ticketing webhook is configured. |
| `api/_search.py` | FAISS vector search fused with BM25. |
| `kb/Nomura_IT_Knowledge_Base.pdf` | Source knowledge base. |
| `kb/kb.json` | Generated. The parsed articles. |
| `kb/kb.faiss` | Generated. The vector index. Not in the repo until built. |
| `kb/kb_index.json` | Generated. Row order and the embedding model used. |
| `scripts/build_kb.py` | Parses the PDF into `kb/kb.json`. |
| `scripts/build_index.py` | Embeds the articles and builds the FAISS index. |
| `scripts/test_retrieval.py` | Retrieval accuracy check and floor calibration. |
| `dev_server.py` | Local server. Mounts the same handlers Vercel runs. |

Everything server-side is Python, because FAISS is a Python library and the
chat route has to run it.

## Running locally

```bash
pip3 install -r requirements.txt
python3 dev_server.py        # http://localhost:8000
```

`dev_server.py` serves the page and mounts every `api/<name>.py` module at
`/api/<name>`, so the routes behave locally exactly as they do deployed. It
prints on startup whether the FAISS index and the OpenRouter key are present.

Submissions go to the API route, which forwards to n8n **server-side**. This
matters: the browser cannot call the n8n webhook directly, because its preflight
`OPTIONS` returns `500` with no `Access-Control-Allow-Origin`, so the POST is
blocked before it is sent and surfaces only as "Failed to fetch".

## The chat widget

Ask it an IT question and it answers from the knowledge base.

**Not every message is a lookup.** Treating them all as one is what made the
chat feel robotic, so the route reads the turn first:

| Turn | Response |
| --- | --- |
| A greeting ("Hi", "Hii", "Heyy") | A short hello. Elongated spellings are normalised, so "Hiii" is a greeting, not a failed search. |
| A thank-you or "that worked" | A warm close, no escalation offered. |
| "It didn't work" / "still not working" | Not a new question — it is the end of the steps already given. Escalates on the article in play, quoting the knowledge base's own routing ("this needs the Messaging Team … a P2 ticket"). |
| Anything else | Retrieval. |

A turn only counts as small talk when every word in it is a pleasantry, so
"hi my outlook won't open" still goes through retrieval.

**Retrieval decides whether a turn is a follow-up or a new topic**, not the
wording alone. "Email is not working" reads like a failure report, but it is
someone opening a new problem — escalating it to whichever team owned the
previous question would be worse than not recognising the phrase at all. So a
failure report only escalates when retrieval finds nothing for it, or finds the
article already in play. Otherwise it is answered as the new question it is.
For the same reason, the context retry below never applies to a failure report:
it has no topic of its own, and attaching the previous question to it
manufactures a confident match for something nobody asked.

The widget sends the active article with each message, which is what lets a
three-word "no it didn't work" know which steps it refers to.

**Retrieval** runs two retrievers and fuses them with reciprocal rank fusion:

- **FAISS** over OpenRouter embeddings. `scripts/build_index.py` embeds each
  article once into a flat inner-product index; queries are embedded with the
  same model and searched against it. Vectors are L2-normalised, so inner
  product is cosine similarity. `IndexFlatIP` is exhaustive and exact — at 28
  articles there is nothing to approximate, and an exact index has no recall
  loss to tune.
- **BM25**, lexical. Needs no key and no index, so the chat works before the
  index is built, and it catches exact tokens embeddings blur together — "P1",
  "spooler", "ipconfig".

They keep separate confidence floors, because their scores are on unrelated
scales; either retriever being sure is enough to answer. Measured by
`python3 scripts/test_retrieval.py`, with the index built:

```
top-1 (used when no LLM key is set): 24/24
top-3 (what the model receives):     24/24
out-of-scope correctly declined:     3/3
```

Fusion is what earns the top-1 score: BM25 alone gets 23/24, missing "connected
to vpn but can't reach internal sites", which the vector side ranks correctly.

The cosine floor is measured, not guessed — the weakest genuine question scores
0.438 and the strongest out-of-scope one 0.325, so `VECTOR_FLOOR` sits at 0.38,
in the middle of that gap.

Re-run the script after changing the knowledge base or the embedding model; it
reprints the separation so the floors can be re-centred.

**Answering** depends on `OPENROUTER_API_KEY`:

| Key | Behaviour |
| --- | --- |
| Set | The model (default `openai/gpt-5.1` via OpenRouter) writes the answer, grounded in the retrieved articles only. Instructed to reply `NO_ANSWER` rather than improvise. |
| Not set | The matching article's steps are returned verbatim. |

Both paths answer only from the knowledge base. If the model call fails the
route falls back to the article's steps rather than failing the chat.

**Feedback.** Every assistant message carries thumbs up / thumbs down. Thumbs up
acknowledges and stops. Thumbs down — and any question the knowledge base cannot
answer — offers the two escalation routes:

- **Raise a ticket** → `POST /api/ticket`. Team and priority are inherited from
  the knowledge base article's own escalation route (e.g. *Outlook Not Opening →
  Messaging Team → P2*), defaulting to IT Service Desk / P3.
- **Request a callback** → closes the chat, preselects the matching reason in
  the form and focuses it.

Thumbs ratings are not currently recorded anywhere — the thumbs down drives the
escalation offer, and nothing is stored.

## Environment variables

All optional; see `.env.example`. Set them in the Vercel project for production.

| Variable | Effect when unset |
| --- | --- |
| `OPENROUTER_API_KEY` | No vector search and no written answers — BM25 retrieval, knowledge base steps returned verbatim. |
| `OPENROUTER_MODEL` | Defaults to `openai/gpt-5.1`. `openai/gpt-5-mini` is ~5x cheaper and ample for this task. |
| `EMBEDDING_MODEL` | Defaults to `openai/text-embedding-3-small`. |
| `BM25_FLOOR` / `VECTOR_FLOOR` | Defaults from `api/_search.py`. |
| `WEBHOOK_URL` | Falls back to the n8n URL in `api/callback.py`. |
| `TICKET_WEBHOOK_URL` | "Raise a ticket" reports that ticketing is not connected. It does not invent a reference number. |

## Rebuilding the knowledge base and index

```bash
pip3 install -r requirements.txt pdfplumber

python3 scripts/build_kb.py                 # kb/*.pdf -> kb/kb.json
export OPENROUTER_API_KEY=sk-or-...
python3 scripts/build_index.py              # kb/kb.json -> kb/kb.faiss
python3 scripts/test_retrieval.py           # check accuracy, calibrate floors
```

**Commit `kb/kb.faiss` and `kb/kb_index.json`.** They are build artefacts, but
Vercel has no key at build time, so the deployed function reads them from the
repo. They are small — 28 vectors is about 170 KB at 1536 dimensions.

The PDF parser reads the arrow glyph, which extracts as the characters `fi`.
Only a standalone `fi` is an arrow — replacing every occurrence corrupts
`Office` into `Of→ce` and silently drops whole articles.

## Deploying to Vercel

Deploy the repo root. No build command, no framework preset.

- The page must be `index.html` — Vercel serves that at `/`, and a repo with
  only `callback.html` returns `404 NOT_FOUND` on the root URL.
- Each `api/*.py` file becomes a Python function at `/api/<name>`. Files
  prefixed with `_` are modules, not routes.
- `requirements.txt` pins faiss-cpu and numpy. Python functions have a 500 MB
  uncompressed bundle limit; this bundle is around 100 MB.

## n8n configuration

Settings that each produced a real failure during setup:

- **HTTP Method: POST.** While set to GET, a POST returns
  `404 "This webhook is not registered for POST requests."`
- **Respond: Immediately**, or add a *Respond to Webhook* node. Otherwise:
  `{"code":0,"message":"No Respond to Webhook node found in the workflow"}`
- **Allowed Origins (CORS):** set to the page's domain if you ever want the
  browser to call the webhook directly.

In the HTTP Request node that calls Vapi, wrap every expression so it emits
valid JSON — free text containing a quote would otherwise break the body:

```json
{
  "assistantId": "…",
  "assistantOverrides": {
    "variableValues": {
      "name": {{ JSON.stringify($json.body.name) }},
      "phone": {{ JSON.stringify($json.body.contact_number) }},
      "reason": {{ JSON.stringify($json.body.reason) }}
    }
  },
  "phoneNumberId": "…",
  "customer": { "number": {{ JSON.stringify($json.body.contact_number) }} }
}
```

The JSON Body field must be in **Expression** mode, or n8n sends the `{{ }}`
through as literal text.

## Submitted payload (callback form)

```json
{
  "name": "Naman Sharma",
  "contact_number": "+12125550147",
  "reason": "Password Reset",
  "submitted_at": "2026-10-13T02:16:00.000Z"
}
```

`reason` is one of: Password Reset, Software Installation, General IT Query,
Account Unlock, VPN Not Connecting.

`contact_number` is normalised to E.164 before sending — spaces, brackets and
dashes are stripped, so `+1 (212) 555-0147` is submitted as `+12125550147`. Any
country code is accepted, but it must be present.

## Notes

- A successful callback submission **places a real phone call** to the number
  entered. Test with a number you control.
- The sidebar content (service desk hours, callback time estimate) is
  placeholder text — replace it before use.
- Internal tool. It carries Nomura's branding without being an official Nomura
  property, so keep this repo and any deployment private.
