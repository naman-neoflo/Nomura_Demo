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
| `api/callback.js` | Forwards callback requests to n8n. |
| `api/chat.js` | Answers a question from the knowledge base. |
| `api/ticket.js` | Raises a ticket, when a ticketing webhook is configured. |
| `api/_retrieval.js` | BM25 search over the knowledge base. |
| `api/_kb-data.js` | Generated. The knowledge base as an ES module. |
| `kb/Nomura_IT_Knowledge_Base.pdf` | Source knowledge base. |
| `kb/kb.json` | Generated. The parsed articles, for reading. |
| `scripts/build_kb.py` | Parses the PDF into the two generated files. |
| `scripts/test_retrieval.mjs` | Retrieval accuracy check. |
| `dev-server.mjs` | Local server. Mounts the same handlers Vercel runs. |

## Running locally

```bash
npm install
node dev-server.mjs        # http://localhost:8000
```

`dev-server.mjs` serves the page and mounts every `api/*.js` module at
`/api/<name>`, so the routes behave locally exactly as they do deployed.

Submissions go to the API route, which forwards to n8n **server-side**. This
matters: the browser cannot call the n8n webhook directly, because its preflight
`OPTIONS` returns `500` with no `Access-Control-Allow-Origin`, so the POST is
blocked before it is sent and surfaces only as "Failed to fetch".

## The chat widget

Ask it an IT question and it answers from the knowledge base.

**Retrieval** is BM25 over the 28 parsed articles, with the title and section
weighted above the body, plus a synonym table for the words people actually use
("wifi", "mfa", "pw"). Measured by `npm test`:

```
top-1 (used when no LLM key is set): 23/24
top-3 (what the model receives):     24/24
out-of-scope correctly declined:     3/3
```

**This is lexical search, not a vector database.** Embeddings would need an API
key the project does not have yet, and at 28 short articles with distinctive
terms ("Outlook", "spooler", "VPN") lexical matching is accurate with no cold
start. `retrieve()` in `api/_retrieval.js` is the seam — swapping in embeddings
and a vector store means changing that one file.

**Answering** depends on `ANTHROPIC_API_KEY`:

| Key | Behaviour |
| --- | --- |
| Set | Claude (`claude-opus-5-5`) writes the answer, grounded in the retrieved articles only. Instructed to reply `NO_ANSWER` rather than improvise. |
| Not set | The matching article's steps are returned verbatim. |

Both paths answer only from the knowledge base. If the Claude call fails the
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
| `ANTHROPIC_API_KEY` | Knowledge base steps are returned verbatim instead of a written answer. |
| `ANTHROPIC_MODEL` | Defaults to `claude-opus-5-5`. |
| `WEBHOOK_URL` | Falls back to the n8n URL in `api/callback.js`. |
| `TICKET_WEBHOOK_URL` | "Raise a ticket" reports that ticketing is not connected. It does not invent a reference number. |

## Rebuilding the knowledge base

```bash
pip3 install pdfplumber
npm run build:kb      # kb/*.pdf -> api/_kb-data.js + kb/kb.json
npm test              # re-check retrieval accuracy
```

The parser reads the PDF's arrow glyph, which extracts as the characters `fi`.
Only a standalone `fi` is an arrow — replacing every occurrence corrupts
`Office` into `Of→ce` and silently drops whole articles.

## Deploying to Vercel

Deploy the repo root. No build command, no framework preset.

- The page must be `index.html` — Vercel serves that at `/`, and a repo with
  only `callback.html` returns `404 NOT_FOUND` on the root URL.
- The `api/*.js` routes answer the page's POSTs. Files prefixed with `_` are
  modules, not routes.

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
