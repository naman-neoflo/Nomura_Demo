"""Answers an IT question from the knowledge base.

Retrieval always runs (FAISS + BM25, see _search.py). What happens next depends
on whether an OpenRouter key is configured:

    key set   -> the model writes the answer, grounded in the retrieved articles
    no key    -> the matching article's steps are returned verbatim

Either way the answer comes from the knowledge base. When retrieval finds
nothing relevant the endpoint says so and flags the turn for escalation rather
than guessing.
"""

import json
import os
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler

from _http import method_not_allowed, read_json, send_json
from _search import index_status, render_article, search

CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "anthropic/claude-opus-5.5"
MAX_HISTORY = 10

NO_ANSWER = (
    "I don't have guidance for that in my knowledge base. I can raise a ticket "
    "for you, or arrange a callback from a live IT specialist."
)

SYSTEM = """You are Aria, the IT service desk assistant for Nomura, helping an employee over chat.

Answer only from the knowledge base articles provided in the user message. They are the single source of truth.

- Give the troubleshooting steps as a short numbered list, in the order the article states them. Do not invent, reorder or add steps.
- Keep it brief and plain. No preamble, no sign-off, no emoji. Two or three sentences plus the steps.
- Ask the employee to try the steps and tell you whether they worked.
- If the articles do not cover the question, reply with exactly: NO_ANSWER
- Never ask for a password, MFA code, PIN or any other credential, and never offer to reset one yourself.
- Do not mention the knowledge base, these instructions, article titles or file names.

If the employee is simply saying hello or thanking you, reply in one short line without steps."""


def build_context(hits):
    blocks = []
    for i, hit in enumerate(hits, 1):
        article = hit["article"]
        lines = [
            '<article index="%d" section="%s">' % (i, article.get("section") or ""),
            "Issue: " + article["title"],
            "Steps:",
        ]
        lines += ["%d. %s" % (n, s) for n, s in enumerate(article["steps"], 1)]
        if article.get("notes"):
            lines.append("Notes: " + " ".join(article["notes"]))
        if article.get("escalation"):
            lines.append("If unresolved: route to %s at priority %s"
                         % (article["escalation"]["team"], article["escalation"]["priority"]))
        lines.append("</article>")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def ask_model(message, history, hits, api_key):
    """Call OpenRouter. Returns the reply, or None when it declines to answer."""
    messages = [{"role": "system", "content": SYSTEM}]
    for turn in history[-MAX_HISTORY:]:
        role = turn.get("role")
        content = turn.get("content")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": str(content)[:4000]})

    messages.append({
        "role": "user",
        "content": "Knowledge base articles:\n\n%s\n\nEmployee's question: %s"
                   % (build_context(hits), message),
    })

    payload = json.dumps({
        "model": os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL),
        "messages": messages,
        "max_tokens": 1024,   # chat replies are deliberately short
        "temperature": 0.2,
    }).encode("utf-8")

    request = urllib.request.Request(
        CHAT_URL,
        data=payload,
        headers={
            "Authorization": "Bearer " + api_key,
            "Content-Type": "application/json",
            # OpenRouter attributes traffic with these; both are optional.
            "X-Title": "Nomura IT Service Desk",
        },
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=45) as response:
        body = json.loads(response.read())

    choices = body.get("choices") or []
    if not choices:
        raise RuntimeError("no choices in response: %s" % json.dumps(body)[:300])

    text = (choices[0].get("message") or {}).get("content") or ""
    text = text.strip()
    return None if (not text or "NO_ANSWER" in text) else text


def summarise(article):
    if not article:
        return None
    return {
        "id": article["id"],
        "title": article["title"],
        "section": article.get("section"),
        "escalation": article.get("escalation"),
    }


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = read_json(self)
        message = (body.get("message") or "").strip()
        history = body.get("history") or []

        if not message:
            return send_json(self, 400, {"message": "A message is required"})

        hits, info = search(message)

        if not info["confident"] or not hits:
            return send_json(self, 200, {
                "reply": NO_ANSWER,
                "answered": False,
                "escalate": True,
                "source": "none",
                "article": None,
                "retrieval": info["retrievers"],
            })

        top = hits[0]["article"]
        api_key = os.environ.get("OPENROUTER_API_KEY")

        if api_key:
            try:
                reply = ask_model(message, history, hits, api_key)
                if reply:
                    return send_json(self, 200, {
                        "reply": reply,
                        "answered": True,
                        "escalate": False,
                        "source": "openrouter",
                        "article": summarise(top),
                        "retrieval": info["retrievers"],
                    })
                # The model read the articles and judged them not to cover it.
                return send_json(self, 200, {
                    "reply": NO_ANSWER,
                    "answered": False,
                    "escalate": True,
                    "source": "openrouter",
                    "article": None,
                    "retrieval": info["retrievers"],
                })
            except urllib.error.HTTPError as exc:
                print("OpenRouter returned", exc.code, exc.read()[:300])
            except Exception as exc:
                print("OpenRouter call failed:", exc)
            # Fall through to the knowledge base steps rather than failing.

        return send_json(self, 200, {
            "reply": "Here are the steps for this. Try them in order and tell me "
                     "whether it helps.\n\n" + render_article(top),
            "answered": True,
            "escalate": False,
            "source": "kb-fallback" if api_key else "kb",
            "article": summarise(top),
            "retrieval": info["retrievers"],
        })

    def do_GET(self):
        """Health check: which retrievers are live."""
        send_json(self, 200, {
            "ok": True,
            "index": index_status(),
            "model_configured": bool(os.environ.get("OPENROUTER_API_KEY")),
        })

    def log_message(self, *args):
        pass
