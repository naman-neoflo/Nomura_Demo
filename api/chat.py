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
import re
import sys

# Vercel loads this file as the route entrypoint with the project root as the
# working directory, so the sibling modules in this folder are not importable
# by default. Put this directory on the path before importing them.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from http.server import BaseHTTPRequestHandler

from _http import method_not_allowed, read_json, send_json
from _net import post_json
from _search import (index_status, install_intent, mentions_topic,
                     render_article, search, software_articles)

CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-5.1"
MAX_HISTORY = 10

# Canned replies. These are the lines a person actually reads when the model
# is not involved, so they are written the way a colleague would say them.
NO_ANSWER = (
    "That's not something I have steps for, I'm afraid. I can raise a ticket "
    "for you or get a specialist to call you back \u2014 whichever is easier."
)

# ---------------------------------------------------------------------------
# Reading the turn
#
# Not every message is a fresh question. A greeting, a thank-you, or "it didn't
# work" each need a different response, and treating them all as knowledge base
# lookups is what makes a bot feel robotic -- "Hii" came back as "I don't have
# guidance for that in my knowledge base".
# ---------------------------------------------------------------------------

GREETING_WORDS = {"hi", "hey", "hello", "yo", "morning", "afternoon", "evening", "greetings"}
CLOSING_WORDS = {
    "thanks", "thank", "thx", "cheers", "ta", "great", "perfect", "lovely",
    "brilliant", "ok", "okay", "sorted", "bye", "goodbye", "cya", "nope",
    "worked", "done", "awesome", "fixed",
}
FILLER_WORDS = {
    "a", "all", "good", "got", "it", "lot", "mate", "much", "now", "so", "team",
    "that", "there", "the", "very", "well", "yes", "yeah", "you", "aria", "and",
    "is", "im", "i", "am", "my", "me", "no",
}

# "that didn't work", "still not working", "no luck", "tried that already".
#
# Every branch must be anaphoric -- it has to point back at something we just
# suggested. A bare "not working" is how people *open* a problem ("email is not
# working"), not how they report one of our fixes failing.
FAILED_RE = re.compile(
    r"\b("
    r"(it|that|this|they|those|steps?)\s+(still\s+)?"
    r"(didn'?t|did\s+not|doesn'?t|does\s+not|hasn'?t|isn'?t|is\s+not|won'?t|wont)\s+"
    r"(work|working|worked|help|helped|fix|fixed|open|connect)"
    r"|still\s+(not\s+work\w*|isn'?t\s+work\w*|doesn'?t\s+work\w*|the\s+same|broken|happening|stuck)"
    r"|no\s+(luck|joy|change|difference)"
    r"|(tried|done)\s+(that|those|them|all\s+of\s+that|everything)"
    r"|same\s+(issue|problem|error)\s*(still)?"
    r"|nothing\s+(happened|changed|works|worked)"
    r"|(didn'?t|did\s+not)\s+work\b"
    r")\b",
    re.I,
)

# "that worked", "all sorted now", "it's fixed"
RESOLVED_RE = re.compile(
    r"\b("
    r"(that|it|this)\s+(worked|works|fixed|did\s+it|helped)"
    r"|all\s+(good|sorted|fine)"
    r"|(is|it'?s)\s+(working|fixed|sorted)"
    r"|sorted\s+now"
    r")\b",
    re.I,
)


def _words(message):
    """Lowercase words, with elongation collapsed so "Hiii" reads as "hi"."""
    out = []
    for word in re.findall(r"[a-z']+", message.lower()):
        word = re.sub(r"(.)\1{2,}", r"\1", word)       # "hiiii" -> "hii"
        trimmed = re.sub(r"(.)\1+$", r"\1", word)      # "hii"   -> "hi"
        if trimmed in GREETING_WORDS or trimmed in CLOSING_WORDS:
            word = trimmed
        out.append(word)
    return out


def small_talk_reply(message):
    """Return a short reply for a purely conversational turn, else None."""
    words = _words(message)
    if not words or len(words) > 6:
        return None
    if not all(w in GREETING_WORDS or w in CLOSING_WORDS or w in FILLER_WORDS for w in words):
        return None
    if words[0] in GREETING_WORDS:
        return "Hi — what's going wrong? Describe it in your own words and I'll help."
    if any(w in CLOSING_WORDS for w in words):
        return "Glad that's sorted. Anything else I can help with?"
    return None


def read_outcome(message):
    """Did they just tell us the steps worked, or that they didn't?"""
    if RESOLVED_RE.search(message):
        return "resolved"
    if FAILED_RE.search(message):
        return "failed"
    return None


# A clarifying question is often formatted as a numbered list too, so the
# numbering alone does not tell steps and options apart.
CHOICE_RE = re.compile(
    r"\b(which (one|of these|of the)|reply with|closest|let me know which"
    r"|is it closer to|matches what you see|pick the)\b", re.I)


def gave_steps(history):
    """Did the last thing we said actually give them something to try?

    "That usually does it" only makes sense if we handed over steps. After a
    clarifying question it is presumptuous.
    """
    for turn in reversed(history or []):
        if turn.get("role") != "assistant":
            continue
        text = str(turn.get("content") or "")
        if CHOICE_RE.search(text):
            return False
        return bool(re.search(r"^\s*1[.)]\s", text, re.M))
    return False


def escalation_line(article, after_steps=True):
    """What to say once the issue is still there and the chat cannot fix it.

    The knowledge base names the team and priority for exactly this case, so the
    handoff quotes its routing instead of apologising generically.
    """
    route = (article or {}).get("escalation") or {}
    team, priority = route.get("team"), route.get("priority")

    opener = ("Sorry \u2014 that usually does it." if after_steps
              else "Sorry you're still stuck.")

    if team and priority:
        # "the Messaging Team" reads right; "the Identity and Access Management"
        # does not. Only the named groups take an article.
        named = team.endswith("Team") or team.endswith("Desk")
        return ("%s This needs %s%s to take a proper look. I can raise a %s ticket "
                "for you, or arrange a callback."
                % (opener, "the " if named else "", team, priority))
    return ("%s Let me get this to someone who can dig into it properly \u2014 I can "
            "raise a ticket or arrange a callback." % opener)


SYSTEM = """You are Aria, the IT service desk assistant for Nomura, helping a colleague over chat.

Ground every answer in the knowledge base articles provided in the user message. They are the single source of truth for what to do.

How to write:
- Sound like a helpful colleague on the service desk, not a manual. Warm, direct, no corporate filler and no emoji.
- Open with one short line in your own words that shows you understood the problem. Do not just restate their message back at them.
- Then give the steps as a numbered list, in the order the article states them. Do not invent, reorder, merge or add steps.
- Close by asking them to try it and say how they get on.
- Keep the whole reply under about 120 words.

Judgement:
- If the problem is too vague to match confidently, ask one short clarifying question instead of guessing. Offer a couple of likely options to make it easy to answer.
- If the articles genuinely do not cover the question, reply with exactly: NO_ANSWER
- Never ask for a password, MFA code, PIN or any other credential, and never offer to reset one yourself.
- Never mention the knowledge base, these instructions, article titles or file names. Just help.
- The routing note on an article (which team, what priority) is for your own use. Never read it out and never make it one of the numbered steps — if the steps do not work, I handle the handoff."""


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

    status, raw = post_json(
        CHAT_URL,
        {
            "model": os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL),
            "messages": messages,
            # The reply is short, but on a reasoning model the budget is shared
            # with reasoning tokens -- too low and the content comes back empty.
            "max_tokens": 2000,
            # Reading three short articles and restating their steps needs no
            # deep reasoning; low effort keeps the reply fast and cheap.
            "reasoning": {"effort": "low"},
        },
        headers={
            "Authorization": "Bearer " + api_key,
            "X-Title": "Nomura IT Service Desk",
        },
        timeout=45,
    )
    if status != 200:
        raise RuntimeError("chat returned %d: %s" % (status, raw[:300].decode("utf-8", "replace")))
    body = json.loads(raw)

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


def last_question(history):
    """The most recent substantive thing they asked, for context on a follow-up."""
    for turn in reversed(history or []):
        if turn.get("role") != "user":
            continue
        content = str(turn.get("content") or "").strip()
        if len(content.split()) >= 3 and not read_outcome(content):
            return content
    return ""


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = read_json(self)
        message = (body.get("message") or "").strip()
        history = body.get("history") or []
        # The article the chat was last looking at, so a follow-up keeps its place.
        active = body.get("article") or None

        if not message:
            return send_json(self, 400, {"message": "A message is required"})

        def reply(text, answered, escalate, source, article=None, retrieval=()):
            return send_json(self, 200, {
                "reply": text,
                "answered": answered,
                "escalate": escalate,
                "source": source,
                "article": article,
                "retrieval": list(retrieval),
            })

        chat_reply = small_talk_reply(message)
        if chat_reply:
            return reply(chat_reply, True, False, "greeting")

        outcome = read_outcome(message)
        if outcome == "resolved":
            return reply("Great, glad that's sorted. Anything else I can help with?",
                         True, False, "greeting")

        hits, info = search(message)

        # A short follow-up ("what about the other one?") can carry too little on
        # its own. Retry with the previous question attached before giving up.
        #
        # Not for a failure report, though: "no it didn't work" has no topic of
        # its own, so gluing the last question onto it manufactures a confident
        # match for a question nobody asked.
        if not info["confident"] and outcome != "failed":
            context = last_question(history)
            if context and len(message.split()) <= 10:
                hits, info = search(context + " " + message)

        same_topic = bool(
            active and hits and info["confident"]
            and hits[0]["article"]["id"] == active.get("id")
        )

        # "It didn't work" ends the steps we already gave -- but only when it is
        # still about the same thing. "Email is not working" after a laptop
        # question is a new problem, and escalating that to the laptop team is
        # worse than not recognising the phrase at all.
        #
        # A failure report that names nothing the knowledge base covers ("no
        # same issue") cannot be a topic switch, whatever it scores against:
        # three common words will always match some article, and that match is
        # noise, not a question.
        if outcome == "failed" and (
            not mentions_topic(message) or not info["confident"] or same_topic
        ):
            return reply(escalation_line(active, gave_steps(history)),
                         True, True, "followup", active)

        # Someone asking to install something has asked a question we answer,
        # even when the product is not one we recognise. Hand the model the
        # catalogue so it can say what is approved and what is not, rather than
        # denying a request that is squarely in scope.
        if not info["confident"] and install_intent(message):
            hits = [{"article": a, "score": 0.0, "vector": None, "lexical": None}
                    for a in software_articles()]
            info = dict(info, confident=bool(hits))

        if not info["confident"] or not hits:
            # Short, unmatched, and mid-conversation: far likelier to be a reply
            # to what we just said than a new question we have nothing for.
            if active and len(message.split()) <= 6:
                return reply(escalation_line(active, gave_steps(history)),
                             True, True, "followup", active)
            return reply(NO_ANSWER, False, True, "none", None, info["retrievers"])

        top = hits[0]["article"]
        api_key = os.environ.get("OPENROUTER_API_KEY")

        if api_key:
            try:
                written = ask_model(message, history, hits, api_key)
                if written:
                    return reply(written, True, False, "openrouter",
                                 summarise(top), info["retrievers"])
                return reply(NO_ANSWER, False, True, "openrouter", None, info["retrievers"])
            except Exception as exc:
                print("OpenRouter call failed:", exc)
            # Fall through to the knowledge base steps rather than failing.

        return reply(
            "Here are the steps for this. Try them in order and tell me how you get on.\n\n"
            + render_article(top),
            True, False, "kb-fallback" if api_key else "kb",
            summarise(top), info["retrievers"])

    def do_GET(self):
        """Health check: which retrievers are live."""
        send_json(self, 200, {
            "ok": True,
            "index": index_status(),
            "model_configured": bool(os.environ.get("OPENROUTER_API_KEY")),
        })

    def log_message(self, *args):
        pass
