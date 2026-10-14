#!/usr/bin/env python3
"""Checks how the chat route reads a turn, without calling the model.

    python3 scripts/test_conversation.py

Retrieval accuracy is covered by test_retrieval.py. This covers the decisions
made around it: greeting, thank-you, "it didn't work", and -- the one that has
broken twice -- telling a follow-up apart from a new topic raised with the same
words. Every case here is a regression that reached a real conversation.
"""

import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "api"))

import chat as chat_route          # noqa: E402
from _search import (install_intent, mentions_topic, search,  # noqa: E402
                     software_articles)

OUTLOOK = {"id": "kb-004", "title": "Outlook Password Keeps Asking",
           "section": "Email & Outlook",
           "escalation": {"team": "Identity and Access Management", "priority": "P2"}}
LAPTOP = {"id": "kb-018", "title": "Laptop Running Slow", "section": "Laptop & Hardware",
          "escalation": {"team": "End User Computing", "priority": "P3"}}


def decide(message, active=None):
    """Mirror the routing in api/chat.py, minus the model call.

    Returns one of: greeting, resolved, followup, answer, none.
    """
    # "thanks that worked" is all pleasantries, so the small-talk path answers
    # it before read_outcome is consulted. Both close the thread; tell them
    # apart by which reply came back.
    if chat_route.declines_escalation(message):
        return "continue"
    if chat_route.requests_escalation(message):
        return "handoff"
    pleasantry = chat_route.small_talk_reply(message)
    if pleasantry:
        return "greeting" if pleasantry.startswith("Hi ") else "resolved"
    if chat_route.read_outcome(message) == "resolved":
        return "resolved"
    outcome = chat_route.read_outcome(message)

    hits, info = search(message)
    same_topic = bool(active and hits and info["confident"]
                      and hits[0]["article"]["id"] == active.get("id"))
    if outcome == "failed" and (not mentions_topic(message)
                                or not info["confident"] or same_topic):
        return "followup"
    if not info["confident"] and install_intent(message) and software_articles():
        return "answer"
    if not info["confident"] or not hits:
        if active and len(message.split()) <= 6:
            return "followup"
        return "none"
    return "answer"


CASES = [
    # greetings, including the elongated spellings people actually type
    ("Hii", None, "greeting"),
    ("hiii there", None, "greeting"),
    ("Heyy", None, "greeting"),
    ("hello", None, "greeting"),
    ("thanks that worked", OUTLOOK, "resolved"),
    ("it worked, cheers", OUTLOOK, "resolved"),

    # genuine follow-ups: no topic of their own, so they end the current thread
    ("no it didn't work", OUTLOOK, "followup"),
    ("still not working", OUTLOOK, "followup"),
    ("no luck", OUTLOOK, "followup"),
    ("tried that already", OUTLOOK, "followup"),
    ("nothing changed", OUTLOOK, "followup"),
    ("no same issue", OUTLOOK, "followup"),
    ("same issue", OUTLOOK, "followup"),

    # a new problem phrased like a failure report must NOT escalate to the
    # team that owned the previous question
    ("email is not working unable to send email", LAPTOP, "answer"),
    ("my vpn is not working", LAPTOP, "answer"),
    ("printer not working", LAPTOP, "answer"),
    ("wifi is not working at my desk", LAPTOP, "answer"),

    # a greeting glued to a real question is a real question
    ("hi my outlook wont open", None, "answer"),

    # software and identity: not in the PDF, so these denied before
    ("i want to install software in my laptop", None, "answer"),
    ("install zoom", None, "answer"),
    ("i need photoshop installed", None, "answer"),
    ("i forgot my password", None, "answer"),
    ("my account is locked", None, "answer"),
    ("my password is expiring soon", None, "answer"),

    # turning down the handoff must not offer the handoff again, and must not
    # surface the routing tables as if they were steps
    ("no don't create a ticket i want the resolution here", LAPTOP, "continue"),
    ("no ticket please", LAPTOP, "continue"),
    ("dont raise a ticket", OUTLOOK, "continue"),
    ("can you just fix it", LAPTOP, "continue"),
    ("yes please raise a ticket", LAPTOP, "handoff"),
    ("can I speak to a person", LAPTOP, "handoff"),
    ("i want a callback", LAPTOP, "handoff"),

    # a problem statement containing success-shaped words is still a problem.
    # "Nothing is working on my laptop" was congratulating people on a fix.
    ("Hi, my laptop has not been working since morning. The internet is not "
     "working. Nothing is working on my laptop. I don't know why.", None, "answer"),
    ("nothing is working on my laptop", None, "answer"),
    ("the internet is not working", None, "answer"),
    ("my laptop is working slowly", None, "answer"),
    ("it's working now thanks", OUTLOOK, "resolved"),
    ("it is working now", OUTLOOK, "resolved"),

    # "it's not working" is our fix failing, not a new problem. The apostrophe
    # swallows the "is", leaving a bare "not" to carry the negation.
    ("it's not working", OUTLOOK, "followup"),
    ("that's not working", OUTLOOK, "followup"),
    ("i have tried everything but it's not working", OUTLOOK, "followup"),

    # more ways of saying the steps have been worked through. "I did all four
    # steps in order" was being answered as a brand new question.
    ("i have gone through all the steps", OUTLOOK, "followup"),
    ("yes I did all of them in order", OUTLOOK, "followup"),
    ("i did the steps already", OUTLOOK, "followup"),

    # still out of scope
    ("how do I book annual leave", None, "none"),
    ("what is the share price of nomura", None, "none"),
    ("can you order me a new desk chair", None, "none"),
]


def main():
    failures = 0
    for message, active, expected in CASES:
        got = decide(message, active)
        ok = got == expected
        failures += not ok
        print("%s %-44s %-9s%s" % ("ok  " if ok else "FAIL", repr(message), got,
                                   "" if ok else "   (expected %s)" % expected))
    print("\n%d/%d passed" % (len(CASES) - failures, len(CASES)))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
