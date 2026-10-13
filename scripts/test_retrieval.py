#!/usr/bin/env python3
"""Checks retrieval against how people actually phrase helpdesk questions.

    python3 scripts/test_retrieval.py

Runs whatever retrievers are available. Without OPENROUTER_API_KEY and a built
index that is BM25 alone; with both, FAISS joins and the script also prints the
cosine separation between in-scope and out-of-scope questions, which is what
VECTOR_FLOOR in api/_search.py should be set from.
"""

import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "api"))

import _search  # noqa: E402

# (question, expected article title or None when it should not be answered)
CASES = [
    ("my outlook won't open", "Outlook Not Opening"),
    ("outlook keeps asking for my password", "Outlook Password Keeps Asking"),
    ("emails are stuck in my outbox", "Emails Stuck in Outbox"),
    ("I'm not receiving any emails", "Outlook Not Syncing / Emails Not Receiving"),
    ("important mail is going to junk", "Emails Going to Junk / Spam"),
    ("teams is stuck on the loading screen", "Teams Not Loading / Stuck on Loading Screen"),
    ("my teams calls keep dropping", "Teams Calls Dropping or Poor Audio"),
    ("I can't join a teams meeting", "Cannot Join a Teams Meeting"),
    ("vpn won't connect", "VPN Not Connecting"),
    ("vpn keeps disconnecting every few minutes", "VPN Disconnects Frequently"),
    ("connected to vpn but can't reach internal sites",
     "VPN Connects but Cannot Access Internal Resources"),
    ("no internet at my desk", "No Internet Access in Office"),
    ("the office internet is really slow", "Slow Internet in Office"),
    ("I can't see the nomura wifi network", "WiFi Not Showing Nomura Network"),
    ("my laptop is really slow", "Laptop Running Slow"),
    ("laptop screen keeps flickering", "Laptop Screen Flickering"),
    ("my laptop isn't charging", "Laptop Not Charging"),
    ("second monitor not detected", "External Monitor Not Detected"),
    ("my mouse stopped working", "Keyboard or Mouse Not Working"),
    ("can't get to the shared drive", "Cannot Access a Shared Drive or Folder"),
    ("mfa isn't working", "Multi-Factor Authentication (MFA) Not Working"),
    ("printer says offline", "Cannot Print — Printer Offline"),
    ("print jobs are stuck", "Print Jobs Stuck in Queue"),
    # Software installation and identity - not in the PDF, added from
    # kb/software_catalog.json and kb/sample_articles.json.
    ("i want to install software in my laptop", "Install approved software on your laptop"),
    ("install zoom", "Install approved software on your laptop"),
    ("can I get tableau installed", "Install approved software on your laptop"),
    ("i forgot my password", "Reset your Nomura password"),
    ("how do I reset my password", "Reset your Nomura password"),
    ("my account is locked", "Account locked after too many failed sign-ins"),
    ("my password is expiring soon", "Password expiring soon"),
    # Out of scope - should be declined. The routing and SLA tables are marked
    # internal, so questions about tickets no longer surface them as answers.
    ("what is a p1 ticket", None),
    ("no don't create a ticket i want the resolution here", None),
    ("how do I book annual leave", None),
    ("what is the share price of nomura", None),
    ("can you order me a new desk chair", None),
]


def main():
    status = _search.index_status()
    print("FAISS index: " + ("%d vectors, %s, %d dims"
                             % (status["vectors"], status["model"], status["dimensions"])
                             if status["available"] else "not built (%s)" % status["reason"]))
    print("retrievers: BM25" + (" + FAISS" if status["available"] else " only"))
    print()

    top1 = top3 = in_scope = out_ok = out_total = 0
    in_vec, out_vec, in_lex, out_lex = [], [], [], []

    for question, expected in CASES:
        hits, info = _search.search(question)
        answered = hits[0]["article"]["title"] if (info["confident"] and hits) else None
        vec = info["top_vector"]
        lex = info["top_lexical"]
        detail = "bm25 %5.2f" % (lex or 0)
        if vec is not None:
            detail += "  cos %.3f" % vec

        if expected is None:
            out_total += 1
            ok = answered is None
            out_ok += ok
            if vec is not None:
                out_vec.append(vec)
            out_lex.append(lex or 0)
            print("%s  %s  [out of scope] %s" % ("ok  " if ok else "FAIL", detail, question))
            if not ok:
                print("        answered with: %s" % answered)
            continue

        in_scope += 1
        if vec is not None:
            in_vec.append(vec)
        in_lex.append(lex or 0)
        hit1 = answered == expected
        hit3 = info["confident"] and any(h["article"]["title"] == expected for h in hits)
        top1 += hit1
        top3 += hit3
        print("%s  %s  %s" % ("ok  " if hit1 else ("top3" if hit3 else "FAIL"), detail, question))
        if not hit1:
            print("        expected: %s\n        got:      %s" % (expected, answered))

    print()
    print("top-1 (used when no LLM key is set): %d/%d" % (top1, in_scope))
    print("top-3 (what the model receives):     %d/%d" % (top3, in_scope))
    print("out-of-scope correctly declined:     %d/%d" % (out_ok, out_total))
    print("floors: bm25 %.1f, cosine %.2f" % (_search.BM25_FLOOR, _search.VECTOR_FLOOR))

    def separation(label, inside, outside, floor):
        if not inside or not outside:
            return
        lo, hi = min(inside), max(outside)
        print()
        print("%s separation:" % label)
        print("  weakest in-scope question:  %.3f" % lo)
        print("  strongest out-of-scope one: %.3f" % hi)
        if lo > hi:
            print("  -> a floor of %.3f sits mid-gap (currently %.3f)" % ((lo + hi) / 2, floor))
        else:
            print("  NOTE: these overlap, so no single floor separates them on this")
            print("  signal alone. The other retriever has to carry the rejection;")
            print("  that is what the two independent floors in _search.py are for.")

    separation("BM25", in_lex, out_lex, _search.BM25_FLOOR)
    separation("Cosine", in_vec, out_vec, _search.VECTOR_FLOOR)

    ok = top3 == in_scope and out_ok == out_total
    print("\n%s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
