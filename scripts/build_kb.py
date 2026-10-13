#!/usr/bin/env python3
"""Parse the IT knowledge base PDF into retrievable chunks.

    python3 scripts/build_kb.py

Writes api/kb.json. Re-run whenever the PDF changes.

Each article in the PDF looks like:

    Outlook Not Opening
    1. Close Outlook completely from Task Manager
    2. Press Win + R -> type outlook.exe /safe -> press Enter
    * Still not opening -> Messaging Team -> P2

so a chunk carries its section, title, ordered steps, and the escalation
route (team + priority) to use when the steps do not resolve the issue.
"""

import json
import os
import re
import sys

try:
    import pdfplumber
except ImportError:
    sys.exit("pdfplumber is required:  pip3 install pdfplumber")

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF = os.path.join(HERE, "kb", "Nomura_IT_Knowledge_Base.pdf")
OUT = os.path.join(HERE, "kb", "kb.json")
OUT_JS = os.path.join(HERE, "api", "_kb-data.js")

# The PDF's arrow glyph extracts as the "fi" ligature, and its bullet as "n".
ARROW = "→"
SECTION_RE = re.compile(r"^(\d+)\.\s+([A-Z][A-Z &]+)$")
STEP_RE = re.compile(r"^(\d+)\.\s+(.*)$")
BULLET_RE = re.compile(r"^n\s+(.*)$")
ARROW_RE = re.compile(r"\bfi\b")
# "Still not opening -> Messaging Team -> P2"
ESCALATION_RE = re.compile(r"^(.*?)\s*" + ARROW + r"\s*(.+?)\s*" + ARROW + r"\s*(P[123])\s*$")


ACRONYMS = {"Vpn": "VPN", "It": "IT", "Sla": "SLA", "Mfa": "MFA", "Wifi": "WiFi"}


def title_case(text):
    return " ".join(ACRONYMS.get(w, w) for w in text.title().split())


def clean(line):
    # The PDF's arrow glyph extracts as the characters "fi". Only a standalone
    # "fi" is an arrow -- replacing every occurrence would corrupt "Office"
    # into "Of->ce", "Specific" into "Speci->c" and "confirmed" into "con->rmed".
    line = ARROW_RE.sub(ARROW, line)
    return re.sub(r"\s+", " ", line).strip()


def read_lines():
    with pdfplumber.open(PDF) as pdf:
        pages = [p.extract_text() or "" for p in pdf.pages]
    out = []
    for raw in "\n".join(pages).split("\n"):
        line = clean(raw)
        if line:
            out.append(line)
    return out


SKIP = (
    "NOMURA IT HELPDESK",
    "General IT Knowledge Base",
    "Nomura IT Helpdesk " + ARROW,
    "For internal use only",
)


def parse():
    lines = read_lines()
    articles, section, cur = [], None, None

    def flush():
        nonlocal cur
        if cur and cur["steps"]:
            articles.append(cur)
        cur = None

    i = 0
    while i < len(lines):
        line = lines[i]

        if any(line.startswith(s) for s in SKIP):
            i += 1
            continue

        m = SECTION_RE.match(line)
        if m:
            flush()
            section = title_case(m.group(2).strip())
            # The priority/assignment tables are reference material, not an article.
            if "Ticket Priority" in section or "Sla" in section:
                section = "Ticket Priority & SLA"
            i += 1
            continue

        m = STEP_RE.match(line)
        if m and cur:
            text = m.group(2)
            # Join wrapped continuation lines.
            while i + 1 < len(lines):
                nxt = lines[i + 1]
                if STEP_RE.match(nxt) or BULLET_RE.match(nxt) or SECTION_RE.match(nxt):
                    break
                if is_title(nxt, lines, i + 1):
                    break
                text += " " + nxt
                i += 1
            cur["steps"].append(text.strip())
            i += 1
            continue

        m = BULLET_RE.match(line)
        if m and cur:
            note = m.group(1)
            esc = ESCALATION_RE.match(note)
            if esc:
                cur["escalation"] = {
                    "when": esc.group(1).strip(),
                    "team": esc.group(2).strip(),
                    "priority": esc.group(3),
                }
            else:
                cur["notes"].append(note)
            i += 1
            continue

        if is_title(line, lines, i):
            flush()
            cur = {"section": section, "title": line, "steps": [], "notes": [], "escalation": None}
            i += 1
            continue

        i += 1

    flush()
    return articles


def is_title(line, lines, idx):
    """A title is a short non-step line directly followed by step 1."""
    if STEP_RE.match(line) or BULLET_RE.match(line) or SECTION_RE.match(line):
        return False
    if len(line) > 80 or ARROW in line:
        return False
    for j in range(idx + 1, min(idx + 3, len(lines))):
        m = STEP_RE.match(lines[j])
        if m and m.group(1) == "1":
            return True
        if STEP_RE.match(lines[j]):
            return False
    return False


def reference_chunks():
    """The SLA and routing tables, kept as their own lookups."""
    return [
        {
            "section": "Ticket Priority & SLA",
            "title": "Ticket priority levels and response times",
            "steps": [
                "P1 - Employee completely cannot work - SLA 1 hour",
                "P2 - Major disruption, partially working - SLA 4 hours",
                "P3 - Minor issue, workaround available - SLA next business day",
            ],
            "notes": [],
            "escalation": None,
        },
        {
            "section": "Ticket Priority & SLA",
            "title": "Which team handles which issue",
            "steps": [
                "Email, Outlook, Teams, Slack - Messaging Team",
                "VPN, WiFi, Internet, Network - Network Team",
                "Laptop, Printer, Monitor, Hardware - End User Computing",
                "Access, Permissions, MFA, Accounts - Identity and Access Management",
                "Unknown or general IT issues - IT Service Desk",
            ],
            "notes": [],
            "escalation": None,
        },
    ]


def main():
    articles = parse() + reference_chunks()
    for n, a in enumerate(articles, 1):
        a["id"] = "kb-%03d" % n
        # Flat text is what retrieval scores against.
        a["text"] = " ".join([a["section"] or "", a["title"]] + a["steps"] + a["notes"])

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"articles": articles}, f, indent=1, ensure_ascii=False)

    # Also emit an ES module. The serverless bundler always follows a static
    # import, whereas a JSON file read at runtime depends on file tracing.
    with open(OUT_JS, "w", encoding="utf-8") as f:
        f.write("// Generated by scripts/build_kb.py -- do not edit by hand.\n")
        f.write("// Source: kb/Nomura_IT_Knowledge_Base.pdf\n\n")
        f.write("export const ARTICLES = ")
        json.dump(articles, f, indent=1, ensure_ascii=False)
        f.write(";\n")

    print("wrote %s and %s  (%d articles)" % (
        os.path.relpath(OUT, HERE), os.path.relpath(OUT_JS, HERE), len(articles)))
    sec = {}
    for a in articles:
        sec[a["section"]] = sec.get(a["section"], 0) + 1
    for k, v in sec.items():
        print("  %-28s %d" % (k, v))
    missing = [a["title"] for a in articles if not a["escalation"] and a["section"] != "Ticket Priority & SLA"]
    if missing:
        print("  no escalation route parsed for:", missing)


if __name__ == "__main__":
    main()
