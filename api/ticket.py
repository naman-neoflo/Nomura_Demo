"""Raises a Freshdesk ticket from the chat widget.

Calls the Freshdesk API directly -- no middleware. Routing comes from the
knowledge base: each article names the team and priority to use when its steps
do not resolve the issue, and that priority maps onto Freshdesk's scale.

A ticket is only real if Freshdesk is configured. With the credentials unset
this returns 503 rather than inventing a reference number, so the widget can
offer the callback form instead of telling someone their ticket was filed when
it was not.
"""

import base64
import json
import os
import sys
import time

# Vercel loads this file as the route entrypoint with the project root as the
# working directory, so the sibling modules in this folder are not importable
# by default. Put this directory on the path before importing them.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from http.server import BaseHTTPRequestHandler

from _http import method_not_allowed, read_json, send_json
from _net import post_json

DOMAIN = os.environ.get("FRESHDESK_DOMAIN", "").strip().rstrip("/")
API_KEY = os.environ.get("FRESHDESK_API_KEY", "").strip()
GROUP_ID = os.environ.get("FRESHDESK_GROUP_ID", "").strip()
PRODUCT_ID = os.environ.get("FRESHDESK_PRODUCT_ID", "").strip()

# The knowledge base works in P1/P2/P3; Freshdesk uses 1 Low to 4 Urgent.
PRIORITY = {"P1": 4, "P2": 3, "P3": 2}
STATUS_OPEN = 2

DEFAULT_ROUTE = {"team": "IT Service Desk", "priority": "P3"}


def configured():
    return bool(DOMAIN and API_KEY)


def api_url():
    host = DOMAIN if "." in DOMAIN else "%s.freshdesk.com" % DOMAIN
    host = host.replace("https://", "").replace("http://", "")
    return "https://%s/api/v2/tickets" % host


def auth_header():
    # Freshdesk takes the API key as the basic-auth username; the password is
    # ignored, and "X" is what their own documentation uses.
    token = base64.b64encode(("%s:X" % API_KEY).encode()).decode()
    return "Basic " + token


def describe(body, article, route):
    """The ticket body. Everything the agent needs to pick it up cold."""
    lines = [body["issue"], ""]
    if article.get("title"):
        lines.append("Knowledge base article used: %s" % article["title"])
    lines.append("Suggested assignment: %s (%s)" % (route["team"], route["priority"]))
    lines.append("Raised from: the IT Service Desk chat assistant")

    transcript = body.get("transcript") or []
    if transcript:
        lines += ["", "Conversation:"]
        for turn in transcript[-20:]:
            who = "Employee" if turn.get("role") == "user" else "Assistant"
            lines.append("%s: %s" % (who, str(turn.get("content") or "").strip()))

    return "\n".join(lines)


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = read_json(self)
        issue = (body.get("issue") or "").strip()
        name = (body.get("name") or "").strip()
        contact = (body.get("contact") or "").strip()
        article = body.get("article") or {}

        if not issue:
            return send_json(self, 400, {"message": "A description of the issue is required"})

        # Freshdesk needs a requester. A phone number is accepted in place of an
        # email, but then the name is mandatory.
        if not name or not contact:
            return send_json(self, 400, {
                "message": "A name and contact number are required to raise a ticket",
                "needs": ["name", "contact"],
            })

        if not configured():
            return send_json(self, 503, {
                "message": "Ticketing is not connected yet",
                "configured": False,
            })

        escalation = article.get("escalation") or {}
        route = {
            "team": escalation.get("team") or DEFAULT_ROUTE["team"],
            "priority": escalation.get("priority") or DEFAULT_ROUTE["priority"],
        }

        payload = {
            "name": name,
            "phone": contact,
            "subject": "IT Service Desk: %s" % issue[:160],
            "description": describe(body, article, route),
            "status": STATUS_OPEN,
            "priority": PRIORITY.get(route["priority"], 2),
            "tags": ["it-service-desk", "chat-assistant", route["team"]],
        }
        if GROUP_ID.isdigit():
            payload["group_id"] = int(GROUP_ID)
        if PRODUCT_ID.isdigit():
            payload["product_id"] = int(PRODUCT_ID)

        try:
            status, raw = post_json(
                api_url(), payload,
                headers={"Authorization": auth_header()},
                timeout=30,
            )
        except Exception as exc:
            print("Freshdesk unreachable:", exc)
            return send_json(self, 502, {"message": "Could not reach the ticketing system"})

        if status >= 400:
            # Freshdesk returns a useful errors array; log it, do not leak it.
            print("Freshdesk returned", status, raw[:500])
            return send_json(self, 502, {
                "message": "The ticketing system rejected the request",
            })

        reference = None
        try:
            reference = json.loads(raw).get("id")
        except (ValueError, AttributeError):
            pass

        print("Freshdesk ticket %s created for %s (%s, %s)"
              % (reference, name, route["team"], route["priority"]))

        return send_json(self, 200, {
            "configured": True,
            "reference": reference,
            "team": route["team"],
            "priority": route["priority"],
        })

    def do_GET(self):
        """Health check: is Freshdesk wired up.

        Deliberately says nothing else. This route is public, and the helpdesk
        domain, group id and product id are infrastructure details that give an
        attacker a head start for no benefit to anyone legitimate.
        """
        send_json(self, 200, {"configured": configured()})

    def log_message(self, *args):
        pass
