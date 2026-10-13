"""Raises a support ticket from the chat widget.

Routing (team and priority) comes from the knowledge base article the chat was
looking at, so a ticket inherits the escalation the KB specifies.

A ticket is only real if TICKET_WEBHOOK_URL points at something that records
it. With no webhook configured this returns 503 rather than inventing a
reference number, so the widget can offer the callback form instead of telling
someone their ticket was filed when it was not.
"""

import json
import os
import time
from http.server import BaseHTTPRequestHandler

from _http import method_not_allowed, read_json, send_json
from _net import post_json

TICKET_WEBHOOK = os.environ.get("TICKET_WEBHOOK_URL")
DEFAULT_ROUTE = {"team": "IT Service Desk", "priority": "P3"}


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = read_json(self)
        issue = (body.get("issue") or "").strip()
        article = body.get("article") or {}

        if not issue:
            return send_json(self, 400, {"message": "A description of the issue is required"})

        if not TICKET_WEBHOOK:
            return send_json(self, 503, {
                "message": "Ticketing is not connected yet",
                "configured": False,
            })

        escalation = article.get("escalation") or {}
        route = {
            "team": escalation.get("team") or DEFAULT_ROUTE["team"],
            "priority": escalation.get("priority") or DEFAULT_ROUTE["priority"],
        }

        try:
            status, raw = post_json(TICKET_WEBHOOK, {
                "source": "it-service-desk-chat",
                "name": body.get("name"),
                "contact_number": body.get("contact"),
                "issue": issue,
                "kb_article": article.get("title"),
                "kb_article_id": article.get("id"),
                "assignment_group": route["team"],
                "priority": route["priority"],
                "transcript": (body.get("transcript") or [])[-20:],
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }, timeout=30)
        except Exception as exc:
            print("Ticket webhook unreachable:", exc)
            return send_json(self, 502, {"message": "Could not reach the ticketing system"})

        if status >= 400:
            print("Ticket webhook returned", status, raw[:300])
            return send_json(self, 502, {"message": "The ticketing system rejected the request"})

        # Use the reference the ticketing system returns; never fabricate one.
        reference = None
        try:
            data = json.loads(raw)
            for field in ("reference", "ticket", "number", "id"):
                if data.get(field):
                    reference = data[field]
                    break
        except (ValueError, AttributeError):
            pass

        send_json(self, 200, {
            "configured": True,
            "reference": reference,
            "team": route["team"],
            "priority": route["priority"],
        })

    def do_GET(self):
        method_not_allowed(self)

    def log_message(self, *args):
        pass
