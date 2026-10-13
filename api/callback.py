"""Forwards a callback request to the n8n webhook.

The browser cannot call the webhook directly -- its preflight OPTIONS returns
500 with no Access-Control-Allow-Origin, so the POST is blocked before it is
sent. Forwarding server-side sidesteps CORS entirely and keeps the webhook URL
out of the page.
"""

import os
import sys

# Vercel loads this file as the route entrypoint with the project root as the
# working directory, so the sibling modules in this folder are not importable
# by default. Put this directory on the path before importing them.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from http.server import BaseHTTPRequestHandler

from _http import method_not_allowed, read_json, send_text
from _net import post_json

WEBHOOK = os.environ.get("WEBHOOK_URL", "https://n8n-sprint.neoflo.ai/webhook/initiate")


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            status, body = post_json(WEBHOOK, read_json(self), timeout=30)
        except Exception as exc:
            print("Forward to n8n failed:", exc)
            return send_text(self, 502, '{"message":"Could not reach the callback service"}')

        send_text(self, status, body)

    def do_GET(self):
        method_not_allowed(self)

    def log_message(self, *args):
        pass
