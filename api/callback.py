"""Forwards a callback request to the n8n webhook.

The browser cannot call the webhook directly -- its preflight OPTIONS returns
500 with no Access-Control-Allow-Origin, so the POST is blocked before it is
sent. Forwarding server-side sidesteps CORS entirely and keeps the webhook URL
out of the page.
"""

import json
import os
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler

from _http import method_not_allowed, read_json, send_text

WEBHOOK = os.environ.get("WEBHOOK_URL", "https://n8n-sprint.neoflo.ai/webhook/initiate")


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        payload = json.dumps(read_json(self)).encode("utf-8")

        request = urllib.request.Request(
            WEBHOOK,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                status, body = response.status, response.read()
        except urllib.error.HTTPError as exc:
            status, body = exc.code, exc.read()
        except Exception as exc:
            print("Forward to n8n failed:", exc)
            return send_text(self, 502, '{"message":"Could not reach the callback service"}')

        send_text(self, status, body)

    def do_GET(self):
        method_not_allowed(self)

    def log_message(self, *args):
        pass
