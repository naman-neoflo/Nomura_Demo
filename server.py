#!/usr/bin/env python3
"""Serve callback.html locally and proxy form submissions to the n8n webhook.

The browser never talks to n8n directly, so CORS never comes into play:
the POST goes to this server, and this server forwards it server-side.

    python3 server.py          # then open http://localhost:8000
"""

import json
import ssl
import sys
import urllib.error
import urllib.request
from http.server import HTTPServer, SimpleHTTPRequestHandler

# This Python build does not trust any root CAs by default (a common macOS
# python.org install issue), which makes every https call fail. Use certifi's
# bundle when it is installed, otherwise fall back to the default context.
try:
    import certifi
    SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    SSL_CTX = ssl.create_default_context()

WEBHOOK = "https://n8n-sprint.neoflo.ai/webhook/initiate"
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000


class Handler(SimpleHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/api/callback":
            self.send_error(404)
            return

        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length)

        req = urllib.request.Request(
            WEBHOOK,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=30, context=SSL_CTX) as res:
                status, payload = res.status, res.read()
        except urllib.error.HTTPError as e:
            status, payload = e.code, e.read()
        except Exception as e:
            status = 502
            payload = json.dumps({"error": str(e)}).encode()

        print(f"  -> forwarded to n8n: HTTP {status} {payload[:200]!r}")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path == "/":
            self.path = "/callback.html"
        return SimpleHTTPRequestHandler.do_GET(self)

    def log_message(self, fmt, *args):
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))


if __name__ == "__main__":
    print(f"Serving http://localhost:{PORT}  (proxying POSTs to {WEBHOOK})")
    HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
