"""Shared helpers for the Vercel Python functions in this directory.

Vercel maps each api/*.py file to a route and loads its `handler` class, a
BaseHTTPRequestHandler subclass. Files prefixed with `_` are modules, not
routes.
"""

import json


def read_json(request):
    """Parse a JSON request body. A missing or malformed body gives {}."""
    try:
        length = int(request.headers.get("Content-Length") or 0)
    except (TypeError, ValueError):
        return {}
    if length <= 0:
        return {}
    try:
        return json.loads(request.rfile.read(length))
    except (ValueError, OSError):
        return {}


def send_json(request, status, payload):
    body = json.dumps(payload).encode("utf-8")
    request.send_response(status)
    request.send_header("Content-Type", "application/json; charset=utf-8")
    request.send_header("Content-Length", str(len(body)))
    request.end_headers()
    request.wfile.write(body)


def send_text(request, status, body, content_type="application/json; charset=utf-8"):
    data = body.encode("utf-8") if isinstance(body, str) else body
    request.send_response(status)
    request.send_header("Content-Type", content_type)
    request.send_header("Content-Length", str(len(data)))
    request.end_headers()
    request.wfile.write(data)


def method_not_allowed(request, allowed="POST"):
    request.send_response(405)
    request.send_header("Allow", allowed)
    request.send_header("Content-Type", "application/json; charset=utf-8")
    request.end_headers()
    request.wfile.write(b'{"message":"Method not allowed"}')
