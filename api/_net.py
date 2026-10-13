"""Outbound HTTP for the API routes.

Everything server-side here talks to exactly one kind of endpoint: POST JSON,
read JSON back. This keeps that in one place, along with the TLS context.

The TLS part matters locally: a python.org install on macOS trusts no root CAs
by default, so every https call fails with CERTIFICATE_VERIFY_FAILED. Vercel's
runtime is fine, but certifi makes both behave the same.
"""

import json
import ssl
import time
import urllib.error
import urllib.request

try:
    import certifi
    SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    SSL_CTX = ssl.create_default_context()


RETRY_STATUSES = {408, 429, 500, 502, 503, 504}


def post_json(url, payload, headers=None, timeout=30, retries=1):
    """POST a JSON body. Returns (status, raw bytes).

    An HTTP error status is returned rather than raised -- callers here decide
    what a 4xx or 5xx means. Connection failures still raise.

    One retry by default, on the statuses that mean "try again" and on dropped
    connections. Embedding a query is on the critical path for vector search,
    and a single transient failure silently costs the retrieval quality the
    index was built for.
    """
    body = json.dumps(payload).encode("utf-8") if not isinstance(payload, bytes) else payload

    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST",
    )

    last_error = None
    for attempt in range(retries + 1):
        if attempt:
            time.sleep(0.6 * attempt)
        try:
            with urllib.request.urlopen(request, timeout=timeout, context=SSL_CTX) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as exc:
            status, raw = exc.code, exc.read()
            if status not in RETRY_STATUSES or attempt == retries:
                return status, raw
            last_error = "HTTP %d" % status
        except Exception as exc:
            if attempt == retries:
                raise
            last_error = str(exc)
        print("retrying %s after %s" % (url.rsplit("/", 1)[-1], last_error))

    return 0, b""
