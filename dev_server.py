#!/usr/bin/env python3
"""Local development server.

Mirrors how Vercel serves this project: static files from the project root, and
each api/<name>.py module mounted at /api/<name> using the same handler class
Vercel loads. Running the real handlers locally means the routes behave here
exactly as they do once deployed.

    python3 dev_server.py          -> http://localhost:8000
"""

import importlib.util
import os
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler

ROOT = os.path.dirname(os.path.abspath(__file__))
API_DIR = os.path.join(ROOT, "api")
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000

# Deliberately NOT adding api/ to sys.path. Vercel loads each route with the
# project root as the working directory, so a route that does not put its own
# directory on the path fails there with ModuleNotFoundError. Loading them the
# same way here means that failure shows up locally instead of in production.


def load_routes():
    routes = {}
    for filename in sorted(os.listdir(API_DIR)):
        if not filename.endswith(".py") or filename.startswith("_"):
            continue
        name = filename[:-3]
        spec = importlib.util.spec_from_file_location(name, os.path.join(API_DIR, filename))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        if hasattr(module, "handler"):
            routes["/api/" + name] = module.handler
    return routes


ROUTES = load_routes()


class Dev(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=ROOT, **kwargs)

    def _dispatch(self, method):
        route = ROUTES.get(self.path.split("?")[0])
        if route is None:
            return False
        # The route classes are BaseHTTPRequestHandler subclasses, but they are
        # only ever used as a bag of methods -- borrow the method and run it
        # against this already-constructed request.
        handler_method = getattr(route, "do_" + method, None)
        if handler_method is None:
            self.send_response(405)
            self.send_header("Allow", "POST")
            self.end_headers()
            return True
        try:
            handler_method(self)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            if not getattr(self, "_headers_sent", False):
                try:
                    self.send_response(500)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(b'{"message":"Handler error"}')
                except OSError:
                    pass
        return True

    def do_POST(self):
        if not self._dispatch("POST"):
            self.send_error(404)

    def do_GET(self):
        if not self._dispatch("GET"):
            super().do_GET()

    def log_message(self, fmt, *args):
        sys.stderr.write("%s\n" % (fmt % args))


if __name__ == "__main__":
    print("http://localhost:%d" % PORT)
    print("routes: " + ", ".join(sorted(ROUTES)))
    import _search
    status = _search.index_status()
    print("FAISS index: " + ("%d vectors (%s)" % (status["vectors"], status["model"])
                             if status["available"] else "not built - " + str(status["reason"])))
    print("OPENROUTER_API_KEY " + ("set" if os.environ.get("OPENROUTER_API_KEY")
                                   else "not set - knowledge base steps will be served directly"))
    HTTPServer(("127.0.0.1", PORT), Dev).serve_forever()
