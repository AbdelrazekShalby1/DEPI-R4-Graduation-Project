from __future__ import annotations

import sys
from pathlib import Path
from socketserver import ThreadingMixIn
from wsgiref.simple_server import WSGIRequestHandler, WSGIServer, make_server

BASE_DIR = Path(__file__).resolve().parent
sys.stdout = (BASE_DIR / "server.log").open("a", encoding="utf-8")
sys.stderr = (BASE_DIR / "server.err").open("a", encoding="utf-8")

from app import app


class ThreadingWSGIServer(ThreadingMixIn, WSGIServer):
    daemon_threads = True


class LoggingRequestHandler(WSGIRequestHandler):
    def log_message(self, format: str, *args) -> None:
        sys.stderr.write("%s - - [%s] %s\n" % (self.address_string(), self.log_date_time_string(), format % args))
        sys.stderr.flush()


print("Serving on http://127.0.0.1:8000", flush=True)
with make_server(
    "127.0.0.1",
    8000,
    app,
    server_class=ThreadingWSGIServer,
    handler_class=LoggingRequestHandler,
) as server:
    server.serve_forever()
