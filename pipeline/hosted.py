"""Serve the Morpheus UI and both APIs from one Railway service."""
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
import shutil
from urllib.parse import urlparse

from . import object_server, server

ROOT = Path(__file__).resolve().parents[1]


def seed_output():
    source = Path(os.environ.get("MORPHEUS_SEED_DIR", "/seed-output"))
    target = ROOT / "output"
    if not source.is_dir():
        return
    for item in source.rglob("*"):
        destination = target / item.relative_to(source)
        if item.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
        elif item.is_file() and not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, destination)


class HostedHandler(server.Handler, object_server.Handler, SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT / "dist"), **kwargs)

    def _allowed_origin(self):
        origin = self.headers.get("Origin")
        if not origin:
            return True
        parsed = urlparse(origin)
        return parsed.scheme in {"http", "https"} and parsed.netloc == self.headers.get("Host")

    def allowed(self):
        return self._allowed_origin()

    def do_GET(self):
        path = urlparse(self.path).path
        if path.startswith("/api/objects/"):
            return object_server.Handler.do_GET(self)
        if path.startswith("/api/"):
            return server.Handler.do_GET(self)
        return SimpleHTTPRequestHandler.do_GET(self)

    def do_POST(self):
        if urlparse(self.path).path.startswith("/api/objects/"):
            return object_server.Handler.do_POST(self)
        return server.Handler.do_POST(self)


def main():
    seed_output()
    port = int(os.environ.get("PORT", "8080"))
    httpd = ThreadingHTTPServer(("0.0.0.0", port), HostedHandler)
    print(f"Morpheus listening on {port}", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
