"""Serve the Morpheus UI and both APIs from one Railway service."""
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import hashlib
import os
from pathlib import Path
import shutil
import urllib.request
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


def prepare_collider():
    configured = os.environ.get("MORPHEUS_COLLIDER_PATH")
    if not configured:
        return
    destination = Path(configured)
    expected = "38b9f692edeb671b7106f3e36cfe7f4c55be20e98fe90c014a8714b09874f6fa"
    if destination.is_file() and hashlib.sha256(destination.read_bytes()).hexdigest() == expected:
        return
    url = "https://wlt-ai-cdn.art/example_exports/rustic_kitchen_with_natural_light/rustic_kitchen_with_natural_light_collider.glb"
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read(8_000_001)
    if len(data) > 8_000_000 or hashlib.sha256(data).hexdigest() != expected:
        raise ValueError("Downloaded collider did not match the expected checksum")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    temporary.write_bytes(data)
    temporary.replace(destination)


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
    prepare_collider()
    port = int(os.environ.get("PORT", "8080"))
    httpd = ThreadingHTTPServer(("0.0.0.0", port), HostedHandler)
    print(f"Morpheus listening on {port}", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
