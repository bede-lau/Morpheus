"""Local native validation service. Run from project root: python -m pipeline.server.

Use the frontend dev-server proxy for /api. The service accepts requests only
from localhost browser origins and does not fetch or open caller-provided paths.
"""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from urllib.parse import urlparse

from .export import ManifestError, package_environment, validate_physics
from .read_api import EnvironmentStore

MAX_BODY = 5 * 1024 * 1024


class Handler(BaseHTTPRequestHandler):
    server_version = "MorpheusLocal/1"
    environment_store = EnvironmentStore()

    def _allowed_origin(self):
        origin = self.headers.get("Origin")
        if not origin:
            return True
        parsed = urlparse(origin)
        return parsed.scheme in {"http", "https"} and parsed.hostname in {"localhost", "127.0.0.1", "::1"}

    def _send(self, status, body, content_type="application/json", extra=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        origin = self.headers.get("Origin")
        if origin and self._allowed_origin():
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status, value):
        self._send(status, json.dumps(value, allow_nan=False).encode())

    def do_OPTIONS(self):
        if not self._allowed_origin():
            self._json(403, {"error": "Only localhost browser origins are allowed"})
            return
        self._send(204, b"", extra={"Access-Control-Allow-Methods": "POST, GET, OPTIONS", "Access-Control-Allow-Headers": "Content-Type"})

    def do_GET(self):
        if not self._allowed_origin():
            self._json(403, {"error": "Only localhost browser origins are allowed"})
            return
        path = urlparse(self.path).path
        try:
            result = self.environment_store.read(path)
        except Exception as exc:
            self.log_error("Environment read failed: %s", exc)
            self._json(500, {"error": "Local environment read failed; see server log"})
            return
        if result is not None:
            self._send(*result)
            return
        if path != "/api/health":
            self._json(404, {"error": "Not found"})
            return
        try:
            import mujoco
            self._json(200, {"status": "ready", "engine": "MuJoCo", "engineVersion": mujoco.__version__})
        except ImportError:
            self._json(200, {"status": "schema_only", "engine": "MuJoCo", "message": "Install requirements.txt for native physics validation"})

    def do_POST(self):
        if not self._allowed_origin():
            self._json(403, {"error": "Only localhost browser origins are allowed"})
            return
        if self.path not in {"/api/validate", "/api/export"}:
            self._json(404, {"error": "Not found"})
            return
        try:
            size = int(self.headers.get("Content-Length", "-1"))
        except ValueError:
            size = -1
        if size < 0:
            self._json(411, {"error": "Content-Length is required"})
            return
        if size > MAX_BODY:
            self._json(413, {"error": "Scene JSON exceeds the 5 MB limit"})
            return
        if self.headers.get_content_type() != "application/json":
            self._json(415, {"error": "Expected application/json"})
            return
        try:
            manifest = json.loads(self.rfile.read(size))
            if self.path == "/api/validate":
                self._json(200, validate_physics(manifest))
            else:
                package, report = package_environment(manifest)
                if report["status"] == "failed":
                    self._json(422, {"error": "Environment failed native validation; correct the scene before exporting", "validation": report})
                    return
                self._send(200, package, "application/zip", {"Content-Disposition": 'attachment; filename="morpheus-environment.zip"', "X-Morpheus-Validation": report["status"]})
        except (ManifestError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            self._json(400, {"status": "failed", "schemaValid": False, "error": str(exc)})
        except Exception as exc:
            self.log_error("Validation error: %s", exc)
            self._json(500, {"status": "failed", "error": "Local environment processing failed; see server log"})


def main():
    parser = argparse.ArgumentParser(description="Morpheus local MuJoCo validation API")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    # Import before announcing readiness: first native import can be slow on a
    # fresh macOS installation and should not consume a request timeout.
    try:
        import mujoco  # noqa: F401
    except ImportError:
        pass
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Morpheus validation API: http://127.0.0.1:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
