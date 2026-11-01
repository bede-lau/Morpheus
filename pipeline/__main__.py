"""Usage: python -m pipeline validate scene.json | package scene.json -o scene.zip"""
import argparse
import json
from pathlib import Path
import sys

from .export import ManifestError, package_environment, validate_physics


def main():
    parser = argparse.ArgumentParser(description="Export and validate Morpheus MuJoCo environments")
    parser.add_argument("command", choices=["validate", "package"])
    parser.add_argument("manifest", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    args = parser.parse_args()
    try:
        manifest = json.loads(args.manifest.read_text())
        if args.command == "package":
            data, report = package_environment(manifest)
            output = args.output or args.manifest.with_suffix(".zip")
            output.write_bytes(data)
            print(json.dumps({"output": str(output), "validation": report}, indent=2))
        else:
            report = validate_physics(manifest)
            print(json.dumps(report, indent=2))
        return 1 if report.get("status") == "failed" else 0
    except (ManifestError, OSError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
