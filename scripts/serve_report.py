#!/usr/bin/env python
"""Serve REPORT.md as rendered HTML via a lightweight HTTP server."""

from __future__ import annotations

import argparse
import http.server
import socketserver
from pathlib import Path


INDEX_TEMPLATE = """<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <title>Compute–Communication Overlap Investigation Report</title>
  <style>
    :root { color-scheme: light dark; }
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 0 auto; max-width: 900px; padding: 2rem; line-height: 1.6; }
    pre { background: #1e1e1e; color: #f1f1f1; padding: 1rem; overflow-x: auto; }
    code { background: rgba(27,31,35,.05); padding: 0.2em 0.4em; border-radius: 4px; }
    img { max-width: 100%; height: auto; }
    table { border-collapse: collapse; }
    table, th, td { border: 1px solid currentColor; padding: 0.4rem; }
  </style>
  <script src=\"https://cdn.jsdelivr.net/npm/marked/marked.min.js\"></script>
</head>
<body>
  <main id=\"content\">Loading report…</main>
  <script>
    async function loadReport() {
      const response = await fetch('REPORT.md');
      const text = await response.text();
      const html = marked.parse(text, { mangle: false, headerIds: true });
      document.getElementById('content').innerHTML = html;
    }
    loadReport();
  </script>
</body>
</html>
"""


class ReportRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, base_path: Path, **kwargs):
        self.base_path = base_path
        super().__init__(*args, directory=str(base_path), **kwargs)

    def do_GET(self):  # noqa: N802 - http.server signature
        if self.path in {"/", "/index.html"}:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(INDEX_TEMPLATE.encode("utf-8"))
        else:
            super().do_GET()


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve REPORT.md over HTTP")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to serve on (default: 8000)")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Project root containing REPORT.md")
    args = parser.parse_args()

    report_path = args.root / "REPORT.md"
    if not report_path.exists():
        parser.error(f"REPORT.md not found at {report_path}")

    handler = lambda *h_args, **h_kwargs: ReportRequestHandler(*h_args, base_path=args.root, **h_kwargs)
    with socketserver.ThreadingTCPServer((args.host, args.port), handler) as httpd:
        print(f"Serving REPORT.md from {args.root} at http://{args.host}:{args.port}/")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down.")


if __name__ == "__main__":
    main()
