#!/usr/bin/env python3
"""Local dev server for the PDF search tool.

Sets the headers needed for SharedArrayBuffer (multi-threaded wasm in
onnxruntime-web) and proper MIME types for .mjs/.wasm.  Usage:

    python3 serve.py [port]        # default 8000
"""
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler


class Handler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def guess_type(self, path):
        if path.endswith(".mjs"):
            return "text/javascript"
        if path.endswith(".wasm"):
            return "application/wasm"
        if path.endswith(".jsonl"):
            return "application/octet-stream"
        return super().guess_type(path)

    def log_message(self, fmt, *args):  # one line per request, quieter
        sys.stderr.write("%s %s\n" % (self.address_string(), fmt % args))


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    HTTPServer(("127.0.0.1", port), Handler).serve_forever()
