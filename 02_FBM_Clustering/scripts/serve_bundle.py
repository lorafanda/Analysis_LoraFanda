"""
serve_bundle.py - serve the analysis repo over HTTP with CORS, so a page served from another
port (the site checkout, GitHub Pages) can fetch a local data bundle and the QC figures.

    "C:/Users/fanda/AppData/Local/Programs/Python/Python311/python.exe" \
        02_FBM_Clustering/scripts/serve_bundle.py 8000

Then open the visualizer with
    LM_visualizer.html?bundle=http://localhost:8000/02_FBM_Clustering/outputs/250_recon/fsaverage/activity_viz/
                       &hg=http://localhost:8000/01_FBM_Analysis/outputs/04_ersp_LM/&review=1

A plain `python -m http.server` sends no Access-Control-Allow-Origin header, and the browser
then refuses the cross-port fetch. Cache-Control: no-store, so a rebuilt bundle shows at once.
"""
import os
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class CORS(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    directory = sys.argv[2] if len(sys.argv) > 2 else REPO
    print(f"serving {directory} on http://localhost:{port}/  (CORS *, no cache) - Ctrl-C to stop")
    ThreadingHTTPServer(("127.0.0.1", port), partial(CORS, directory=directory)).serve_forever()
