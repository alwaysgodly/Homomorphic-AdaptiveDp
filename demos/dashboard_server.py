"""
Live dashboard for the HE secure-aggregation demo (standard library only).

    python demos/dashboard_server.py          then open  http://localhost:8000
    press "Start training" in the page (or run  python demos/he_demo.py  yourself)

The page only DISPLAYS events that he_demo.py writes to results/live_events.jsonl while the
real encryption / aggregation / decryption code runs - nothing on the page is pre-filled.
"""
import json, os, subprocess, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
EVENTS = os.path.join(ROOT, "results", "live_events.jsonl")
PROC = {"p": None}


def running():
    return PROC["p"] is not None and PROC["p"].poll() is None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path in ("/", "/index.html"):
            with open(os.path.join(HERE, "dashboard.html"), "rb") as f:
                return self._send(200, f.read(), "text/html; charset=utf-8")
        if u.path == "/events":
            since = int(parse_qs(u.query).get("since", ["0"])[0])
            evs = []
            if os.path.exists(EVENTS):
                with open(EVENTS) as f:
                    lines = f.read().splitlines()
                for ln in lines[since:]:
                    try:
                        evs.append(json.loads(ln))
                    except ValueError:
                        break            # half-written last line; pick it up next poll
            return self._send(200, json.dumps({"events": evs, "next": since + len(evs), "running": running()}))
        self._send(404, "not found", "text/plain")

    def do_POST(self):
        if urlparse(self.path).path != "/run":
            return self._send(404, "not found", "text/plain")
        n = int(self.headers.get("Content-Length", 0))
        try:
            req = json.loads(self.rfile.read(n) or b"{}")
        except ValueError:
            req = {}
        if running():
            return self._send(409, json.dumps({"error": "a run is already in progress"}))
        key_bits = int(req.get("key_bits", 1024))
        rounds = max(1, min(30, int(req.get("rounds", 10))))
        cmd = [sys.executable, os.path.join(HERE, "he_demo.py"), "--key-bits", str(key_bits),
               "--rounds", str(rounds), "--no-plot"]
        if req.get("dp_noise"):
            cmd += ["--dp-noise", str(float(req["dp_noise"]))]
        os.makedirs(os.path.dirname(EVENTS), exist_ok=True)
        open(EVENTS, "w").close()            # clear the previous run so the page never shows stale events
        PROC["p"] = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self._send(200, json.dumps({"started": True, "cmd": " ".join(cmd[1:])}))


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    print(f"Dashboard: http://localhost:{port}   (Ctrl+C to stop)")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
