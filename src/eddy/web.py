"""A web page served by the robot: its camera, where it is, a box to type commands in,
and a big stop button. Open it from a laptop or a phone on the same network.

    eddy web -r ros2          then open the link it prints

Nothing to install: this is Python's own web server. The link carries a short code that is
new on every start, so only someone who has seen the robot's terminal can drive it. The
code travels unencrypted, like everything else on the page: fine on the robot's own
network, not a defence against someone who can read the network's traffic.
"""
from __future__ import annotations

import hmac
import json
import math
import os
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional
from urllib.parse import parse_qs, urlparse


class WebPage:
    """Serves the page for one Eddy. start() returns at once; the server runs in the background."""

    def __init__(self, eddy, host: str = "0.0.0.0", port: int = 8080, code: Optional[str] = None):
        """code: what the link must carry (default: $EDDY_WEB_CODE, else a new random one)."""
        self.eddy = eddy
        self.code = code or os.environ.get("EDDY_WEB_CODE") or secrets.token_urlsafe(4)
        self.results = []                   # the last few outcomes, newest last, whoever asked
        self._running = False
        eddy.listeners.append(self._remember)
        page = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):   # keep the robot's terminal for the robot
                pass

            def _send(self, status: int, body: bytes, kind: str = "application/json") -> None:
                self.send_response(status)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def _allowed(self, query: dict) -> bool:
                given = (query.get("t") or [""])[0]
                return hmac.compare_digest(given.encode(), page.code.encode())

            def do_GET(self):
                url = urlparse(self.path)
                query = parse_qs(url.query)
                if url.path == "/":
                    # The page itself holds nothing secret; everything it asks for needs the code.
                    return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
                if not self._allowed(query):
                    return self._send(401, b'{"error": "wrong or missing code"}')
                if url.path == "/state":
                    return self._send(200, json.dumps(page.state()).encode())
                if url.path == "/camera.mjpg":
                    return page._stream(self)
                self._send(404, b'{"error": "no such page"}')

            def do_POST(self):
                url = urlparse(self.path)
                if not self._allowed(parse_qs(url.query)):
                    return self._send(401, b'{"error": "wrong or missing code"}')
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                except ValueError:
                    length = -1
                if not 0 <= length <= 500:
                    return self._send(413, b'{"error": "a command is one short sentence"}')
                text = self.rfile.read(length).decode(errors="replace").strip()
                if url.path == "/stop":
                    # Carried out before answering, so the page knows the stop has been sent.
                    result = page.eddy.stop("web")
                    return self._send(200, json.dumps({"ok": result.ok, "message": result.message}).encode())
                if url.path == "/tell":
                    if not text:
                        return self._send(400, b'{"error": "nothing to do"}')
                    # A drive can take a while: carry it out in the background and answer now.
                    # The outcome shows up in /state.
                    threading.Thread(target=page.eddy.tell, args=(text, "web"), daemon=True).start()
                    return self._send(202, b'{"ok": true}')
                self._send(404, b'{"error": "no such page"}')

        self.server = ThreadingHTTPServer((host, port), Handler)
        self.server.daemon_threads = True
        self.port = self.server.server_address[1]

    def start(self) -> "WebPage":
        self._running = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return self

    def stop(self) -> None:
        self._running = False
        self.server.shutdown()
        self.server.server_close()

    def links(self) -> list:
        """The addresses to open, one per network this machine is on."""
        from .link import PeerNode
        hosts = [a for a in PeerNode._local_addresses() if a != "127.0.0.1"] or ["127.0.0.1"]
        return [f"http://{host}:{self.port}/?t={self.code}" for host in hosts]

    def _remember(self, result) -> None:
        self.results.append({"ok": result.ok, "message": result.message, "at": time.strftime("%H:%M:%S")})
        del self.results[:-8]

    def state(self) -> dict:
        """Everything the page shows, as one JSON object."""
        eddy = self.eddy
        out = {"name": eddy.model.name, "capabilities": eddy.model.capabilities(), "busy": eddy._doing is not None,
               "results": list(self.results), "camera": eddy.adapter.camera_frame() is not None,
               "camera_note": eddy.adapter.camera_note}
        try:
            s = eddy.adapter.state()
            out.update(position=s.has_odometry, x=round(s.x, 3), y=round(s.y, 3),
                       heading=round(math.degrees(s.heading), 1), range_ahead=s.range_ahead)
        except Exception as exc:
            out["error"] = f"cannot read the robot: {exc}"
        return out

    def _stream(self, handler) -> None:
        """Send camera pictures one after another for as long as the browser stays connected."""
        handler.send_response(200)
        handler.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        handler.send_header("Cache-Control", "no-store")
        handler.send_header("Connection", "close")
        handler.end_headers()
        handler.close_connection = True
        last, sent = None, 0.0
        try:
            while self._running:
                frame = self.eddy.adapter.camera_frame()
                now = time.monotonic()
                # A new picture goes out at once; the same one is repeated every second so a
                # browser on a quiet camera does not give up.
                if frame is not None and (frame is not last or now - sent > 1.0):
                    handler.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                        + str(len(frame)).encode() + b"\r\n\r\n" + frame + b"\r\n")
                    handler.wfile.flush()
                    last, sent = frame, now
                time.sleep(0.04)
        except OSError:
            pass            # the browser went away


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Eddy</title>
<style>
  :root { --bg:#f5f5f2; --card:#fff; --ink:#1c1c1a; --dim:#6b6b66; --line:#deded8; --stop:#c62828; --go:#1f5fbf; --bad:#b3261e; }
  @media (prefers-color-scheme: dark) { :root { --bg:#161615; --card:#222220; --ink:#ecece8; --dim:#9a9a94; --line:#3a3a36; --go:#6ea2f0; --bad:#ff8a80; } }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--ink); font:16px/1.4 system-ui, sans-serif; }
  main { max-width:720px; margin:0 auto; padding:16px; display:grid; gap:12px; }
  h1 { font-size:20px; margin:0; } .dim { color:var(--dim); font-size:14px; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:12px; }
  #cam { width:100%; border-radius:8px; display:none; background:#000; }
  #stop { width:100%; padding:22px; font-size:26px; font-weight:700; letter-spacing:.04em; color:#fff; background:var(--stop);
          border:0; border-radius:12px; cursor:pointer; touch-action:manipulation; }
  #stop:active { filter:brightness(.85); }
  form { display:flex; gap:8px; } input { flex:1; min-width:0; padding:12px; font-size:16px; border:1px solid var(--line);
          border-radius:8px; background:var(--bg); color:var(--ink); }
  button.go { padding:12px 16px; font-size:16px; border:0; border-radius:8px; background:var(--go); color:#fff; cursor:pointer; }
  #quick { display:flex; flex-wrap:wrap; gap:8px; margin-top:8px; }
  #quick button { flex:1 1 40%; padding:12px; font-size:15px; border:1px solid var(--line); border-radius:8px;
          background:var(--bg); color:var(--ink); cursor:pointer; touch-action:manipulation; }
  ul { list-style:none; margin:0; padding:0; display:grid; gap:6px; } li { font-size:15px; overflow-wrap:anywhere; }
  li.bad { color:var(--bad); } li time { color:var(--dim); font-size:13px; margin-right:6px; }
  canvas { width:100%; height:180px; display:block; }
</style></head><body><main>
  <div><h1 id="name">Eddy</h1><div class="dim" id="can">Connecting...</div></div>
  <div class="card" id="gate" style="display:none">
    <p style="margin-top:0">Enter the code shown in the robot's terminal (the part of the link after <b>t=</b>).</p>
    <form id="codeform"><input id="code" autocomplete="off" autocapitalize="off" aria-label="Code"><button class="go">Open</button></form>
  </div>
  <div id="app" style="display:none; gap:12px">
    <button id="stop">STOP</button>
    <div class="card"><img id="cam" alt="The robot's camera"><div class="dim" id="camnote"></div></div>
    <div class="card">
      <form id="tell"><input id="say" placeholder="drive forward 50 cm" autocomplete="off" autocapitalize="off" aria-label="Command">
        <button class="go">Send</button></form>
      <div id="quick"></div>
    </div>
    <div class="card"><div id="where">&nbsp;</div><canvas id="map" aria-label="Path driven"></canvas></div>
    <div class="card"><ul id="results"><li class="dim">Nothing yet.</li></ul></div>
  </div>
</main><script>
let code = new URLSearchParams(location.search).get('t') || '';
const $ = id => document.getElementById(id), q = () => '?t=' + encodeURIComponent(code);
const trail = []; let camOn = false, quickMade = false;
function gate(show) { $('gate').style.display = show ? '' : 'none'; $('app').style.display = show ? 'none' : 'grid'; }
$('codeform').onsubmit = e => { e.preventDefault(); code = $('code').value.trim(); history.replaceState(null, '', '/' + q()); poll(); };
async function post(path, text) {
  try { const r = await fetch(path + q(), {method:'POST', body:text || ''}); if (r.status === 401) gate(true); }
  catch (e) { $('can').textContent = 'Lost contact with the robot.'; }
}
$('stop').onclick = () => post('/stop');
$('tell').onsubmit = e => { e.preventDefault(); const t = $('say').value.trim(); if (t) { post('/tell', t); $('say').value = ''; } };
function quick(caps) {
  if (quickMade) return; quickMade = true;
  const b = [];
  if (caps.includes('drive')) b.push('drive forward 20 cm', 'go back 20 cm', 'turn left 30 degrees', 'turn right 30 degrees');
  if (caps.includes('gripper')) b.push('open the gripper', 'close gripper');
  b.push('status');
  for (const t of b) { const el = document.createElement('button'); el.textContent = t; el.onclick = () => post('/tell', t); $('quick').append(el); }
}
function draw(s) {
  const c = $('map'), w = c.width = c.clientWidth * devicePixelRatio, h = c.height = c.clientHeight * devicePixelRatio, g = c.getContext('2d');
  if (!s.position) { c.style.display = 'none'; return; }
  const last = trail[trail.length - 1];
  if (!last || Math.hypot(last[0] - s.x, last[1] - s.y) > 0.01) trail.push([s.x, s.y]);
  if (trail.length > 2000) trail.shift();
  let x0 = s.x, x1 = s.x, y0 = s.y, y1 = s.y;
  for (const [x, y] of trail) { x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y); }
  const span = Math.max(x1 - x0, y1 - y0, 1) * 1.3, k = Math.min(w, h) / span, cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
  const px = (x, y) => [w / 2 - (y - cy) * k, h / 2 - (x - cx) * k];      // forward is up, left is left
  const ink = getComputedStyle(document.body).color;
  g.strokeStyle = ink; g.globalAlpha = .45; g.lineWidth = 2 * devicePixelRatio; g.beginPath();
  trail.forEach(([x, y], i) => { const [a, b] = px(x, y); i ? g.lineTo(a, b) : g.moveTo(a, b); }); g.stroke(); g.globalAlpha = 1;
  const [a, b] = px(s.x, s.y), r = 9 * devicePixelRatio, t = s.heading * Math.PI / 180;
  g.fillStyle = ink; g.beginPath();
  for (const [d, m] of [[0, 1.4], [2.5, 1], [-2.5, 1]]) g.lineTo(a - Math.sin(t + d) * r * m, b - Math.cos(t + d) * r * m);
  g.fill();
}
async function poll() {
  let s;
  try { const r = await fetch('/state' + q()); if (r.status === 401) { gate(true); $('can').textContent = 'This page needs the code.'; return; } s = await r.json(); }
  catch (e) { $('can').textContent = 'Lost contact with the robot.'; return; }
  gate(false);
  $('name').textContent = 'Eddy on ' + s.name; document.title = 'Eddy: ' + s.name;
  $('can').textContent = 'Can: ' + s.capabilities.join(', ') + (s.busy ? '. Moving now.' : '.');
  quick(s.capabilities);
  if (s.camera && !camOn) { camOn = true; $('cam').src = '/camera.mjpg' + q(); $('cam').style.display = 'block'; }
  $('camnote').textContent = s.camera ? '' : s.camera_note;
  $('where').textContent = s.error || (s.position ? `x ${s.x.toFixed(2)} m, y ${s.y.toFixed(2)} m, facing ${s.heading.toFixed(0)} degrees` : 'This robot reports no position.')
      + (s.range_ahead != null ? `. Nearest thing ahead: ${s.range_ahead.toFixed(2)} m.` : '');
  draw(s);
  if (s.results.length) $('results').replaceChildren(...s.results.slice().reverse().map(r => {
    const li = document.createElement('li'), t = document.createElement('time');
    t.textContent = r.at; li.append(t, r.message); if (!r.ok) li.className = 'bad'; return li; }));
}
poll(); setInterval(poll, 500);
</script></body></html>
"""
