"""Robot-to-robot: Eddy robots on the same network find each other and talk.

Discovery: every node announces itself about once a second on a multicast group
(no router setup, no central server). Messages: one JSON line over TCP, answered
with one JSON line.

Message types
    ping    are you there
    state   what are you, where are you
    tell    please carry out this sentence     (only obeyed when the robot was started with --obey)
    stop    stop now                           (always honoured from a trusted peer)
    event   something happened to me           (kept in the receiver's inbox)

Trust: if a key is set, every message is signed with HMAC-SHA256 and carries a
timestamp and a one-time nonce, so a message cannot be forged or replayed. A robot
never obeys `tell` without a key: an unauthenticated network must not move motors.
The key is a shared secret, so anyone holding it is fully trusted, and messages
are signed but not encrypted.
"""
from __future__ import annotations

import collections
import hashlib
import hmac
import json
import os
import socket
import socketserver
import struct
import threading
import time
import uuid
from typing import Callable, Dict, Optional

# The multicast address and UDP port every Eddy announces itself on. 239.255.x.x is the
# range reserved for use inside one local network.
GROUP, BEACON_PORT = "239.255.42.99", 42424
MAX_SKEW = 30.0        # seconds a signed message stays valid


class PeerError(Exception):
    """A peer could not be reached, or it refused the message."""
    pass


def _canonical(msg: dict) -> bytes:
    """The exact bytes that get signed: the message without its signature, keys sorted,
    no spaces. Sender and receiver must build identical bytes or the signature will not match."""
    return json.dumps({k: v for k, v in msg.items() if k != "sig"}, sort_keys=True, separators=(",", ":")).encode()


class PeerNode:
    """One robot's presence on the network: announces itself, finds others, sends and
    receives messages. It knows nothing about robots; attach() connects it to an Eddy."""

    def __init__(self, name: str, key: Optional[str] = None, port: int = 0, host: str = "0.0.0.0",
                 beacon_port: int = BEACON_PORT, interval: float = 1.0, capabilities=()):
        """name: what peers call this robot. key: shared secret for signing (None = unsigned).
        port: TCP port for messages (0 = let the system pick). interval: seconds between beacons."""
        self.name = name
        self.key = key.encode() if key else None
        self.id = uuid.uuid4().hex                          # unique per run, to recognise our own beacons
        self.interval = interval
        self.beacon_port = beacon_port
        self.capabilities = list(capabilities)
        self.peers: Dict[str, dict] = {}                    # name -> {host, port, caps, seen}
        self.inbox = collections.deque(maxlen=200)          # (sender, type, body)
        # Message type -> function that answers it. "ping" is built in; attach() adds the rest.
        self.handlers: Dict[str, Callable[[str, object], object]] = {"ping": lambda sender, body: "pong"}
        self._nonces = collections.OrderedDict()            # nonces already seen, oldest first
        self._lock = threading.Lock()
        self._running = False
        node = self     # the Handler class below needs a way back to this node

        class Handler(socketserver.StreamRequestHandler):
            """Handles one incoming TCP connection: read one line, answer with one line."""

            def handle(self):
                self.connection.settimeout(10)
                try:
                    line = self.rfile.readline(1 << 20)     # at most 1 MB
                    reply = node._receive(line)
                except Exception as exc:
                    reply = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                self.wfile.write((json.dumps(reply) + "\n").encode())

        class Server(socketserver.ThreadingTCPServer):
            """TCP server with one thread per connection, so a slow command does not block a stop."""
            allow_reuse_address = True
            daemon_threads = True

        self._server = Server((host, port), Handler)
        self.port = self._server.server_address[1]          # the real port, if 0 was asked for

    # ---- lifecycle -----------------------------------------------------------

    def start(self, discovery: bool = True) -> "PeerNode":
        """Start answering messages and, unless discovery=False, announcing and listening for peers."""
        self._running = True
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        if discovery:
            try:
                self._own_addresses = set(self._local_addresses())
                self._rx = self._open_beacon_socket()
                threading.Thread(target=self._listen_beacons, daemon=True).start()
                threading.Thread(target=self._send_beacons, daemon=True).start()
            except OSError as exc:      # no multicast on this network: peers can still be added by address
                print(f"[link] discovery unavailable ({exc}); add peers by address instead", flush=True)
        return self

    def stop(self) -> None:
        """Stop the server and the discovery threads, and release the sockets."""
        self._running = False
        self._server.shutdown()
        self._server.server_close()
        rx = getattr(self, "_rx", None)     # only exists if discovery started
        if rx is not None:
            rx.close()

    def on(self, kind: str, handler: Callable[[str, object], object]) -> None:
        """handler(sender_name, body) -> reply body. Raise PeerError to refuse."""
        self.handlers[kind] = handler

    def add_peer(self, name: str, host: str, port: int) -> None:
        """Register a peer by address, for networks where discovery does not work.
        Manual peers are never forgotten for being quiet."""
        self.peers[name] = {"host": host, "port": port, "caps": [], "seen": time.time(), "manual": True}

    # ---- discovery -----------------------------------------------------------

    @staticmethod
    def _local_addresses():
        """This machine's IPv4 addresses, loopback included (so robots on one machine find each other)."""
        found = ["127.0.0.1"]
        # (Resolving our own hostname would also work, but can stall for seconds on some systems.)
        try:                                    # the address used to reach the local network
            # "Connecting" a UDP socket sends nothing; it just makes the system choose which
            # of our addresses it would use, which we then read back.
            probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            probe.connect(("10.255.255.255", 1))
            found.append(probe.getsockname()[0])
            probe.close()
        except OSError:
            pass
        return list(dict.fromkeys(found))       # remove duplicates, keep order

    def _open_beacon_socket(self) -> socket.socket:
        """Open the UDP socket that hears other robots' announcements."""
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if hasattr(socket, "SO_REUSEPORT"):
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)     # several robots on one machine
        s.bind(("", self.beacon_port))
        # Join the multicast group on each interface; one that fails (e.g. down) is skipped.
        joined = 0
        for address in self._local_addresses() + ["0.0.0.0"]:          # listen on every interface
            try:
                s.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP,
                             struct.pack("4s4s", socket.inet_aton(GROUP), socket.inet_aton(address)))
                joined += 1
            except OSError:
                pass
        if not joined:
            s.close()
            raise OSError("could not join the discovery group on any network interface")
        s.settimeout(0.5)       # wake up twice a second so the listener can notice a stop
        return s

    def _send_beacons(self) -> None:
        """Thread: announce this robot on every network interface until stopped."""
        # One sending socket per interface, each pinned to that interface.
        senders = []
        for address in self._local_addresses():                        # announce on every interface
            tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
            try:
                tx.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 1)    # stay on the local network
                tx.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_LOOP, 1)
                tx.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF, socket.inet_aton(address))
                senders.append(tx)
            except OSError:
                tx.close()
        while self._running:
            # The beacon says who we are and which TCP port to send messages to.
            beacon = json.dumps({"eddy": 1, "id": self.id, "name": self.name, "port": self.port,
                                 "caps": self.capabilities}).encode()
            for tx in senders:
                try:
                    tx.sendto(beacon, (GROUP, self.beacon_port))
                except OSError:
                    pass                    # that interface is down for a moment: keep trying
            time.sleep(self.interval)
        for tx in senders:
            tx.close()

    def _listen_beacons(self) -> None:
        """Thread: record every robot whose announcement we hear."""
        while self._running:
            try:
                data, (host, _) = self._rx.recvfrom(4096)
            except socket.timeout:
                self._forget_stale()        # quiet moment: drop peers that have gone silent
                continue
            except OSError:
                return                      # the socket was closed by stop()
            try:
                b = json.loads(data)
                # Ignore anything that is not an Eddy beacon, and our own beacons.
                if b.get("eddy") != 1 or b.get("id") == self.id:
                    continue
                if host in self._own_addresses:
                    host = "127.0.0.1"      # a robot on this same machine: always reachable over loopback
                self.peers[str(b["name"])] = {"host": host, "port": int(b["port"]), "caps": list(b.get("caps", [])),
                                              "seen": time.time(), "id": b.get("id")}
            except (ValueError, KeyError, TypeError):
                continue                    # not ours, or malformed: ignore

    def _forget_stale(self) -> None:
        """Drop discovered peers not heard from for a while (10 beacons, at least 10 s)."""
        cutoff = time.time() - max(10.0, self.interval * 10)
        for name in [n for n, p in self.peers.items() if p["seen"] < cutoff and not p.get("manual")]:
            self.peers.pop(name, None)

    def wait_for(self, name: str, timeout: float = 5.0) -> bool:
        """Wait until a peer with this name has been discovered. False if it never shows up."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if name in self.peers:
                return True
            time.sleep(0.05)
        return False

    # ---- messages ------------------------------------------------------------

    def _sign(self, msg: dict) -> dict:
        """Stamp an outgoing message with sender, time and a random nonce, and sign it if we have a key."""
        msg.update({"from": self.name, "ts": time.time(), "nonce": os.urandom(8).hex()})
        if self.key:
            msg["sig"] = hmac.new(self.key, _canonical(msg), hashlib.sha256).hexdigest()
        return msg

    def _verify(self, msg: dict) -> bool:
        """True if the message is proven to come from someone holding our key, and is fresh."""
        if not self.key or "sig" not in msg:
            return False
        # 1. The signature must match (compare_digest avoids leaking how many characters matched).
        expected = hmac.new(self.key, _canonical(msg), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected, str(msg["sig"])):
            return False
        # 2. It must be recent.
        if abs(time.time() - float(msg.get("ts", 0))) > MAX_SKEW:
            return False
        # 3. Its nonce must not have been used before.
        with self._lock:
            nonce = str(msg.get("nonce"))
            if nonce in self._nonces:
                return False                # a replay of a message we already acted on
            self._nonces[nonce] = time.time()
            while len(self._nonces) > 5000:             # keep memory bounded: forget the oldest
                self._nonces.popitem(last=False)
        return True

    def _receive(self, line: bytes) -> dict:
        """Handle one incoming message line and build the reply."""
        msg = json.loads(line)
        if not isinstance(msg, dict) or "type" not in msg:
            return {"ok": False, "error": "not an Eddy message"}
        trusted = self._verify(msg)
        # A robot that has a key accepts nothing unsigned or wrongly signed.
        if self.key and not trusted:
            return {"ok": False, "error": "refused: bad or missing signature (do both robots use the same --key?)"}
        kind, sender, body = str(msg["type"]), str(msg.get("from", "?")), msg.get("body")
        # A robot without a key still answers harmless questions, but never moves for a stranger.
        if kind in ("tell", "stop") and not trusted:
            return {"ok": False, "error": "refused: this robot only takes commands from peers that share a --key"}
        handler = self.handlers.get(kind)
        if handler is None:
            # No handler for this type: keep it for whoever reads the inbox.
            self.inbox.append((sender, kind, body))
            return {"ok": True, "body": None}
        try:
            return {"ok": True, "body": handler(sender, body)}
        except PeerError as exc:
            return {"ok": False, "error": str(exc)}

    def send(self, peer: str, kind: str, body=None, timeout: float = 60.0):
        """Send one message to a named peer and return its reply body. Raises PeerError if refused."""
        target = self.peers.get(peer)
        if target is None:
            raise PeerError(f"no peer called '{peer}' (known: {', '.join(sorted(self.peers)) or 'none'})")
        raw = (json.dumps(self._sign({"type": kind, "body": body})) + "\n").encode()
        try:
            # 5 s to connect; then up to `timeout` for the answer, since a command may take a while.
            with socket.create_connection((target["host"], target["port"]), timeout=5) as sock:
                sock.settimeout(timeout)
                sock.sendall(raw)
                reply = json.loads(sock.makefile("rb").readline() or b"{}")
        except (OSError, ValueError) as exc:
            raise PeerError(f"could not reach {peer}: {exc}") from exc
        if not reply.get("ok"):
            raise PeerError(reply.get("error", "no answer"))
        return reply.get("body")

    def request(self, peer: str, sentence: str) -> str:
        """Ask another robot to carry out a sentence; returns what it answered."""
        try:
            body = self.send(peer, "tell", sentence)
            return f"{peer}: {body.get('message')}"
        except PeerError as exc:
            return f"{peer}: {exc}"

    def broadcast(self, kind: str, body=None) -> Dict[str, object]:
        """Send to every known peer. Returns {peer: reply or the PeerError}."""
        out = {}
        for peer in list(self.peers):
            try:
                out[peer] = self.send(peer, kind, body, timeout=10.0)
            except PeerError as exc:
                out[peer] = exc
        return out


def attach(node: PeerNode, eddy, obey: bool = False) -> None:
    """Connect a peer node to an Eddy: answer state queries, honour stop, optionally obey commands."""
    node.capabilities = eddy.model.capabilities()

    def on_state(sender, body):
        """Answer "what are you and where are you"."""
        s = eddy.adapter.state()
        return {"name": eddy.model.name, "capabilities": eddy.model.capabilities(), "x": s.x, "y": s.y,
                "heading": s.heading, "range_ahead": s.range_ahead, "joints": s.joints}

    def on_tell(sender, body):
        """Carry out a sentence from a peer, but only if this robot was started with --obey."""
        if not obey:
            raise PeerError("this robot is not accepting commands from peers (start it with --obey)")
        result = eddy.tell(str(body))
        return {"ok": result.ok, "message": result.message}

    def on_stop(sender, body):
        """Stop on a trusted peer's request, whether or not --obey was given."""
        return {"message": eddy.stop().message}

    def on_event(sender, body):
        """Remember what another robot reported."""
        node.inbox.append((sender, "event", body))

    node.on("state", on_state)
    node.on("tell", on_tell)
    node.on("stop", on_stop)
    node.on("event", on_event)

    def announce(result) -> None:           # tell the others when something notable happens to me
        """Called after every command this robot finishes. If it met an obstacle, tell every
        peer where, in a background thread so the robot is not held up by the network."""
        if "in the way" in result.message:
            s = eddy.adapter.state()
            body = {"what": "obstacle", "x": s.x, "y": s.y, "heading": s.heading}
            threading.Thread(target=node.broadcast, args=("event", body), daemon=True).start()
    eddy.listeners.append(announce)


def selftest(check) -> None:
    """Two simulated robots on this machine find each other and cooperate. Used by `eddy selftest`.

    check(name, passed, detail="") records one result.
    """
    from .adapters.sim import SimAdapter
    from .brain import Eddy
    from .senses.speak import Speaker

    port = 42000 + os.getpid() % 2000           # private beacon port so parallel runs do not mix
    made = []                                   # every node created, so all are stopped at the end

    def robot(name, key, obey):
        """Create a simulated robot with its own Eddy and network node."""
        sim = SimAdapter(name=name)
        eddy = Eddy(sim, Speaker("silent"))
        eddy.start()
        node = PeerNode(name, key=key, host="127.0.0.1", beacon_port=port, interval=0.2).start()
        attach(node, eddy, obey=obey)
        made.append(node)
        return sim, eddy, node

    try:
        # Discovery: two robots with the same key should find each other by themselves.
        sim_a, eddy_a, a = robot("alpha", "s3cret", obey=True)
        sim_b, eddy_b, b = robot("bravo", "s3cret", obey=True)
        found = a.wait_for("bravo", 5) and b.wait_for("alpha", 5)
        check("peers: two robots discover each other", found, f"alpha sees {sorted(a.peers)}, bravo sees {sorted(b.peers)}")
        if not found:                              # multicast blocked here: fall back to addresses
            a.add_peer("bravo", "127.0.0.1", b.port)
            b.add_peer("alpha", "127.0.0.1", a.port)

        # Cooperation: read state, command each other in both directions.
        state = a.send("bravo", "state")
        check("peers: one robot reads another's state", state["name"] == "bravo" and "drive" in state["capabilities"])
        answer = a.request("bravo", "drive forward 30 cm")
        check("peers: one robot commands another", abs(sim_b.x - 0.3) < 0.03 and sim_a.x == 0.0, answer)
        answer = b.request("alpha", "reach 0.2 0.1 0.3")
        check("peers: and the other way round", "Reached" in answer, answer)

        # Events: bravo drives into a wall; alpha should receive an obstacle event.
        eddy_b.tell("turn left 90 degrees")
        eddy_b.tell("drive forward 5 meters")        # bravo meets a wall and tells alpha about it
        deadline = time.time() + 3
        while time.time() < deadline and not any(k == "event" for _, k, _ in a.inbox):
            time.sleep(0.05)
        events = [body for sender, kind, body in a.inbox if kind == "event" and sender == "bravo"]
        check("peers: a robot warns the others about an obstacle", bool(events) and events[0]["what"] == "obstacle",
              str(events[:1]))

        # Trust
        # A robot with a different key must not be able to move bravo.
        _, _, stranger = robot("mallory", "wrong-key", obey=True)
        stranger.add_peer("bravo", "127.0.0.1", b.port)
        before = sim_b.x
        answer = stranger.request("bravo", "drive forward 50 cm")
        check("peers: a robot with the wrong key is refused", "refused" in answer and sim_b.x == before, answer)

        # Nor may a robot with no key at all.
        _, _, nokey = robot("anon", None, obey=True)
        nokey.add_peer("bravo", "127.0.0.1", b.port)
        check("peers: an unsigned command is refused", "refused" in nokey.request("bravo", "turn left"))

        # A robot with the right key but without --obey answers and stops, but does not move.
        sim_c, _, careful = robot("charlie", "s3cret", obey=False)
        a.add_peer("charlie", "127.0.0.1", careful.port)
        answer = a.request("charlie", "drive forward 50 cm")
        check("peers: a robot not started with --obey does not move", "not accepting" in answer and sim_c.x == 0.0, answer)
        check("peers: but it still honours a stop", "Stopped" in a.send("charlie", "stop")["message"])

        # Replay and tampering: a correctly signed message works once; a second copy, or a
        # copy with its contents changed, is refused.
        captured = a._sign({"type": "tell", "body": "drive forward 20 cm"})
        line = (json.dumps(captured) + "\n").encode()
        first = b._receive(line)
        second = b._receive(line)
        check("peers: a recorded command cannot be replayed", first["ok"] and not second["ok"], second.get("error", ""))
        tampered = dict(captured, body="drive forward 5 meters", nonce="ffff")
        check("peers: a tampered command is refused", not b._receive((json.dumps(tampered) + "\n").encode())["ok"])
        check("peers: garbage does not crash a node", not b._receive(b'{"nope": 1}\n')["ok"] and a.send("bravo", "ping") == "pong")
    finally:
        for node in made:
            node.stop()
