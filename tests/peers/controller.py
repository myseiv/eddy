"""Run on a third machine: find two robots on the network and work with both. See run.sh."""
import sys

from eddy.link import PeerNode

node = PeerNode("controller", key="demo-key").start()
ok = node.wait_for("alpha", 15) and node.wait_for("bravo", 15)
print("discovered:", {n: (p["host"], p["caps"]) for n, p in node.peers.items()}, flush=True)
if not ok:
    sys.exit("FAIL: did not discover both robots")
results = [
    node.request("alpha", "drive forward 40 cm"),
    node.request("bravo", "drive forward 40 cm"),
    node.request("bravo", "reach 0.2 0.1 0.3"),
]
print("\n".join(results), flush=True)
a, b = node.send("alpha", "state"), node.send("bravo", "state")
print(f"alpha at x={a['x']:.2f}, bravo ({b['name']}) at x={b['x']:.2f}", flush=True)
stops = node.broadcast("stop")
print("stop all:", {k: (v if isinstance(v, Exception) else v["message"]) for k, v in stops.items()}, flush=True)
intruder = PeerNode("intruder", key="not-the-key").start(discovery=False)
intruder.peers = dict(node.peers)
refused = intruder.request("alpha", "drive forward 1 meter")
print(refused, flush=True)
good = (abs(a["x"] - 0.4) < 0.05 and abs(b["x"] - 0.4) < 0.06 and "Reached" in results[2] and b["name"] == "mockbot"
        and "refused" in refused and all(not isinstance(v, Exception) for v in stops.values()))
print("PASS: two robots on separate machines cooperated" if good else "FAIL")
sys.exit(0 if good else 1)
