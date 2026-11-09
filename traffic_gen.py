#!/usr/bin/env python3
"""
traffic_gen.py
Workload Generator — Tasks 2.1 (mice), 2.2 (elephants + collision), 2.4 (testing).

Mice   : Poisson arrivals (exponential inter-arrival times) of 8-64 KB TCP/UDP
         request bursts between random host pairs. FCT = connect -> app-level ACK.
         -> results/mice_results.csv
Elephant: long-lived iperf (iperf2) TCP transfers with `-i 0.5 -y C` interval logging.
         -> results/elephant_results.csv
Collision: H1->H9 and H2->H10 started together. Both sources sit on e_0_0 and both
         destinations on e_2_0, so they compete for the same uplinks/core paths.
Core load: OVS port counters of the 4 core switches before/after each run.
         -> results/core_load.csv

Usage
  # full experiment: starts the k=4 fat-tree itself (controller must be running)
  sudo python3 traffic_gen.py run --scenario collision --scheme ecmp --duration 30
  sudo python3 traffic_gen.py run --scenario mixed     --scheme proposed

  # Task 2.4 smoke tests
  sudo python3 traffic_gen.py run --topo two --scenario mixed --scheme test --duration 10
  python3 traffic_gen.py local --duration 5            # loopback, no Mininet needed

  # internal helpers (started on Mininet hosts by `run`)
  python3 traffic_gen.py server --port 6000
  python3 traffic_gen.py mice-client --schedule s.json --out part.csv

Host aliases: hosts sorted as h_0_0_0, h_0_0_1, ... are H1..H16 (H1=10.0.0.2, H9=10.2.0.2).
Run the same --seed for static / ecmp / proposed so every scheme sees identical traffic.
"""

import argparse
import csv
import json
import os
import random
import re
import shutil
import socket
import struct
import subprocess
import sys
import threading
import time
from collections import defaultdict, namedtuple

try:
    import config
except ImportError:
    config = None


def _cfg(name, default):
    return getattr(config, name, default) if config is not None else default


SCRIPT_PATH = os.path.abspath(__file__)
MICE_PORT = _cfg("MICE_PORT", 6000)
IPERF_PORT = _cfg("IPERF_PORT", 5001)
RESULTS_DIR = _cfg("RESULTS_DIR", "results")
MICE_MIN_BYTES = 8 * 1024
MICE_MAX_BYTES = 64 * 1024
UDP_CHUNK = 1200
DEFAULT_COLLISION_PAIRS = [(1, 9), (2, 10)]

_HDR = struct.Struct("!Q")  # TCP request header: payload size
_UDP_HDR = struct.Struct("!IIQ")  # UDP datagram header: tag, seq, total size
_ACK = b"\x06"

MICE_FIELDS = [
    "scheme",
    "run_id",
    "flow_id",
    "src",
    "dst",
    "src_ip",
    "dst_ip",
    "proto",
    "size_bytes",
    "start_epoch",
    "start_offset_s",
    "fct_ms",
    "status",
]
ELEPHANT_FIELDS = [
    "scheme",
    "run_id",
    "flow_id",
    "src",
    "dst",
    "src_ip",
    "dst_ip",
    "interval_start",
    "interval_end",
    "time_s",
    "epoch",
    "bytes",
    "throughput_mbps",
]
CORE_FIELDS = [
    "scheme",
    "run_id",
    "switch",
    "duration_s",
    "tx_bytes",
    "rx_bytes",
    "avg_tx_mbps",
    "avg_rx_mbps",
]

HostInfo = namedtuple("HostInfo", ["name", "ip", "alias"])


# =========================================================== CSV helpers
def write_rows(path, fields, rows, append=True):
    """Write rows to a CSV; appends (without a second header) when the file exists."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    new = not (append and os.path.exists(path) and os.path.getsize(path) > 0)
    with open(path, "w" if new else "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        if new:
            writer.writeheader()
        writer.writerows(rows)
    return path


def read_rows(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


# ====================================================== mice: server side
def _recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("peer closed early")
        buf.extend(chunk)
    return bytes(buf)


def _serve_tcp_conn(conn):
    with conn:
        try:
            conn.settimeout(30)
            (size,) = _HDR.unpack(_recv_exact(conn, _HDR.size))
            remaining = size
            while remaining > 0:
                chunk = conn.recv(min(65536, remaining))
                if not chunk:
                    return
                remaining -= len(chunk)
            conn.sendall(_ACK)
        except OSError:
            pass


def run_server(port=MICE_PORT, bind="0.0.0.0", stop_event=None, ready_event=None):
    """Mice sink: TCP and UDP on the same port; replies with a 1-byte ACK per request."""
    stop_event = stop_event or threading.Event()
    tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    tcp.bind((bind, port))
    tcp.listen(256)
    tcp.settimeout(0.5)
    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    udp.bind((bind, port))
    udp.settimeout(0.5)

    def udp_loop():
        pending = {}
        while not stop_event.is_set():
            try:
                data, addr = udp.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                break
            if len(data) < _UDP_HDR.size:
                continue
            tag, _seq, total = _UDP_HDR.unpack_from(data)
            key = (addr, tag)
            got = pending.get(key, 0) + len(data) - _UDP_HDR.size
            if got >= total:
                pending.pop(key, None)
                udp.sendto(_ACK + struct.pack("!I", tag), addr)
            else:
                pending[key] = got
            if len(pending) > 10000:  # drop state of lost flows
                pending.clear()

    udp_thread = threading.Thread(target=udp_loop, daemon=True)
    udp_thread.start()
    if ready_event is not None:
        ready_event.set()
    try:
        while not stop_event.is_set():
            try:
                conn, _ = tcp.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            threading.Thread(target=_serve_tcp_conn, args=(conn,), daemon=True).start()
    finally:
        stop_event.set()
        tcp.close()
        udp.close()
        udp_thread.join(1.0)


# ====================================================== mice: client side
def send_mice_flow(dst_ip, size, port=MICE_PORT, proto="tcp", timeout=10.0):
    """
    Send one mice request of `size` bytes and wait for the server's ACK.
    Returns (fct_ms, status) with status in {'ok', 'timeout', 'error', 'bad_ack'}.
    """
    start = time.perf_counter()
    try:
        if proto == "tcp":
            with socket.create_connection((dst_ip, port), timeout=timeout) as s:
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                s.sendall(_HDR.pack(size) + bytes(size))
                ack = _recv_exact(s, 1)
        elif proto == "udp":
            tag = random.getrandbits(32)
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                sent, seq = 0, 0
                while sent < size:
                    n = min(UDP_CHUNK, size - sent)
                    s.sendto(_UDP_HDR.pack(tag, seq, size) + bytes(n), (dst_ip, port))
                    sent, seq = sent + n, seq + 1
                deadline = start + timeout
                while True:
                    s.settimeout(max(0.01, deadline - time.perf_counter()))
                    data, _ = s.recvfrom(64)
                    if (
                        len(data) >= 5
                        and data[:1] == _ACK
                        and struct.unpack("!I", data[1:5])[0] == tag
                    ):
                        ack = _ACK
                        break
        else:
            raise ValueError("proto must be tcp or udp")
        fct = (time.perf_counter() - start) * 1000.0
        return fct, ("ok" if ack == _ACK else "bad_ack")
    except socket.timeout:
        return (time.perf_counter() - start) * 1000.0, "timeout"
    except OSError:
        return (time.perf_counter() - start) * 1000.0, "error"


def generate_mice_schedule(
    hosts,
    duration,
    rate,
    min_bytes=MICE_MIN_BYTES,
    max_bytes=MICE_MAX_BYTES,
    seed=None,
    proto="tcp",
):
    """
    Poisson arrival process: aggregate `rate` flows/s, exponential inter-arrival times,
    uniformly random (src != dst) host pairs and sizes in [min_bytes, max_bytes].
    hosts: list of HostInfo.
    """
    if len(hosts) < 2:
        raise ValueError("need at least two hosts")
    if rate <= 0:
        raise ValueError("rate must be > 0")
    rng = random.Random(seed)
    flows = []
    t = rng.expovariate(rate)
    while t < duration:
        src, dst = rng.sample(hosts, 2)
        flows.append(
            {
                "flow_id": "M%05d" % len(flows),
                "src": src.alias,
                "dst": dst.alias,
                "src_name": src.name,
                "src_ip": src.ip,
                "dst_ip": dst.ip,
                "size_bytes": rng.randint(min_bytes, max_bytes),
                "start_offset_s": round(t, 6),
                "proto": proto,
            }
        )
        t += rng.expovariate(rate)
    return flows


def run_mice_flows(flows, t0, scheme, run_id, port=MICE_PORT, timeout=10.0):
    """Fire each flow at t0 + start_offset_s (own thread, so bursts overlap) and log FCTs."""
    rows, lock, threads = [], threading.Lock(), []

    def worker(f):
        start_epoch = time.time()
        fct, status = send_mice_flow(
            f["dst_ip"], f["size_bytes"], port, f["proto"], timeout
        )
        row = {
            "scheme": scheme,
            "run_id": run_id,
            "flow_id": f["flow_id"],
            "src": f["src"],
            "dst": f["dst"],
            "src_ip": f["src_ip"],
            "dst_ip": f["dst_ip"],
            "proto": f["proto"],
            "size_bytes": f["size_bytes"],
            "start_epoch": "%.6f" % start_epoch,
            "start_offset_s": "%.6f" % (start_epoch - t0),
            "fct_ms": "%.3f" % fct,
            "status": status,
        }
        with lock:
            rows.append(row)

    for f in sorted(flows, key=lambda x: x["start_offset_s"]):
        delay = t0 + f["start_offset_s"] - time.time()
        if delay > 0:
            time.sleep(delay)
        th = threading.Thread(target=worker, args=(f,), daemon=True)
        th.start()
        threads.append(th)
    for th in threads:
        th.join(timeout + 5)
    rows.sort(key=lambda r: float(r["start_offset_s"]))
    return rows


# ============================================================ elephants
def iperf_client_cmd(dst_ip, port=IPERF_PORT, duration=30, interval=0.5):
    return [
        "iperf",
        "-c",
        dst_ip,
        "-p",
        str(port),
        "-t",
        str(duration),
        "-i",
        str(interval),
        "-y",
        "C",
    ]


def parse_iperf_csv(text, interval=0.5):
    """
    Parse iperf2 `-y C` output:
      timestamp,src_ip,src_port,dst_ip,dst_port,id,start-end,bytes,bits_per_second
    The final whole-run summary line is dropped.
    """
    samples = []
    for line in text.splitlines():
        parts = line.strip().split(",")
        if len(parts) < 9:
            continue
        try:
            a, b = (float(x) for x in parts[6].split("-"))
            nbytes = int(float(parts[7]))
            bps = float(parts[8])
        except ValueError:
            continue
        if b - a > interval * 1.5:
            continue  # summary line (0.0-30.0)
        samples.append(
            {
                "interval_start": a,
                "interval_end": b,
                "bytes": nbytes,
                "throughput_mbps": bps / 1e6,
            }
        )
    return samples


def elephant_rows(samples, spec, launch_epoch, t0, scheme, run_id):
    rows = []
    for s in samples:
        rows.append(
            {
                "scheme": scheme,
                "run_id": run_id,
                "flow_id": spec["flow_id"],
                "src": spec["src"],
                "dst": spec["dst"],
                "src_ip": spec["src_ip"],
                "dst_ip": spec["dst_ip"],
                "interval_start": s["interval_start"],
                "interval_end": s["interval_end"],
                "time_s": "%.3f" % (launch_epoch - t0 + s["interval_end"]),
                "epoch": "%.3f" % (launch_epoch + s["interval_end"]),
                "bytes": s["bytes"],
                "throughput_mbps": "%.4f" % s["throughput_mbps"],
            }
        )
    return rows


def start_elephant(src_node, dst_node, spec, port=IPERF_PORT, interval=0.5):
    """Launch one long-lived iperf TCP transfer. The RECEIVER logs the throughput."""
    server = dst_node.popen(
        ["iperf", "-s", "-p", str(port), "-y", "C", "-i", str(interval)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    time.sleep(0.3)  # give the server time to start listening
    cmd = iperf_client_cmd(spec["dst_ip"], port, spec["duration"], interval)
    proc = src_node.popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return {"proc": proc, "server": server, "launch_epoch": time.time(), "spec": spec}


def collect_elephant(handle, t0, scheme, run_id, interval=0.5):
    proc, spec = handle["proc"], handle["spec"]
    try:
        out, err = proc.communicate(timeout=spec["duration"] + 20)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, err = proc.communicate()
    # use the receiver's report (what really arrived), not the sender's
    time.sleep(1.0)
    handle["server"].terminate()
    out, err = handle["server"].communicate()
    text = out.decode(errors="replace") if isinstance(out, bytes) else (out or "")
    samples = parse_iperf_csv(text, interval)
    if not samples:
        msg = err.decode(errors="replace") if isinstance(err, bytes) else err
        print(
            "  [!] no iperf samples for %s: %s"
            % (spec["flow_id"], (msg or "").strip()[:200])
        )
    return elephant_rows(samples, spec, handle["launch_epoch"], t0, scheme, run_id)


def _elephant_spec(src, dst, offset, duration):
    return {
        "flow_id": "%s->%s" % (src.alias, dst.alias),
        "src": src.alias,
        "dst": dst.alias,
        "src_name": src.name,
        "dst_name": dst.name,
        "src_ip": src.ip,
        "dst_ip": dst.ip,
        "offset": round(offset, 3),
        "duration": max(1.0, round(duration, 1)),
    }


def collision_specs(hosts, duration, pairs=None, stagger=0.0):
    """
    Collision trigger: by default H1->H9 and H2->H10 start together (stagger=0).
    Use stagger>0 to start the 2nd flow later, so the plot shows the moment of collapse.
    """
    pairs = pairs or DEFAULT_COLLISION_PAIRS
    if max(max(p) for p in pairs) > len(hosts):
        print(
            "  [!] collision pairs %s need %d hosts; using H1->H2"
            % (pairs, max(max(p) for p in pairs))
        )
        pairs = [(1, 2)]
    specs = []
    for k, (s, d) in enumerate(pairs):
        offset = k * stagger
        specs.append(
            _elephant_spec(hosts[s - 1], hosts[d - 1], offset, duration - offset)
        )
    return specs


def _pod(ip):
    parts = ip.split(".")
    return parts[1] if len(parts) == 4 else None


def random_elephant_specs(hosts, count, duration, seed=None):
    """`count` long-lived inter-pod transfers from distinct sources, staggered in the first third."""
    rng = random.Random(None if seed is None else seed + 1)
    specs = []
    for src in rng.sample(hosts, min(count, len(hosts))):
        others = [h for h in hosts if h.name != src.name]
        inter_pod = [h for h in others if _pod(h.ip) != _pod(src.ip)]
        dst = rng.choice(inter_pod or others)
        offset = rng.uniform(0, duration / 3.0)
        specs.append(_elephant_spec(src, dst, offset, duration - offset - 1))
    return specs


def launch_elephants(specs, t0, nodes, interval=0.5):
    """Start each spec at t0 + offset from a background thread. Returns (thread, handles)."""
    handles = [None] * len(specs)

    def launcher():
        for i, spec in sorted(enumerate(specs), key=lambda x: x[1]["offset"]):
            delay = t0 + spec["offset"] - time.time()
            if delay > 0:
                time.sleep(delay)
            handles[i] = start_elephant(
                nodes[spec["src_name"]],
                nodes[spec["dst_name"]],
                spec,
                IPERF_PORT + 1 + i,
                interval,
            )
            print(
                "  elephant %-10s started (%s -> %s, %.0fs)"
                % (spec["flow_id"], spec["src_ip"], spec["dst_ip"], spec["duration"])
            )

    th = threading.Thread(target=launcher, daemon=True)
    th.start()
    return th, handles


# ============================================================ core load
_PORT_RE = re.compile(r'port\s+"?([^:"]+?)"?:\s*rx pkts=\d+,\s*bytes=(\d+)')
_TX_RE = re.compile(r"tx pkts=\d+,\s*bytes=(\d+)")


def parse_dump_ports(text):
    """Parse `ovs-ofctl dump-ports` output -> {port: {'rx_bytes', 'tx_bytes'}} (LOCAL dropped)."""
    ports, current = {}, None
    for line in text.splitlines():
        m = _PORT_RE.search(line)
        if m:
            current = m.group(1).strip()
            ports[current] = {"rx_bytes": int(m.group(2)), "tx_bytes": 0}
            continue
        m = _TX_RE.search(line)
        if m and current is not None:
            ports[current]["tx_bytes"] = int(m.group(1))
    ports.pop("LOCAL", None)
    return ports


def snapshot_ports(switch_names):
    snap = {}
    for sw in switch_names:
        try:
            out = subprocess.run(
                ["ovs-ofctl", "-O", "OpenFlow13", "dump-ports", sw],
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout
        except (OSError, subprocess.SubprocessError):
            out = ""
        snap[sw] = parse_dump_ports(out)
    return snap


def core_load_rows(before, after, duration, scheme, run_id):
    rows = []
    for sw in sorted(after):
        tx = rx = 0
        for port, cnt in after[sw].items():
            prev = before.get(sw, {}).get(port, {"tx_bytes": 0, "rx_bytes": 0})
            tx += max(0, cnt["tx_bytes"] - prev["tx_bytes"])
            rx += max(0, cnt["rx_bytes"] - prev["rx_bytes"])
        rows.append(
            {
                "scheme": scheme,
                "run_id": run_id,
                "switch": sw,
                "duration_s": "%.2f" % duration,
                "tx_bytes": tx,
                "rx_bytes": rx,
                "avg_tx_mbps": "%.4f" % (tx * 8 / duration / 1e6),
                "avg_rx_mbps": "%.4f" % (rx * 8 / duration / 1e6),
            }
        )
    return rows


# ============================================================ Mininet glue
def _natural_key(name):
    return (tuple(int(x) for x in re.findall(r"\d+", name)), name)


def mininet_hosts(net):
    nodes = sorted(net.hosts, key=lambda h: _natural_key(h.name))
    hosts = [HostInfo(h.name, h.IP(), "H%d" % (i + 1)) for i, h in enumerate(nodes)]
    return hosts, {h.name: h for h in nodes}


def build_net(topo="fattree"):
    from mininet.link import TCLink
    from mininet.log import setLogLevel
    from mininet.net import Mininet
    from mininet.node import OVSSwitch, RemoteController

    setLogLevel("info")
    bw = _cfg("LINK_BANDWIDTH", 10.0)
    if topo == "fattree":
        from fattree_topo import FatTreeTopo, OpenFlow13Switch

        net = Mininet(
            topo=FatTreeTopo(k=_cfg("K_FAT_TREE", 4)),
            switch=OpenFlow13Switch,
            controller=RemoteController(
                "c0",
                ip=_cfg("CONTROLLER_IP", "127.0.0.1"),
                port=_cfg("CONTROLLER_PORT", 6653),
            ),
            link=TCLink,
            autoSetMacs=False,
            autoStaticArp=False,
        )
    elif topo == "two":
        # 2-host testbed; standalone OVS = learning switch, no controller needed
        from functools import partial
        from mininet.topo import SingleSwitchTopo

        net = Mininet(
            topo=SingleSwitchTopo(k=2, lopts={"bw": bw}),
            switch=partial(OVSSwitch, failMode="standalone"),
            controller=None,
            link=TCLink,
            autoSetMacs=True,
        )
    else:
        raise ValueError("unknown topo %r" % topo)
    net.start()
    return net


def run_experiment(
    net,
    scenario="mixed",
    scheme="proposed",
    duration=30.0,
    mice_rate=20.0,
    mice_proto="tcp",
    mice_min=MICE_MIN_BYTES,
    mice_max=MICE_MAX_BYTES,
    mice_timeout=10.0,
    n_elephants=4,
    collision_pairs=None,
    stagger=0.0,
    interval=0.5,
    seed=42,
    results_dir=RESULTS_DIR,
    append=True,
    warmup=True,
    with_mice=None,
):
    """
    Run one benchmark on an already-started Mininet `net`.
    scenario: 'mice' | 'elephant' | 'collision' | 'mixed'
    """
    run_id = time.strftime("%Y%m%d-%H%M%S")
    hosts, nodes = mininet_hosts(net)
    tmp = os.path.join(results_dir, "tmp", run_id)
    os.makedirs(tmp, exist_ok=True)
    if with_mice is None:
        with_mice = scenario != "elephant"
    print(
        "\n=== run %s | scenario=%s scheme=%s duration=%.0fs hosts=%d ==="
        % (run_id, scenario, scheme, duration, len(hosts))
    )

    if warmup:
        print("*** warm-up pingAll (ARP / initial rules)")
        loss = net.pingAll(timeout="1")
        if loss > 0:
            print(
                "  [!] %.0f%% ping loss — check the controller before trusting results"
                % loss
            )

    if scenario == "collision":
        specs = collision_specs(hosts, duration, collision_pairs, stagger)
    elif scenario in ("elephant", "mixed"):
        specs = random_elephant_specs(hosts, n_elephants, duration, seed)
    else:
        specs = []

    servers = []
    for node in nodes.values():
        if with_mice:
            servers.append(
                node.popen(
                    [sys.executable, SCRIPT_PATH, "server", "--port", str(MICE_PORT)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            )
    time.sleep(1.0)

    core_names = [s.name for s in net.switches if s.name.startswith("c_")] or [
        s.name for s in net.switches
    ]
    before, before_ts = snapshot_ports(core_names), time.time()
    t0 = time.time() + 2.0

    mice_procs = []
    if with_mice:
        flows = generate_mice_schedule(
            hosts, duration, mice_rate, mice_min, mice_max, seed, mice_proto
        )
        print(
            "*** %d mice flows scheduled (%.1f flows/s, %s)"
            % (len(flows), mice_rate, mice_proto)
        )
        by_src = defaultdict(list)
        for f in flows:
            by_src[f["src_name"]].append(f)
        for src_name, fl in by_src.items():
            sched = os.path.join(tmp, "mice_%s.json" % src_name)
            part = os.path.join(tmp, "mice_%s.csv" % src_name)
            with open(sched, "w") as fh:
                json.dump(
                    {
                        "t0": t0,
                        "scheme": scheme,
                        "run_id": run_id,
                        "port": MICE_PORT,
                        "timeout": mice_timeout,
                        "flows": fl,
                    },
                    fh,
                )
            log = open(os.path.join(tmp, "mice_%s.log" % src_name), "w")
            proc = nodes[src_name].popen(
                [
                    sys.executable,
                    SCRIPT_PATH,
                    "mice-client",
                    "--schedule",
                    sched,
                    "--out",
                    part,
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            mice_procs.append((proc, part, log))

    launcher, handles = launch_elephants(specs, t0, nodes, interval)

    end = t0 + duration
    while time.time() < end:
        time.sleep(min(5.0, max(0.0, end - time.time())))
        print("  ... %.0f / %.0f s" % (min(duration, time.time() - t0), duration))
    launcher.join()

    e_rows = []
    for h in handles:
        if h is not None:
            e_rows += collect_elephant(h, t0, scheme, run_id, interval)
    m_rows = []
    for proc, part, log in mice_procs:
        try:
            proc.wait(timeout=mice_timeout + 15)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()
        if os.path.exists(part):
            m_rows += read_rows(part)
    after = snapshot_ports(core_names)
    c_rows = core_load_rows(before, after, time.time() - before_ts, scheme, run_id)

    for p in servers:
        p.terminate()

    out = {}
    if m_rows:
        out["mice"] = write_rows(
            os.path.join(results_dir, "mice_results.csv"), MICE_FIELDS, m_rows, append
        )
    if e_rows:
        out["elephant"] = write_rows(
            os.path.join(results_dir, "elephant_results.csv"),
            ELEPHANT_FIELDS,
            e_rows,
            append,
        )
    if c_rows:
        out["core"] = write_rows(
            os.path.join(results_dir, "core_load.csv"), CORE_FIELDS, c_rows, append
        )
    print_summary(m_rows, e_rows, c_rows)
    return out


def print_summary(m_rows, e_rows, c_rows):
    ok = sorted(float(r["fct_ms"]) for r in m_rows if r["status"] == "ok")
    if m_rows:
        p = (
            (lambda q: ok[min(len(ok) - 1, int(q * len(ok)))])
            if ok
            else (lambda q: float("nan"))
        )
        print(
            "mice : %d/%d ok   FCT p50=%.1f ms  p99=%.1f ms"
            % (len(ok), len(m_rows), p(0.5), p(0.99))
        )
    per_flow = defaultdict(list)
    for r in e_rows:
        per_flow[r["flow_id"]].append(float(r["throughput_mbps"]))
    for fid, v in sorted(per_flow.items()):
        print(
            "eleph: %-10s mean %.2f Mbps over %d samples"
            % (fid, sum(v) / len(v), len(v))
        )
    for r in c_rows:
        print("core : %-8s %s Mbps" % (r["switch"], r["avg_tx_mbps"]))


# ============================================================ local test
def _free_port():
    while True:
        t = socket.socket()
        t.bind(("127.0.0.1", 0))
        port = t.getsockname()[1]
        u = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            u.bind(("127.0.0.1", port))
            return port
        except OSError:
            continue
        finally:
            t.close()
            u.close()


def run_local(
    scheme="local",
    duration=5.0,
    rate=10.0,
    proto="tcp",
    seed=42,
    results_dir=RESULTS_DIR,
    append=True,
    elephant_secs=5,
    interval=0.5,
):
    """Loopback sanity test (Task 2.4) — no Mininet or root needed."""
    port = _free_port()
    stop, ready = threading.Event(), threading.Event()
    th = threading.Thread(
        target=run_server,
        kwargs={
            "port": port,
            "bind": "127.0.0.1",
            "stop_event": stop,
            "ready_event": ready,
        },
        daemon=True,
    )
    th.start()
    ready.wait(5)
    run_id = time.strftime("%Y%m%d-%H%M%S")
    hosts = [HostInfo("lo_a", "127.0.0.1", "H1"), HostInfo("lo_b", "127.0.0.1", "H2")]
    t0 = time.time() + 0.5
    flows = generate_mice_schedule(hosts, duration, rate, seed=seed, proto=proto)
    m_rows = run_mice_flows(flows, t0, scheme, run_id, port, timeout=5.0)
    stop.set()
    th.join(2)
    out = {
        "mice": write_rows(
            os.path.join(results_dir, "mice_results.csv"), MICE_FIELDS, m_rows, append
        )
    }

    e_rows = []
    if shutil.which("iperf"):
        eport = _free_port()
        srv = subprocess.Popen(
            ["iperf", "-s", "-p", str(eport)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(0.5)
        spec = _elephant_spec(hosts[0], hosts[1], 0, elephant_secs)
        launch = time.time()
        res = subprocess.run(
            iperf_client_cmd("127.0.0.1", eport, elephant_secs, interval),
            capture_output=True,
            text=True,
            timeout=elephant_secs + 15,
        )
        srv.terminate()
        e_rows = elephant_rows(
            parse_iperf_csv(res.stdout, interval), spec, launch, launch, scheme, run_id
        )
        out["elephant"] = write_rows(
            os.path.join(results_dir, "elephant_results.csv"),
            ELEPHANT_FIELDS,
            e_rows,
            append,
        )
    else:
        print("[!] iperf not installed: skipped elephant test (sudo apt install iperf)")
    print_summary(m_rows, e_rows, [])
    return out


# ============================================================ CLI
def _parse_pairs(text):
    pairs = []
    for item in text.split(","):
        s, d = item.strip().upper().replace("H", "").split("-")
        pairs.append((int(s), int(d)))
    return pairs


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run a benchmark inside Mininet (needs sudo)")
    r.add_argument("--topo", choices=["fattree", "two"], default="fattree")
    r.add_argument(
        "--scenario",
        choices=["mice", "elephant", "collision", "mixed"],
        default="mixed",
    )
    r.add_argument(
        "--scheme",
        default=_cfg("EXPERIMENT_SCHEME", "proposed"),
        help="label for this run, e.g. static | ecmp | proposed",
    )
    r.add_argument("--duration", type=float, default=30.0)
    r.add_argument(
        "--mice-rate",
        type=float,
        default=20.0,
        help="aggregate mice arrivals per second",
    )
    r.add_argument("--mice-proto", choices=["tcp", "udp"], default="tcp")
    r.add_argument("--mice-min", type=int, default=MICE_MIN_BYTES)
    r.add_argument("--mice-max", type=int, default=MICE_MAX_BYTES)
    r.add_argument("--mice-timeout", type=float, default=10.0)
    r.add_argument("--no-mice", action="store_true")
    r.add_argument(
        "--elephants", type=int, default=4, help="random elephants (elephant/mixed)"
    )
    r.add_argument(
        "--pairs",
        type=_parse_pairs,
        default=None,
        help='collision pairs, e.g. "1-9,2-10"',
    )
    r.add_argument(
        "--stagger", type=float, default=0.0, help="delay between collision flows (s)"
    )
    r.add_argument("--interval", type=float, default=0.5)
    r.add_argument("--seed", type=int, default=42)
    r.add_argument("--results-dir", default=RESULTS_DIR)
    r.add_argument(
        "--fresh", action="store_true", help="overwrite CSVs instead of appending"
    )
    r.add_argument("--no-warmup", action="store_true")
    r.add_argument("--cli", action="store_true", help="open the Mininet CLI afterwards")

    lo = sub.add_parser("local", help="loopback test without Mininet")
    lo.add_argument("--scheme", default="local")
    lo.add_argument("--duration", type=float, default=5.0)
    lo.add_argument("--mice-rate", type=float, default=10.0)
    lo.add_argument("--mice-proto", choices=["tcp", "udp"], default="tcp")
    lo.add_argument("--seed", type=int, default=42)
    lo.add_argument("--results-dir", default=RESULTS_DIR)
    lo.add_argument("--fresh", action="store_true")

    sv = sub.add_parser("server", help="mice sink (run on a host)")
    sv.add_argument("--port", type=int, default=MICE_PORT)

    mc = sub.add_parser("mice-client", help="replay a mice schedule (run on a host)")
    mc.add_argument("--schedule", required=True)
    mc.add_argument("--out", required=True)

    args = ap.parse_args(argv)

    if args.cmd == "server":
        try:
            run_server(args.port)
        except KeyboardInterrupt:
            pass
    elif args.cmd == "mice-client":
        with open(args.schedule) as fh:
            s = json.load(fh)
        rows = run_mice_flows(
            s["flows"], s["t0"], s["scheme"], s["run_id"], s["port"], s["timeout"]
        )
        write_rows(args.out, MICE_FIELDS, rows, append=False)
    elif args.cmd == "local":
        print(
            run_local(
                args.scheme,
                args.duration,
                args.mice_rate,
                args.mice_proto,
                args.seed,
                args.results_dir,
                append=not args.fresh,
            )
        )
    elif args.cmd == "run":
        if hasattr(os, "geteuid") and os.geteuid() != 0:
            sys.exit("Mininet needs root: sudo python3 traffic_gen.py run ...")
        net = build_net(args.topo)
        try:
            out = run_experiment(
                net,
                args.scenario,
                args.scheme,
                args.duration,
                args.mice_rate,
                args.mice_proto,
                args.mice_min,
                args.mice_max,
                args.mice_timeout,
                args.elephants,
                args.pairs,
                args.stagger,
                args.interval,
                args.seed,
                args.results_dir,
                append=not args.fresh,
                warmup=not args.no_warmup,
                with_mice=False if args.no_mice else None,
            )
            print("results:", out)
            if args.cli:
                from mininet.cli import CLI

                CLI(net)
        finally:
            net.stop()


if __name__ == "__main__":
    main()
