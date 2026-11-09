# classifier.py

```py
#!/usr/bin/env python3
"""
classifier.py
Telemetry & Flow Classifier — Tasks 3.2 (rate pipeline) and 3.3 (classification).

Pure Python: no Ryu / Mininet imports, so it can be unit-tested anywhere.

Rate calculation (per flow, per poll):

    Rate [Mbps] = (byte_count_now - byte_count_last) * 8 / (t_now - t_last) / 1e6

Classification:
    Rate >= ELEPHANT_THRESHOLD * LINK_BANDWIDTH for N consecutive samples -> ELEPHANT
    otherwise the flow stays MICE.
    An ELEPHANT is demoted back to MICE after MICE_DEMOTION_SAMPLES slow samples
    (hysteresis, so a flow does not flap between classes), and is reported as
    EXPIRED when the switch stops reporting it for FLOW_EXPIRY seconds.

Typical use (flow_monitor.py does this for you):

    clf = FlowClassifier()
    transitions = clf.update({five_tuple: byte_count, ...}, timestamp)
    for flow_id, old, new in transitions: ...
    clf.get_active_elephants()   # {FiveTuple: demand_mbps}
"""

import time
from collections import namedtuple

try:
    import config
except ImportError:  # allow use outside the project directory
    config = None


def _cfg(name, default):
    return getattr(config, name, default) if config is not None else default


ELEPHANT = 'ELEPHANT'
MICE = 'MICE'
EXPIRED = 'EXPIRED'

FiveTuple = namedtuple('FiveTuple', ['src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol'])


def bytes_to_mbps(delta_bytes, delta_t):
    """Convert a byte delta over delta_t seconds into Mbps."""
    return (delta_bytes * 8.0) / (delta_t * 1e6)


class RateTracker(object):
    """
    Task 3.2 — keeps history[key] = (last_byte_count, last_timestamp) and turns
    cumulative byte counters into instantaneous rates. Used for flows (key =
    5-tuple) and for ports (key = (dpid, port_no, 'tx'|'rx')).
    """

    def __init__(self):
        self.history = {}

    def update(self, key, byte_count, timestamp):
        """Store the new sample and return the rate in Mbps (None for the first sample)."""
        prev = self.history.get(key)
        if prev is None:
            self.history[key] = (byte_count, timestamp)
            return None
        last_bytes, last_ts = prev
        dt = timestamp - last_ts
        if dt <= 0:
            return None  # duplicate / out-of-order sample: keep the old reference point
        self.history[key] = (byte_count, timestamp)
        delta = byte_count - last_bytes
        if delta < 0:
            return None  # counter reset (flow rule re-installed): restart from this sample
        return bytes_to_mbps(delta, dt)

    def forget(self, key):
        self.history.pop(key, None)

    def __contains__(self, key):
        return key in self.history

    def __len__(self):
        return len(self.history)


class FlowRecord(object):
    __slots__ = ('flow_id', 'label', 'rate_mbps', 'peak_rate_mbps', 'above', 'below',
                 'samples', 'first_seen', 'last_seen', 'last_change', 'meta')

    def __init__(self, flow_id, now):
        self.flow_id = flow_id
        self.label = MICE
        self.rate_mbps = 0.0
        self.peak_rate_mbps = 0.0
        self.above = 0          # consecutive samples >= threshold
        self.below = 0          # consecutive samples <  threshold
        self.samples = 0
        self.first_seen = now
        self.last_seen = now
        self.last_change = now
        self.meta = {}


class FlowClassifier(object):
    """Task 3.3 — standalone Elephant/Mice classifier."""

    def __init__(self, link_bandwidth=None, threshold=None, consecutive_samples=None,
                 demotion_samples=None, expiry=None):
        self.link_bandwidth = float(link_bandwidth if link_bandwidth is not None
                                    else _cfg('LINK_BANDWIDTH', 10.0))
        self.threshold = float(threshold if threshold is not None
                               else _cfg('ELEPHANT_THRESHOLD', 0.1))
        self.consecutive_samples = int(consecutive_samples if consecutive_samples is not None
                                       else _cfg('ELEPHANT_CONSECUTIVE_SAMPLES', 2))
        self.demotion_samples = int(demotion_samples if demotion_samples is not None
                                    else _cfg('MICE_DEMOTION_SAMPLES', 3))
        self.expiry = float(expiry if expiry is not None else _cfg('FLOW_EXPIRY', 5.0))
        if self.consecutive_samples < 1 or self.demotion_samples < 1:
            raise ValueError('consecutive/demotion samples must be >= 1')

        self.rates = RateTracker()
        self.flows = {}

    # ------------------------------------------------------------------ state
    @property
    def rate_threshold_mbps(self):
        return self.threshold * self.link_bandwidth

    @property
    def flow_history(self):
        """flow_history[flow_id] = (last_byte_count, last_timestamp)"""
        return self.rates.history

    # --------------------------------------------------------------- updating
    def update(self, byte_counts, timestamp=None, meta=None):
        """
        Feed one polling round of cumulative byte counters.

        byte_counts: {flow_id: cumulative_byte_count}
        timestamp:   seconds (defaults to time.time())
        meta:        optional {flow_id: dict} stored with the flow (dpid, packets, ...)

        Returns a list of class transitions: (flow_id, old_label, new_label),
        where new_label is ELEPHANT, MICE or EXPIRED.
        """
        now = time.time() if timestamp is None else float(timestamp)
        transitions = []
        for flow_id, byte_count in byte_counts.items():
            rec = self._record(flow_id, now, meta)
            rate = self.rates.update(flow_id, byte_count, now)
            if rate is not None:
                change = self._apply_rate(rec, rate, now)
                if change:
                    transitions.append(change)
        transitions.extend(self.expire(now))
        return transitions

    def update_rate(self, flow_id, rate_mbps, timestamp=None, meta=None):
        """Feed an already-computed rate (bypasses the byte-counter pipeline)."""
        now = time.time() if timestamp is None else float(timestamp)
        rec = self._record(flow_id, now, meta)
        return self._apply_rate(rec, float(rate_mbps), now)

    def _record(self, flow_id, now, meta):
        rec = self.flows.get(flow_id)
        if rec is None:
            rec = self.flows[flow_id] = FlowRecord(flow_id, now)
        rec.last_seen = now
        if meta and flow_id in meta:
            rec.meta = meta[flow_id]
        return rec

    def _apply_rate(self, rec, rate, now):
        rec.rate_mbps = rate
        rec.peak_rate_mbps = max(rec.peak_rate_mbps, rate)
        rec.samples += 1
        if rate >= self.rate_threshold_mbps:
            rec.above += 1
            rec.below = 0
        else:
            rec.below += 1
            rec.above = 0

        if rec.label == MICE and rec.above >= self.consecutive_samples:
            rec.label, rec.last_change = ELEPHANT, now
            return (rec.flow_id, MICE, ELEPHANT)
        if rec.label == ELEPHANT and rec.below >= self.demotion_samples:
            rec.label, rec.last_change = MICE, now
            return (rec.flow_id, ELEPHANT, MICE)
        return None

    def expire(self, now=None):
        """Forget flows not reported for `expiry` seconds. Returns ELEPHANT->EXPIRED transitions."""
        now = time.time() if now is None else float(now)
        stale = [fid for fid, rec in self.flows.items() if now - rec.last_seen > self.expiry]
        transitions = []
        for fid in stale:
            rec = self.flows.pop(fid)
            self.rates.forget(fid)
            if rec.label == ELEPHANT:
                transitions.append((fid, ELEPHANT, EXPIRED))
        return transitions

    # ------------------------------------------------------------------ query
    def get_active_elephants(self, use_peak=False):
        """
        {flow 5-tuple: current bandwidth demand in Mbps} for every ELEPHANT.
        use_peak=True returns the highest observed rate instead, which is a better
        demand estimate for a flow that is currently squeezed by a collision.
        """
        return {fid: (rec.peak_rate_mbps if use_peak else rec.rate_mbps)
                for fid, rec in self.flows.items() if rec.label == ELEPHANT}

    def get_elephant_details(self):
        return {fid: {'rate_mbps': rec.rate_mbps, 'peak_rate_mbps': rec.peak_rate_mbps,
                      'since': rec.last_change, 'samples': rec.samples, 'meta': dict(rec.meta)}
                for fid, rec in self.flows.items() if rec.label == ELEPHANT}

    def classify(self, flow_id):
        rec = self.flows.get(flow_id)
        return rec.label if rec else None

    def get_rate(self, flow_id):
        rec = self.flows.get(flow_id)
        return rec.rate_mbps if rec else None

    def counts(self):
        out = {ELEPHANT: 0, MICE: 0}
        for rec in self.flows.values():
            out[rec.label] += 1
        return out

    def remove_flow(self, flow_id):
        self.flows.pop(flow_id, None)
        self.rates.forget(flow_id)

    def reset(self):
        self.flows.clear()
        self.rates = RateTracker()

    def __len__(self):
        return len(self.flows)

```

# config.py

```py
# config.py
LINK_BANDWIDTH = 10.0      # Emulated link capacity in Mbps
K_FAT_TREE = 4             # k=4 Fat-Tree (16 hosts, 20 switches)
LINK_DELAY = '2ms'
CONTROLLER_IP = '127.0.0.1'
CONTROLLER_PORT = 6653

# --- Telemetry & classification (flow_monitor.py / classifier.py) ---
MONITOR_INTERVAL = 1.0               # seconds between OpenFlow stats polls
ELEPHANT_THRESHOLD = 0.1             # fraction of LINK_BANDWIDTH (Hedera uses 10%) -> 1 Mbps
ELEPHANT_CONSECUTIVE_SAMPLES = 2     # N samples >= threshold before a flow becomes ELEPHANT
MICE_DEMOTION_SAMPLES = 3            # samples below threshold before an ELEPHANT is demoted
FLOW_EXPIRY = 5.0                    # seconds without stats before a flow is forgotten
CLASSIFY_AT_INGRESS_EDGE_ONLY = True # count each flow once, at the edge switch of its source host
MONITOR_LOG_CSV = True               # write port_stats.csv / flow_stats.csv / events.csv

# --- Workload generator & plotter (traffic_gen.py / plotter.py) ---
RESULTS_DIR = 'results'
EXPERIMENT_SCHEME = 'proposed'       # label for this run: static | ecmp | proposed
MICE_PORT = 6000
IPERF_PORT = 5001

```

# fattree_topo.py

```py
"""
The main goal of this file is to build k=4 fat tree datacenter network:
- 4 core switches
- 8 aggregation switches
- 8 edge switches
- 16 hosts
- configured with OpenFlow 1.3 and deterministic DPIDs
"""

import sys
from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import OVSSwitch, RemoteController
from mininet.link import TCLink
from mininet.log import setLogLevel, info
from mininet.cli import CLI

from config import LINK_BANDWIDTH, K_FAT_TREE, LINK_DELAY, CONTROLLER_IP, CONTROLLER_PORT


class FatTreeTopo(Topo):
    def __init__(self, k=K_FAT_TREE, **opts):
        self.k = k
        self.num_pods = k
        self.num_core_switches = (k // 2) ** 2
        self.num_agg_per_pod = k // 2
        self.num_edge_per_pod = k // 2
        self.num_hosts_per_edge = k // 2
        
        
        self.core_switches = []
        self.agg_switches = {}
        self.edge_switches = {}
        self.host_dict = {}
        
        super(FatTreeTopo, self).__init__(**opts)


    def build(self):
        info("***Building Fat-Tree topology with k=%d\n" % self.k)
        info('***Creating core switches\n')
        self._create_core_switches()
        info('***Creating pods\n')
        self._create_pods()
        info('***Creating core to aggregation links\n')
        self._create_core_to_agg_links()
    
    def _dpid_format(self, layer_id, pod_id, switch_id):
        """
            format the DPID as a 16-digit hexadecimal string with leading zeros.
            format: 10 zeros <layer:02x> <pod:02x> <switch:02x>
        """
        return f"{0:010x}{layer_id:02x}{pod_id:02x}{switch_id:02x}"

    def _create_core_switches(self):
        
        """
            creates (k/2)^2 core switches in (k/2) x (k/2) grid
            for k=4: 4 core switches arranged as (i, j) where i, j in [0, 1].

        """
        half_k = self.k // 2
        for i in range(half_k):
            for j in range(half_k):
                core_id = i * half_k + j
                name = f'c_{i}_{j}'
                dpid = self._dpid_format(layer_id=1, pod_id=0, switch_id=core_id)
                self.core_switches.append(name)
                self.addSwitch(name, dpid=dpid)
    
    
    def _create_pods(self):
        """
            creates k pods, each containing (k/2) aggregation switches and (k/2) edge switches
            complete bipartite graph between aggregation and edge switches
            (k/2) hosts connected to each edge switch
        """
        
        for pod_id in range(self.num_pods):
            for agg_id in range(self.num_agg_per_pod):
                agg_name = f'a_{pod_id}_{agg_id}'
                dpid = self._dpid_format(layer_id=2, pod_id=pod_id, switch_id=agg_id)
                self.addSwitch(agg_name, dpid=dpid)
                self.agg_switches[(pod_id, agg_id)] = agg_name
                
            for edge_id in range(self.num_edge_per_pod):
                edge_name = f'e_{pod_id}_{edge_id}'
                dpid = self._dpid_format(layer_id=3, pod_id=pod_id, switch_id=edge_id + 1)
                self.addSwitch(edge_name, dpid=dpid)
                self.edge_switches[(pod_id, edge_id)] = edge_name
                
                # Connect edge switches to aggregation switches in the same pod
                for agg_id in range(self.num_agg_per_pod):
                    agg_name = self.agg_switches[(pod_id, agg_id)]
                    self.addLink(edge_name, agg_name, bw=LINK_BANDWIDTH, delay=LINK_DELAY)
                
                # Create hosts and connect to edge switch
                for host_id in range(self.num_hosts_per_edge):
                    host_name = f'h_{pod_id}_{edge_id}_{host_id}'
                    
                    ip_addr = f'10.{pod_id}.{edge_id}.{host_id + 2}/8'
                    mac_addr = f'00:00:00:{pod_id:02x}:{edge_id:02x}:{(host_id + 2):02x}'
                    
                    self.addHost(host_name, ip=ip_addr, mac=mac_addr)
                    self.host_dict[(pod_id, edge_id, host_id)] = host_name
                    self.addLink(host_name, edge_name, bw=LINK_BANDWIDTH, delay=LINK_DELAY)        
    
    def _create_core_to_agg_links(self):
        """
            connects core switches to aggregation switches:
            core switch (i, j) connects to aggregation switch (pod_id, i) for all pods
        """
        half_k = self.k // 2
        
        for i in range(half_k):
            for j in range(half_k):
                core_name = f'c_{i}_{j}'
                for pod_id in range(self.num_pods):
                    agg_id = i  # Each core switch connects to the aggregation switch with the same index
                    agg_name = self.agg_switches[(pod_id, agg_id)]
                    self.addLink(core_name, agg_name, bw=LINK_BANDWIDTH, delay=LINK_DELAY)        
                    
                    
class OpenFlow13Switch(OVSSwitch):
    """Custom switch class to enforce OpenFlow 1.3"""
    def __init__(self, name, **params):
        super(OpenFlow13Switch, self).__init__(name, protocols='OpenFlow13', **params)
        

def run_fattree():
    setLogLevel('info')
    
    topo = FatTreeTopo(k=K_FAT_TREE)
    
    # Add remote controller
    info('*** Adding remote controller\n')
    controller = RemoteController('c0', ip=CONTROLLER_IP, port=CONTROLLER_PORT)     
    
    net = Mininet(
        topo=topo, 
        switch=OpenFlow13Switch, 
        controller=controller, 
        link=TCLink, 
        autoSetMacs=False,
        autoStaticArp=False
    )
    
  
    info('*** Starting network\n')
    net.start()
    
    
    info('\n ==== Network Topology Summary ====\n')
    info(f"Total Switches: {len(topo.core_switches) + len(topo.agg_switches) + len(topo.edge_switches)}\n")    
    info(f"Total Hosts: {len(topo.hosts())}\n")
    info(f"Link Capacity: {LINK_BANDWIDTH} Mbps, Delay: {LINK_DELAY} ms\n")
    info(f"Sample Hosts:\n")
    for h in net.hosts:
        info(f"Host: {h.name:10}, IP: {h.IP():14}, MAC: {h.MAC()}\n")
    info('\n ==== End of Summary ====\n')

    info('*** Running CLI\n')
    CLI(net)
    
    info('*** Stopping network\n')
    net.stop()
    
topos = {'fattree': (lambda: FatTreeTopo(k=K_FAT_TREE))}

if __name__ == '__main__':
    run_fattree()
```

# flow_monitor.py

```py
#!/usr/bin/env python3
"""
flow_monitor.py
Telemetry — Task 3.1 (flow & port stats poller) feeding classifier.py.

Every MONITOR_INTERVAL seconds it sends OFPFlowStatsRequest and OFPPortStatsRequest
to the switches, then:
  * EventOFPPortStatsReply -> TX/RX bytes per port -> Mbps + utilization
  * EventOFPFlowStatsReply -> 5-tuple (src_ip, dst_ip, src_port, dst_port, protocol)
                              + byte_count -> FlowClassifier -> ELEPHANT / MICE

Run standalone (monitoring only) or next to the controller:
    ryu-manager main_controller.py flow_monitor.py --ofp-tcp-listen-port 6653

How main_controller.py can consume it (pick one):
  1. Subscribe to events:
        @set_ev_cls(flow_monitor.EventFlowClassChanged)
        def on_class_change(self, ev): ev.flow_id, ev.old, ev.new, ev.rate_mbps, ev.dpid
  2. Look the app up and poll it:
        mon = app_manager.lookup_service_brick('flow_monitor')
        mon.get_active_elephants()      # {FiveTuple: Mbps}
        mon.get_port_utilization()      # {(dpid, port): {...}}
  3. Subclass FlowMonitor and override on_flow_class_change().

IMPORTANT: flow stats are per flow *entry*. The controller must install rules that
match on the 5-tuple (eth_type=0x0800, ipv4_src, ipv4_dst, ip_proto, tcp/udp ports)
at least at the ingress edge switch, otherwise there is nothing per-flow to classify
(simple_switch_13 only matches on MAC addresses).
"""

import csv
import os
import time

from ryu.base import app_manager
from ryu.controller import event, ofp_event
from ryu.controller.handler import DEAD_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from ryu.lib import hub
from ryu.ofproto import ofproto_v1_3

import config
from classifier import ELEPHANT, EXPIRED, MICE, FiveTuple, FlowClassifier, RateTracker


def _cfg(name, default):
    return getattr(config, name, default)


K = _cfg('K_FAT_TREE', 4)
LAYER_CORE, LAYER_AGG, LAYER_EDGE = 1, 2, 3


# --------------------------------------------------------------------- helpers
def dpid_layer(dpid):
    return (dpid >> 16) & 0xff


def dpid_to_name(dpid):
    """Inverse of FatTreeTopo._dpid_format: <layer:02x><pod:02x><switch:02x>."""
    layer, pod, sw = dpid_layer(dpid), (dpid >> 8) & 0xff, dpid & 0xff
    half = K // 2
    if layer == LAYER_CORE:
        return 'c_%d_%d' % (sw // half, sw % half)
    if layer == LAYER_AGG:
        return 'a_%d_%d' % (pod, sw)
    if layer == LAYER_EDGE:
        return 'e_%d_%d' % (pod, sw - 1)
    return 'dpid_%x' % dpid


def ip_to_ingress_edge_dpid(ip):
    """10.<pod>.<edge>.<host> -> DPID of the edge switch the host hangs off."""
    try:
        a, pod, edge, _ = (int(x) for x in str(ip).split('.'))
    except (ValueError, AttributeError):
        return None
    if a != 10:
        return None
    return (LAYER_EDGE << 16) | (pod << 8) | (edge + 1)


def _match_get(match, key, default=None):
    try:
        value = match[key]
    except (KeyError, TypeError):
        return default
    if isinstance(value, tuple):  # masked field: (value, mask)
        value = value[0]
    return value


def extract_five_tuple(match):
    """OFPMatch -> FiveTuple, or None if the entry is not an IPv4 flow rule."""
    src_ip, dst_ip = _match_get(match, 'ipv4_src'), _match_get(match, 'ipv4_dst')
    if src_ip is None or dst_ip is None:
        return None
    proto = int(_match_get(match, 'ip_proto', 0) or 0)
    if proto == 6:
        sport, dport = _match_get(match, 'tcp_src', 0), _match_get(match, 'tcp_dst', 0)
    elif proto == 17:
        sport, dport = _match_get(match, 'udp_src', 0), _match_get(match, 'udp_dst', 0)
    else:
        sport, dport = 0, 0
    return FiveTuple(src_ip, dst_ip, int(sport or 0), int(dport or 0), proto)


# ---------------------------------------------------------------------- events
class EventFlowClassChanged(event.EventBase):
    """A flow was promoted to ELEPHANT, demoted to MICE, or an ELEPHANT EXPIRED."""

    def __init__(self, flow_id, old, new, rate_mbps, dpid):
        super(EventFlowClassChanged, self).__init__()
        self.flow_id, self.old, self.new = flow_id, old, new
        self.rate_mbps, self.dpid = rate_mbps, dpid


class EventStatsUpdated(event.EventBase):
    """Fired after each processed reply; kind is 'flow' or 'port'."""

    def __init__(self, dpid, kind):
        super(EventStatsUpdated, self).__init__()
        self.dpid, self.kind = dpid, kind


# ------------------------------------------------------------------------- app
class FlowMonitor(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]
    _EVENTS = [EventFlowClassChanged, EventStatsUpdated]

    def __init__(self, *args, **kwargs):
        super(FlowMonitor, self).__init__(*args, **kwargs)
        self.name = 'flow_monitor'
        self.interval = float(_cfg('MONITOR_INTERVAL', 1.0))
        self.ingress_only = bool(_cfg('CLASSIFY_AT_INGRESS_EDGE_ONLY', True))
        self.link_bw = float(_cfg('LINK_BANDWIDTH', 10.0))
        self.scheme = os.environ.get('SDN_SCHEME', _cfg('EXPERIMENT_SCHEME', 'proposed'))

        self.datapaths = {}
        self.classifier = FlowClassifier()
        self.port_rates = RateTracker()
        self.port_stats = {}   # (dpid, port_no) -> {tx_mbps, rx_mbps, utilization, ...}
        self.flow_stats = {}   # FiveTuple -> {dpid, bytes, packets, rate_mbps, label}
        self._flow_parts = {}  # multipart reassembly buffers
        self._port_parts = {}

        self._logs = {}
        if _cfg('MONITOR_LOG_CSV', True):
            self._open_logs(_cfg('RESULTS_DIR', 'results'))

        self.monitor_thread = hub.spawn(self._monitor)

    # ------------------------------------------------------------ switch registry
    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def _state_change_handler(self, ev):
        dp = ev.datapath
        if ev.state == MAIN_DISPATCHER and dp.id is not None:
            if dp.id not in self.datapaths:
                self.logger.info('monitor: register %s (%016x)', dpid_to_name(dp.id), dp.id)
            self.datapaths[dp.id] = dp
        elif ev.state == DEAD_DISPATCHER and dp.id in self.datapaths:
            self.logger.info('monitor: unregister %s', dpid_to_name(dp.id))
            del self.datapaths[dp.id]

    # ------------------------------------------------------------------ polling
    def _monitor(self):
        while True:
            for dp in list(self.datapaths.values()):
                self._request_stats(dp)
            hub.sleep(self.interval)

    def _wants_flow_stats(self, dpid):
        if not self.ingress_only:
            return True
        has_edges = any(dpid_layer(d) == LAYER_EDGE for d in self.datapaths)
        return dpid_layer(dpid) == LAYER_EDGE or not has_edges

    def _request_stats(self, dp):
        ofp, parser = dp.ofproto, dp.ofproto_parser
        if self._wants_flow_stats(dp.id):
            dp.send_msg(parser.OFPFlowStatsRequest(dp))
        dp.send_msg(parser.OFPPortStatsRequest(dp, 0, ofp.OFPP_ANY))

    # ------------------------------------------------------------- flow replies
    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def _flow_stats_reply_handler(self, ev):
        msg = ev.msg
        dpid = msg.datapath.id
        self._flow_parts.setdefault(dpid, []).extend(msg.body)
        if msg.flags & ofproto_v1_3.OFPMPF_REPLY_MORE:
            return  # wait for the rest of the multipart reply
        self.process_flow_stats(dpid, self._flow_parts.pop(dpid), time.time())

    def process_flow_stats(self, dpid, body, now):
        byte_counts, meta = {}, {}
        for stat in body:
            if stat.priority == 0:
                continue  # table-miss entry
            ft = extract_five_tuple(stat.match)
            if ft is None:
                continue
            if self.ingress_only:
                ingress = ip_to_ingress_edge_dpid(ft.src_ip)
                if ingress is not None and ingress in self.datapaths and ingress != dpid:
                    continue  # count each flow once, at its ingress edge switch
            # a rerouted flow has two rules here (priority 10 + 100): add them up
            byte_counts[ft] = byte_counts.get(ft, 0) + stat.byte_count
            old = meta.get(ft, {'packets': 0})
            meta[ft] = {'dpid': dpid, 'packets': old['packets'] + stat.packet_count,
                        'duration_s': stat.duration_sec + stat.duration_nsec * 1e-9}

        transitions = self.classifier.update(byte_counts, now, meta)

        for ft, nbytes in byte_counts.items():
            self.flow_stats[ft] = {'dpid': dpid, 'bytes': nbytes,
                                   'packets': meta[ft]['packets'],
                                   'rate_mbps': self.classifier.get_rate(ft) or 0.0,
                                   'label': self.classifier.classify(ft) or MICE}
            self._log('flows', [round(now, 3), self.scheme, dpid_to_name(dpid)] + list(ft) +
                      [nbytes, round(self.flow_stats[ft]['rate_mbps'], 4), self.flow_stats[ft]['label']])
        for ft in [f for f in self.flow_stats if f not in self.classifier.flows]:
            del self.flow_stats[ft]

        for flow_id, old, new in transitions:
            rate = self.classifier.get_rate(flow_id) or 0.0
            self.logger.info('monitor: %s %s:%s -> %s:%s/%s  %s -> %s (%.2f Mbps)',
                             dpid_to_name(dpid), flow_id.src_ip, flow_id.src_port,
                             flow_id.dst_ip, flow_id.dst_port, flow_id.protocol, old, new, rate)
            self.log_event('%s %s->%s' % (new, flow_id.src_ip, flow_id.dst_ip), now)
            self.on_flow_class_change(flow_id, old, new, rate, dpid)
            self.send_event_to_observers(EventFlowClassChanged(flow_id, old, new, rate, dpid))
        self.send_event_to_observers(EventStatsUpdated(dpid, 'flow'))

    # ------------------------------------------------------------- port replies
    @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
    def _port_stats_reply_handler(self, ev):
        msg = ev.msg
        dpid = msg.datapath.id
        self._port_parts.setdefault(dpid, []).extend(msg.body)
        if msg.flags & ofproto_v1_3.OFPMPF_REPLY_MORE:
            return
        self.process_port_stats(dpid, self._port_parts.pop(dpid), time.time())

    def process_port_stats(self, dpid, body, now):
        for stat in body:
            port = stat.port_no
            if port > ofproto_v1_3.OFPP_MAX:
                continue  # LOCAL / reserved ports
            tx = self.port_rates.update((dpid, port, 'tx'), stat.tx_bytes, now)
            rx = self.port_rates.update((dpid, port, 'rx'), stat.rx_bytes, now)
            if tx is None or rx is None:
                continue
            util = max(tx, rx) / self.link_bw if self.link_bw else 0.0
            self.port_stats[(dpid, port)] = {'tx_mbps': tx, 'rx_mbps': rx, 'utilization': util,
                                             'tx_bytes': stat.tx_bytes, 'rx_bytes': stat.rx_bytes,
                                             'timestamp': now}
            self._log('ports', [round(now, 3), self.scheme, dpid, dpid_to_name(dpid), port,
                                round(tx, 4), round(rx, 4), round(util, 4)])
        self.send_event_to_observers(EventStatsUpdated(dpid, 'port'))

    # --------------------------------------------------------------- public API
    def on_flow_class_change(self, flow_id, old, new, rate_mbps, dpid):
        """Hook for subclasses (e.g. trigger rerouting when new == ELEPHANT)."""

    def get_active_elephants(self, use_peak=False):
        """{FiveTuple: demand_mbps} for every flow currently classified ELEPHANT."""
        return self.classifier.get_active_elephants(use_peak=use_peak)

    def get_flow_stats(self):
        return dict(self.flow_stats)

    def get_port_utilization(self, dpid=None):
        if dpid is None:
            return dict(self.port_stats)
        return {k: v for k, v in self.port_stats.items() if k[0] == dpid}

    def get_core_load(self):
        """{core switch name: aggregate TX Mbps over all its ports}."""
        load = {}
        for (dpid, _port), s in self.port_stats.items():
            if dpid_layer(dpid) == LAYER_CORE:
                name = dpid_to_name(dpid)
                load[name] = load.get(name, 0.0) + s['tx_mbps']
        return dict(sorted(load.items()))

    def log_event(self, text, timestamp=None):
        """Append to events.csv; plotter.py draws these as markers (e.g. 'reroute H1->H9')."""
        self._log('events', [round(timestamp or time.time(), 3), self.scheme, text])

    # ------------------------------------------------------------------ logging
    def _open_logs(self, results_dir):
        specs = {
            'ports': ('port_stats.csv', ['epoch', 'scheme', 'dpid', 'switch', 'port',
                                         'tx_mbps', 'rx_mbps', 'utilization']),
            'flows': ('flow_stats.csv', ['epoch', 'scheme', 'switch', 'src_ip', 'dst_ip',
                                         'src_port', 'dst_port', 'protocol', 'bytes',
                                         'rate_mbps', 'label']),
            'events': ('events.csv', ['epoch', 'scheme', 'event']),
        }
        try:
            os.makedirs(results_dir, exist_ok=True)
            for key, (fname, header) in specs.items():
                path = os.path.join(results_dir, fname)
                new = not os.path.exists(path) or os.path.getsize(path) == 0
                fh = open(path, 'a', newline='')
                writer = csv.writer(fh)
                if new:
                    writer.writerow(header)
                    fh.flush()
                self._logs[key] = (fh, writer)
        except OSError as exc:
            self.logger.warning('monitor: CSV logging disabled (%s)', exc)
            self._logs = {}

    def close_logs(self):
        for fh, _ in self._logs.values():
            fh.close()
        self._logs = {}

    def _log(self, key, row):
        entry = self._logs.get(key)
        if entry:
            entry[1].writerow(row)
            entry[0].flush()


```

# init.py

```py
# girl need to implement here and then i guess to run
# maybe we will have seperate codes
# like for generating the topology
# another one for the algorithm
# another one is for the simulation
```

# live_dashboard.py

```py
#!/usr/bin/env python3
"""
live_dashboard.py
Real-time dynamic visualization dashboard for professor demonstration.
Reads results/port_stats.csv and results/events.csv live.
"""

import os
import csv
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import defaultdict

RESULTS_DIR = 'results'
PORT_CSV = os.path.join(RESULTS_DIR, 'port_stats.csv')
EVENT_CSV = os.path.join(RESULTS_DIR, 'events.csv')
FLOW_CSV = os.path.join(RESULTS_DIR, 'elephant_results.csv')

fig, (ax_bars, ax_tp) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("SDN Datacenter Fabric: Real-Time Telemetry & Dynamic Rerouting", fontsize=14, fontweight='bold')

def read_last_core_stats():
    """Reads latest TX throughput for core switches from port_stats.csv."""
    if not os.path.exists(PORT_CSV):
        return {}
    core_loads = defaultdict(float)
    try:
        with open(PORT_CSV, 'r') as f:
            reader = list(csv.DictReader(f))
            if not reader:
                return {}
            # Get data from the last 2 seconds
            latest_epoch = float(reader[-1]['epoch'])
            recent = [r for r in reader if latest_epoch - float(r['epoch']) <= 1.5]
            for r in recent:
                sw = r['switch']
                if sw.startswith('c_'):
                    core_loads[sw] += float(r.get('tx_mbps', 0))
    except (ValueError, OSError, KeyError):
        pass
    return core_loads

def read_throughput_series():
    """Reads time-series throughput for elephant flows."""
    if not os.path.exists(FLOW_CSV):
        return {}, []
    series = defaultdict(list)
    try:
        with open(FLOW_CSV, 'r') as f:
            reader = csv.DictReader(f)
            for r in reader:
                fid = r['flow_id']
                t = float(r['time_s'])
                tp = float(r['throughput_mbps'])
                series[fid].append((t, tp))
    except (ValueError, OSError, KeyError):
        pass
    return series

def update(frame):
    ax_bars.clear()
    ax_tp.clear()

    # -------------------------------------------------------------
    # Panel 1: Live Core Switch Link Utilization
    # -------------------------------------------------------------
    core_loads = read_last_core_stats()
    core_switches = ['c_0_0', 'c_0_1', 'c_1_0', 'c_1_1']
    loads = [core_loads.get(sw, 0.0) for sw in core_switches]
    
    # Dynamic Bar Colors: Green (Normal) -> Orange (Loaded) -> Red (Bottlenecked)
    colors = []
    for l in loads:
        if l >= 9.0:
            colors.append('#e74c3c')  # Red (Congested)
        elif l >= 5.0:
            colors.append('#f39c12')  # Orange
        else:
            colors.append('#2ecc71')  # Green (Healthy)

    bars = ax_bars.bar(core_switches, loads, color=colors, width=0.5, edgecolor='black')
    ax_bars.axhline(10.0, color='red', linestyle='--', linewidth=1.5, label='Link Capacity (10 Mbps)')
    ax_bars.set_ylim(0, 12.0)
    ax_bars.set_ylabel("Aggregate TX Load (Mbps)", fontsize=11)
    ax_bars.set_title("Core Switch Real-Time Load Balancing", fontsize=12)
    ax_bars.grid(axis='y', alpha=0.3)
    ax_bars.legend(loc='upper right')

    for bar, val in zip(bars, loads):
        ax_bars.text(bar.get_x() + bar.get_width() / 2, val + 0.3, f"{val:.1f}M", 
                     ha='center', va='bottom', fontweight='bold', fontsize=9)

    # -------------------------------------------------------------
    # Panel 2: Elephant Flow Throughput vs Time
    # -------------------------------------------------------------
    series = read_throughput_series()
    for fid, pts in series.items():
        if pts:
            pts.sort(key=lambda x: x[0])
            xs = [p[0] for p in pts][-30:]  # keep recent 30 samples
            ys = [p[1] for p in pts][-30:]
            ax_tp.plot(xs, ys, marker='o', markersize=3, label=fid, linewidth=2)

    ax_tp.axhline(10.0, color='gray', linestyle=':', label='Link Limit (10M)')
    ax_tp.set_ylim(0, 12.0)
    ax_tp.set_xlabel("Time (seconds)", fontsize=11)
    ax_tp.set_ylabel("Throughput (Mbps)", fontsize=11)
    ax_tp.set_title("Elephant Flow Throughput (Recovery Curve)", fontsize=12)
    ax_tp.grid(alpha=0.3)
    if series:
        ax_tp.legend(loc='lower right')

ani = animation.FuncAnimation(fig, update, interval=1000)

if __name__ == '__main__':
    print("Starting Live SDN Dashboard... (Close window to exit)")
    plt.tight_layout()
    plt.show()
```

# main_controller.py

```py
#!/usr/bin/env python3
"""
main_controller.py
Ryu controller for our project. It does 3 things:
  1. answers ARP requests itself (so we never flood in the fat-tree, it has loops)
  2. installs normal paths for every new flow (priority 10)
       static   -> always the first path
       ecmp     -> path chosen by hashing the 5-tuple
       proposed -> same as ecmp, but elephants get rerouted
  3. (proposed only) when flow_monitor finds an elephant, ask reroute_engine
     for the best path and install it with priority 100

Run:  SDN_SCHEME=proposed ryu-manager main_controller.py --ofp-tcp-listen-port 6653
      (SDN_SCHEME can be static, ecmp or proposed)
"""

import zlib

from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from ryu.lib.packet import arp, ether_types, ethernet, ipv4, packet, tcp, udp

from classifier import ELEPHANT, FiveTuple
from flow_monitor import FlowMonitor
from reroute_engine import RerouteEngine

MICE_PRIORITY = 10
ELEPHANT_PRIORITY = 100
MICE_IDLE_TIMEOUT = 10
ELEPHANT_IDLE_TIMEOUT = 5


class MainController(FlowMonitor):
    # we extend FlowMonitor, so stats polling + classification already run here

    def __init__(self, *args, **kwargs):
        super(MainController, self).__init__(*args, **kwargs)
        if self.scheme not in ('static', 'ecmp', 'proposed'):
            raise ValueError('SDN_SCHEME must be static, ecmp or proposed')

        self.engine = RerouteEngine()

        self.ip_to_host = self.engine.ip_to_host
        self.ip_to_mac = self.engine.ip_to_mac

        self.logger.info('controller started, scheme = %s', self.scheme)

    # ---------------------------------------------------------------- helpers
    def add_flow(self, dp, priority, match, out_port, idle_timeout=0):
        parser = dp.ofproto_parser
        actions = [parser.OFPActionOutput(out_port)]
        inst = [parser.OFPInstructionActions(dp.ofproto.OFPIT_APPLY_ACTIONS, actions)]
        mod = parser.OFPFlowMod(datapath=dp, priority=priority, match=match,
                                instructions=inst, idle_timeout=idle_timeout)
        dp.send_msg(mod)

    def make_match(self, parser, flow):
        fields = {'eth_type': ether_types.ETH_TYPE_IP,
                  'ipv4_src': flow.src_ip, 'ipv4_dst': flow.dst_ip,
                  'ip_proto': flow.protocol}
        if flow.protocol == 6:
            fields['tcp_src'] = flow.src_port
            fields['tcp_dst'] = flow.dst_port
        elif flow.protocol == 17:
            fields['udp_src'] = flow.src_port
            fields['udp_dst'] = flow.dst_port
        return parser.OFPMatch(**fields)

    def next_port(self, path, switch, dst_ip):
        # port on `switch` that leads to the next hop of the path
        hops = path + [self.ip_to_host[dst_ip]]
        i = hops.index(switch)
        return self.engine.port_map[(switch, hops[i + 1])]

    def pick_path(self, flow):
        paths = self.engine.get_candidate_paths(flow.src_ip, flow.dst_ip)
        if not paths:
            return None
        if self.scheme == 'static':
            return paths[0]
        # ecmp: same flow always gets the same path, different flows are spread
        h = zlib.crc32(str(tuple(flow)).encode())
        return paths[h % len(paths)]

    def install_path(self, flow, path, priority, idle_timeout):
        for sw in path:
            dp = self.datapaths.get(self.engine.name_to_dpid[sw])
            if dp is None:
                continue
            match = self.make_match(dp.ofproto_parser, flow)
            self.add_flow(dp, priority, match, self.next_port(path, sw, flow.dst_ip), idle_timeout)

    # ------------------------------------------------------------ switch join
    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        dp = ev.msg.datapath
        parser = dp.ofproto_parser
        # table-miss: send unknown packets to the controller
        self.add_flow(dp, 0, parser.OFPMatch(), dp.ofproto.OFPP_CONTROLLER)

    # -------------------------------------------------------------- packet in
    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        msg = ev.msg
        dp = msg.datapath
        in_port = msg.match['in_port']
        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocol(ethernet.ethernet)
        if eth is None:
            return

        if eth.ethertype == ether_types.ETH_TYPE_ARP:
            self.handle_arp(dp, in_port, eth, pkt.get_protocol(arp.arp))
            return

        ip = pkt.get_protocol(ipv4.ipv4)
        if ip is None or ip.src not in self.ip_to_host or ip.dst not in self.ip_to_host:
            return  # ignore IPv6, LLDP, unknown hosts

        src_port, dst_port = 0, 0
        t = pkt.get_protocol(tcp.tcp)
        u = pkt.get_protocol(udp.udp)
        if t:
            src_port, dst_port = t.src_port, t.dst_port
        elif u:
            src_port, dst_port = u.src_port, u.dst_port
        flow = FiveTuple(ip.src, ip.dst, src_port, dst_port, ip.proto)

        path = self.pick_path(flow)
        if path is None:
            return
        self.install_path(flow, path, MICE_PRIORITY, MICE_IDLE_TIMEOUT)

        # send this first packet out too
        switch = self.engine.dpid_to_name.get(dp.id)
        if switch not in path:
            return
        parser = dp.ofproto_parser
        actions = [parser.OFPActionOutput(self.next_port(path, switch, flow.dst_ip))]
        data = msg.data if msg.buffer_id == dp.ofproto.OFP_NO_BUFFER else None
        out = parser.OFPPacketOut(datapath=dp, buffer_id=msg.buffer_id, in_port=in_port,
                                  actions=actions, data=data)
        dp.send_msg(out)

    def handle_arp(self, dp, in_port, eth, arp_pkt):
        # proxy ARP: we know every host's MAC, so reply directly
        if arp_pkt is None or arp_pkt.opcode != arp.ARP_REQUEST:
            return
        mac = self.ip_to_mac.get(arp_pkt.dst_ip)
        if mac is None:
            return
        reply = packet.Packet()
        reply.add_protocol(ethernet.ethernet(ethertype=ether_types.ETH_TYPE_ARP,
                                             dst=eth.src, src=mac))
        reply.add_protocol(arp.arp(opcode=arp.ARP_REPLY, src_mac=mac, src_ip=arp_pkt.dst_ip,
                                   dst_mac=arp_pkt.src_mac, dst_ip=arp_pkt.src_ip))
        reply.serialize()
        parser = dp.ofproto_parser
        out = parser.OFPPacketOut(datapath=dp, buffer_id=dp.ofproto.OFP_NO_BUFFER,
                                  in_port=dp.ofproto.OFPP_CONTROLLER,
                                  actions=[parser.OFPActionOutput(in_port)], data=reply.data)
        dp.send_msg(out)

    # -------------------------------------------------- elephant rerouting
    def on_flow_class_change(self, flow_id, old, new, rate_mbps, dpid):
        # called by FlowMonitor every time a flow changes class
        if self.scheme != 'proposed':
            return
        if new == ELEPHANT:
            self.reroute(flow_id, rate_mbps)
        else:
            # flow became MICE again or finished -> give its bandwidth back
            self.engine.release_flow(flow_id)

    def reroute(self, flow, demand_mbps):
        path, free_bw = self.engine.compute_optimal_path(flow, demand_mbps, flow.src_ip, flow.dst_ip)
        if path is None:
            return
        self.install_path(flow, path, ELEPHANT_PRIORITY, ELEPHANT_IDLE_TIMEOUT)

        cores = [n for n in path if n.startswith('c_')]
        via = cores[0] if cores else path[0]
        src = self.ip_to_host[flow.src_ip]
        dst = self.ip_to_host[flow.dst_ip]
        self.logger.info('REROUTE %s -> %s (%.2f Mbps) via %s : %s',
                         src, dst, demand_mbps, via, ' -> '.join(path))
        self.log_event('reroute %s->%s via %s' % (src, dst, via))
```

# plotter.py

```py
#!/usr/bin/env python3
"""
plotter.py
Visualization — Task 2.3. Parses the CSV logs written by traffic_gen.py
(and optionally events.csv from flow_monitor.py) and draws:

  1. throughput_vs_time.png  — elephant throughput over time, one panel per scheme,
                               with link capacity, aggregate and controller events
  2. fct_cdf.png             — CDF of mice FCT: Static vs ECMP vs Proposed
  3. core_load.png           — aggregate load on each core switch + imbalance (std dev)

By default only the most recent run of each scheme is plotted (use --all-runs to merge).

Usage:
  python3 plotter.py                       # reads ./results, writes ./results/plots
  python3 plotter.py --results-dir results --schemes static ecmp proposed
"""

import argparse
import csv
import os
import re
import sys
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

try:
    import config
except ImportError:
    config = None

LINK_BW = float(getattr(config, 'LINK_BANDWIDTH', 10.0)) if config else 10.0
RESULTS_DIR = getattr(config, 'RESULTS_DIR', 'results') if config else 'results'

SCHEME_ORDER = ['static', 'ecmp', 'proposed']
SCHEME_LABELS = {'static': 'Static', 'ecmp': 'ECMP', 'proposed': 'Proposed (SDN rerouting)'}
SCHEME_COLORS = {'static': '#d62728', 'ecmp': '#ff7f0e', 'proposed': '#2ca02c'}

MICE_REQUIRED = ('scheme', 'run_id', 'size_bytes', 'fct_ms', 'status')
ELEPHANT_REQUIRED = ('scheme', 'run_id', 'flow_id', 'time_s', 'throughput_mbps')
CORE_REQUIRED = ('scheme', 'run_id', 'switch', 'avg_tx_mbps')


# ------------------------------------------------------------------ loading
def read_csv(path, required=()):
    """Read a CSV into dicts and validate that the required columns exist."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, newline='') as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            raise ValueError('%s is empty' % path)
        missing = [c for c in required if c not in reader.fieldnames]
        if missing:
            raise ValueError('%s: missing columns %s' % (path, missing))
        return list(reader)


def _f(value, default=float('nan')):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def select_runs(rows, schemes=None, all_runs=False):
    """Filter by scheme and keep only the latest run_id of each scheme."""
    if schemes:
        rows = [r for r in rows if r['scheme'] in schemes]
    if all_runs:
        return rows
    latest = {}
    for r in rows:
        if r['run_id'] > latest.get(r['scheme'], ''):
            latest[r['scheme']] = r['run_id']
    return [r for r in rows if r['run_id'] == latest[r['scheme']]]


def order_schemes(names):
    names = set(names)
    return [s for s in SCHEME_ORDER if s in names] + sorted(n for n in names if n not in SCHEME_ORDER)


def scheme_label(s):
    return SCHEME_LABELS.get(s, s)


def scheme_color(s, i):
    return SCHEME_COLORS.get(s, plt.cm.tab10(i % 10))


def _natural(name):
    return [int(x) if x.isdigit() else x for x in re.split(r'(\d+)', name)]


def load_events(path):
    if not path or not os.path.exists(path):
        return []
    events = []
    for r in read_csv(path, ('epoch', 'event')):
        t = _f(r['epoch'])
        if not np.isnan(t):
            events.append({'epoch': t, 'event': r['event'], 'scheme': r.get('scheme', '') or ''})
    return events


# --------------------------------------------------------------- 1. throughput
def plot_throughput_vs_time(elephant_csv, out_path, schemes=None, all_runs=False,
                            events_csv=None, link_bw=LINK_BW):
    rows = select_runs(read_csv(elephant_csv, ELEPHANT_REQUIRED), schemes, all_runs)
    if not rows:
        raise ValueError('no elephant rows to plot in %s' % elephant_csv)

    series = defaultdict(lambda: defaultdict(list))
    t0, window = {}, {}
    for r in rows:
        t, y = _f(r['time_s']), _f(r['throughput_mbps'])
        if np.isnan(t) or np.isnan(y):
            continue
        s = r['scheme']
        series[s][r['flow_id']].append((t, y))
        ep = _f(r.get('epoch'))
        if not np.isnan(ep):
            t0[s] = ep - t
            lo, hi = window.get(s, (ep, ep))
            window[s] = (min(lo, ep - t), max(hi, ep))
    events = load_events(events_csv)
    names = order_schemes(series)

    fig, axes = plt.subplots(len(names), 1, figsize=(10, 2.9 * len(names) + 0.6),
                             sharex=True, squeeze=False)
    for ax, s in zip(axes[:, 0], names):
        flows = series[s]
        ymax = link_bw
        curves = []
        for fid in sorted(flows, key=_natural):
            pts = sorted(flows[fid])
            xs, ys = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
            ax.plot(xs, ys, marker='o', markersize=2.5, linewidth=1.4, label=fid)
            curves.append((xs, ys))
            ymax = max(ymax, ys.max())
        if len(curves) > 1:
            step = np.median(np.concatenate([np.diff(c[0]) for c in curves if len(c[0]) > 1] or [[0.5]]))
            grid = np.arange(min(c[0].min() for c in curves), max(c[0].max() for c in curves) + 1e-9,
                             max(step, 1e-3))
            agg = sum(np.interp(grid, xs, ys, left=0.0, right=0.0) for xs, ys in curves)
            ax.plot(grid, agg, 'k--', linewidth=1.1, alpha=0.7, label='Aggregate')
            ymax = max(ymax, agg.max())
        ax.axhline(link_bw, linestyle=':', color='gray', label='Link capacity (%g Mbps)' % link_bw)
        ax.set_ylim(0, ymax * 1.15)

        if s in t0:
            lo, hi = window[s]
            for ev in events:
                if ev['scheme'] not in ('', s) or not (lo - 1 <= ev['epoch'] <= hi + 1):
                    continue
                x = ev['epoch'] - t0[s]
                ax.axvline(x, color='purple', linestyle='--', alpha=0.6, linewidth=1)
                ax.text(x, 0.97, ' ' + ev['event'], rotation=90, va='top', ha='left', fontsize=7,
                        color='purple', transform=ax.get_xaxis_transform())

        ax.set_title(scheme_label(s), fontsize=11)
        ax.set_ylabel('Throughput (Mbps)')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc='lower right', ncol=min(4, len(curves) + 2))
    axes[-1, 0].set_xlabel('Time since experiment start (s)')
    fig.suptitle('Elephant flow throughput vs. time', fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# ----------------------------------------------------------------- 2. FCT CDF
def plot_fct_cdf(mice_csv, out_path, schemes=None, all_runs=False, log_x=True):
    rows = select_runs(read_csv(mice_csv, MICE_REQUIRED), schemes, all_runs)
    data, failed = defaultdict(list), defaultdict(int)
    for r in rows:
        fct = _f(r['fct_ms'])
        if r['status'] == 'ok' and np.isfinite(fct) and fct > 0:
            data[r['scheme']].append(fct)
        else:
            failed[r['scheme']] += 1
    if not any(data.values()):
        raise ValueError('no completed mice flows in %s' % mice_csv)

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, s in enumerate(order_schemes(set(data) | set(failed))):
        v = np.sort(np.array(data[s]))
        if v.size == 0:
            continue
        y = np.arange(1, v.size + 1) / v.size
        p50, p99 = np.percentile(v, [50, 99])
        extra = ', failed=%d' % failed[s] if failed[s] else ''
        ax.step(v, y, where='post', color=scheme_color(s, i), linewidth=2,
                label='%s  p50=%.1f ms, p99=%.1f ms (n=%d%s)' % (scheme_label(s), p50, p99, v.size, extra))
    if log_x:
        ax.set_xscale('log')
    ax.set_xlabel('Flow completion time (ms)' + (' — log scale' if log_x else ''))
    ax.set_ylabel('CDF')
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3, which='both')
    ax.legend(fontsize=8, loc='upper left')
    ax.set_title('Mice flow completion time (8–64 KB requests)')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------- 3. core load
def plot_core_load(core_csv, out_path, schemes=None, all_runs=False, metric='avg_tx_mbps'):
    rows = select_runs(read_csv(core_csv, CORE_REQUIRED), schemes, all_runs)
    values = defaultdict(lambda: defaultdict(list))
    for r in rows:
        v = _f(r.get(metric))
        if not np.isnan(v):
            values[r['scheme']][r['switch']].append(v)
    if not values:
        raise ValueError('no core load rows in %s' % core_csv)

    names = order_schemes(values)
    switches = sorted({sw for s in values for sw in values[s]}, key=_natural)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={'width_ratios': [3, 1.4]})
    x = np.arange(len(switches))
    width = 0.8 / len(names)
    stds, covs = [], []
    for i, s in enumerate(names):
        vals = np.array([np.mean(values[s][sw]) if values[s].get(sw) else 0.0 for sw in switches])
        ax1.bar(x + (i - (len(names) - 1) / 2.0) * width, vals, width,
                label=scheme_label(s), color=scheme_color(s, i))
        std = float(np.std(vals))
        stds.append(std)
        covs.append(std / vals.mean() if vals.mean() > 0 else 0.0)
    ax1.set_xticks(x)
    ax1.set_xticklabels(switches)
    ax1.set_ylabel('Avg. aggregate load (Mbps)')
    ax1.set_xlabel('Core switch')
    ax1.set_title('Load per core switch')
    ax1.grid(axis='y', alpha=0.3)
    ax1.legend(fontsize=8)

    bars = ax2.bar(range(len(names)), stds, color=[scheme_color(s, i) for i, s in enumerate(names)])
    for b, std, cov in zip(bars, stds, covs):
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height(), 'σ=%.2f\nCoV=%.2f' % (std, cov),
                 ha='center', va='bottom', fontsize=8)
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels([scheme_label(s).split(' ')[0] for s in names])
    ax2.set_ylabel('Std. dev. across cores (Mbps)')
    ax2.set_title('Load imbalance (lower = better)')
    ax2.set_ylim(0, max(stds + [0.1]) * 1.35)
    ax2.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description='Plot SDN benchmark results.')
    ap.add_argument('--results-dir', default=RESULTS_DIR)
    ap.add_argument('--mice', help='default: <results-dir>/mice_results.csv')
    ap.add_argument('--elephant', help='default: <results-dir>/elephant_results.csv')
    ap.add_argument('--core', help='default: <results-dir>/core_load.csv')
    ap.add_argument('--events', help='default: <results-dir>/events.csv (optional)')
    ap.add_argument('--out-dir', help='default: <results-dir>/plots')
    ap.add_argument('--schemes', nargs='*', help='e.g. static ecmp proposed')
    ap.add_argument('--all-runs', action='store_true', help='merge every run instead of the latest')
    ap.add_argument('--linear-x', action='store_true', help='linear x axis for the FCT CDF')
    args = ap.parse_args(argv)

    d = args.results_dir
    out_dir = args.out_dir or os.path.join(d, 'plots')
    os.makedirs(out_dir, exist_ok=True)
    jobs = [
        ('Throughput vs time', plot_throughput_vs_time, args.elephant or os.path.join(d, 'elephant_results.csv'),
         'throughput_vs_time.png', {'events_csv': args.events or os.path.join(d, 'events.csv')}),
        ('FCT CDF', plot_fct_cdf, args.mice or os.path.join(d, 'mice_results.csv'),
         'fct_cdf.png', {'log_x': not args.linear_x}),
        ('Core load', plot_core_load, args.core or os.path.join(d, 'core_load.csv'),
         'core_load.png', {}),
    ]
    failures = 0
    for title, fn, src, name, kw in jobs:
        if not os.path.exists(src):
            print('[skip] %-18s %s not found' % (title, src))
            continue
        try:
            path = fn(src, os.path.join(out_dir, name), schemes=args.schemes, all_runs=args.all_runs, **kw)
            print('[ok]   %-18s -> %s' % (title, path))
        except (ValueError, KeyError) as exc:
            failures += 1
            print('[fail] %-18s %s' % (title, exc))
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())

```

# README.md

```md
��# sdn_final_project

# Module 1: k=4 Fat-Tree Datacenter Fabric (Topology Setup)

This module builds a canonical $k=4$ Fat-Tree datacenter network in Mininet using Open vSwitch (OVS) configured with OpenFlow 1.3.

---

## 1. Topology Specifications

- **Architecture:** Canonical $k=4$ Fat-Tree (Al-Fares et al.)
- **Switches (20 Total):**
  - 4 Core Switches (`c_0_0` to `c_1_1`)
  - 8 Aggregation Switches (`a_0_0` to `a_3_1`)
  - 8 Edge Switches (`e_0_0` to `e_3_1`)
- **Hosts (16 Total):**
  - `h_0_0_0` to `h_3_1_1`
  - IP Scheme: `10.<pod>.<edge>.<host_id + 2>/24`
- **Link Parameters:**
  - Bandwidth: 10.0 Mbps (`TCLink`)
  - Propagation Delay: 2 ms per link
- **OpenFlow Protocol:** OpenFlow 1.3 with 64-bit deterministic DPIDs

---

## 2. Prerequisites

Ensure Open vSwitch, Mininet, and Ryu SDN controller are installed:

\`\`\`bash
sudo mn -c
sudo apt-get install openvswitch-switch
pip3 install ryu
\`\`\`

## 3. How to Run
SDN development requires two terminal windows:

Terminal 1: Start the SDN Controller (Ryu)
Start Ryu listening on TCP port 6653:
\`\`\`bash
ryu-manager --verbose ryu.app.simple_switch_13 --ofp-tcp-listen-port 6653
\`\`\`

Terminal 2: Launch the Mininet Network Fabric
Navigate to your project directory and run:
\`\`\`bash
# Clean up any leftover Mininet instances
sudo mn -c
\`\`\`

# Launch the Fat-Tree topology
sudo python3 fattree_topo.py
## 4. Verification Commands (Inside Mininet CLI)
Once you reach the mininet> prompt, run the following commands to verify:
A. Verify Node Counts
\`\`\`bash
mininet> nodes
\`\`\`
Expected: 20 switches (c_*, a_*, e_*), 16 hosts (h_*), and 1 controller (c0).


B. Verify Link Wiring and Ports
\`\`\`bash
mininet> net
\`\`\`
Expected: 4 uplinks per aggregation switch, 2 downlinks + 2 uplinks per edge switch, and 4 downlinks per core switch.

C. Verify IP & MAC Addressing
\`\`\`bash
mininet> dump
\`\`\`
Expected: All hosts display deterministic IPs in 10.x.y.z format.

D. Verify Controller Connectivity
\`\`\`bash
mininet> sh ovs-vsctl show
\`\`\`
Expected: Look for is_connected: true under each bridge.

E. Verify Flow Rule Installation from Ryu
\`\`\`bash
mininet> sh ovs-ofctl -O OpenFlow13 dump-flows c_0_0
\`\`\`
Expected: A table-miss entry (priority=0, actions=CONTROLLER:65535) installed by Ryu.

## 5. How to Exit Cleanly
Always exit the Mininet CLI and clear kernel state:
\`\`\`bash
mininet> exit
sudo mn -c
\`\`\`
```

# reroute_engine.py

```py
#!/usr/bin/env python3
"""
reroute_engine.py
"""

import networkx as nx
import config

try:
    DEFAULT_BW = config.LINK_BANDWIDTH
    K = config.K_FAT_TREE
except AttributeError:
    DEFAULT_BW = 10.0
    K = 4


class RerouteEngine:
    def __init__(self, k=K, link_bw=DEFAULT_BW):
        self.k = k
        self.link_bw = link_bw
        self.graph = nx.DiGraph()

        self.switches = []
        self.hosts = []
        self.dpid_to_name = {}
        self.name_to_dpid = {}
        self.host_to_edge = {}
        self.ip_to_edge = {}
        self.ip_to_mac = {}
        self.ip_to_host = {}
        self.host_to_ip = {}
        self.port_map = {}       # (u, v) -> egress port on u to reach v

        self.active_reservations = {}
        self._path_cache = {}

        self._build_fattree_fabric()
        self._precompute_all_shortest_paths()

    def _dpid_format(self, layer_id, pod_id, switch_id):
        """16-digit hex DPID matching fattree_topo.py."""
        return f"{0:010x}{layer_id:02x}{pod_id:02x}{switch_id:02x}"

    def _build_fattree_fabric(self):
        """Constructs the Fat-Tree graph, DPIDs, and deterministic port mappings."""
        half_k = self.k // 2
        num_pods = self.k
        agg_switches = {}
        edge_switches = {}

        # -------------------------------------------------------------
        # 1. Core Switches
        # -------------------------------------------------------------
        for i in range(half_k):
            for j in range(half_k):
                core_id = i * half_k + j
                sw_name = f'c_{i}_{j}'
                dpid_hex = self._dpid_format(layer_id=1, pod_id=0, switch_id=core_id)
                dpid_int = int(dpid_hex, 16)

                self.switches.append(sw_name)
                self.dpid_to_name[dpid_int] = sw_name
                self.name_to_dpid[sw_name] = dpid_int
                self.graph.add_node(sw_name, type='switch', layer='core')

        # -------------------------------------------------------------
        # 2. Pods (Aggregation, Edge, Hosts)
        # -------------------------------------------------------------
        for p in range(num_pods):
            # Aggregation switches
            for a in range(half_k):
                sw_name = f'a_{p}_{a}'
                dpid_hex = self._dpid_format(layer_id=2, pod_id=p, switch_id=a)
                dpid_int = int(dpid_hex, 16)
                agg_switches[(p, a)] = sw_name
                self.switches.append(sw_name)
                self.dpid_to_name[dpid_int] = sw_name
                self.name_to_dpid[sw_name] = dpid_int
                self.graph.add_node(sw_name, type='switch', layer='agg')

            # Edge switches and Hosts
            for e in range(half_k):
                sw_name = f'e_{p}_{e}'
                dpid_hex = self._dpid_format(layer_id=3, pod_id=p, switch_id=e + 1)
                dpid_int = int(dpid_hex, 16)
                edge_switches[(p, e)] = sw_name
                self.switches.append(sw_name)
                self.dpid_to_name[dpid_int] = sw_name
                self.name_to_dpid[sw_name] = dpid_int
                self.graph.add_node(sw_name, type='switch', layer='edge')

                # Connect Edge <-> Agg within Pod
                for a in range(half_k):
                    agg_name = agg_switches[(p, a)]
                    # Edge port to agg: a + 1
                    # Agg port to edge: e + 1
                    p_edge = a + 1
                    p_agg = e + 1
                    self._add_bidirectional_link(sw_name, agg_name, p_edge, p_agg)

                # Connect Hosts <-> Edge
                for h in range(half_k):
                    h_name = f'h_{p}_{e}_{h}'
                    ip_addr = f'10.{p}.{e}.{h + 2}'
                    mac_addr = f'00:00:00:{p:02x}:{e:02x}:{(h + 2):02x}'

                    self.hosts.append(h_name)
                    self.host_to_edge[h_name] = sw_name
                    self.ip_to_edge[ip_addr] = sw_name
                    self.ip_to_mac[ip_addr] = mac_addr
                    self.ip_to_host[ip_addr] = h_name
                    self.host_to_ip[h_name] = ip_addr

                    self.graph.add_node(h_name, type='host', ip=ip_addr, mac=mac_addr)
                    # Edge downlink ports: 3, 4 (half_k + h + 1)
                    p_edge = half_k + h + 1
                    p_host = 1
                    self._add_bidirectional_link(sw_name, h_name, p_edge, p_host)

        # -------------------------------------------------------------
        # 3. Core <-> Aggregation Links
        # -------------------------------------------------------------
        for i in range(half_k):
            for j in range(half_k):
                core_name = f'c_{i}_{j}'
                for p in range(num_pods):
                    agg_name = agg_switches[(p, i)]
                    # Core port to pod p: p + 1
                    # Agg uplink port: half_k + j + 1
                    p_core = p + 1
                    p_agg = half_k + j + 1
                    self._add_bidirectional_link(core_name, agg_name, p_core, p_agg)

    def _add_bidirectional_link(self, u, v, port_u, port_v):
        """Adds symmetric directed edges and records physical port numbers."""
        self.port_map[(u, v)] = port_u
        self.port_map[(v, u)] = port_v
        self.graph.add_edge(u, v, capacity=self.link_bw, used_bw=0.0, residual_bw=self.link_bw)
        self.graph.add_edge(v, u, capacity=self.link_bw, used_bw=0.0, residual_bw=self.link_bw)

    def _precompute_all_shortest_paths(self):
        edge_switches = [n for n in self.switches if n.startswith('e_')]
        for src in edge_switches:
            for dst in edge_switches:
                if src == dst:
                    self._path_cache[(src, dst)] = [[src]]
                else:
                    paths = list(nx.all_shortest_paths(self.graph, source=src, target=dst))
                    self._path_cache[(src, dst)] = paths

    def resolve_to_edge(self, node_or_ip):
        if node_or_ip in self.ip_to_edge:
            return self.ip_to_edge[node_or_ip]
        if node_or_ip in self.host_to_edge:
            return self.host_to_edge[node_or_ip]
        return node_or_ip

    def get_candidate_paths(self, src_node_or_ip, dst_node_or_ip):
        src_edge = self.resolve_to_edge(src_node_or_ip)
        dst_edge = self.resolve_to_edge(dst_node_or_ip)
        return self._path_cache.get((src_edge, dst_edge), [])

    def get_path_bottleneck_residual(self, path):
        if len(path) <= 1:
            return float('inf')
        bottleneck = float('inf')
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            residual = self.graph[u][v]['residual_bw']
            if residual < bottleneck:
                bottleneck = residual
        return bottleneck

    def compute_optimal_path(self, flow_id, demand_mbps, src, dst):
        candidate_paths = self.get_candidate_paths(src, dst)
        if not candidate_paths:
            return None, 0.0

        if flow_id in self.active_reservations:
            self.release_flow(flow_id)

        qualified_paths = []
        all_path_capacities = []

        for p in candidate_paths:
            b_cap = self.get_path_bottleneck_residual(p)
            all_path_capacities.append((p, b_cap))
            if b_cap >= demand_mbps:
                qualified_paths.append((p, b_cap))

        if qualified_paths:
            best_path, best_cap = max(qualified_paths, key=lambda x: x[1])
        else:
            best_path, best_cap = max(all_path_capacities, key=lambda x: x[1])

        self._book_bandwidth(flow_id, best_path, demand_mbps)
        return best_path, best_cap

    def _book_bandwidth(self, flow_id, path, demand_mbps):
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            self.graph[u][v]['used_bw'] += demand_mbps
            self.graph[u][v]['residual_bw'] = max(0.0, self.graph[u][v]['capacity'] - self.graph[u][v]['used_bw'])

        self.active_reservations[flow_id] = {
            'path': path,
            'demand': demand_mbps
        }

    def release_flow(self, flow_id):
        if flow_id not in self.active_reservations:
            return
        info = self.active_reservations.pop(flow_id)
        path = info['path']
        demand = info['demand']

        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            self.graph[u][v]['used_bw'] = max(0.0, self.graph[u][v]['used_bw'] - demand)
            self.graph[u][v]['residual_bw'] = min(self.graph[u][v]['capacity'], self.graph[u][v]['capacity'] - self.graph[u][v]['used_bw'])

    def get_port(self, u, v):
        return self.port_map.get((u, v))

    def get_path_ports(self, path):
        out_instructions = []
        for i in range(len(path) - 1):
            curr_node, next_node = path[i], path[i + 1]
            port = self.get_port(curr_node, next_node)
            out_instructions.append((curr_node, port))
        return out_instructions


if __name__ == '__main__':
    print("Testing pure Python RerouteEngine...")
    engine = RerouteEngine()
    print(f"Nodes: {engine.graph.number_of_nodes()}, Edges: {engine.graph.number_of_edges()}")
    p, cap = engine.compute_optimal_path("test", 5.0, "10.0.0.2", "10.2.0.2")
    print(f"Path: {' -> '.join(p)}, Bottleneck: {cap} Mbps")
```

# results.zip

This is a binary file of the type: Compressed Archive

# results\core_load.csv

```csv
scheme,run_id,switch,duration_s,tx_bytes,rx_bytes,avg_tx_mbps,avg_rx_mbps
ecmp,20261109-082704,c_0_0,35.61,18422748,18437558,4.1391,4.1424
ecmp,20261109-082704,c_0_1,35.61,3105193,3105259,0.6976,0.6977
ecmp,20261109-082704,c_1_0,35.61,17035342,17041340,3.8273,3.8287
ecmp,20261109-082704,c_1_1,35.61,2135898,2135964,0.4799,0.4799
proposed,20261109-082859,c_0_0,43.01,31989257,31995247,5.9505,5.9516
proposed,20261109-082859,c_0_1,43.01,4442209,4442209,0.8263,0.8263
proposed,20261109-082859,c_1_0,43.01,33015773,33021631,6.1415,6.1426
proposed,20261109-082859,c_1_1,43.01,4025205,4025205,0.7488,0.7488

```

# results\elephant_results.csv

```csv
scheme,run_id,flow_id,src,dst,src_ip,dst_ip,interval_start,interval_end,time_s,epoch,bytes,throughput_mbps
ecmp,20261109-082704,H1->H9,H1,H9,10.0.0.2,10.2.0.2,0.0,0.5,3.517,1794241653.081,340280,5.4445
ecmp,20261109-082704,H1->H9,H1,H9,10.0.0.2,10.2.0.2,0.5,1.0,4.017,1794241653.581,207064,3.3130
ecmp,20261109-082704,H1->H9,H1,H9,10.0.0.2,10.2.0.2,1.0,1.5,4.517,1794241654.081,195480,3.1277
ecmp,20261109-082704,H1->H9,H1,H9,10.0.0.2,10.2.0.2,1.5,2.0,5.017,1794241654.581,65160,1.0426
ecmp,20261109-082704,H1->H9,H1,H9,10.0.0.2,10.2.0.2,2.0,2.5,5.517,1794241655.081,130320,2.0851
ecmp,20261109-082704,H1->H9,H1,H9,10.0.0.2,10.2.0.2,2.5,3.0,6.017,1794241655.581,195480,3.1277
ecmp,20261109-082704,H1->H9,H1,H9,10.0.0.2,10.2.0.2,3.0,3.5,6.517,1794241656.081,130320,2.0851
ecmp,20261109-082704,H1->H9,H1,H9,10.0.0.2,10.2.0.2,3.5,4.0,7.017,1794241656.581,195480,3.1277
ecmp,20261109-082704,H1->H9,H1,H9,10.0.0.2,10.2.0.2,4.0,4.5,7.517,1794241657.081,195480,3.1277
ecmp,20261109-082704,H1->H9,H1,H9,10.0.0.2,10.2.0.2,4.5,5.0,8.017,1794241657.581,260640,4.1702
ecmp,20261109-082704,H1->H9,H1,H9,10.0.0.2,10.2.0.2,5.0,5.5,8.517,1794241658.081,456120,7.2979
ecmp,20261109-082704,H1->H9,H1,H9,10.0.0.2,10.2.0.2,5.5,6.0,9.017,1794241658.581,260640,4.1702
ecmp,20261109-082704,H1->H9,H1,H9,10.0.0.2,10.2.0.2,6.0,6.5,9.517,1794241659.081,456120,7.2979
ecmp,20261109-082704,H1->H9,H1,H9,10.0.0.2,10.2.0.2,6.5,7.0,10.017,1794241659.581,586440,9.3830
ecmp,20261109-082704,H1->H9,H1,H9,10.0.0.2,10.2.0.2,7.0,7.5,10.517,1794241660.081,456120,7.2979
ecmp,20261109-082704,H1->H9,H1,H9,10.0.0.2,10.2.0.2,7.5,8.0,11.017,1794241660.581,586440,9.3830
ecmp,20261109-082704,H1->H9,H1,H9,10.0.0.2,10.2.0.2,8.0,8.5,11.517,1794241661.081,456120,7.2979
ecmp,20261109-082704,H1->H9,H1,H9,10.0.0.2,10.2.0.2,8.5,9.0,12.017,1794241661.581,716760,11.4682
ecmp,20261109-082704,H1->H9,H1,H9,10.0.0.2,10.2.0.2,9.0,9.5,12.517,1794241662.081,651600,10.4256
ecmp,20261109-082704,H1->H9,H1,H9,10.0.0.2,10.2.0.2,9.5,10.0,13.017,1794241662.581,586440,9.3830
ecmp,20261109-082704,H1->H9,H1,H9,10.0.0.2,10.2.0.2,10.0,10.5,13.517,1794241663.081,781920,12.5107
ecmp,20261109-082704,H1->H9,H1,H9,10.0.0.2,10.2.0.2,10.5,11.0,14.017,1794241663.581,586440,9.3830
ecmp,20261109-082704,H1->H9,H1,H9,10.0.0.2,10.2.0.2,11.0,11.5,14.517,1794241664.081,716760,11.4682
ecmp,20261109-082704,H1->H9,H1,H9,10.0.0.2,10.2.0.2,11.5,12.0,15.017,1794241664.581,912240,14.5958
ecmp,20261109-082704,H1->H9,H1,H9,10.0.0.2,10.2.0.2,12.0,12.5,15.517,1794241665.081,716760,11.4682
ecmp,20261109-082704,H1->H9,H1,H9,10.0.0.2,10.2.0.2,12.5,13.0,16.017,1794241665.581,325800,5.2128
ecmp,20261109-082704,H1->H9,H1,H9,10.0.0.2,10.2.0.2,13.0,13.5,16.517,1794241666.081,977400,15.6384
ecmp,20261109-082704,H1->H9,H1,H9,10.0.0.2,10.2.0.2,13.5,14.0,17.017,1794241666.581,651600,10.4256
ecmp,20261109-082704,H1->H9,H1,H9,10.0.0.2,10.2.0.2,14.0,14.5,17.517,1794241667.081,1172880,18.7661
ecmp,20261109-082704,H1->H9,H1,H9,10.0.0.2,10.2.0.2,14.5,15.0,18.017,1794241667.581,1042560,16.6810
ecmp,20261109-082704,H1->H9,H1,H9,10.0.0.2,10.2.0.2,15.0,15.5,18.517,1794241668.081,1172880,18.7661
ecmp,20261109-082704,H1->H9,H1,H9,10.0.0.2,10.2.0.2,15.5,16.0,19.017,1794241668.581,977400,15.6384
ecmp,20261109-082704,H1->H9,H1,H9,10.0.0.2,10.2.0.2,16.0,16.5,19.517,1794241669.081,1107720,17.7235
ecmp,20261109-082704,H1->H9,H1,H9,10.0.0.2,10.2.0.2,16.5,17.0,20.017,1794241669.581,1175136,18.8022
ecmp,20261109-082704,H1->H9,H1,H9,10.0.0.2,10.2.0.2,17.0,17.5,20.517,1794241670.081,714504,11.4321
ecmp,20261109-082704,H1->H9,H1,H9,10.0.0.2,10.2.0.2,17.5,18.0,21.017,1794241670.581,1629000,26.0640
ecmp,20261109-082704,H1->H9,H1,H9,10.0.0.2,10.2.0.2,18.0,18.5,21.517,1794241671.081,1107720,17.7235
ecmp,20261109-082704,H1->H9,H1,H9,10.0.0.2,10.2.0.2,18.5,19.0,22.017,1794241671.581,1172880,18.7661
ecmp,20261109-082704,H1->H9,H1,H9,10.0.0.2,10.2.0.2,19.0,19.5,22.517,1794241672.081,977400,15.6384
ecmp,20261109-082704,H1->H9,H1,H9,10.0.0.2,10.2.0.2,19.5,20.0,23.017,1794241672.581,651600,10.4256
ecmp,20261109-082704,H2->H10,H2,H10,10.0.0.3,10.2.0.3,0.0,0.5,3.841,1794241653.405,471352,7.5416
ecmp,20261109-082704,H2->H10,H2,H10,10.0.0.3,10.2.0.3,0.5,1.0,4.341,1794241653.905,96264,1.5402
ecmp,20261109-082704,H2->H10,H2,H10,10.0.0.3,10.2.0.3,1.0,1.5,4.841,1794241654.405,195480,3.1277
ecmp,20261109-082704,H2->H10,H2,H10,10.0.0.3,10.2.0.3,1.5,2.0,5.341,1794241654.905,130320,2.0851
ecmp,20261109-082704,H2->H10,H2,H10,10.0.0.3,10.2.0.3,2.0,2.5,5.841,1794241655.405,130320,2.0851
ecmp,20261109-082704,H2->H10,H2,H10,10.0.0.3,10.2.0.3,2.5,3.0,6.341,1794241655.905,195480,3.1277
ecmp,20261109-082704,H2->H10,H2,H10,10.0.0.3,10.2.0.3,3.0,3.5,6.841,1794241656.405,130320,2.0851
ecmp,20261109-082704,H2->H10,H2,H10,10.0.0.3,10.2.0.3,3.5,4.0,7.341,1794241656.905,195480,3.1277
ecmp,20261109-082704,H2->H10,H2,H10,10.0.0.3,10.2.0.3,4.0,4.5,7.841,1794241657.405,195480,3.1277
ecmp,20261109-082704,H2->H10,H2,H10,10.0.0.3,10.2.0.3,4.5,5.0,8.341,1794241657.905,456120,7.2979
ecmp,20261109-082704,H2->H10,H2,H10,10.0.0.3,10.2.0.3,5.0,5.5,8.841,1794241658.405,325800,5.2128
ecmp,20261109-082704,H2->H10,H2,H10,10.0.0.3,10.2.0.3,5.5,6.0,9.341,1794241658.905,325800,5.2128
ecmp,20261109-082704,H2->H10,H2,H10,10.0.0.3,10.2.0.3,6.0,6.5,9.841,1794241659.405,325800,5.2128
ecmp,20261109-082704,H2->H10,H2,H10,10.0.0.3,10.2.0.3,6.5,7.0,10.341,1794241659.905,260640,4.1702
ecmp,20261109-082704,H2->H10,H2,H10,10.0.0.3,10.2.0.3,7.0,7.5,10.841,1794241660.405,325800,5.2128
ecmp,20261109-082704,H2->H10,H2,H10,10.0.0.3,10.2.0.3,7.5,8.0,11.341,1794241660.905,390960,6.2554
ecmp,20261109-082704,H2->H10,H2,H10,10.0.0.3,10.2.0.3,8.0,8.5,11.841,1794241661.405,325800,5.2128
ecmp,20261109-082704,H2->H10,H2,H10,10.0.0.3,10.2.0.3,8.5,9.0,12.341,1794241661.905,456120,7.2979
ecmp,20261109-082704,H2->H10,H2,H10,10.0.0.3,10.2.0.3,9.0,9.5,12.841,1794241662.405,260640,4.1702
ecmp,20261109-082704,H2->H10,H2,H10,10.0.0.3,10.2.0.3,9.5,10.0,13.341,1794241662.905,325800,5.2128
ecmp,20261109-082704,H2->H10,H2,H10,10.0.0.3,10.2.0.3,10.0,10.5,13.841,1794241663.405,586440,9.3830
ecmp,20261109-082704,H2->H10,H2,H10,10.0.0.3,10.2.0.3,10.5,11.0,14.341,1794241663.905,456120,7.2979
ecmp,20261109-082704,H2->H10,H2,H10,10.0.0.3,10.2.0.3,11.0,11.5,14.841,1794241664.405,325800,5.2128
ecmp,20261109-082704,H2->H10,H2,H10,10.0.0.3,10.2.0.3,11.5,12.0,15.341,1794241664.905,586440,9.3830
ecmp,20261109-082704,H2->H10,H2,H10,10.0.0.3,10.2.0.3,12.0,12.5,15.841,1794241665.405,521280,8.3405
ecmp,20261109-082704,H2->H10,H2,H10,10.0.0.3,10.2.0.3,12.5,13.0,16.341,1794241665.905,586440,9.3830
ecmp,20261109-082704,H2->H10,H2,H10,10.0.0.3,10.2.0.3,13.0,13.5,16.841,1794241666.405,716760,11.4682
ecmp,20261109-082704,H2->H10,H2,H10,10.0.0.3,10.2.0.3,13.5,14.0,17.341,1794241666.905,586440,9.3830
ecmp,20261109-082704,H2->H10,H2,H10,10.0.0.3,10.2.0.3,14.0,14.5,17.841,1794241667.405,781920,12.5107
ecmp,20261109-082704,H2->H10,H2,H10,10.0.0.3,10.2.0.3,14.5,15.0,18.341,1794241667.905,716760,11.4682
ecmp,20261109-082704,H2->H10,H2,H10,10.0.0.3,10.2.0.3,15.0,15.5,18.841,1794241668.405,781920,12.5107
ecmp,20261109-082704,H2->H10,H2,H10,10.0.0.3,10.2.0.3,15.5,16.0,19.341,1794241668.905,716760,11.4682
ecmp,20261109-082704,H2->H10,H2,H10,10.0.0.3,10.2.0.3,16.0,16.5,19.841,1794241669.405,456120,7.2979
ecmp,20261109-082704,H2->H10,H2,H10,10.0.0.3,10.2.0.3,16.5,17.0,20.341,1794241669.905,847080,13.5533
ecmp,20261109-082704,H2->H10,H2,H10,10.0.0.3,10.2.0.3,17.0,17.5,20.841,1794241670.405,716760,11.4682
ecmp,20261109-082704,H2->H10,H2,H10,10.0.0.3,10.2.0.3,17.5,18.0,21.341,1794241670.905,1172880,18.7661
ecmp,20261109-082704,H2->H10,H2,H10,10.0.0.3,10.2.0.3,18.0,18.5,21.841,1794241671.405,847080,13.5533
ecmp,20261109-082704,H2->H10,H2,H10,10.0.0.3,10.2.0.3,18.5,19.0,22.341,1794241671.905,1042560,16.6810
ecmp,20261109-082704,H2->H10,H2,H10,10.0.0.3,10.2.0.3,19.0,19.5,22.841,1794241672.405,717512,11.4802
ecmp,20261109-082704,H2->H10,H2,H10,10.0.0.3,10.2.0.3,19.5,20.0,23.341,1794241672.905,1237288,19.7966
proposed,20261109-082859,H1->H9,H1,H9,10.0.0.2,10.2.0.2,0.0,0.5,0.592,1794241761.069,917504,14.6801
proposed,20261109-082859,H1->H9,H1,H9,10.0.0.2,10.2.0.2,0.5,1.0,1.092,1794241761.569,605792,9.6927
proposed,20261109-082859,H1->H9,H1,H9,10.0.0.2,10.2.0.2,1.0,1.5,1.592,1794241762.069,456120,7.2979
proposed,20261109-082859,H1->H9,H1,H9,10.0.0.2,10.2.0.2,1.5,2.0,2.092,1794241762.569,390960,6.2554
proposed,20261109-082859,H1->H9,H1,H9,10.0.0.2,10.2.0.2,2.0,2.5,2.592,1794241763.069,325800,5.2128
proposed,20261109-082859,H1->H9,H1,H9,10.0.0.2,10.2.0.2,2.5,3.0,3.092,1794241763.569,456120,7.2979
proposed,20261109-082859,H1->H9,H1,H9,10.0.0.2,10.2.0.2,3.0,3.5,3.592,1794241764.069,390960,6.2554
proposed,20261109-082859,H1->H9,H1,H9,10.0.0.2,10.2.0.2,3.5,4.0,4.092,1794241764.569,260640,4.1702
proposed,20261109-082859,H1->H9,H1,H9,10.0.0.2,10.2.0.2,4.0,4.5,4.592,1794241765.069,195480,3.1277
proposed,20261109-082859,H1->H9,H1,H9,10.0.0.2,10.2.0.2,4.5,5.0,5.092,1794241765.569,260640,4.1702
proposed,20261109-082859,H1->H9,H1,H9,10.0.0.2,10.2.0.2,5.0,5.5,5.592,1794241766.069,260640,4.1702
proposed,20261109-082859,H1->H9,H1,H9,10.0.0.2,10.2.0.2,5.5,6.0,6.092,1794241766.569,456120,7.2979
proposed,20261109-082859,H1->H9,H1,H9,10.0.0.2,10.2.0.2,6.0,6.5,6.592,1794241767.069,390960,6.2554
proposed,20261109-082859,H1->H9,H1,H9,10.0.0.2,10.2.0.2,6.5,7.0,7.092,1794241767.569,456120,7.2979
proposed,20261109-082859,H1->H9,H1,H9,10.0.0.2,10.2.0.2,7.0,7.5,7.592,1794241768.069,195480,3.1277
proposed,20261109-082859,H1->H9,H1,H9,10.0.0.2,10.2.0.2,7.5,8.0,8.092,1794241768.569,325800,5.2128
proposed,20261109-082859,H1->H9,H1,H9,10.0.0.2,10.2.0.2,8.0,8.5,8.592,1794241769.069,586440,9.3830
proposed,20261109-082859,H1->H9,H1,H9,10.0.0.2,10.2.0.2,8.5,9.0,9.092,1794241769.569,390960,6.2554
proposed,20261109-082859,H1->H9,H1,H9,10.0.0.2,10.2.0.2,9.0,9.5,9.592,1794241770.069,651600,10.4256
proposed,20261109-082859,H1->H9,H1,H9,10.0.0.2,10.2.0.2,9.5,10.0,10.092,1794241770.569,390960,6.2554
proposed,20261109-082859,H1->H9,H1,H9,10.0.0.2,10.2.0.2,10.0,10.5,10.592,1794241771.069,847080,13.5533
proposed,20261109-082859,H1->H9,H1,H9,10.0.0.2,10.2.0.2,10.5,11.0,11.092,1794241771.569,390960,6.2554
proposed,20261109-082859,H1->H9,H1,H9,10.0.0.2,10.2.0.2,11.0,11.5,11.592,1794241772.069,847080,13.5533
proposed,20261109-082859,H1->H9,H1,H9,10.0.0.2,10.2.0.2,11.5,12.0,12.092,1794241772.569,456120,7.2979
proposed,20261109-082859,H1->H9,H1,H9,10.0.0.2,10.2.0.2,12.0,12.5,12.592,1794241773.069,912240,14.5958
proposed,20261109-082859,H1->H9,H1,H9,10.0.0.2,10.2.0.2,12.5,13.0,13.092,1794241773.569,651600,10.4256
proposed,20261109-082859,H1->H9,H1,H9,10.0.0.2,10.2.0.2,13.0,13.5,13.592,1794241774.069,847080,13.5533
proposed,20261109-082859,H1->H9,H1,H9,10.0.0.2,10.2.0.2,13.5,14.0,14.092,1794241774.569,912240,14.5958
proposed,20261109-082859,H1->H9,H1,H9,10.0.0.2,10.2.0.2,14.0,14.5,14.592,1794241775.069,716760,11.4682
proposed,20261109-082859,H1->H9,H1,H9,10.0.0.2,10.2.0.2,14.5,15.0,15.092,1794241775.569,1107720,17.7235
proposed,20261109-082859,H1->H9,H1,H9,10.0.0.2,10.2.0.2,15.0,15.5,15.592,1794241776.069,521280,8.3405
proposed,20261109-082859,H1->H9,H1,H9,10.0.0.2,10.2.0.2,15.5,16.0,16.092,1794241776.569,260640,4.1702
proposed,20261109-082859,H1->H9,H1,H9,10.0.0.2,10.2.0.2,16.0,16.5,16.592,1794241777.069,1433520,22.9363
proposed,20261109-082859,H1->H9,H1,H9,10.0.0.2,10.2.0.2,16.5,17.0,17.092,1794241777.569,912240,14.5958
proposed,20261109-082859,H1->H9,H1,H9,10.0.0.2,10.2.0.2,17.0,17.5,17.592,1794241778.069,586440,9.3830
proposed,20261109-082859,H1->H9,H1,H9,10.0.0.2,10.2.0.2,17.5,18.0,18.092,1794241778.569,1629000,26.0640
proposed,20261109-082859,H1->H9,H1,H9,10.0.0.2,10.2.0.2,18.0,18.5,18.592,1794241779.069,912240,14.5958
proposed,20261109-082859,H1->H9,H1,H9,10.0.0.2,10.2.0.2,18.5,19.0,19.092,1794241779.569,1107720,17.7235
proposed,20261109-082859,H1->H9,H1,H9,10.0.0.2,10.2.0.2,19.0,19.5,19.592,1794241780.069,1498680,23.9789
proposed,20261109-082859,H1->H9,H1,H9,10.0.0.2,10.2.0.2,19.5,20.0,20.092,1794241780.569,716760,11.4682
proposed,20261109-082859,H1->H9,H1,H9,10.0.0.2,10.2.0.2,20.0,20.5,20.592,1794241781.069,1368360,21.8938
proposed,20261109-082859,H1->H9,H1,H9,10.0.0.2,10.2.0.2,20.5,21.0,21.092,1794241781.569,1433520,22.9363
proposed,20261109-082859,H1->H9,H1,H9,10.0.0.2,10.2.0.2,21.0,21.5,21.592,1794241782.069,1303200,20.8512
proposed,20261109-082859,H1->H9,H1,H9,10.0.0.2,10.2.0.2,21.5,22.0,22.092,1794241782.569,195480,3.1277
proposed,20261109-082859,H1->H9,H1,H9,10.0.0.2,10.2.0.2,22.0,22.5,22.592,1794241783.069,195480,3.1277
proposed,20261109-082859,H1->H9,H1,H9,10.0.0.2,10.2.0.2,22.5,23.0,23.092,1794241783.569,390960,6.2554
proposed,20261109-082859,H1->H9,H1,H9,10.0.0.2,10.2.0.2,23.0,23.5,23.592,1794241784.069,65160,1.0426
proposed,20261109-082859,H1->H9,H1,H9,10.0.0.2,10.2.0.2,23.5,24.0,24.092,1794241784.569,521280,8.3405
proposed,20261109-082859,H1->H9,H1,H9,10.0.0.2,10.2.0.2,24.0,24.5,24.592,1794241785.069,325800,5.2128
proposed,20261109-082859,H1->H9,H1,H9,10.0.0.2,10.2.0.2,24.5,25.0,25.092,1794241785.569,456120,7.2979
proposed,20261109-082859,H2->H10,H2,H10,10.0.0.3,10.2.0.3,0.0,0.5,3.542,1794241764.019,496664,7.9466
proposed,20261109-082859,H2->H10,H2,H10,10.0.0.3,10.2.0.3,0.5,1.0,4.042,1794241764.519,136112,2.1778
proposed,20261109-082859,H2->H10,H2,H10,10.0.0.3,10.2.0.3,1.0,1.5,4.542,1794241765.019,65160,1.0426
proposed,20261109-082859,H2->H10,H2,H10,10.0.0.3,10.2.0.3,1.5,2.0,5.042,1794241765.519,130320,2.0851
proposed,20261109-082859,H2->H10,H2,H10,10.0.0.3,10.2.0.3,2.0,2.5,5.542,1794241766.019,524288,8.3886
proposed,20261109-082859,H2->H10,H2,H10,10.0.0.3,10.2.0.3,2.5,3.0,6.042,1794241766.519,909232,14.5477
proposed,20261109-082859,H2->H10,H2,H10,10.0.0.3,10.2.0.3,3.0,3.5,6.542,1794241767.019,325800,5.2128
proposed,20261109-082859,H2->H10,H2,H10,10.0.0.3,10.2.0.3,3.5,4.0,7.042,1794241767.519,390960,6.2554
proposed,20261109-082859,H2->H10,H2,H10,10.0.0.3,10.2.0.3,4.0,4.5,7.542,1794241768.019,390960,6.2554
proposed,20261109-082859,H2->H10,H2,H10,10.0.0.3,10.2.0.3,4.5,5.0,8.042,1794241768.519,130320,2.0851
proposed,20261109-082859,H2->H10,H2,H10,10.0.0.3,10.2.0.3,5.0,5.5,8.542,1794241769.019,521280,8.3405
proposed,20261109-082859,H2->H10,H2,H10,10.0.0.3,10.2.0.3,5.5,6.0,9.042,1794241769.519,195480,3.1277
proposed,20261109-082859,H2->H10,H2,H10,10.0.0.3,10.2.0.3,6.0,6.5,9.542,1794241770.019,521280,8.3405
proposed,20261109-082859,H2->H10,H2,H10,10.0.0.3,10.2.0.3,6.5,7.0,10.042,1794241770.519,325800,5.2128
proposed,20261109-082859,H2->H10,H2,H10,10.0.0.3,10.2.0.3,7.0,7.5,10.542,1794241771.019,260640,4.1702
proposed,20261109-082859,H2->H10,H2,H10,10.0.0.3,10.2.0.3,7.5,8.0,11.042,1794241771.519,325800,5.2128
proposed,20261109-082859,H2->H10,H2,H10,10.0.0.3,10.2.0.3,8.0,8.5,11.542,1794241772.019,521280,8.3405
proposed,20261109-082859,H2->H10,H2,H10,10.0.0.3,10.2.0.3,8.5,9.0,12.042,1794241772.519,260640,4.1702
proposed,20261109-082859,H2->H10,H2,H10,10.0.0.3,10.2.0.3,9.0,9.5,12.542,1794241773.019,586440,9.3830
proposed,20261109-082859,H2->H10,H2,H10,10.0.0.3,10.2.0.3,9.5,10.0,13.042,1794241773.519,716760,11.4682
proposed,20261109-082859,H2->H10,H2,H10,10.0.0.3,10.2.0.3,10.0,10.5,13.542,1794241774.019,521280,8.3405
proposed,20261109-082859,H2->H10,H2,H10,10.0.0.3,10.2.0.3,10.5,11.0,14.042,1794241774.519,977400,15.6384
proposed,20261109-082859,H2->H10,H2,H10,10.0.0.3,10.2.0.3,11.0,11.5,14.542,1794241775.019,456120,7.2979
proposed,20261109-082859,H2->H10,H2,H10,10.0.0.3,10.2.0.3,11.5,12.0,15.042,1794241775.519,912240,14.5958
proposed,20261109-082859,H2->H10,H2,H10,10.0.0.3,10.2.0.3,12.0,12.5,15.542,1794241776.019,586440,9.3830
proposed,20261109-082859,H2->H10,H2,H10,10.0.0.3,10.2.0.3,12.5,13.0,16.042,1794241776.519,521280,8.3405
proposed,20261109-082859,H2->H10,H2,H10,10.0.0.3,10.2.0.3,13.0,13.5,16.542,1794241777.019,1172880,18.7661
proposed,20261109-082859,H2->H10,H2,H10,10.0.0.3,10.2.0.3,13.5,14.0,17.042,1794241777.519,586440,9.3830
proposed,20261109-082859,H2->H10,H2,H10,10.0.0.3,10.2.0.3,14.0,14.5,17.542,1794241778.019,651600,10.4256
proposed,20261109-082859,H2->H10,H2,H10,10.0.0.3,10.2.0.3,14.5,15.0,18.042,1794241778.519,781920,12.5107
proposed,20261109-082859,H2->H10,H2,H10,10.0.0.3,10.2.0.3,15.0,15.5,18.542,1794241779.019,977400,15.6384
proposed,20261109-082859,H2->H10,H2,H10,10.0.0.3,10.2.0.3,15.5,16.0,19.042,1794241779.519,651600,10.4256
proposed,20261109-082859,H2->H10,H2,H10,10.0.0.3,10.2.0.3,16.0,16.5,19.542,1794241780.019,1238040,19.8086
proposed,20261109-082859,H2->H10,H2,H10,10.0.0.3,10.2.0.3,16.5,17.0,20.042,1794241780.519,847080,13.5533
proposed,20261109-082859,H2->H10,H2,H10,10.0.0.3,10.2.0.3,17.0,17.5,20.542,1794241781.019,1172880,18.7661
proposed,20261109-082859,H2->H10,H2,H10,10.0.0.3,10.2.0.3,17.5,18.0,21.042,1794241781.519,1368360,21.8938
proposed,20261109-082859,H2->H10,H2,H10,10.0.0.3,10.2.0.3,18.0,18.5,21.542,1794241782.019,1238040,19.8086
proposed,20261109-082859,H2->H10,H2,H10,10.0.0.3,10.2.0.3,18.5,19.0,22.042,1794241782.519,716760,11.4682
proposed,20261109-082859,H2->H10,H2,H10,10.0.0.3,10.2.0.3,19.0,19.5,22.542,1794241783.019,781920,12.5107
proposed,20261109-082859,H2->H10,H2,H10,10.0.0.3,10.2.0.3,19.5,20.0,23.042,1794241783.519,1107720,17.7235
proposed,20261109-082859,H2->H10,H2,H10,10.0.0.3,10.2.0.3,20.0,20.5,23.542,1794241784.019,1238040,19.8086
proposed,20261109-082859,H2->H10,H2,H10,10.0.0.3,10.2.0.3,20.5,21.0,24.042,1794241784.519,1563840,25.0214
proposed,20261109-082859,H2->H10,H2,H10,10.0.0.3,10.2.0.3,21.0,21.5,24.542,1794241785.019,651600,10.4256
proposed,20261109-082859,H2->H10,H2,H10,10.0.0.3,10.2.0.3,21.5,22.0,25.042,1794241785.519,521280,8.3405

```

# results\events.csv

```csv
epoch,scheme,event
1794241659.535,proposed,ELEPHANT 10.0.0.2->10.2.0.2
1794241659.599,proposed,reroute h_0_0_0->h_2_0_0 via c_0_0
1794241659.535,proposed,ELEPHANT 10.0.0.3->10.2.0.3
1794241659.607,proposed,reroute h_0_0_1->h_2_0_1 via c_1_0
1794241690.971,proposed,MICE 10.0.0.2->10.2.0.2
1794241690.971,proposed,MICE 10.0.0.3->10.2.0.3
1794241763.738,proposed,ELEPHANT 10.0.0.2->10.2.0.2
1794241763.777,proposed,reroute h_0_0_0->h_2_0_0 via c_0_0
1794241765.935,proposed,ELEPHANT 10.0.0.3->10.2.0.3
1794241765.98,proposed,reroute h_0_0_1->h_2_0_1 via c_1_0
1794241797.182,proposed,ELEPHANT 10.1.0.2->10.3.1.2
1794241797.273,proposed,reroute h_1_0_0->h_3_1_0 via c_0_0
1794241799.27,proposed,MICE 10.1.0.2->10.3.1.2

```

# results\flow_stats.csv

```csv
epoch,scheme,switch,src_ip,dst_ip,src_port,dst_port,protocol,bytes,rate_mbps,label

```

# results\mice_results.csv

```csv
scheme,run_id,flow_id,src,dst,src_ip,dst_ip,proto,size_bytes,start_epoch,start_offset_s,fct_ms,status
ecmp,20261109-082704,M00000,H1,H12,10.0.0.2,10.2.1.3,tcp,26216,1794241653.326111,3.761805,2290.063,ok
ecmp,20261109-082704,M00004,H1,H9,10.0.0.2,10.2.0.2,tcp,21223,1794241653.537622,3.973315,2028.922,ok
ecmp,20261109-082704,M00006,H1,H13,10.0.0.2,10.3.0.2,tcp,61002,1794241653.572134,4.007827,2351.127,ok
ecmp,20261109-082704,M00032,H1,H11,10.0.0.2,10.2.1.2,tcp,55425,1794241653.572925,4.008618,4085.294,ok
ecmp,20261109-082704,M00038,H1,H10,10.0.0.2,10.2.0.3,tcp,29435,1794241653.599082,4.034775,3088.481,ok
ecmp,20261109-082704,M00063,H1,H2,10.0.0.2,10.0.0.3,tcp,57577,1794241653.607254,4.042948,2997.022,ok
ecmp,20261109-082704,M00066,H1,H7,10.0.0.2,10.1.1.2,tcp,25572,1794241653.635220,4.070913,4199.910,ok
ecmp,20261109-082704,M00084,H1,H11,10.0.0.2,10.2.1.2,tcp,61741,1794241654.226077,4.661771,4424.486,ok
ecmp,20261109-082704,M00133,H1,H15,10.0.0.2,10.3.1.2,tcp,57407,1794241656.770678,7.206371,4528.972,ok
ecmp,20261109-082704,M00148,H1,H14,10.0.0.2,10.3.0.3,tcp,58366,1794241657.312661,7.748355,5603.995,ok
ecmp,20261109-082704,M00161,H1,H12,10.0.0.2,10.2.1.3,tcp,43543,1794241657.782977,8.218670,6236.165,ok
ecmp,20261109-082704,M00191,H1,H2,10.0.0.2,10.0.0.3,tcp,27475,1794241659.234214,9.669907,5118.546,ok
ecmp,20261109-082704,M00228,H1,H2,10.0.0.2,10.0.0.3,tcp,23194,1794241661.149650,11.585344,5570.705,ok
ecmp,20261109-082704,M00229,H1,H13,10.0.0.2,10.3.0.2,tcp,52250,1794241661.310312,11.746005,11253.497,ok
ecmp,20261109-082704,M00239,H1,H11,10.0.0.2,10.2.1.2,tcp,65166,1794241661.761961,12.197654,18068.727,ok
ecmp,20261109-082704,M00244,H1,H11,10.0.0.2,10.2.1.2,tcp,48095,1794241661.956939,12.392632,13184.061,ok
ecmp,20261109-082704,M00254,H1,H5,10.0.0.2,10.1.0.2,tcp,55774,1794241662.505646,12.941339,11107.457,ok
ecmp,20261109-082704,M00264,H1,H12,10.0.0.2,10.2.1.3,tcp,17639,1794241663.086473,13.522166,10011.349,timeout
ecmp,20261109-082704,M00274,H1,H16,10.0.0.2,10.3.1.3,tcp,29450,1794241663.604568,14.040261,15410.073,ok
ecmp,20261109-082704,M00308,H1,H10,10.0.0.2,10.2.0.3,tcp,44984,1794241665.209693,15.645386,10014.261,timeout
ecmp,20261109-082704,M00323,H1,H16,10.0.0.2,10.3.1.3,tcp,59755,1794241665.947559,16.383252,10017.909,timeout
ecmp,20261109-082704,M00327,H1,H5,10.0.0.2,10.1.0.2,tcp,64448,1794241666.064760,16.500454,10010.342,timeout
ecmp,20261109-082704,M00330,H1,H6,10.0.0.2,10.1.0.3,tcp,59964,1794241666.354681,16.790375,10016.605,timeout
ecmp,20261109-082704,M00358,H1,H10,10.0.0.2,10.2.0.3,tcp,51303,1794241667.744195,18.179889,10012.095,timeout
ecmp,20261109-082704,M00363,H1,H2,10.0.0.2,10.0.0.3,tcp,57089,1794241667.974266,18.409960,10011.384,timeout
ecmp,20261109-082704,M00001,H5,H12,10.1.0.2,10.2.1.3,tcp,14909,1794241653.093467,3.529160,1939.856,ok
ecmp,20261109-082704,M00027,H5,H9,10.1.0.2,10.2.0.2,tcp,40535,1794241653.195858,3.631551,1904.952,ok
ecmp,20261109-082704,M00043,H5,H11,10.1.0.2,10.2.1.2,tcp,39340,1794241653.211385,3.647078,1751.239,ok
ecmp,20261109-082704,M00094,H5,H9,10.1.0.2,10.2.0.2,tcp,10553,1794241654.537225,4.972918,4832.694,ok
ecmp,20261109-082704,M00095,H5,H7,10.1.0.2,10.1.1.2,tcp,16544,1794241654.625590,5.061284,3387.826,ok
ecmp,20261109-082704,M00120,H5,H4,10.1.0.2,10.0.1.3,tcp,35746,1794241656.053709,6.489403,4881.049,ok
ecmp,20261109-082704,M00145,H5,H12,10.1.0.2,10.2.1.3,tcp,22213,1794241657.250202,7.685895,5329.681,ok
ecmp,20261109-082704,M00154,H5,H10,10.1.0.2,10.2.0.3,tcp,43196,1794241657.521438,7.957132,4864.915,ok
ecmp,20261109-082704,M00178,H5,H7,10.1.0.2,10.1.1.2,tcp,19735,1794241658.502654,8.938348,7893.467,ok
ecmp,20261109-082704,M00201,H5,H16,10.1.0.2,10.3.1.3,tcp,11174,1794241659.705052,10.140746,6948.871,ok
ecmp,20261109-082704,M00222,H5,H15,10.1.0.2,10.3.1.2,tcp,25600,1794241660.715654,11.151347,5730.049,ok
ecmp,20261109-082704,M00248,H5,H8,10.1.0.2,10.1.1.3,tcp,55155,1794241662.332215,12.767908,10012.093,timeout
ecmp,20261109-082704,M00252,H5,H14,10.1.0.2,10.3.0.3,tcp,12767,1794241662.450946,12.886639,6220.921,ok
ecmp,20261109-082704,M00261,H5,H8,10.1.0.2,10.1.1.3,tcp,10618,1794241662.852601,13.288294,10012.362,timeout
ecmp,20261109-082704,M00265,H5,H2,10.1.0.2,10.0.0.3,tcp,38960,1794241663.113474,13.549167,6916.144,ok
ecmp,20261109-082704,M00279,H5,H4,10.1.0.2,10.0.1.3,tcp,38413,1794241663.760905,14.196599,10018.436,timeout
ecmp,20261109-082704,M00293,H5,H11,10.1.0.2,10.2.1.2,tcp,14559,1794241664.438796,14.874490,10009.975,timeout
ecmp,20261109-082704,M00324,H5,H13,10.1.0.2,10.3.0.2,tcp,45192,1794241665.969000,16.404694,10017.090,timeout
ecmp,20261109-082704,M00328,H5,H9,10.1.0.2,10.2.0.2,tcp,24618,1794241666.277224,16.712917,10010.654,timeout
ecmp,20261109-082704,M00332,H5,H12,10.1.0.2,10.2.1.3,tcp,35780,1794241666.445064,16.880757,10018.693,timeout
ecmp,20261109-082704,M00340,H5,H2,10.1.0.2,10.0.0.3,tcp,16462,1794241667.022475,17.458169,10015.127,timeout
ecmp,20261109-082704,M00348,H5,H11,10.1.0.2,10.2.1.2,tcp,39273,1794241667.343733,17.779426,10017.093,timeout
ecmp,20261109-082704,M00389,H5,H4,10.1.0.2,10.0.1.3,tcp,14929,1794241668.924459,19.360153,10009.718,timeout
ecmp,20261109-082704,M00002,H3,H10,10.0.1.2,10.2.0.3,tcp,35843,1794241653.089258,3.524951,1872.347,ok
ecmp,20261109-082704,M00003,H3,H4,10.0.1.2,10.0.1.3,tcp,23439,1794241653.289912,3.725605,2024.360,ok
ecmp,20261109-082704,M00014,H3,H14,10.0.1.2,10.3.0.3,tcp,23448,1794241653.290694,3.726388,2325.679,ok
ecmp,20261109-082704,M00030,H3,H7,10.0.1.2,10.1.1.2,tcp,33201,1794241653.336548,3.772242,2199.448,ok
ecmp,20261109-082704,M00041,H3,H2,10.0.1.2,10.0.0.3,tcp,56158,1794241653.394707,3.830401,2118.592,ok
ecmp,20261109-082704,M00042,H3,H13,10.0.1.2,10.3.0.2,tcp,43103,1794241653.408688,3.844382,2395.925,ok
ecmp,20261109-082704,M00075,H3,H7,10.0.1.2,10.1.1.2,tcp,51273,1794241653.776851,4.212544,4132.017,ok
ecmp,20261109-082704,M00079,H3,H1,10.0.1.2,10.0.0.2,tcp,38226,1794241653.993755,4.429449,4083.656,ok
ecmp,20261109-082704,M00144,H3,H16,10.0.1.2,10.3.1.3,tcp,18076,1794241657.232232,7.667925,6200.639,ok
ecmp,20261109-082704,M00156,H3,H11,10.0.1.2,10.2.1.2,tcp,36281,1794241657.584055,8.019748,6768.850,ok
ecmp,20261109-082704,M00175,H3,H9,10.0.1.2,10.2.0.2,tcp,50629,1794241658.439792,8.875485,10015.907,timeout
ecmp,20261109-082704,M00182,H3,H5,10.0.1.2,10.1.0.2,tcp,37735,1794241658.795822,9.231516,9146.567,ok
ecmp,20261109-082704,M00223,H3,H1,10.0.1.2,10.0.0.2,tcp,19065,1794241660.807676,11.243370,10013.408,timeout
ecmp,20261109-082704,M00233,H3,H8,10.0.1.2,10.1.1.3,tcp,30995,1794241661.455274,11.890968,10010.671,timeout
ecmp,20261109-082704,M00267,H3,H14,10.0.1.2,10.3.0.3,tcp,29724,1794241663.257218,13.692912,10010.213,timeout
ecmp,20261109-082704,M00270,H3,H4,10.0.1.2,10.0.1.3,tcp,49555,1794241663.408959,13.844652,10010.126,timeout
ecmp,20261109-082704,M00304,H3,H13,10.0.1.2,10.3.0.2,tcp,34050,1794241665.055485,15.491179,10012.071,timeout
ecmp,20261109-082704,M00333,H3,H8,10.0.1.2,10.1.1.3,tcp,37570,1794241666.490440,16.926134,10015.096,timeout
ecmp,20261109-082704,M00339,H3,H15,10.0.1.2,10.3.1.2,tcp,13486,1794241667.012589,17.448283,10013.735,timeout
ecmp,20261109-082704,M00349,H3,H2,10.0.1.2,10.0.0.3,tcp,36495,1794241667.344811,17.780505,10016.441,timeout
ecmp,20261109-082704,M00366,H3,H15,10.0.1.2,10.3.1.2,tcp,11639,1794241668.059592,18.495285,10020.063,timeout
ecmp,20261109-082704,M00379,H3,H14,10.0.1.2,10.3.0.3,tcp,28708,1794241668.411229,18.846922,10010.282,timeout
ecmp,20261109-082704,M00385,H3,H7,10.0.1.2,10.1.1.2,tcp,65242,1794241668.801034,19.236727,10012.883,timeout
ecmp,20261109-082704,M00387,H3,H9,10.0.1.2,10.2.0.2,tcp,49691,1794241668.893669,19.329362,10010.274,timeout
ecmp,20261109-082704,M00391,H3,H16,10.0.1.2,10.3.1.3,tcp,58841,1794241668.947339,19.383032,10021.124,timeout
ecmp,20261109-082704,M00400,H3,H10,10.0.1.2,10.2.0.3,tcp,47125,1794241669.351952,19.787645,10014.605,timeout
ecmp,20261109-082704,M00005,H14,H4,10.3.0.3,10.0.1.3,tcp,37631,1794241653.669615,4.105309,2553.095,ok
ecmp,20261109-082704,M00007,H14,H6,10.3.0.3,10.1.0.3,tcp,26402,1794241653.716586,4.152280,2432.065,ok
ecmp,20261109-082704,M00045,H14,H4,10.3.0.3,10.0.1.3,tcp,43535,1794241653.741604,4.177298,2481.304,ok
ecmp,20261109-082704,M00054,H14,H4,10.3.0.3,10.0.1.3,tcp,14373,1794241653.764023,4.199717,2363.186,ok
ecmp,20261109-082704,M00055,H14,H6,10.3.0.3,10.1.0.3,tcp,35951,1794241653.764764,4.200458,2404.242,ok
ecmp,20261109-082704,M00090,H14,H16,10.3.0.3,10.3.1.3,tcp,26324,1794241654.457207,4.892901,2663.785,ok
ecmp,20261109-082704,M00093,H14,H9,10.3.0.3,10.2.0.2,tcp,8825,1794241654.535165,4.970858,4355.262,ok
ecmp,20261109-082704,M00099,H14,H10,10.3.0.3,10.2.0.3,tcp,57309,1794241654.838875,5.274568,3810.239,ok
ecmp,20261109-082704,M00119,H14,H6,10.3.0.3,10.1.0.3,tcp,34563,1794241656.004622,6.440315,2949.374,ok
ecmp,20261109-082704,M00138,H14,H16,10.3.0.3,10.3.1.3,tcp,44097,1794241657.019914,7.455608,5439.353,ok
ecmp,20261109-082704,M00146,H14,H6,10.3.0.3,10.1.0.3,tcp,43752,1794241657.250762,7.686455,4194.987,ok
ecmp,20261109-082704,M00160,H14,H5,10.3.0.3,10.1.0.2,tcp,62908,1794241657.769073,8.204767,4354.439,ok
ecmp,20261109-082704,M00176,H14,H16,10.3.0.3,10.3.1.3,tcp,40317,1794241658.461354,8.897047,4959.850,ok
ecmp,20261109-082704,M00210,H14,H10,10.3.0.3,10.2.0.3,tcp,47259,1794241660.247734,10.683428,10010.199,timeout
ecmp,20261109-082704,M00213,H14,H5,10.3.0.3,10.1.0.2,tcp,45457,1794241660.393759,10.829453,4920.696,ok
ecmp,20261109-082704,M00217,H14,H8,10.3.0.3,10.1.1.3,tcp,53370,1794241660.505251,10.940945,10009.981,timeout
ecmp,20261109-082704,M00227,H14,H12,10.3.0.3,10.2.1.3,tcp,29315,1794241661.108846,11.544540,7427.808,ok
ecmp,20261109-082704,M00232,H14,H11,10.3.0.3,10.2.1.2,tcp,61554,1794241661.424044,11.859737,8017.038,ok
ecmp,20261109-082704,M00240,H14,H1,10.3.0.3,10.0.0.2,tcp,20485,1794241661.802708,12.238402,12968.908,ok
ecmp,20261109-082704,M00246,H14,H2,10.3.0.3,10.0.0.3,tcp,22917,1794241662.031606,12.467300,8464.887,ok
ecmp,20261109-082704,M00253,H14,H6,10.3.0.3,10.1.0.3,tcp,59815,1794241662.468075,12.903769,6353.983,ok
ecmp,20261109-082704,M00259,H14,H1,10.3.0.3,10.0.0.2,tcp,29044,1794241662.709296,13.144989,10011.944,timeout
ecmp,20261109-082704,M00311,H14,H2,10.3.0.3,10.0.0.3,tcp,17352,1794241665.299657,15.735351,10020.100,timeout
ecmp,20261109-082704,M00315,H14,H10,10.3.0.3,10.2.0.3,tcp,56837,1794241665.469721,15.905414,10010.248,timeout
ecmp,20261109-082704,M00316,H14,H11,10.3.0.3,10.2.1.2,tcp,14681,1794241665.532403,15.968096,10010.982,timeout
ecmp,20261109-082704,M00317,H14,H5,10.3.0.3,10.1.0.2,tcp,10336,1794241665.617342,16.053036,10010.698,timeout
ecmp,20261109-082704,M00345,H14,H6,10.3.0.3,10.1.0.3,tcp,62417,1794241667.267463,17.703157,10012.867,timeout
ecmp,20261109-082704,M00346,H14,H12,10.3.0.3,10.2.1.3,tcp,14423,1794241667.305809,17.741502,10013.252,timeout
ecmp,20261109-082704,M00352,H14,H16,10.3.0.3,10.3.1.3,tcp,55447,1794241667.414557,17.850250,10009.457,timeout
ecmp,20261109-082704,M00357,H14,H16,10.3.0.3,10.3.1.3,tcp,44955,1794241667.555747,17.991440,10013.378,timeout
ecmp,20261109-082704,M00368,H14,H8,10.3.0.3,10.1.1.3,tcp,47977,1794241668.080264,18.515958,10019.922,timeout
ecmp,20261109-082704,M00008,H11,H2,10.2.1.2,10.0.0.3,tcp,14270,1794241653.636958,4.072652,2695.833,ok
ecmp,20261109-082704,M00022,H11,H7,10.2.1.2,10.1.1.2,tcp,25738,1794241653.683541,4.119235,2892.473,ok
ecmp,20261109-082704,M00023,H11,H4,10.2.1.2,10.0.1.3,tcp,51146,1794241653.729105,4.164798,4006.426,ok
ecmp,20261109-082704,M00051,H11,H2,10.2.1.2,10.0.0.3,tcp,41887,1794241653.729593,4.165287,5541.880,ok
ecmp,20261109-082704,M00070,H11,H1,10.2.1.2,10.0.0.2,tcp,11478,1794241653.753125,4.188818,5231.443,ok
ecmp,20261109-082704,M00076,H11,H15,10.2.1.2,10.3.1.2,tcp,25281,1794241653.825221,4.260915,3635.266,ok
ecmp,20261109-082704,M00077,H11,H4,10.2.1.2,10.0.1.3,tcp,25599,1794241653.836370,4.272064,3790.375,ok
ecmp,20261109-082704,M00091,H11,H13,10.2.1.2,10.3.0.2,tcp,16765,1794241654.455957,4.891650,2582.737,ok
ecmp,20261109-082704,M00102,H11,H13,10.2.1.2,10.3.0.2,tcp,35174,1794241655.043999,5.479692,2791.456,ok
ecmp,20261109-082704,M00110,H11,H1,10.2.1.2,10.0.0.2,tcp,15751,1794241655.554583,5.990276,3368.385,ok
ecmp,20261109-082704,M00117,H11,H10,10.2.1.2,10.2.0.3,tcp,28764,1794241655.908355,6.344049,6110.616,ok
ecmp,20261109-082704,M00124,H11,H8,10.2.1.2,10.1.1.3,tcp,37144,1794241656.343029,6.778722,7690.729,ok
ecmp,20261109-082704,M00127,H11,H2,10.2.1.2,10.0.0.3,tcp,61828,1794241656.464823,6.900517,5108.875,ok
ecmp,20261109-082704,M00137,H11,H13,10.2.1.2,10.3.0.2,tcp,37196,1794241657.066123,7.501817,5585.183,ok
ecmp,20261109-082704,M00158,H11,H4,10.2.1.2,10.0.1.3,tcp,37991,1794241657.604875,8.040569,6324.092,ok
ecmp,20261109-082704,M00184,H11,H1,10.2.1.2,10.0.0.2,tcp,40586,1794241658.860479,9.296172,5614.372,ok
ecmp,20261109-082704,M00194,H11,H6,10.2.1.2,10.1.0.3,tcp,54254,1794241659.307482,9.743176,5079.806,ok
ecmp,20261109-082704,M00234,H11,H16,10.2.1.2,10.3.1.3,tcp,15047,1794241661.476551,11.912244,7293.718,ok
ecmp,20261109-082704,M00235,H11,H7,10.2.1.2,10.1.1.2,tcp,53652,1794241661.573174,12.008868,10009.849,timeout
ecmp,20261109-082704,M00262,H11,H14,10.2.1.2,10.3.0.3,tcp,14769,1794241662.859727,13.295420,10014.952,timeout
ecmp,20261109-082704,M00331,H11,H1,10.2.1.2,10.0.0.2,tcp,19611,1794241666.433631,16.869324,10008.488,timeout
ecmp,20261109-082704,M00341,H11,H7,10.2.1.2,10.1.1.2,tcp,47342,1794241667.043533,17.479226,10014.169,timeout
ecmp,20261109-082704,M00347,H11,H16,10.2.1.2,10.3.1.3,tcp,24896,1794241667.315744,17.751438,10013.620,timeout
ecmp,20261109-082704,M00351,H11,H16,10.2.1.2,10.3.1.3,tcp,16202,1794241667.390559,17.826253,10012.748,timeout
ecmp,20261109-082704,M00386,H11,H4,10.2.1.2,10.0.1.3,tcp,30019,1794241668.820716,19.256410,10011.909,timeout
ecmp,20261109-082704,M00394,H11,H14,10.2.1.2,10.3.0.3,tcp,49297,1794241669.112454,19.548148,5150.560,error
ecmp,20261109-082704,M00009,H12,H14,10.2.1.3,10.3.0.3,tcp,30733,1794241653.484919,3.920612,2769.255,ok
ecmp,20261109-082704,M00012,H12,H10,10.2.1.3,10.2.0.3,tcp,20793,1794241653.672897,4.108590,2549.568,ok
ecmp,20261109-082704,M00016,H12,H3,10.2.1.3,10.0.1.2,tcp,32452,1794241653.673327,4.109021,2484.054,ok
ecmp,20261109-082704,M00047,H12,H8,10.2.1.3,10.1.1.3,tcp,42111,1794241653.673956,4.109649,2930.046,ok
ecmp,20261109-082704,M00096,H12,H15,10.2.1.3,10.3.1.2,tcp,60366,1794241654.630439,5.066132,2683.483,ok
ecmp,20261109-082704,M00113,H12,H16,10.2.1.3,10.3.1.3,tcp,59731,1794241655.878247,6.313941,4856.734,ok
ecmp,20261109-082704,M00162,H12,H4,10.2.1.3,10.0.1.3,tcp,50799,1794241657.783504,8.219197,7447.116,ok
ecmp,20261109-082704,M00167,H12,H3,10.2.1.3,10.0.1.2,tcp,47899,1794241658.047669,8.483363,7276.214,ok
ecmp,20261109-082704,M00174,H12,H9,10.2.1.3,10.2.0.2,tcp,36266,1794241658.385625,8.821318,10010.588,timeout
ecmp,20261109-082704,M00177,H12,H11,10.2.1.3,10.2.1.2,tcp,62502,1794241658.470936,8.906630,5345.177,ok
ecmp,20261109-082704,M00291,H12,H4,10.2.1.3,10.0.1.3,tcp,49913,1794241664.367000,14.802694,10018.888,timeout
ecmp,20261109-082704,M00295,H12,H16,10.2.1.3,10.3.1.3,tcp,16804,1794241664.584758,15.020452,9449.027,ok
ecmp,20261109-082704,M00298,H12,H9,10.2.1.3,10.2.0.2,tcp,41077,1794241664.660699,15.096392,10010.418,timeout
ecmp,20261109-082704,M00307,H12,H2,10.2.1.3,10.0.0.3,tcp,52418,1794241665.199522,15.635216,10021.584,timeout
ecmp,20261109-082704,M00319,H12,H2,10.2.1.3,10.0.0.3,tcp,53128,1794241665.831461,16.267155,10019.324,timeout
ecmp,20261109-082704,M00320,H12,H1,10.2.1.3,10.0.0.2,tcp,34284,1794241665.853999,16.289693,10017.989,timeout
ecmp,20261109-082704,M00350,H12,H13,10.2.1.3,10.3.0.2,tcp,16719,1794241667.348992,17.784686,10012.687,timeout
ecmp,20261109-082704,M00365,H12,H5,10.2.1.3,10.1.0.2,tcp,55482,1794241668.039511,18.475205,10018.991,timeout
ecmp,20261109-082704,M00370,H12,H7,10.2.1.3,10.1.1.2,tcp,15459,1794241668.144688,18.580382,10015.276,timeout
ecmp,20261109-082704,M00377,H12,H2,10.2.1.3,10.0.0.3,tcp,34214,1794241668.286676,18.722369,10011.960,timeout
ecmp,20261109-082704,M00383,H12,H2,10.2.1.3,10.0.0.3,tcp,36812,1794241668.685074,19.120768,10011.585,timeout
ecmp,20261109-082704,M00397,H12,H9,10.2.1.3,10.2.0.2,tcp,13054,1794241669.204898,19.640591,10014.644,timeout
ecmp,20261109-082704,M00010,H2,H12,10.0.0.3,10.2.1.3,tcp,38300,1794241654.014245,4.449939,3153.067,ok
ecmp,20261109-082704,M00013,H2,H11,10.0.0.3,10.2.1.2,tcp,23127,1794241654.015494,4.451188,3475.782,ok
ecmp,20261109-082704,M00021,H2,H4,10.0.0.3,10.0.1.3,tcp,62049,1794241654.113211,4.548904,3364.009,ok
ecmp,20261109-082704,M00028,H2,H14,10.0.0.3,10.3.0.3,tcp,15377,1794241654.141226,4.576920,3037.434,ok
ecmp,20261109-082704,M00050,H2,H4,10.0.0.3,10.0.1.3,tcp,12609,1794241654.149802,4.585495,2862.433,ok
ecmp,20261109-082704,M00056,H2,H11,10.0.0.3,10.2.1.2,tcp,51016,1794241654.161736,4.597430,3884.776,ok
ecmp,20261109-082704,M00069,H2,H10,10.0.0.3,10.2.0.3,tcp,56409,1794241654.162788,4.598482,3958.066,ok
ecmp,20261109-082704,M00097,H2,H15,10.0.0.3,10.3.1.2,tcp,31641,1794241654.809044,5.244738,3558.967,ok
ecmp,20261109-082704,M00115,H2,H12,10.0.0.3,10.2.1.3,tcp,36769,1794241655.904509,6.340202,3455.582,ok
ecmp,20261109-082704,M00147,H2,H4,10.0.0.3,10.0.1.3,tcp,62776,1794241657.277817,7.713510,7986.211,ok
ecmp,20261109-082704,M00163,H2,H13,10.0.0.3,10.3.0.2,tcp,10225,1794241657.783902,8.219595,4660.466,ok
ecmp,20261109-082704,M00208,H2,H3,10.0.0.3,10.0.1.2,tcp,17969,1794241660.164258,10.599952,9468.225,ok
ecmp,20261109-082704,M00230,H2,H13,10.0.0.3,10.3.0.2,tcp,57754,1794241661.403319,11.839012,14635.781,ok
ecmp,20261109-082704,M00289,H2,H15,10.0.0.3,10.3.1.2,tcp,47933,1794241664.284048,14.719741,10018.219,timeout
ecmp,20261109-082704,M00294,H2,H5,10.0.0.3,10.1.0.2,tcp,59874,1794241664.553999,14.989693,10010.615,timeout
ecmp,20261109-082704,M00312,H2,H5,10.0.0.3,10.1.0.2,tcp,40476,1794241665.300181,15.735875,15141.627,ok
ecmp,20261109-082704,M00335,H2,H12,10.0.0.3,10.2.1.3,tcp,59521,1794241666.571074,17.006767,10008.116,timeout
ecmp,20261109-082704,M00337,H2,H8,10.0.0.3,10.1.1.3,tcp,63720,1794241666.856728,17.292421,10010.704,timeout
ecmp,20261109-082704,M00367,H2,H11,10.0.0.3,10.2.1.2,tcp,13538,1794241668.062198,18.497891,10017.770,timeout
ecmp,20261109-082704,M00378,H2,H7,10.0.0.3,10.1.1.2,tcp,9420,1794241668.383128,18.818821,10011.391,timeout
ecmp,20261109-082704,M00402,H2,H8,10.0.0.3,10.1.1.3,tcp,61264,1794241669.540464,19.976158,10015.659,timeout
ecmp,20261109-082704,M00011,H13,H2,10.3.0.2,10.0.0.3,tcp,44370,1794241653.941809,4.377503,3535.654,ok
ecmp,20261109-082704,M00015,H13,H5,10.3.0.2,10.1.0.2,tcp,37906,1794241654.002926,4.438620,3471.138,ok
ecmp,20261109-082704,M00026,H13,H6,10.3.0.2,10.1.0.3,tcp,22565,1794241654.003708,4.439402,2558.729,ok
ecmp,20261109-082704,M00065,H13,H15,10.3.0.2,10.3.1.2,tcp,12034,1794241654.024057,4.459751,3251.531,ok
ecmp,20261109-082704,M00108,H13,H6,10.3.0.2,10.1.0.3,tcp,26450,1794241655.423474,5.859167,3012.131,ok
ecmp,20261109-082704,M00121,H13,H11,10.3.0.2,10.2.1.2,tcp,57221,1794241656.105863,6.541557,4988.453,ok
ecmp,20261109-082704,M00132,H13,H8,10.3.0.2,10.1.1.3,tcp,34383,1794241656.749879,7.185572,7076.384,ok
ecmp,20261109-082704,M00150,H13,H14,10.3.0.2,10.3.0.3,tcp,62962,1794241657.374801,7.810495,4215.877,ok
ecmp,20261109-082704,M00153,H13,H12,10.3.0.2,10.2.1.3,tcp,19008,1794241657.442041,7.877734,5488.500,ok
ecmp,20261109-082704,M00155,H13,H10,10.3.0.2,10.2.0.3,tcp,45179,1794241657.521832,7.957525,6490.208,ok
ecmp,20261109-082704,M00159,H13,H5,10.3.0.2,10.1.0.2,tcp,57475,1794241657.625681,8.061375,4920.134,ok
ecmp,20261109-082704,M00171,H13,H7,10.3.0.2,10.1.1.2,tcp,55051,1794241658.232756,8.668449,10010.930,timeout
ecmp,20261109-082704,M00236,H13,H14,10.3.0.2,10.3.0.3,tcp,58032,1794241661.612116,12.047809,17199.621,ok
ecmp,20261109-082704,M00251,H13,H4,10.3.0.2,10.0.1.3,tcp,47472,1794241662.416142,12.851835,10011.030,timeout
ecmp,20261109-082704,M00257,H13,H8,10.3.0.2,10.1.1.3,tcp,29280,1794241662.591404,13.027097,10012.127,timeout
ecmp,20261109-082704,M00260,H13,H7,10.3.0.2,10.1.1.2,tcp,51687,1794241662.783866,13.219559,10011.252,timeout
ecmp,20261109-082704,M00268,H13,H6,10.3.0.2,10.1.0.3,tcp,49269,1794241663.350484,13.786178,9407.010,ok
ecmp,20261109-082704,M00278,H13,H10,10.3.0.2,10.2.0.3,tcp,52961,1794241663.745568,14.181261,10013.186,timeout
ecmp,20261109-082704,M00285,H13,H14,10.3.0.2,10.3.0.3,tcp,39859,1794241664.078778,14.514472,10014.870,timeout
ecmp,20261109-082704,M00286,H13,H10,10.3.0.2,10.2.0.3,tcp,31716,1794241664.083749,14.519442,10020.530,timeout
ecmp,20261109-082704,M00303,H13,H14,10.3.0.2,10.3.0.3,tcp,60921,1794241665.015175,15.450869,10016.451,timeout
ecmp,20261109-082704,M00362,H13,H2,10.3.0.2,10.0.0.3,tcp,17489,1794241667.910535,18.346229,10010.653,timeout
ecmp,20261109-082704,M00364,H13,H7,10.3.0.2,10.1.1.2,tcp,37922,1794241668.019221,18.454915,10018.832,timeout
ecmp,20261109-082704,M00376,H13,H9,10.3.0.2,10.2.0.2,tcp,54307,1794241668.267443,18.703136,10010.085,timeout
ecmp,20261109-082704,M00380,H13,H12,10.3.0.2,10.2.1.3,tcp,50144,1794241668.456457,18.892151,10020.157,timeout
ecmp,20261109-082704,M00017,H9,H12,10.2.0.2,10.2.1.3,tcp,52988,1794241654.399445,4.835139,2853.113,ok
ecmp,20261109-082704,M00025,H9,H12,10.2.0.2,10.2.1.3,tcp,46503,1794241654.524289,4.959982,2789.438,ok
ecmp,20261109-082704,M00033,H9,H13,10.2.0.2,10.3.0.2,tcp,50198,1794241654.545497,4.981191,2601.540,ok
ecmp,20261109-082704,M00035,H9,H16,10.2.0.2,10.3.1.3,tcp,58127,1794241654.576644,5.012338,2821.498,ok
ecmp,20261109-082704,M00081,H9,H3,10.2.0.2,10.0.1.2,tcp,31064,1794241654.583445,5.019139,5648.509,ok
ecmp,20261109-082704,M00092,H9,H3,10.2.0.2,10.0.1.2,tcp,56769,1794241654.632523,5.068216,6722.573,ok
ecmp,20261109-082704,M00109,H9,H6,10.2.0.2,10.1.0.3,tcp,50232,1794241655.523998,5.959692,4848.904,ok
ecmp,20261109-082704,M00111,H9,H3,10.2.0.2,10.0.1.2,tcp,46241,1794241655.668571,6.104265,5498.165,ok
ecmp,20261109-082704,M00112,H9,H1,10.2.0.2,10.0.0.2,tcp,15296,1794241655.825051,6.260744,3716.100,ok
ecmp,20261109-082704,M00140,H9,H13,10.2.0.2,10.3.0.2,tcp,24395,1794241657.076841,7.512534,5575.246,ok
ecmp,20261109-082704,M00141,H9,H13,10.2.0.2,10.3.0.2,tcp,59157,1794241657.177103,7.612797,4947.952,ok
ecmp,20261109-082704,M00179,H9,H10,10.2.0.2,10.2.0.3,tcp,61158,1794241658.565897,9.001590,8197.472,ok
ecmp,20261109-082704,M00181,H9,H6,10.2.0.2,10.1.0.3,tcp,64009,1794241658.785947,9.221641,5197.545,ok
ecmp,20261109-082704,M00186,H9,H6,10.2.0.2,10.1.0.3,tcp,26519,1794241658.976603,9.412296,5434.569,ok
ecmp,20261109-082704,M00187,H9,H16,10.2.0.2,10.3.1.3,tcp,8857,1794241659.089299,9.524992,6157.183,ok
ecmp,20261109-082704,M00198,H9,H12,10.2.0.2,10.2.1.3,tcp,46831,1794241659.478840,9.914534,7403.433,ok
ecmp,20261109-082704,M00215,H9,H11,10.2.0.2,10.2.1.2,tcp,13512,1794241660.489538,10.925231,10015.083,timeout
ecmp,20261109-082704,M00249,H9,H7,10.2.0.2,10.1.1.2,tcp,62896,1794241662.353374,12.789068,10011.317,timeout
ecmp,20261109-082704,M00266,H9,H6,10.2.0.2,10.1.0.3,tcp,49037,1794241663.191226,13.626919,10015.924,timeout
ecmp,20261109-082704,M00288,H9,H1,10.2.0.2,10.0.0.2,tcp,45280,1794241664.189426,14.625120,10009.177,timeout
ecmp,20261109-082704,M00292,H9,H11,10.2.0.2,10.2.1.2,tcp,57709,1794241664.373458,14.809152,10012.591,timeout
ecmp,20261109-082704,M00306,H9,H10,10.2.0.2,10.2.0.3,tcp,33162,1794241665.115861,15.551554,10011.846,timeout
ecmp,20261109-082704,M00359,H9,H1,10.2.0.2,10.0.0.2,tcp,20015,1794241667.829380,18.265073,10011.236,timeout
ecmp,20261109-082704,M00369,H9,H4,10.2.0.2,10.0.1.3,tcp,57667,1794241668.111300,18.546994,10010.505,timeout
ecmp,20261109-082704,M00374,H9,H5,10.2.0.2,10.1.0.2,tcp,29695,1794241668.252533,18.688227,10010.547,timeout
ecmp,20261109-082704,M00018,H6,H9,10.1.0.3,10.2.0.2,tcp,55976,1794241654.778360,5.214054,2947.136,ok
ecmp,20261109-082704,M00029,H6,H13,10.1.0.3,10.3.0.2,tcp,52788,1794241654.813022,5.248716,3088.023,ok
ecmp,20261109-082704,M00044,H6,H5,10.1.0.3,10.1.0.2,tcp,42773,1794241654.825068,5.260761,2238.192,ok
ecmp,20261109-082704,M00064,H6,H7,10.1.0.3,10.1.1.2,tcp,40018,1794241654.891866,5.327559,5398.267,ok
ecmp,20261109-082704,M00071,H6,H1,10.1.0.3,10.0.0.2,tcp,41473,1794241655.004804,5.440497,3668.579,ok
ecmp,20261109-082704,M00072,H6,H2,10.1.0.3,10.0.0.3,tcp,47188,1794241655.017180,5.452874,3663.223,ok
ecmp,20261109-082704,M00101,H6,H15,10.1.0.3,10.3.1.2,tcp,35212,1794241655.034236,5.469930,2241.221,ok
ecmp,20261109-082704,M00126,H6,H11,10.1.0.3,10.2.1.2,tcp,13749,1794241656.449116,6.884810,5111.771,ok
ecmp,20261109-082704,M00139,H6,H12,10.1.0.3,10.2.1.3,tcp,64650,1794241657.059355,7.495049,5514.561,ok
ecmp,20261109-082704,M00185,H6,H8,10.1.0.3,10.1.1.3,tcp,22093,1794241658.959561,9.395255,10016.430,timeout
ecmp,20261109-082704,M00197,H6,H4,10.1.0.3,10.0.1.3,tcp,22372,1794241659.423302,9.858996,7913.903,ok
ecmp,20261109-082704,M00205,H6,H1,10.1.0.3,10.0.0.2,tcp,24738,1794241659.966074,10.401767,5428.081,ok
ecmp,20261109-082704,M00216,H6,H9,10.1.0.3,10.2.0.2,tcp,13111,1794241660.496232,10.931926,8285.662,ok
ecmp,20261109-082704,M00245,H6,H5,10.1.0.3,10.1.0.2,tcp,44291,1794241662.031841,12.467534,4369.993,ok
ecmp,20261109-082704,M00277,H6,H3,10.1.0.3,10.0.1.2,tcp,19665,1794241663.739093,14.174786,10019.442,timeout
ecmp,20261109-082704,M00282,H6,H2,10.1.0.3,10.0.0.3,tcp,37141,1794241663.863620,14.299313,18934.618,timeout
ecmp,20261109-082704,M00299,H6,H5,10.1.0.3,10.1.0.2,tcp,62238,1794241664.787021,15.222715,7964.414,ok
ecmp,20261109-082704,M00361,H6,H14,10.1.0.3,10.3.0.3,tcp,17580,1794241667.866644,18.302338,10008.821,timeout
ecmp,20261109-082704,M00382,H6,H13,10.1.0.3,10.3.0.2,tcp,48701,1794241668.657528,19.093222,10018.287,timeout
ecmp,20261109-082704,M00019,H15,H7,10.3.1.2,10.1.1.2,tcp,25883,1794241654.005558,4.441252,4972.863,ok
ecmp,20261109-082704,M00024,H15,H3,10.3.1.2,10.0.1.2,tcp,25551,1794241654.034585,4.470279,3968.561,ok
ecmp,20261109-082704,M00031,H15,H9,10.3.1.2,10.2.0.2,tcp,24668,1794241654.076834,4.512527,3958.474,ok
ecmp,20261109-082704,M00059,H15,H3,10.3.1.2,10.0.1.2,tcp,35840,1794241654.078097,4.513791,3967.849,ok
ecmp,20261109-082704,M00060,H15,H4,10.3.1.2,10.0.1.3,tcp,65502,1794241654.108181,4.543874,4190.622,ok
ecmp,20261109-082704,M00061,H15,H13,10.3.1.2,10.3.0.2,tcp,64641,1794241654.112232,4.547926,3547.358,ok
ecmp,20261109-082704,M00067,H15,H5,10.3.1.2,10.1.0.2,tcp,35914,1794241654.139256,4.574949,5055.123,ok
ecmp,20261109-082704,M00106,H15,H6,10.3.1.2,10.1.0.3,tcp,28192,1794241655.285895,5.721589,3147.121,ok
ecmp,20261109-082704,M00131,H15,H7,10.3.1.2,10.1.1.2,tcp,49464,1794241656.709280,7.144973,7141.327,ok
ecmp,20261109-082704,M00135,H15,H1,10.3.1.2,10.0.0.2,tcp,44722,1794241656.879374,7.315068,4551.267,ok
ecmp,20261109-082704,M00157,H15,H3,10.3.1.2,10.0.1.2,tcp,11487,1794241657.584610,8.020303,7656.392,ok
ecmp,20261109-082704,M00183,H15,H10,10.3.1.2,10.2.0.3,tcp,48190,1794241658.806356,9.242049,7868.607,ok
ecmp,20261109-082704,M00212,H15,H5,10.3.1.2,10.1.0.2,tcp,64563,1794241660.352245,10.787938,5355.278,ok
ecmp,20261109-082704,M00231,H15,H16,10.3.1.2,10.3.1.3,tcp,26421,1794241661.413784,11.849478,11253.864,ok
ecmp,20261109-082704,M00237,H15,H2,10.3.1.2,10.0.0.3,tcp,28804,1794241661.647408,12.083101,8116.831,ok
ecmp,20261109-082704,M00263,H15,H2,10.3.1.2,10.0.0.3,tcp,42660,1794241662.963594,13.399287,12128.876,ok
ecmp,20261109-082704,M00280,H15,H5,10.3.1.2,10.1.0.2,tcp,51894,1794241663.808154,14.243848,10012.930,timeout
ecmp,20261109-082704,M00281,H15,H16,10.3.1.2,10.3.1.3,tcp,27042,1794241663.809404,14.245098,8116.403,ok
ecmp,20261109-082704,M00284,H15,H14,10.3.1.2,10.3.0.3,tcp,27991,1794241664.066995,14.502688,10015.991,timeout
ecmp,20261109-082704,M00310,H15,H16,10.3.1.2,10.3.1.3,tcp,54137,1794241665.281790,15.717484,10016.970,timeout
ecmp,20261109-082704,M00314,H15,H6,10.3.1.2,10.1.0.3,tcp,52132,1794241665.314645,15.750339,10015.434,timeout
ecmp,20261109-082704,M00342,H15,H9,10.3.1.2,10.2.0.2,tcp,47869,1794241667.076767,17.512460,10012.091,timeout
ecmp,20261109-082704,M00392,H15,H13,10.3.1.2,10.3.0.2,tcp,45136,1794241668.987058,19.422751,10007.298,timeout
ecmp,20261109-082704,M00393,H15,H11,10.3.1.2,10.2.1.2,tcp,45186,1794241669.069404,19.505097,10014.222,timeout
ecmp,20261109-082704,M00395,H15,H2,10.3.1.2,10.0.0.3,tcp,38923,1794241669.135267,19.570961,10010.087,timeout
ecmp,20261109-082704,M00398,H15,H1,10.3.1.2,10.0.0.2,tcp,11920,1794241669.224094,19.659787,10016.344,timeout
ecmp,20261109-082704,M00401,H15,H10,10.3.1.2,10.2.0.3,tcp,44519,1794241669.386642,19.822335,10011.555,timeout
ecmp,20261109-082704,M00020,H8,H11,10.1.1.3,10.2.1.2,tcp,29444,1794241654.641491,5.077185,4210.908,ok
ecmp,20261109-082704,M00048,H8,H4,10.1.1.3,10.0.1.3,tcp,12388,1794241654.810570,5.246264,4439.587,ok
ecmp,20261109-082704,M00049,H8,H10,10.1.1.3,10.2.0.3,tcp,22624,1794241654.827423,5.263116,5650.524,ok
ecmp,20261109-082704,M00073,H8,H7,10.1.1.3,10.1.1.2,tcp,16048,1794241654.853138,5.288832,5621.309,ok
ecmp,20261109-082704,M00074,H8,H10,10.1.1.3,10.2.0.3,tcp,47154,1794241654.873672,5.309366,4586.857,ok
ecmp,20261109-082704,M00082,H8,H6,10.1.1.3,10.1.0.3,tcp,26868,1794241654.892873,5.328567,4020.196,ok
ecmp,20261109-082704,M00098,H8,H11,10.1.1.3,10.2.1.2,tcp,14928,1794241654.946166,5.381860,3892.695,ok
ecmp,20261109-082704,M00100,H8,H14,10.1.1.3,10.3.0.3,tcp,18841,1794241654.948005,5.383698,3994.060,ok
ecmp,20261109-082704,M00103,H8,H5,10.1.1.3,10.1.0.2,tcp,18625,1794241655.116835,5.552529,4151.978,ok
ecmp,20261109-082704,M00107,H8,H4,10.1.1.3,10.0.1.3,tcp,9742,1794241655.366498,5.802192,6649.198,ok
ecmp,20261109-082704,M00128,H8,H11,10.1.1.3,10.2.1.2,tcp,28535,1794241656.624887,7.060580,6373.799,ok
ecmp,20261109-082704,M00142,H8,H5,10.1.1.3,10.1.0.2,tcp,37019,1794241657.215780,7.651474,6196.480,ok
ecmp,20261109-082704,M00151,H8,H16,10.1.1.3,10.3.1.3,tcp,22572,1794241657.396406,7.832100,6432.681,ok
ecmp,20261109-082704,M00165,H8,H3,10.1.1.3,10.0.1.2,tcp,39227,1794241657.980699,8.416392,10014.107,timeout
ecmp,20261109-082704,M00172,H8,H2,10.1.1.3,10.0.0.3,tcp,53883,1794241658.242519,8.678212,19326.873,timeout
ecmp,20261109-082704,M00192,H8,H5,10.1.1.3,10.1.0.2,tcp,51705,1794241659.243080,9.678773,10009.769,timeout
ecmp,20261109-082704,M00200,H8,H6,10.1.1.3,10.1.0.3,tcp,19951,1794241659.688475,10.124169,8987.852,ok
ecmp,20261109-082704,M00211,H8,H13,10.1.1.3,10.3.0.2,tcp,42439,1794241660.323139,10.758833,10017.638,timeout
ecmp,20261109-082704,M00220,H8,H15,10.1.1.3,10.3.1.2,tcp,25529,1794241660.602065,11.037758,10015.751,timeout
ecmp,20261109-082704,M00258,H8,H10,10.1.1.3,10.2.0.3,tcp,33290,1794241662.699337,13.135031,10010.811,timeout
ecmp,20261109-082704,M00276,H8,H9,10.1.1.3,10.2.0.2,tcp,35196,1794241663.702654,14.138348,10014.199,timeout
ecmp,20261109-082704,M00302,H8,H14,10.1.1.3,10.3.0.3,tcp,52515,1794241664.950279,15.385973,10010.614,timeout
ecmp,20261109-082704,M00309,H8,H11,10.1.1.3,10.2.1.2,tcp,12334,1794241665.229832,15.665526,10015.071,timeout
ecmp,20261109-082704,M00313,H8,H15,10.1.1.3,10.3.1.2,tcp,43420,1794241665.305334,15.741028,8957.467,error
ecmp,20261109-082704,M00326,H8,H6,10.1.1.3,10.1.0.3,tcp,58941,1794241666.056338,16.492031,10010.511,timeout
ecmp,20261109-082704,M00372,H8,H7,10.1.1.3,10.1.1.2,tcp,47469,1794241668.183780,18.619474,10010.330,timeout
ecmp,20261109-082704,M00384,H8,H7,10.1.1.3,10.1.1.2,tcp,46784,1794241668.780162,19.215855,10010.407,timeout
ecmp,20261109-082704,M00034,H10,H7,10.2.0.3,10.1.1.2,tcp,18557,1794241654.815422,5.251116,6807.553,ok
ecmp,20261109-082704,M00040,H10,H4,10.2.0.3,10.0.1.3,tcp,11988,1794241654.974685,5.410379,3439.402,ok
ecmp,20261109-082704,M00078,H10,H8,10.2.0.3,10.1.1.3,tcp,28912,1794241655.003395,5.439088,4398.572,ok
ecmp,20261109-082704,M00083,H10,H16,10.2.0.3,10.3.1.3,tcp,61086,1794241655.008529,5.444222,4452.301,ok
ecmp,20261109-082704,M00087,H10,H16,10.2.0.3,10.3.1.3,tcp,21995,1794241655.013457,5.449151,4543.435,ok
ecmp,20261109-082704,M00122,H10,H7,10.2.0.3,10.1.1.2,tcp,44101,1794241656.224448,6.660141,9442.562,ok
ecmp,20261109-082704,M00123,H10,H5,10.2.0.3,10.1.0.2,tcp,21966,1794241656.311318,6.747011,4746.223,ok
ecmp,20261109-082704,M00143,H10,H4,10.2.0.3,10.0.1.3,tcp,25999,1794241657.216706,7.652400,8046.719,ok
ecmp,20261109-082704,M00169,H10,H2,10.2.0.3,10.0.0.3,tcp,46116,1794241658.106220,8.541913,4761.587,ok
ecmp,20261109-082704,M00170,H10,H16,10.2.0.3,10.3.1.3,tcp,52582,1794241658.106953,8.542646,6478.590,ok
ecmp,20261109-082704,M00195,H10,H5,10.2.0.3,10.1.0.2,tcp,23300,1794241659.338788,9.774482,5944.988,ok
ecmp,20261109-082704,M00199,H10,H2,10.2.0.3,10.0.0.3,tcp,62759,1794241659.673839,10.109533,10202.772,ok
ecmp,20261109-082704,M00202,H10,H12,10.2.0.3,10.2.1.3,tcp,16467,1794241659.883356,10.319049,10011.102,timeout
ecmp,20261109-082704,M00204,H10,H8,10.2.0.3,10.1.1.3,tcp,39565,1794241659.936497,10.372191,10018.054,timeout
ecmp,20261109-082704,M00209,H10,H2,10.2.0.3,10.0.0.3,tcp,24457,1794241660.237249,10.672943,9230.188,ok
ecmp,20261109-082704,M00218,H10,H1,10.2.0.3,10.0.0.2,tcp,23362,1794241660.550392,10.986086,15056.904,ok
ecmp,20261109-082704,M00219,H10,H12,10.2.0.3,10.2.1.3,tcp,64527,1794241660.566563,11.002256,10010.458,timeout
ecmp,20261109-082704,M00224,H10,H8,10.2.0.3,10.1.1.3,tcp,16341,1794241660.882170,11.317864,10012.174,timeout
ecmp,20261109-082704,M00225,H10,H12,10.2.0.3,10.2.1.3,tcp,34570,1794241660.911982,11.347675,10013.364,timeout
ecmp,20261109-082704,M00243,H10,H8,10.2.0.3,10.1.1.3,tcp,53952,1794241661.916571,12.352265,10011.307,timeout
ecmp,20261109-082704,M00271,H10,H4,10.2.0.3,10.0.1.3,tcp,57132,1794241663.465171,13.900865,10011.733,timeout
ecmp,20261109-082704,M00275,H10,H6,10.2.0.3,10.1.0.3,tcp,32757,1794241663.674353,14.110047,10008.397,timeout
ecmp,20261109-082704,M00283,H10,H11,10.2.0.3,10.2.1.2,tcp,35999,1794241664.009702,14.445395,10011.185,timeout
ecmp,20261109-082704,M00287,H10,H1,10.2.0.3,10.0.0.2,tcp,62589,1794241664.128456,14.564150,10017.960,timeout
ecmp,20261109-082704,M00290,H10,H13,10.2.0.3,10.3.0.2,tcp,60546,1794241664.353206,14.788900,10011.849,timeout
ecmp,20261109-082704,M00296,H10,H6,10.2.0.3,10.1.0.3,tcp,57178,1794241664.595561,15.031254,10011.193,timeout
ecmp,20261109-082704,M00300,H10,H12,10.2.0.3,10.2.1.3,tcp,65234,1794241664.920468,15.356161,10011.926,timeout
ecmp,20261109-082704,M00336,H10,H8,10.2.0.3,10.1.1.3,tcp,50353,1794241666.698995,17.134689,10015.909,timeout
ecmp,20261109-082704,M00353,H10,H16,10.2.0.3,10.3.1.3,tcp,28668,1794241667.417026,17.852719,10016.982,timeout
ecmp,20261109-082704,M00360,H10,H15,10.2.0.3,10.3.1.2,tcp,30462,1794241667.848709,18.284403,10010.979,timeout
ecmp,20261109-082704,M00396,H10,H13,10.2.0.3,10.3.0.2,tcp,26193,1794241669.164924,19.600617,10010.930,timeout
ecmp,20261109-082704,M00399,H10,H2,10.2.0.3,10.0.0.3,tcp,50440,1794241669.246163,19.681856,10015.220,timeout
ecmp,20261109-082704,M00036,H4,H14,10.0.1.3,10.3.0.3,tcp,49171,1794241654.790192,5.225885,3256.031,ok
ecmp,20261109-082704,M00039,H4,H15,10.0.1.3,10.3.1.2,tcp,31980,1794241654.869306,5.305000,2795.555,ok
ecmp,20261109-082704,M00057,H4,H1,10.0.1.3,10.0.0.2,tcp,34578,1794241654.877275,5.312969,2876.001,ok
ecmp,20261109-082704,M00058,H4,H16,10.0.1.3,10.3.1.3,tcp,20748,1794241654.907757,5.343450,2807.636,ok
ecmp,20261109-082704,M00062,H4,H1,10.0.1.3,10.0.0.2,tcp,50930,1794241655.010311,5.446004,2846.632,ok
ecmp,20261109-082704,M00080,H4,H2,10.0.1.3,10.0.0.3,tcp,43426,1794241655.024587,5.460280,3750.991,ok
ecmp,20261109-082704,M00085,H4,H15,10.0.1.3,10.3.1.2,tcp,16992,1794241655.035168,5.470862,2820.993,ok
ecmp,20261109-082704,M00086,H4,H12,10.0.1.3,10.2.1.3,tcp,44448,1794241655.036616,5.472309,3062.696,ok
ecmp,20261109-082704,M00104,H4,H7,10.0.1.3,10.1.1.2,tcp,65350,1794241655.203833,5.639526,4388.293,ok
ecmp,20261109-082704,M00114,H4,H7,10.0.1.3,10.1.1.2,tcp,45979,1794241655.887312,6.323006,6226.325,ok
ecmp,20261109-082704,M00118,H4,H12,10.0.1.3,10.2.1.3,tcp,27873,1794241655.956444,6.392137,4809.732,ok
ecmp,20261109-082704,M00134,H4,H13,10.0.1.3,10.3.0.2,tcp,36054,1794241656.868457,7.304151,5243.466,ok
ecmp,20261109-082704,M00136,H4,H8,10.0.1.3,10.1.1.3,tcp,16930,1794241656.895825,7.331519,8788.662,ok
ecmp,20261109-082704,M00168,H4,H13,10.0.1.3,10.3.0.2,tcp,61885,1794241658.095232,8.530925,7308.148,ok
ecmp,20261109-082704,M00173,H4,H13,10.0.1.3,10.3.0.2,tcp,45280,1794241658.314702,8.750395,8980.787,ok
ecmp,20261109-082704,M00214,H4,H13,10.0.1.3,10.3.0.2,tcp,21809,1794241660.441004,10.876698,10010.129,timeout
ecmp,20261109-082704,M00238,H4,H13,10.0.1.3,10.3.0.2,tcp,34680,1794241661.664865,12.100559,10019.915,timeout
ecmp,20261109-082704,M00247,H4,H8,10.0.1.3,10.1.1.3,tcp,15890,1794241662.279733,12.715426,10011.515,timeout
ecmp,20261109-082704,M00272,H4,H13,10.0.1.3,10.3.0.2,tcp,49725,1794241663.473598,13.909292,10013.722,timeout
ecmp,20261109-082704,M00273,H4,H8,10.0.1.3,10.1.1.3,tcp,19094,1794241663.535675,13.971369,10014.195,timeout
ecmp,20261109-082704,M00301,H4,H8,10.0.1.3,10.1.1.3,tcp,13126,1794241664.939935,15.375629,10011.049,timeout
ecmp,20261109-082704,M00305,H4,H8,10.0.1.3,10.1.1.3,tcp,32347,1794241665.057148,15.492841,10010.647,timeout
ecmp,20261109-082704,M00321,H4,H14,10.0.1.3,10.3.0.3,tcp,62206,1794241665.866080,16.301773,10016.338,timeout
ecmp,20261109-082704,M00329,H4,H11,10.0.1.3,10.2.1.2,tcp,64975,1794241666.353872,16.789566,10017.113,timeout
ecmp,20261109-082704,M00334,H4,H8,10.0.1.3,10.1.1.3,tcp,41211,1794241666.562106,16.997799,10016.810,timeout
ecmp,20261109-082704,M00338,H4,H8,10.0.1.3,10.1.1.3,tcp,54879,1794241666.887951,17.323644,10011.508,timeout
ecmp,20261109-082704,M00343,H4,H14,10.0.1.3,10.3.0.3,tcp,51099,1794241667.106403,17.542096,10013.168,timeout
ecmp,20261109-082704,M00355,H4,H6,10.0.1.3,10.1.0.3,tcp,63594,1794241667.472416,17.908110,6119.215,error
ecmp,20261109-082704,M00356,H4,H13,10.0.1.3,10.3.0.2,tcp,26448,1794241667.516378,17.952071,10011.227,timeout
ecmp,20261109-082704,M00381,H4,H7,10.0.1.3,10.1.1.2,tcp,9558,1794241668.487636,18.923329,10010.520,timeout
ecmp,20261109-082704,M00037,H7,H3,10.1.1.2,10.0.1.2,tcp,32696,1794241655.103102,5.538795,6287.925,ok
ecmp,20261109-082704,M00046,H7,H12,10.1.1.2,10.2.1.3,tcp,28620,1794241655.126806,5.562500,4182.095,ok
ecmp,20261109-082704,M00088,H7,H11,10.1.1.2,10.2.1.2,tcp,49757,1794241655.143480,5.579174,3831.163,ok
ecmp,20261109-082704,M00116,H7,H6,10.1.1.2,10.1.0.3,tcp,36457,1794241655.905243,6.340936,4598.789,ok
ecmp,20261109-082704,M00125,H7,H9,10.1.1.2,10.2.0.2,tcp,39202,1794241656.374485,6.810178,6593.626,ok
ecmp,20261109-082704,M00129,H7,H3,10.1.1.2,10.0.1.2,tcp,9792,1794241656.635393,7.071087,10010.260,timeout
ecmp,20261109-082704,M00164,H7,H14,10.1.1.2,10.3.0.3,tcp,9527,1794241657.931162,8.366855,7666.470,ok
ecmp,20261109-082704,M00166,H7,H8,10.1.1.2,10.1.1.3,tcp,54032,1794241658.035669,8.471363,10012.025,timeout
ecmp,20261109-082704,M00188,H7,H2,10.1.1.2,10.0.0.3,tcp,24009,1794241659.119712,9.555406,19482.249,timeout
ecmp,20261109-082704,M00196,H7,H6,10.1.1.2,10.1.0.3,tcp,16027,1794241659.349221,9.784915,10011.476,timeout
ecmp,20261109-082704,M00221,H7,H16,10.1.1.2,10.3.1.3,tcp,15713,1794241660.682295,11.117989,10008.251,timeout
ecmp,20261109-082704,M00242,H7,H5,10.1.1.2,10.1.0.2,tcp,44185,1794241661.915073,12.350767,10010.199,timeout
ecmp,20261109-082704,M00297,H7,H3,10.1.1.2,10.0.1.2,tcp,59736,1794241664.615967,15.051660,10011.355,timeout
ecmp,20261109-082704,M00318,H7,H8,10.1.1.2,10.1.1.3,tcp,37335,1794241665.677196,16.112890,10010.362,timeout
ecmp,20261109-082704,M00322,H7,H11,10.1.1.2,10.2.1.2,tcp,50101,1794241665.895708,16.331401,10017.731,timeout
ecmp,20261109-082704,M00325,H7,H10,10.1.1.2,10.2.0.3,tcp,22344,1794241665.978276,16.413970,10018.365,timeout
ecmp,20261109-082704,M00344,H7,H16,10.1.1.2,10.3.1.3,tcp,37785,1794241667.159345,17.595039,6119.993,error
ecmp,20261109-082704,M00354,H7,H3,10.1.1.2,10.0.1.2,tcp,51231,1794241667.438688,17.874382,10016.315,timeout
ecmp,20261109-082704,M00373,H7,H5,10.1.1.2,10.1.0.2,tcp,22035,1794241668.184484,18.620177,10012.237,timeout
ecmp,20261109-082704,M00388,H7,H15,10.1.1.2,10.3.1.2,tcp,59062,1794241668.904523,19.340216,10010.428,timeout
ecmp,20261109-082704,M00390,H7,H3,10.1.1.2,10.0.1.2,tcp,47670,1794241668.934786,19.370479,10012.521,timeout
ecmp,20261109-082704,M00052,H16,H4,10.3.1.3,10.0.1.3,tcp,43531,1794241654.872023,5.307716,3056.314,ok
ecmp,20261109-082704,M00053,H16,H4,10.3.1.3,10.0.1.3,tcp,59612,1794241654.991488,5.427181,2746.191,ok
ecmp,20261109-082704,M00068,H16,H3,10.3.1.3,10.0.1.2,tcp,20637,1794241654.992381,5.428074,2669.170,ok
ecmp,20261109-082704,M00089,H16,H5,10.3.1.3,10.1.0.2,tcp,63642,1794241655.006234,5.441927,2266.741,ok
ecmp,20261109-082704,M00105,H16,H4,10.3.1.3,10.0.1.3,tcp,21271,1794241655.203471,5.639165,3722.157,ok
ecmp,20261109-082704,M00130,H16,H10,10.3.1.3,10.2.0.3,tcp,63880,1794241656.636130,7.071823,5087.110,ok
ecmp,20261109-082704,M00149,H16,H1,10.3.1.3,10.0.0.2,tcp,31244,1794241657.353821,7.789515,4659.230,ok
ecmp,20261109-082704,M00152,H16,H1,10.3.1.3,10.0.0.2,tcp,33676,1794241657.420268,7.855962,6995.487,ok
ecmp,20261109-082704,M00180,H16,H8,10.3.1.3,10.1.1.3,tcp,36737,1794241658.690386,9.126079,9543.380,ok
ecmp,20261109-082704,M00189,H16,H9,10.3.1.3,10.2.0.2,tcp,57879,1794241659.192549,9.628242,9639.457,ok
ecmp,20261109-082704,M00190,H16,H11,10.3.1.3,10.2.1.2,tcp,54843,1794241659.203179,9.638873,5935.858,ok
ecmp,20261109-082704,M00193,H16,H9,10.3.1.3,10.2.0.2,tcp,42986,1794241659.286445,9.722138,10027.633,ok
ecmp,20261109-082704,M00203,H16,H2,10.3.1.3,10.0.0.3,tcp,65386,1794241659.937501,10.373195,3998.741,ok
ecmp,20261109-082704,M00206,H16,H2,10.3.1.3,10.0.0.3,tcp,62070,1794241660.101731,10.537425,5325.240,ok
ecmp,20261109-082704,M00207,H16,H2,10.3.1.3,10.0.0.3,tcp,46007,1794241660.112562,10.548255,5286.238,ok
ecmp,20261109-082704,M00226,H16,H8,10.3.1.3,10.1.1.3,tcp,13463,1794241661.055547,11.491240,10010.876,timeout
ecmp,20261109-082704,M00241,H16,H11,10.3.1.3,10.2.1.2,tcp,37161,1794241661.838263,12.273957,9565.955,ok
ecmp,20261109-082704,M00250,H16,H4,10.3.1.3,10.0.1.3,tcp,38127,1794241662.375785,12.811479,10020.528,timeout
ecmp,20261109-082704,M00255,H16,H14,10.3.1.3,10.3.0.3,tcp,17930,1794241662.526714,12.962407,9407.590,ok
ecmp,20261109-082704,M00256,H16,H6,10.3.1.3,10.1.0.3,tcp,29978,1794241662.554839,12.990533,6893.952,ok
ecmp,20261109-082704,M00269,H16,H14,10.3.1.3,10.3.0.3,tcp,43648,1794241663.405322,13.841016,10014.157,timeout
ecmp,20261109-082704,M00371,H16,H9,10.3.1.3,10.2.0.2,tcp,51913,1794241668.163908,18.599601,10011.900,timeout
ecmp,20261109-082704,M00375,H16,H12,10.3.1.3,10.2.1.3,tcp,36415,1794241668.258473,18.694166,10010.739,timeout
proposed,20261109-082859,M00000,H1,H12,10.0.0.2,10.2.1.3,tcp,26216,1794241760.530064,0.052716,730.413,ok
proposed,20261109-082859,M00004,H1,H9,10.0.0.2,10.2.0.2,tcp,21223,1794241760.640020,0.162672,687.616,ok
proposed,20261109-082859,M00006,H1,H13,10.0.0.2,10.3.0.2,tcp,61002,1794241760.751349,0.274001,787.000,ok
proposed,20261109-082859,M00032,H1,H11,10.0.0.2,10.2.1.2,tcp,55425,1794241762.217594,1.740246,1891.290,ok
proposed,20261109-082859,M00038,H1,H10,10.0.0.2,10.2.0.3,tcp,29435,1794241762.380731,1.903382,1359.515,ok
proposed,20261109-082859,M00063,H1,H2,10.0.0.2,10.0.0.3,tcp,57577,1794241764.115300,3.637952,794.820,ok
proposed,20261109-082859,M00066,H1,H7,10.0.0.2,10.1.1.2,tcp,25572,1794241764.250646,3.773298,1277.048,ok
proposed,20261109-082859,M00084,H1,H11,10.0.0.2,10.2.1.2,tcp,61741,1794241765.131340,4.653992,1855.174,ok
proposed,20261109-082859,M00133,H1,H15,10.0.0.2,10.3.1.2,tcp,57407,1794241767.676341,7.198993,1251.099,ok
proposed,20261109-082859,M00148,H1,H14,10.0.0.2,10.3.0.3,tcp,58366,1794241768.225462,7.748113,895.284,ok
proposed,20261109-082859,M00161,H1,H12,10.0.0.2,10.2.1.3,tcp,43543,1794241768.687579,8.210230,3339.715,ok
proposed,20261109-082859,M00191,H1,H2,10.0.0.2,10.0.0.3,tcp,27475,1794241770.145346,9.667998,1983.674,ok
proposed,20261109-082859,M00228,H1,H2,10.0.0.2,10.0.0.3,tcp,23194,1794241772.060172,11.582824,3846.689,ok
proposed,20261109-082859,M00229,H1,H13,10.0.0.2,10.3.0.2,tcp,52250,1794241772.222794,11.745446,3699.360,ok
proposed,20261109-082859,M00239,H1,H11,10.0.0.2,10.2.1.2,tcp,65166,1794241772.680507,12.203158,6413.902,ok
proposed,20261109-082859,M00244,H1,H11,10.0.0.2,10.2.1.2,tcp,48095,1794241772.870986,12.393638,6848.102,ok
proposed,20261109-082859,M00254,H1,H5,10.0.0.2,10.1.0.2,tcp,55774,1794241773.416048,12.938700,6329.526,ok
proposed,20261109-082859,M00264,H1,H12,10.0.0.2,10.2.1.3,tcp,17639,1794241774.001638,13.524290,8083.558,ok
proposed,20261109-082859,M00274,H1,H16,10.0.0.2,10.3.1.3,tcp,29450,1794241774.502242,14.024894,7531.627,ok
proposed,20261109-082859,M00308,H1,H10,10.0.0.2,10.2.0.3,tcp,44984,1794241776.123765,15.646417,13462.720,timeout
proposed,20261109-082859,M00323,H1,H16,10.0.0.2,10.3.1.3,tcp,59755,1794241776.853958,16.376609,13445.206,ok
proposed,20261109-082859,M00327,H1,H5,10.0.0.2,10.1.0.2,tcp,64448,1794241776.977012,16.499664,13700.536,ok
proposed,20261109-082859,M00330,H1,H6,10.0.0.2,10.1.0.3,tcp,59964,1794241777.266248,16.788899,13455.128,ok
proposed,20261109-082859,M00358,H1,H10,10.0.0.2,10.2.0.3,tcp,51303,1794241778.652256,18.174908,17168.183,timeout
proposed,20261109-082859,M00363,H1,H2,10.0.0.2,10.0.0.3,tcp,57089,1794241778.886918,18.409570,19203.247,timeout
proposed,20261109-082859,M00453,H1,H10,10.0.0.2,10.2.0.3,tcp,31314,1794241782.080174,21.602826,10010.859,timeout
proposed,20261109-082859,M00456,H1,H7,10.0.0.2,10.1.1.2,tcp,65426,1794241782.182641,21.705292,17270.909,ok
proposed,20261109-082859,M00516,H1,H2,10.0.0.2,10.0.0.3,tcp,11171,1794241785.092506,24.615158,10018.358,timeout
proposed,20261109-082859,M00001,H5,H12,10.1.0.2,10.2.1.3,tcp,14909,1794241760.544162,0.066814,687.504,ok
proposed,20261109-082859,M00027,H5,H9,10.1.0.2,10.2.0.2,tcp,40535,1794241761.954499,1.477151,374.423,ok
proposed,20261109-082859,M00043,H5,H11,10.1.0.2,10.2.1.2,tcp,39340,1794241762.631560,2.154212,887.108,ok
proposed,20261109-082859,M00094,H5,H9,10.1.0.2,10.2.0.2,tcp,10553,1794241765.449003,4.971655,266.485,ok
proposed,20261109-082859,M00095,H5,H7,10.1.0.2,10.1.1.2,tcp,16544,1794241765.536906,5.059557,156.268,ok
proposed,20261109-082859,M00120,H5,H4,10.1.0.2,10.0.1.3,tcp,35746,1794241766.964681,6.487333,368.798,ok
proposed,20261109-082859,M00145,H5,H12,10.1.0.2,10.2.1.3,tcp,22213,1794241768.163184,7.685836,500.258,ok
proposed,20261109-082859,M00154,H5,H10,10.1.0.2,10.2.0.3,tcp,43196,1794241768.434592,7.957244,516.308,ok
proposed,20261109-082859,M00178,H5,H7,10.1.0.2,10.1.1.2,tcp,19735,1794241769.411348,8.934000,327.238,ok
proposed,20261109-082859,M00201,H5,H16,10.1.0.2,10.3.1.3,tcp,11174,1794241770.613469,10.136121,1201.203,ok
proposed,20261109-082859,M00222,H5,H15,10.1.0.2,10.3.1.2,tcp,25600,1794241771.635092,11.157743,2429.561,ok
proposed,20261109-082859,M00248,H5,H8,10.1.0.2,10.1.1.3,tcp,55155,1794241773.240862,12.763514,2079.887,ok
proposed,20261109-082859,M00252,H5,H14,10.1.0.2,10.3.0.3,tcp,12767,1794241773.365433,12.888085,1958.377,ok
proposed,20261109-082859,M00261,H5,H8,10.1.0.2,10.1.1.3,tcp,10618,1794241773.770767,13.293419,1417.009,ok
proposed,20261109-082859,M00265,H5,H2,10.1.0.2,10.0.0.3,tcp,38960,1794241774.035906,13.558558,4479.743,ok
proposed,20261109-082859,M00279,H5,H4,10.1.0.2,10.0.1.3,tcp,38413,1794241774.669499,14.192150,2229.999,ok
proposed,20261109-082859,M00293,H5,H11,10.1.0.2,10.2.1.2,tcp,14559,1794241775.349692,14.872343,3043.339,ok
proposed,20261109-082859,M00324,H5,H13,10.1.0.2,10.3.0.2,tcp,45192,1794241776.875041,16.397692,2837.353,ok
proposed,20261109-082859,M00328,H5,H9,10.1.0.2,10.2.0.2,tcp,24618,1794241777.183690,16.706342,2644.402,ok
proposed,20261109-082859,M00332,H5,H12,10.1.0.2,10.2.1.3,tcp,35780,1794241777.359762,16.882414,2417.738,ok
proposed,20261109-082859,M00340,H5,H2,10.1.0.2,10.0.0.3,tcp,16462,1794241777.937830,17.460482,11818.410,ok
proposed,20261109-082859,M00348,H5,H11,10.1.0.2,10.2.1.2,tcp,39273,1794241778.251430,17.774082,2986.024,ok
proposed,20261109-082859,M00389,H5,H4,10.1.0.2,10.0.1.3,tcp,14929,1794241779.835131,19.357783,2915.721,ok
proposed,20261109-082859,M00404,H5,H1,10.1.0.2,10.0.0.2,tcp,37717,1794241780.523270,20.045922,14525.808,ok
proposed,20261109-082859,M00412,H5,H6,10.1.0.2,10.1.0.3,tcp,13533,1794241780.774077,20.296729,4233.147,ok
proposed,20261109-082859,M00413,H5,H15,10.1.0.2,10.3.1.2,tcp,31117,1794241780.815362,20.338014,7227.895,ok
proposed,20261109-082859,M00423,H5,H10,10.1.0.2,10.2.0.3,tcp,32880,1794241781.199247,20.721899,7165.192,ok
proposed,20261109-082859,M00433,H5,H14,10.1.0.2,10.3.0.3,tcp,24576,1794241781.518522,21.041173,5338.120,ok
proposed,20261109-082859,M00443,H5,H3,10.1.0.2,10.0.1.2,tcp,10702,1794241781.851795,21.374447,4806.154,ok
proposed,20261109-082859,M00447,H5,H9,10.1.0.2,10.2.0.2,tcp,55852,1794241781.888463,21.411114,7283.348,ok
proposed,20261109-082859,M00463,H5,H2,10.1.0.2,10.0.0.3,tcp,55118,1794241782.461371,21.984023,10014.986,timeout
proposed,20261109-082859,M00487,H5,H9,10.1.0.2,10.2.0.2,tcp,32240,1794241783.772287,23.294938,9612.925,ok
proposed,20261109-082859,M00501,H5,H10,10.1.0.2,10.2.0.3,tcp,13842,1794241784.477203,23.999855,10011.165,timeout
proposed,20261109-082859,M00506,H5,H15,10.1.0.2,10.3.1.2,tcp,62524,1794241784.688847,24.211498,9812.356,ok
proposed,20261109-082859,M00514,H5,H3,10.1.0.2,10.0.1.2,tcp,51362,1794241785.038102,24.560753,9760.773,ok
proposed,20261109-082859,M00002,H3,H10,10.0.1.2,10.2.0.3,tcp,35843,1794241760.602803,0.125455,735.655,ok
proposed,20261109-082859,M00003,H3,H4,10.0.1.2,10.0.1.3,tcp,23439,1794241760.646273,0.168925,449.606,ok
proposed,20261109-082859,M00014,H3,H14,10.0.1.2,10.3.0.3,tcp,23448,1794241761.022501,0.545153,366.355,ok
proposed,20261109-082859,M00030,H3,H7,10.0.1.2,10.1.1.2,tcp,33201,1794241761.998230,1.520881,176.336,ok
proposed,20261109-082859,M00041,H3,H2,10.0.1.2,10.0.0.3,tcp,56158,1794241762.527499,2.050151,270.608,ok
proposed,20261109-082859,M00042,H3,H13,10.0.1.2,10.3.0.2,tcp,43103,1794241762.558580,2.081232,711.550,ok
proposed,20261109-082859,M00075,H3,H7,10.0.1.2,10.1.1.2,tcp,51273,1794241764.692222,4.214874,783.250,ok
proposed,20261109-082859,M00079,H3,H1,10.0.1.2,10.0.0.2,tcp,38226,1794241764.902589,4.425241,677.773,ok
proposed,20261109-082859,M00144,H3,H16,10.0.1.2,10.3.1.3,tcp,18076,1794241768.145192,7.667843,533.337,ok
proposed,20261109-082859,M00156,H3,H11,10.0.1.2,10.2.1.2,tcp,36281,1794241768.497039,8.019691,1445.362,ok
proposed,20261109-082859,M00175,H3,H9,10.0.1.2,10.2.0.2,tcp,50629,1794241769.352345,8.874997,671.976,ok
proposed,20261109-082859,M00182,H3,H5,10.0.1.2,10.1.0.2,tcp,37735,1794241769.705465,9.228116,2124.714,ok
proposed,20261109-082859,M00223,H3,H1,10.0.1.2,10.0.0.2,tcp,19065,1794241771.718966,11.241618,3890.383,ok
proposed,20261109-082859,M00233,H3,H8,10.0.1.2,10.1.1.3,tcp,30995,1794241772.361998,11.884649,1932.399,ok
proposed,20261109-082859,M00267,H3,H14,10.0.1.2,10.3.0.3,tcp,29724,1794241774.161397,13.684049,2752.231,ok
proposed,20261109-082859,M00270,H3,H4,10.0.1.2,10.0.1.3,tcp,49555,1794241774.319345,13.841997,2496.951,ok
proposed,20261109-082859,M00304,H3,H13,10.0.1.2,10.3.0.2,tcp,34050,1794241775.966547,15.489199,2422.499,ok
proposed,20261109-082859,M00333,H3,H8,10.0.1.2,10.1.1.3,tcp,37570,1794241777.396340,16.918991,2433.303,ok
proposed,20261109-082859,M00339,H3,H15,10.0.1.2,10.3.1.2,tcp,13486,1794241777.917138,17.439790,2154.692,ok
proposed,20261109-082859,M00349,H3,H2,10.0.1.2,10.0.0.3,tcp,36495,1794241778.261637,17.784289,12010.919,ok
proposed,20261109-082859,M00366,H3,H15,10.0.1.2,10.3.1.2,tcp,11639,1794241778.967356,18.490007,1933.564,ok
proposed,20261109-082859,M00379,H3,H14,10.0.1.2,10.3.0.3,tcp,28708,1794241779.323857,18.846509,3647.270,ok
proposed,20261109-082859,M00385,H3,H7,10.0.1.2,10.1.1.2,tcp,65242,1794241779.713224,19.235876,3017.559,ok
proposed,20261109-082859,M00387,H3,H9,10.0.1.2,10.2.0.2,tcp,49691,1794241779.806662,19.329313,4378.011,ok
proposed,20261109-082859,M00391,H3,H16,10.0.1.2,10.3.1.3,tcp,58841,1794241779.853353,19.376004,3694.798,ok
proposed,20261109-082859,M00400,H3,H10,10.0.1.2,10.2.0.3,tcp,47125,1794241780.260862,19.783514,3819.545,ok
proposed,20261109-082859,M00415,H3,H5,10.0.1.2,10.1.0.2,tcp,44821,1794241780.899305,20.421957,7775.481,ok
proposed,20261109-082859,M00417,H3,H11,10.0.1.2,10.2.1.2,tcp,52131,1794241780.972367,20.495018,4700.351,ok
proposed,20261109-082859,M00441,H3,H6,10.0.1.2,10.1.0.3,tcp,22907,1794241781.798011,21.320663,5006.916,ok
proposed,20261109-082859,M00446,H3,H1,10.0.1.2,10.0.0.2,tcp,24964,1794241781.877685,21.400337,14163.239,ok
proposed,20261109-082859,M00474,H3,H12,10.0.1.2,10.2.1.3,tcp,36012,1794241783.189945,22.712597,9987.500,ok
proposed,20261109-082859,M00476,H3,H6,10.0.1.2,10.1.0.3,tcp,51506,1794241783.212049,22.734701,9462.857,ok
proposed,20261109-082859,M00493,H3,H8,10.0.1.2,10.1.1.3,tcp,19507,1794241783.857686,23.380338,6277.364,ok
proposed,20261109-082859,M00504,H3,H16,10.0.1.2,10.3.1.3,tcp,28968,1794241784.597693,24.120345,7976.770,ok
proposed,20261109-082859,M00511,H3,H5,10.0.1.2,10.1.0.2,tcp,21056,1794241784.972739,24.495391,10010.894,timeout
proposed,20261109-082859,M00513,H3,H9,10.0.1.2,10.2.0.2,tcp,50199,1794241785.029387,24.552039,9071.814,ok
proposed,20261109-082859,M00005,H14,H4,10.3.0.3,10.0.1.3,tcp,37631,1794241760.710380,0.233032,715.134,ok
proposed,20261109-082859,M00007,H14,H6,10.3.0.3,10.1.0.3,tcp,26402,1794241760.755906,0.278558,612.406,ok
proposed,20261109-082859,M00045,H14,H4,10.3.0.3,10.0.1.3,tcp,43535,1794241762.882602,2.405254,899.990,ok
proposed,20261109-082859,M00054,H14,H4,10.3.0.3,10.0.1.3,tcp,14373,1794241763.206567,2.729219,316.017,ok
proposed,20261109-082859,M00055,H14,H6,10.3.0.3,10.1.0.3,tcp,35951,1794241763.209312,2.731964,386.225,ok
proposed,20261109-082859,M00090,H14,H16,10.3.0.3,10.3.1.3,tcp,26324,1794241765.360690,4.883342,287.717,ok
proposed,20261109-082859,M00093,H14,H9,10.3.0.3,10.2.0.2,tcp,8825,1794241765.443796,4.966448,854.493,ok
proposed,20261109-082859,M00099,H14,H10,10.3.0.3,10.2.0.3,tcp,57309,1794241765.749325,5.271977,1115.810,ok
proposed,20261109-082859,M00119,H14,H6,10.3.0.3,10.1.0.3,tcp,34563,1794241766.904932,6.427583,415.239,ok
proposed,20261109-082859,M00138,H14,H16,10.3.0.3,10.3.1.3,tcp,44097,1794241767.927982,7.450633,625.586,ok
proposed,20261109-082859,M00146,H14,H6,10.3.0.3,10.1.0.3,tcp,43752,1794241768.163650,7.686301,546.035,ok
proposed,20261109-082859,M00160,H14,H5,10.3.0.3,10.1.0.2,tcp,62908,1794241768.683475,8.206127,1155.881,ok
proposed,20261109-082859,M00176,H14,H16,10.3.0.3,10.3.1.3,tcp,40317,1794241769.379956,8.902608,394.719,ok
proposed,20261109-082859,M00210,H14,H10,10.3.0.3,10.2.0.3,tcp,47259,1794241771.157857,10.680509,1993.657,ok
proposed,20261109-082859,M00213,H14,H5,10.3.0.3,10.1.0.2,tcp,45457,1794241771.303285,10.825936,1355.821,ok
proposed,20261109-082859,M00217,H14,H8,10.3.0.3,10.1.1.3,tcp,53370,1794241771.417458,10.940110,1663.294,ok
proposed,20261109-082859,M00227,H14,H12,10.3.0.3,10.2.1.3,tcp,29315,1794241772.016051,11.538703,2121.710,ok
proposed,20261109-082859,M00232,H14,H11,10.3.0.3,10.2.1.2,tcp,61554,1794241772.330977,11.853628,1888.371,ok
proposed,20261109-082859,M00240,H14,H1,10.3.0.3,10.0.0.2,tcp,20485,1794241772.721148,12.243800,5061.336,ok
proposed,20261109-082859,M00246,H14,H2,10.3.0.3,10.0.0.3,tcp,22917,1794241772.951728,12.474380,3582.446,ok
proposed,20261109-082859,M00253,H14,H6,10.3.0.3,10.1.0.3,tcp,59815,1794241773.380033,12.902684,2255.592,ok
proposed,20261109-082859,M00259,H14,H1,10.3.0.3,10.0.0.2,tcp,29044,1794241773.618990,13.141642,6603.269,ok
proposed,20261109-082859,M00311,H14,H2,10.3.0.3,10.0.0.3,tcp,17352,1794241776.218735,15.741387,7447.865,ok
proposed,20261109-082859,M00315,H14,H10,10.3.0.3,10.2.0.3,tcp,56837,1794241776.376840,15.899491,3652.688,ok
proposed,20261109-082859,M00316,H14,H11,10.3.0.3,10.2.1.2,tcp,14681,1794241776.449602,15.972254,3019.878,ok
proposed,20261109-082859,M00317,H14,H5,10.3.0.3,10.1.0.2,tcp,10336,1794241776.532804,16.055456,2927.972,ok
proposed,20261109-082859,M00345,H14,H6,10.3.0.3,10.1.0.3,tcp,62417,1794241778.188469,17.711121,3147.583,ok
proposed,20261109-082859,M00346,H14,H12,10.3.0.3,10.2.1.3,tcp,14423,1794241778.209203,17.731855,2504.100,ok
proposed,20261109-082859,M00352,H14,H16,10.3.0.3,10.3.1.3,tcp,55447,1794241778.332496,17.855148,2735.111,ok
proposed,20261109-082859,M00357,H14,H16,10.3.0.3,10.3.1.3,tcp,44955,1794241778.467573,17.990224,2591.505,ok
proposed,20261109-082859,M00368,H14,H8,10.3.0.3,10.1.1.3,tcp,47977,1794241778.998698,18.521349,2257.298,ok
proposed,20261109-082859,M00438,H14,H2,10.3.0.3,10.0.0.3,tcp,22743,1794241781.674802,21.197454,12282.715,ok
proposed,20261109-082859,M00448,H14,H2,10.3.0.3,10.0.0.3,tcp,59059,1794241781.935467,21.458119,15276.688,ok
proposed,20261109-082859,M00454,H14,H3,10.3.0.3,10.0.1.2,tcp,51732,1794241782.088630,21.611282,4134.897,ok
proposed,20261109-082859,M00503,H14,H13,10.3.0.3,10.3.0.2,tcp,35024,1794241784.517945,24.040596,7137.616,ok
proposed,20261109-082859,M00518,H14,H10,10.3.0.3,10.2.0.3,tcp,50068,1794241785.310083,24.832735,8728.028,ok
proposed,20261109-082859,M00008,H11,H2,10.2.1.2,10.0.0.3,tcp,14270,1794241760.761618,0.284270,516.481,ok
proposed,20261109-082859,M00022,H11,H7,10.2.1.2,10.1.1.2,tcp,25738,1794241761.583738,1.106390,520.811,ok
proposed,20261109-082859,M00023,H11,H4,10.2.1.2,10.0.1.3,tcp,51146,1794241761.596873,1.119525,560.081,ok
proposed,20261109-082859,M00051,H11,H2,10.2.1.2,10.0.0.3,tcp,41887,1794241763.147359,2.670011,448.303,ok
proposed,20261109-082859,M00070,H11,H1,10.2.1.2,10.0.0.2,tcp,11478,1794241764.503013,4.025665,1415.491,ok
proposed,20261109-082859,M00076,H11,H15,10.2.1.2,10.3.1.2,tcp,25281,1794241764.733202,4.255854,607.063,ok
proposed,20261109-082859,M00077,H11,H4,10.2.1.2,10.0.1.3,tcp,25599,1794241764.745760,4.268412,616.378,ok
proposed,20261109-082859,M00091,H11,H13,10.2.1.2,10.3.0.2,tcp,16765,1794241765.361040,4.883691,321.667,ok
proposed,20261109-082859,M00102,H11,H13,10.2.1.2,10.3.0.2,tcp,35174,1794241765.945831,5.468482,363.300,ok
proposed,20261109-082859,M00110,H11,H1,10.2.1.2,10.0.0.2,tcp,15751,1794241766.472119,5.994771,889.860,ok
proposed,20261109-082859,M00117,H11,H10,10.2.1.2,10.2.0.3,tcp,28764,1794241766.813729,6.336381,607.539,ok
proposed,20261109-082859,M00124,H11,H8,10.2.1.2,10.1.1.3,tcp,37144,1794241767.252369,6.775021,484.896,ok
proposed,20261109-082859,M00127,H11,H2,10.2.1.2,10.0.0.3,tcp,61828,1794241767.383170,6.905822,1719.857,ok
proposed,20261109-082859,M00137,H11,H13,10.2.1.2,10.3.0.2,tcp,37196,1794241767.879983,7.402634,817.459,ok
proposed,20261109-082859,M00158,H11,H4,10.2.1.2,10.0.1.3,tcp,37991,1794241768.518127,8.040779,292.228,ok
proposed,20261109-082859,M00184,H11,H1,10.2.1.2,10.0.0.2,tcp,40586,1794241769.773612,9.296264,2471.485,ok
proposed,20261109-082859,M00194,H11,H6,10.2.1.2,10.1.0.3,tcp,54254,1794241770.226825,9.749477,653.997,ok
proposed,20261109-082859,M00234,H11,H16,10.2.1.2,10.3.1.3,tcp,15047,1794241772.398841,11.921492,1803.822,ok
proposed,20261109-082859,M00235,H11,H7,10.2.1.2,10.1.1.2,tcp,53652,1794241772.492419,12.015070,1697.748,ok
proposed,20261109-082859,M00262,H11,H14,10.2.1.2,10.3.0.3,tcp,14769,1794241773.774620,13.297271,1542.696,ok
proposed,20261109-082859,M00331,H11,H1,10.2.1.2,10.0.0.2,tcp,19611,1794241777.355780,16.878432,11077.934,ok
proposed,20261109-082859,M00341,H11,H7,10.2.1.2,10.1.1.2,tcp,47342,1794241777.958220,17.480872,3058.238,ok
proposed,20261109-082859,M00347,H11,H16,10.2.1.2,10.3.1.3,tcp,24896,1794241778.230125,17.752777,4230.099,ok
proposed,20261109-082859,M00351,H11,H16,10.2.1.2,10.3.1.3,tcp,16202,1794241778.302703,17.825355,2755.659,ok
proposed,20261109-082859,M00386,H11,H4,10.2.1.2,10.0.1.3,tcp,30019,1794241779.732040,19.254692,4685.469,ok
proposed,20261109-082859,M00394,H11,H14,10.2.1.2,10.3.0.3,tcp,49297,1794241780.022290,19.544942,3383.291,ok
proposed,20261109-082859,M00405,H11,H12,10.2.1.2,10.2.1.3,tcp,13718,1794241780.533867,20.056518,3599.253,ok
proposed,20261109-082859,M00409,H11,H16,10.2.1.2,10.3.1.3,tcp,47375,1794241780.701221,20.223872,3699.778,ok
proposed,20261109-082859,M00410,H11,H2,10.2.1.2,10.0.0.3,tcp,29798,1794241780.703080,20.225732,12530.008,ok
proposed,20261109-082859,M00416,H11,H14,10.2.1.2,10.3.0.3,tcp,16567,1794241780.919802,20.442453,6119.242,ok
proposed,20261109-082859,M00460,H11,H3,10.2.1.2,10.0.1.2,tcp,11097,1794241782.375113,21.897765,6630.085,ok
proposed,20261109-082859,M00461,H11,H14,10.2.1.2,10.3.0.3,tcp,50318,1794241782.396548,21.919200,4712.129,ok
proposed,20261109-082859,M00495,H11,H16,10.2.1.2,10.3.1.3,tcp,30960,1794241784.037089,23.559741,7695.360,ok
proposed,20261109-082859,M00515,H11,H14,10.2.1.2,10.3.0.3,tcp,45129,1794241785.090863,24.613515,8142.562,ok
proposed,20261109-082859,M00523,H11,H14,10.2.1.2,10.3.0.3,tcp,55270,1794241785.424955,24.947607,9433.670,ok
proposed,20261109-082859,M00009,H12,H14,10.2.1.3,10.3.0.3,tcp,30733,1794241760.787182,0.309834,624.348,ok
proposed,20261109-082859,M00012,H12,H10,10.2.1.3,10.2.0.3,tcp,20793,1794241760.887399,0.410051,305.461,ok
proposed,20261109-082859,M00016,H12,H3,10.2.1.3,10.0.1.2,tcp,32452,1794241761.173670,0.696322,300.219,ok
proposed,20261109-082859,M00047,H12,H8,10.2.1.3,10.1.1.3,tcp,42111,1794241762.985719,2.508371,606.056,ok
proposed,20261109-082859,M00096,H12,H15,10.2.1.3,10.3.1.2,tcp,60366,1794241765.539415,5.062067,772.177,ok
proposed,20261109-082859,M00113,H12,H16,10.2.1.3,10.3.1.3,tcp,59731,1794241766.781407,6.304058,741.115,ok
proposed,20261109-082859,M00162,H12,H4,10.2.1.3,10.0.1.3,tcp,50799,1794241768.690922,8.213573,1194.426,ok
proposed,20261109-082859,M00167,H12,H3,10.2.1.3,10.0.1.2,tcp,47899,1794241768.961051,8.483703,939.845,ok
proposed,20261109-082859,M00174,H12,H9,10.2.1.3,10.2.0.2,tcp,36266,1794241769.298844,8.821496,640.755,ok
proposed,20261109-082859,M00177,H12,H11,10.2.1.3,10.2.1.2,tcp,62502,1794241769.381305,8.903957,425.929,ok
proposed,20261109-082859,M00291,H12,H4,10.2.1.3,10.0.1.3,tcp,49913,1794241775.281439,14.804090,3362.230,ok
proposed,20261109-082859,M00295,H12,H16,10.2.1.3,10.3.1.3,tcp,16804,1794241775.495901,15.018553,2953.772,ok
proposed,20261109-082859,M00298,H12,H9,10.2.1.3,10.2.0.2,tcp,41077,1794241775.565460,15.088112,4148.421,ok
proposed,20261109-082859,M00307,H12,H2,10.2.1.3,10.0.0.3,tcp,52418,1794241776.113220,15.635871,12302.656,ok
proposed,20261109-082859,M00319,H12,H2,10.2.1.3,10.0.0.3,tcp,53128,1794241776.740542,16.263193,11538.818,ok
proposed,20261109-082859,M00320,H12,H1,10.2.1.3,10.0.0.2,tcp,34284,1794241776.761609,16.284261,12147.241,ok
proposed,20261109-082859,M00350,H12,H13,10.2.1.3,10.3.0.2,tcp,16719,1794241778.262120,17.784771,3008.677,ok
proposed,20261109-082859,M00365,H12,H5,10.2.1.3,10.1.0.2,tcp,55482,1794241778.947087,18.469739,2288.228,ok
proposed,20261109-082859,M00370,H12,H7,10.2.1.3,10.1.1.2,tcp,15459,1794241779.054684,18.577336,2139.870,ok
proposed,20261109-082859,M00377,H12,H2,10.2.1.3,10.0.0.3,tcp,34214,1794241779.198283,18.720934,11143.380,ok
proposed,20261109-082859,M00383,H12,H2,10.2.1.3,10.0.0.3,tcp,36812,1794241779.596798,19.119450,13034.562,ok
proposed,20261109-082859,M00397,H12,H9,10.2.1.3,10.2.0.2,tcp,13054,1794241780.117642,19.640293,3221.647,ok
proposed,20261109-082859,M00408,H12,H6,10.2.1.3,10.1.0.3,tcp,26731,1794241780.680391,20.203043,5088.506,ok
proposed,20261109-082859,M00418,H12,H1,10.2.1.3,10.0.0.2,tcp,31955,1794241781.003709,20.526361,16160.653,ok
proposed,20261109-082859,M00426,H12,H11,10.2.1.3,10.2.1.2,tcp,55350,1794241781.281738,20.804390,9329.581,ok
proposed,20261109-082859,M00439,H12,H15,10.2.1.3,10.3.1.2,tcp,9939,1794241781.757934,21.280585,5203.156,ok
proposed,20261109-082859,M00455,H12,H2,10.2.1.3,10.0.0.3,tcp,42667,1794241782.145857,21.668509,14222.276,ok
proposed,20261109-082859,M00458,H12,H13,10.2.1.3,10.3.0.2,tcp,12619,1794241782.292969,21.815620,7752.545,ok
proposed,20261109-082859,M00479,H12,H10,10.2.1.3,10.2.0.3,tcp,36696,1794241783.264645,22.787296,8530.276,ok
proposed,20261109-082859,M00481,H12,H9,10.2.1.3,10.2.0.2,tcp,60692,1794241783.328450,22.851102,8717.174,ok
proposed,20261109-082859,M00484,H12,H6,10.2.1.3,10.1.0.3,tcp,33282,1794241783.496497,23.019149,10261.994,ok
proposed,20261109-082859,M00486,H12,H6,10.2.1.3,10.1.0.3,tcp,37286,1794241783.696873,23.219525,10012.508,timeout
proposed,20261109-082859,M00500,H12,H8,10.2.1.3,10.1.1.3,tcp,60860,1794241784.281139,23.803791,9763.952,ok
proposed,20261109-082859,M00524,H12,H7,10.2.1.3,10.1.1.2,tcp,16761,1794241785.448322,24.970974,8325.397,ok
proposed,20261109-082859,M00010,H2,H12,10.0.0.3,10.2.1.3,tcp,38300,1794241760.832213,0.354865,545.886,ok
proposed,20261109-082859,M00013,H2,H11,10.0.0.3,10.2.1.2,tcp,23127,1794241760.949415,0.472067,420.942,ok
proposed,20261109-082859,M00021,H2,H4,10.0.0.3,10.0.1.3,tcp,62049,1794241761.582248,1.104899,519.748,ok
proposed,20261109-082859,M00028,H2,H14,10.0.0.3,10.3.0.3,tcp,15377,1794241761.962439,1.485090,200.439,ok
proposed,20261109-082859,M00050,H2,H4,10.0.0.3,10.0.1.3,tcp,12609,1794241763.031752,2.554404,164.918,ok
proposed,20261109-082859,M00056,H2,H11,10.0.0.3,10.2.1.2,tcp,51016,1794241763.238062,2.760713,420.313,ok
proposed,20261109-082859,M00069,H2,H10,10.0.0.3,10.2.0.3,tcp,56409,1794241764.461177,3.983828,2175.970,ok
proposed,20261109-082859,M00097,H2,H15,10.0.0.3,10.3.1.2,tcp,31641,1794241765.717507,5.240159,1261.214,ok
proposed,20261109-082859,M00115,H2,H12,10.0.0.3,10.2.1.3,tcp,36769,1794241766.810939,6.333591,1636.052,ok
proposed,20261109-082859,M00147,H2,H4,10.0.0.3,10.0.1.3,tcp,62776,1794241768.194236,7.716888,802.304,ok
proposed,20261109-082859,M00163,H2,H13,10.0.0.3,10.3.0.2,tcp,10225,1794241768.693293,8.215945,1156.704,ok
proposed,20261109-082859,M00208,H2,H3,10.0.0.3,10.0.1.2,tcp,17969,1794241771.068794,10.591445,2180.572,ok
proposed,20261109-082859,M00230,H2,H13,10.0.0.3,10.3.0.2,tcp,57754,1794241772.313473,11.836125,2988.694,ok
proposed,20261109-082859,M00289,H2,H15,10.0.0.3,10.3.1.2,tcp,47933,1794241775.197011,14.719663,6586.376,ok
proposed,20261109-082859,M00294,H2,H5,10.0.0.3,10.1.0.2,tcp,59874,1794241775.468917,14.991569,6714.497,ok
proposed,20261109-082859,M00312,H2,H5,10.0.0.3,10.1.0.2,tcp,40476,1794241776.219088,15.741740,7614.195,ok
proposed,20261109-082859,M00335,H2,H12,10.0.0.3,10.2.1.3,tcp,59521,1794241777.485514,17.008165,13372.339,ok
proposed,20261109-082859,M00337,H2,H8,10.0.0.3,10.1.1.3,tcp,63720,1794241777.772129,17.294781,12232.023,ok
proposed,20261109-082859,M00367,H2,H11,10.0.0.3,10.2.1.2,tcp,13538,1794241778.973951,18.496603,7904.444,ok
proposed,20261109-082859,M00378,H2,H7,10.0.0.3,10.1.1.2,tcp,9420,1794241779.302506,18.825158,7929.936,ok
proposed,20261109-082859,M00402,H2,H8,10.0.0.3,10.1.1.3,tcp,61264,1794241780.450509,19.973161,15019.234,ok
proposed,20261109-082859,M00422,H2,H11,10.0.0.3,10.2.1.2,tcp,30262,1794241781.098245,20.620897,14885.945,ok
proposed,20261109-082859,M00469,H2,H14,10.0.0.3,10.3.0.3,tcp,51166,1794241782.960256,22.482908,16275.959,ok
proposed,20261109-082859,M00480,H2,H16,10.0.0.3,10.3.1.3,tcp,52178,1794241783.273813,22.796464,18058.683,ok
proposed,20261109-082859,M00485,H2,H10,10.0.0.3,10.2.0.3,tcp,18247,1794241783.640841,23.163492,10013.642,timeout
proposed,20261109-082859,M00508,H2,H14,10.0.0.3,10.3.0.3,tcp,16329,1794241784.789023,24.311674,10009.538,timeout
proposed,20261109-082859,M00521,H2,H3,10.0.0.3,10.0.1.2,tcp,37001,1794241785.357775,24.880427,14624.355,ok
proposed,20261109-082859,M00011,H13,H2,10.3.0.2,10.0.0.3,tcp,44370,1794241760.870552,0.393203,548.473,ok
proposed,20261109-082859,M00015,H13,H5,10.3.0.2,10.1.0.2,tcp,37906,1794241761.124209,0.646861,326.561,ok
proposed,20261109-082859,M00026,H13,H6,10.3.0.2,10.1.0.3,tcp,22565,1794241761.657552,1.180203,446.493,ok
proposed,20261109-082859,M00065,H13,H15,10.3.0.2,10.3.1.2,tcp,12034,1794241764.240570,3.763222,158.146,ok
proposed,20261109-082859,M00108,H13,H6,10.3.0.2,10.1.0.3,tcp,26450,1794241766.332139,5.854791,167.304,ok
proposed,20261109-082859,M00121,H13,H11,10.3.0.2,10.2.1.2,tcp,57221,1794241767.018454,6.541106,454.685,ok
proposed,20261109-082859,M00132,H13,H8,10.3.0.2,10.1.1.3,tcp,34383,1794241767.660800,7.183452,987.998,ok
proposed,20261109-082859,M00150,H13,H14,10.3.0.2,10.3.0.3,tcp,62962,1794241768.277810,7.800461,395.586,ok
proposed,20261109-082859,M00153,H13,H12,10.3.0.2,10.2.1.3,tcp,19008,1794241768.350841,7.873493,396.116,ok
proposed,20261109-082859,M00155,H13,H10,10.3.0.2,10.2.0.3,tcp,45179,1794241768.436062,7.958714,549.288,ok
proposed,20261109-082859,M00159,H13,H5,10.3.0.2,10.1.0.2,tcp,57475,1794241768.542698,8.065350,1266.201,ok
proposed,20261109-082859,M00171,H13,H7,10.3.0.2,10.1.1.2,tcp,55051,1794241769.137298,8.659950,709.324,ok
proposed,20261109-082859,M00236,H13,H14,10.3.0.2,10.3.0.3,tcp,58032,1794241772.523824,12.046476,1520.209,ok
proposed,20261109-082859,M00251,H13,H4,10.3.0.2,10.0.1.3,tcp,47472,1794241773.329408,12.852059,2058.005,ok
proposed,20261109-082859,M00257,H13,H8,10.3.0.2,10.1.1.3,tcp,29280,1794241773.503269,13.025921,1826.603,ok
proposed,20261109-082859,M00260,H13,H7,10.3.0.2,10.1.1.2,tcp,51687,1794241773.689349,13.212000,3020.834,ok
proposed,20261109-082859,M00268,H13,H6,10.3.0.2,10.1.0.3,tcp,49269,1794241774.261138,13.783790,2417.054,ok
proposed,20261109-082859,M00278,H13,H10,10.3.0.2,10.2.0.3,tcp,52961,1794241774.654957,14.177609,2396.683,ok
proposed,20261109-082859,M00285,H13,H14,10.3.0.2,10.3.0.3,tcp,39859,1794241775.000890,14.523541,3355.638,ok
proposed,20261109-082859,M00286,H13,H10,10.3.0.2,10.2.0.3,tcp,31716,1794241775.001271,14.523923,3458.880,ok
proposed,20261109-082859,M00303,H13,H14,10.3.0.2,10.3.0.3,tcp,60921,1794241775.924350,15.447002,2319.432,ok
proposed,20261109-082859,M00362,H13,H2,10.3.0.2,10.0.0.3,tcp,17489,1794241778.821126,18.343778,11420.539,ok
proposed,20261109-082859,M00364,H13,H7,10.3.0.2,10.1.1.2,tcp,37922,1794241778.924235,18.446887,2289.792,ok
proposed,20261109-082859,M00376,H13,H9,10.3.0.2,10.2.0.2,tcp,54307,1794241779.187740,18.710391,2668.370,ok
proposed,20261109-082859,M00380,H13,H12,10.3.0.2,10.2.1.3,tcp,50144,1794241779.376343,18.898995,3158.785,ok
proposed,20261109-082859,M00411,H13,H5,10.3.0.2,10.1.0.2,tcp,24706,1794241780.711286,20.233938,4337.376,ok
proposed,20261109-082859,M00414,H13,H3,10.3.0.2,10.0.1.2,tcp,47188,1794241780.836558,20.359210,4386.879,ok
proposed,20261109-082859,M00431,H13,H9,10.3.0.2,10.2.0.2,tcp,42759,1794241781.406072,20.928723,5815.797,ok
proposed,20261109-082859,M00440,H13,H9,10.3.0.2,10.2.0.2,tcp,32699,1794241781.784369,21.307020,5500.251,ok
proposed,20261109-082859,M00457,H13,H6,10.3.0.2,10.1.0.3,tcp,24815,1794241782.215740,21.738392,6105.043,ok
proposed,20261109-082859,M00471,H13,H12,10.3.0.2,10.2.1.3,tcp,37010,1794241783.002003,22.524655,8387.602,ok
proposed,20261109-082859,M00472,H13,H6,10.3.0.2,10.1.0.3,tcp,20435,1794241783.117229,22.639880,13899.878,ok
proposed,20261109-082859,M00502,H13,H12,10.3.0.2,10.2.1.3,tcp,55303,1794241784.482701,24.005352,9307.593,ok
proposed,20261109-082859,M00520,H13,H5,10.3.0.2,10.1.0.2,tcp,37915,1794241785.342655,24.865307,10010.847,timeout
proposed,20261109-082859,M00017,H9,H12,10.2.0.2,10.2.1.3,tcp,52988,1794241761.199874,0.722526,295.175,ok
proposed,20261109-082859,M00025,H9,H12,10.2.0.2,10.2.1.3,tcp,46503,1794241761.630858,1.153510,580.355,ok
proposed,20261109-082859,M00033,H9,H13,10.2.0.2,10.3.0.2,tcp,50198,1794241762.222647,1.745299,604.093,ok
proposed,20261109-082859,M00035,H9,H16,10.2.0.2,10.3.1.3,tcp,58127,1794241762.276724,1.799376,567.002,ok
proposed,20261109-082859,M00081,H9,H3,10.2.0.2,10.0.1.2,tcp,31064,1794241764.974870,4.497522,387.569,ok
proposed,20261109-082859,M00092,H9,H3,10.2.0.2,10.0.1.2,tcp,56769,1794241765.412749,4.935400,359.760,ok
proposed,20261109-082859,M00109,H9,H6,10.2.0.2,10.1.0.3,tcp,50232,1794241766.431173,5.953825,211.299,ok
proposed,20261109-082859,M00111,H9,H3,10.2.0.2,10.0.1.2,tcp,46241,1794241766.571244,6.093895,560.701,ok
proposed,20261109-082859,M00112,H9,H1,10.2.0.2,10.0.0.2,tcp,15296,1794241766.735268,6.257920,1292.585,ok
proposed,20261109-082859,M00140,H9,H13,10.2.0.2,10.3.0.2,tcp,24395,1794241767.990785,7.513437,689.564,ok
proposed,20261109-082859,M00141,H9,H13,10.2.0.2,10.3.0.2,tcp,59157,1794241768.083739,7.606391,838.966,ok
proposed,20261109-082859,M00179,H9,H10,10.2.0.2,10.2.0.3,tcp,61158,1794241769.479167,9.001819,391.977,ok
proposed,20261109-082859,M00181,H9,H6,10.2.0.2,10.1.0.3,tcp,64009,1794241769.691671,9.214323,1419.589,ok
proposed,20261109-082859,M00186,H9,H6,10.2.0.2,10.1.0.3,tcp,26519,1794241769.889435,9.412086,1160.795,ok
proposed,20261109-082859,M00187,H9,H16,10.2.0.2,10.3.1.3,tcp,8857,1794241769.997568,9.520220,965.951,ok
proposed,20261109-082859,M00198,H9,H12,10.2.0.2,10.2.1.3,tcp,46831,1794241770.392768,9.915420,698.788,ok
proposed,20261109-082859,M00215,H9,H11,10.2.0.2,10.2.1.2,tcp,13512,1794241771.404730,10.927381,1494.652,ok
proposed,20261109-082859,M00249,H9,H7,10.2.0.2,10.1.1.2,tcp,62896,1794241773.255448,12.778100,2729.000,ok
proposed,20261109-082859,M00266,H9,H6,10.2.0.2,10.1.0.3,tcp,49037,1794241774.105956,13.628608,3329.101,ok
proposed,20261109-082859,M00288,H9,H1,10.2.0.2,10.0.0.2,tcp,45280,1794241775.093693,14.616345,13305.240,ok
proposed,20261109-082859,M00292,H9,H11,10.2.0.2,10.2.1.2,tcp,57709,1794241775.288196,14.810848,3244.481,ok
proposed,20261109-082859,M00306,H9,H10,10.2.0.2,10.2.0.3,tcp,33162,1794241776.024523,15.547175,2478.927,ok
proposed,20261109-082859,M00359,H9,H1,10.2.0.2,10.0.0.2,tcp,20015,1794241778.741698,18.264350,12110.878,ok
proposed,20261109-082859,M00369,H9,H4,10.2.0.2,10.0.1.3,tcp,57667,1794241779.018211,18.540863,2501.827,ok
proposed,20261109-082859,M00374,H9,H5,10.2.0.2,10.1.0.2,tcp,29695,1794241779.166983,18.689635,2151.146,ok
proposed,20261109-082859,M00437,H9,H8,10.2.0.2,10.1.1.3,tcp,41835,1794241781.666591,21.189242,4156.732,ok
proposed,20261109-082859,M00488,H9,H4,10.2.0.2,10.0.1.3,tcp,15626,1794241783.795127,23.317778,8524.550,ok
proposed,20261109-082859,M00512,H9,H3,10.2.0.2,10.0.1.2,tcp,49153,1794241785.020697,24.543349,8707.599,ok
proposed,20261109-082859,M00517,H9,H11,10.2.0.2,10.2.1.2,tcp,22007,1794241785.234700,24.757352,10012.059,timeout
proposed,20261109-082859,M00018,H6,H9,10.1.0.3,10.2.0.2,tcp,55976,1794241761.252422,0.775074,345.056,ok
proposed,20261109-082859,M00029,H6,H13,10.1.0.3,10.3.0.2,tcp,52788,1794241761.969138,1.491789,212.093,ok
proposed,20261109-082859,M00044,H6,H5,10.1.0.3,10.1.0.2,tcp,42773,1794241762.778592,2.301243,364.001,ok
proposed,20261109-082859,M00064,H6,H7,10.1.0.3,10.1.1.2,tcp,40018,1794241764.205417,3.728069,245.401,ok
proposed,20261109-082859,M00071,H6,H1,10.1.0.3,10.0.0.2,tcp,41473,1794241764.540315,4.062967,1748.294,ok
proposed,20261109-082859,M00072,H6,H2,10.1.0.3,10.0.0.3,tcp,47188,1794241764.542665,4.065316,808.104,ok
proposed,20261109-082859,M00101,H6,H15,10.1.0.3,10.3.1.2,tcp,35212,1794241765.945534,5.468186,382.865,ok
proposed,20261109-082859,M00126,H6,H11,10.1.0.3,10.2.1.2,tcp,13749,1794241767.361270,6.883922,338.913,ok
proposed,20261109-082859,M00139,H6,H12,10.1.0.3,10.2.1.3,tcp,64650,1794241767.957684,7.480336,748.998,ok
proposed,20261109-082859,M00185,H6,H8,10.1.0.3,10.1.1.3,tcp,22093,1794241769.870345,9.392997,917.559,ok
proposed,20261109-082859,M00197,H6,H4,10.1.0.3,10.0.1.3,tcp,22372,1794241770.325439,9.848090,527.425,ok
proposed,20261109-082859,M00205,H6,H1,10.1.0.3,10.0.0.2,tcp,24738,1794241770.873169,10.395821,2618.736,ok
proposed,20261109-082859,M00216,H6,H9,10.1.0.3,10.2.0.2,tcp,13111,1794241771.412901,10.935552,1350.254,ok
proposed,20261109-082859,M00245,H6,H5,10.1.0.3,10.1.0.2,tcp,44291,1794241772.952557,12.475209,2245.696,ok
proposed,20261109-082859,M00277,H6,H3,10.1.0.3,10.0.1.2,tcp,19665,1794241774.652173,14.174825,2217.115,ok
proposed,20261109-082859,M00282,H6,H2,10.1.0.3,10.0.0.3,tcp,37141,1794241774.776586,14.299238,3617.340,ok
proposed,20261109-082859,M00299,H6,H5,10.1.0.3,10.1.0.2,tcp,62238,1794241775.688990,15.211641,2594.201,ok
proposed,20261109-082859,M00361,H6,H14,10.1.0.3,10.3.0.3,tcp,17580,1794241778.779802,18.302454,2891.428,ok
proposed,20261109-082859,M00382,H6,H13,10.1.0.3,10.3.0.2,tcp,48701,1794241779.565217,19.087869,3384.902,ok
proposed,20261109-082859,M00406,H6,H1,10.1.0.3,10.0.0.2,tcp,24428,1794241780.565153,20.087805,14427.980,ok
proposed,20261109-082859,M00424,H6,H14,10.1.0.3,10.3.0.3,tcp,53598,1794241781.208172,20.730824,4499.421,ok
proposed,20261109-082859,M00425,H6,H12,10.1.0.3,10.2.1.3,tcp,36875,1794241781.280700,20.803352,7197.512,ok
proposed,20261109-082859,M00432,H6,H14,10.1.0.3,10.3.0.3,tcp,21279,1794241781.432560,20.955212,5351.465,ok
proposed,20261109-082859,M00475,H6,H9,10.1.0.3,10.2.0.2,tcp,27444,1794241783.191020,22.713672,7399.892,ok
proposed,20261109-082859,M00482,H6,H1,10.1.0.3,10.0.0.2,tcp,17549,1794241783.400043,22.922694,10014.817,timeout
proposed,20261109-082859,M00489,H6,H8,10.1.0.3,10.1.1.3,tcp,42122,1794241783.794721,23.317373,9976.564,ok
proposed,20261109-082859,M00497,H6,H13,10.1.0.3,10.3.0.2,tcp,31882,1794241784.195306,23.717958,8260.402,ok
proposed,20261109-082859,M00507,H6,H13,10.1.0.3,10.3.0.2,tcp,16648,1794241784.764339,24.286990,8973.151,ok
proposed,20261109-082859,M00019,H15,H7,10.3.1.2,10.1.1.2,tcp,25883,1794241761.263476,0.786128,225.693,ok
proposed,20261109-082859,M00024,H15,H3,10.3.1.2,10.0.1.2,tcp,25551,1794241761.628247,1.150898,498.876,ok
proposed,20261109-082859,M00031,H15,H9,10.3.1.2,10.2.0.2,tcp,24668,1794241762.046403,1.569055,408.167,ok
proposed,20261109-082859,M00059,H15,H3,10.3.1.2,10.0.1.2,tcp,35840,1794241763.514302,3.036954,629.381,ok
proposed,20261109-082859,M00060,H15,H4,10.3.1.2,10.0.1.3,tcp,65502,1794241763.523643,3.046295,668.858,ok
proposed,20261109-082859,M00061,H15,H13,10.3.1.2,10.3.0.2,tcp,64641,1794241763.656976,3.179628,440.938,ok
proposed,20261109-082859,M00067,H15,H5,10.3.1.2,10.1.0.2,tcp,35914,1794241764.381073,3.903725,311.663,ok
proposed,20261109-082859,M00106,H15,H6,10.3.1.2,10.1.0.3,tcp,28192,1794241766.192414,5.715066,190.054,ok
proposed,20261109-082859,M00131,H15,H7,10.3.1.2,10.1.1.2,tcp,49464,1794241767.618236,7.140888,1073.830,ok
proposed,20261109-082859,M00135,H15,H1,10.3.1.2,10.0.0.2,tcp,44722,1794241767.788987,7.311639,1769.070,ok
proposed,20261109-082859,M00157,H15,H3,10.3.1.2,10.0.1.2,tcp,11487,1794241768.497796,8.020448,250.191,ok
proposed,20261109-082859,M00183,H15,H10,10.3.1.2,10.2.0.3,tcp,48190,1794241769.726137,9.248788,1272.892,ok
proposed,20261109-082859,M00212,H15,H5,10.3.1.2,10.1.0.2,tcp,64563,1794241771.260066,10.782717,1461.843,ok
proposed,20261109-082859,M00231,H15,H16,10.3.1.2,10.3.1.3,tcp,26421,1794241772.320329,11.842980,1337.931,ok
proposed,20261109-082859,M00237,H15,H2,10.3.1.2,10.0.0.3,tcp,28804,1794241772.560163,12.082815,4065.412,ok
proposed,20261109-082859,M00263,H15,H2,10.3.1.2,10.0.0.3,tcp,42660,1794241773.875827,13.398478,4502.929,ok
proposed,20261109-082859,M00280,H15,H5,10.3.1.2,10.1.0.2,tcp,51894,1794241774.719562,14.242214,2235.038,ok
proposed,20261109-082859,M00281,H15,H16,10.3.1.2,10.3.1.3,tcp,27042,1794241774.720242,14.242893,3629.041,ok
proposed,20261109-082859,M00284,H15,H14,10.3.1.2,10.3.0.3,tcp,27991,1794241774.980025,14.502677,3420.934,ok
proposed,20261109-082859,M00310,H15,H16,10.3.1.2,10.3.1.3,tcp,54137,1794241776.197958,15.720610,2229.114,ok
proposed,20261109-082859,M00314,H15,H6,10.3.1.2,10.1.0.3,tcp,52132,1794241776.229105,15.751757,2276.307,ok
proposed,20261109-082859,M00342,H15,H9,10.3.1.2,10.2.0.2,tcp,47869,1794241777.989794,17.512446,2271.605,ok
proposed,20261109-082859,M00392,H15,H13,10.3.1.2,10.3.0.2,tcp,45136,1794241779.901589,19.424241,2922.901,ok
proposed,20261109-082859,M00393,H15,H11,10.3.1.2,10.2.1.2,tcp,45186,1794241779.971492,19.494144,3896.857,ok
proposed,20261109-082859,M00395,H15,H2,10.3.1.2,10.0.0.3,tcp,38923,1794241780.045166,19.567817,13139.720,ok
proposed,20261109-082859,M00398,H15,H1,10.3.1.2,10.0.0.2,tcp,11920,1794241780.134583,19.657235,9982.439,ok
proposed,20261109-082859,M00401,H15,H10,10.3.1.2,10.2.0.3,tcp,44519,1794241780.291845,19.814497,3576.761,ok
proposed,20261109-082859,M00407,H15,H9,10.3.1.2,10.2.0.2,tcp,42450,1794241780.627918,20.150569,3925.682,ok
proposed,20261109-082859,M00427,H15,H10,10.3.1.2,10.2.0.3,tcp,26865,1794241781.295838,20.818489,6550.002,ok
proposed,20261109-082859,M00428,H15,H4,10.3.1.2,10.0.1.3,tcp,43188,1794241781.363875,20.886526,4404.779,ok
proposed,20261109-082859,M00430,H15,H8,10.3.1.2,10.1.1.3,tcp,58617,1794241781.389598,20.912249,4298.967,ok
proposed,20261109-082859,M00450,H15,H2,10.3.1.2,10.0.0.3,tcp,15457,1794241781.965857,21.488509,12558.649,ok
proposed,20261109-082859,M00462,H15,H12,10.3.1.2,10.2.1.3,tcp,39553,1794241782.408528,21.931180,7381.375,ok
proposed,20261109-082859,M00464,H15,H1,10.3.1.2,10.0.0.2,tcp,27423,1794241782.533522,22.056174,10016.048,timeout
proposed,20261109-082859,M00477,H15,H10,10.3.1.2,10.2.0.3,tcp,55224,1794241783.232499,22.755150,8802.319,ok
proposed,20261109-082859,M00478,H15,H6,10.3.1.2,10.1.0.3,tcp,37495,1794241783.263543,22.786195,10011.537,timeout
proposed,20261109-082859,M00483,H15,H1,10.3.1.2,10.0.0.2,tcp,16464,1794241783.495377,23.018028,10013.115,timeout
proposed,20261109-082859,M00491,H15,H4,10.3.1.2,10.0.1.3,tcp,48291,1794241783.835097,23.357749,7602.699,ok
proposed,20261109-082859,M00494,H15,H2,10.3.1.2,10.0.0.3,tcp,61309,1794241783.973264,23.495915,10010.951,timeout
proposed,20261109-082859,M00505,H15,H8,10.3.1.2,10.1.1.3,tcp,52763,1794241784.648370,24.171022,8157.553,ok
proposed,20261109-082859,M00522,H15,H4,10.3.1.2,10.0.1.3,tcp,30486,1794241785.372821,24.895473,8397.753,ok
proposed,20261109-082859,M00020,H8,H11,10.1.1.3,10.2.1.2,tcp,29444,1794241761.492567,1.015219,169.714,ok
proposed,20261109-082859,M00048,H8,H4,10.1.1.3,10.0.1.3,tcp,12388,1794241763.008503,2.531155,230.569,ok
proposed,20261109-082859,M00049,H8,H10,10.1.1.3,10.2.0.3,tcp,22624,1794241763.032812,2.555464,342.336,ok
proposed,20261109-082859,M00073,H8,H7,10.1.1.3,10.1.1.2,tcp,16048,1794241764.555937,4.078589,82.929,ok
proposed,20261109-082859,M00074,H8,H10,10.1.1.3,10.2.0.3,tcp,47154,1794241764.691536,4.214187,958.139,ok
proposed,20261109-082859,M00082,H8,H6,10.1.1.3,10.1.0.3,tcp,26868,1794241765.069043,4.591695,288.014,ok
proposed,20261109-082859,M00098,H8,H11,10.1.1.3,10.2.1.2,tcp,14928,1794241765.728169,5.250821,543.469,ok
proposed,20261109-082859,M00100,H8,H14,10.1.1.3,10.3.0.3,tcp,18841,1794241765.757503,5.280155,573.189,ok
proposed,20261109-082859,M00103,H8,H5,10.1.1.3,10.1.0.2,tcp,18625,1794241766.027842,5.550494,197.728,ok
proposed,20261109-082859,M00107,H8,H4,10.1.1.3,10.0.1.3,tcp,9742,1794241766.276985,5.799636,114.867,ok
proposed,20261109-082859,M00128,H8,H11,10.1.1.3,10.2.1.2,tcp,28535,1794241767.529866,7.052518,217.843,ok
proposed,20261109-082859,M00142,H8,H5,10.1.1.3,10.1.0.2,tcp,37019,1794241768.117685,7.640337,390.842,ok
proposed,20261109-082859,M00151,H8,H16,10.1.1.3,10.3.1.3,tcp,22572,1794241768.308915,7.831567,391.362,ok
proposed,20261109-082859,M00165,H8,H3,10.1.1.3,10.0.1.2,tcp,39227,1794241768.892838,8.415490,1006.219,ok
proposed,20261109-082859,M00172,H8,H2,10.1.1.3,10.0.0.3,tcp,53883,1794241769.149074,8.671726,792.424,ok
proposed,20261109-082859,M00192,H8,H5,10.1.1.3,10.1.0.2,tcp,51705,1794241770.155915,9.678566,656.272,ok
proposed,20261109-082859,M00200,H8,H6,10.1.1.3,10.1.0.3,tcp,19951,1794241770.592660,10.115312,1231.751,ok
proposed,20261109-082859,M00211,H8,H13,10.1.1.3,10.3.0.2,tcp,42439,1794241771.234935,10.757587,1390.935,ok
proposed,20261109-082859,M00220,H8,H15,10.1.1.3,10.3.1.2,tcp,25529,1794241771.510768,11.033420,1132.576,ok
proposed,20261109-082859,M00258,H8,H10,10.1.1.3,10.2.0.3,tcp,33290,1794241773.611827,13.134479,3098.095,ok
proposed,20261109-082859,M00276,H8,H9,10.1.1.3,10.2.0.2,tcp,35196,1794241774.609122,14.131774,2435.085,ok
proposed,20261109-082859,M00302,H8,H14,10.1.1.3,10.3.0.3,tcp,52515,1794241775.862986,15.385637,2645.765,ok
proposed,20261109-082859,M00309,H8,H11,10.1.1.3,10.2.1.2,tcp,12334,1794241776.144939,15.667591,2264.857,ok
proposed,20261109-082859,M00313,H8,H15,10.1.1.3,10.3.1.2,tcp,43420,1794241776.219543,15.742195,3303.687,ok
proposed,20261109-082859,M00326,H8,H6,10.1.1.3,10.1.0.3,tcp,58941,1794241776.969474,16.492126,3148.904,ok
proposed,20261109-082859,M00372,H8,H7,10.1.1.3,10.1.1.2,tcp,47469,1794241779.092827,18.615479,1640.968,ok
proposed,20261109-082859,M00384,H8,H7,10.1.1.3,10.1.1.2,tcp,46784,1794241779.686969,19.209620,3043.650,ok
proposed,20261109-082859,M00420,H8,H3,10.1.1.3,10.0.1.2,tcp,18346,1794241781.056066,20.578718,4565.895,ok
proposed,20261109-082859,M00452,H8,H12,10.1.1.3,10.2.1.3,tcp,17601,1794241782.058736,21.581388,5202.041,ok
proposed,20261109-082859,M00510,H8,H6,10.1.1.3,10.1.0.3,tcp,42198,1794241784.850951,24.373603,10011.732,timeout
proposed,20261109-082859,M00034,H10,H7,10.2.0.3,10.1.1.2,tcp,18557,1794241762.248039,1.770691,374.398,ok
proposed,20261109-082859,M00040,H10,H4,10.2.0.3,10.0.1.3,tcp,11988,1794241762.517712,2.040364,1105.785,ok
proposed,20261109-082859,M00078,H10,H8,10.2.0.3,10.1.1.3,tcp,28912,1794241764.771243,4.293895,643.056,ok
proposed,20261109-082859,M00083,H10,H16,10.2.0.3,10.3.1.3,tcp,61086,1794241765.078979,4.601631,501.579,ok
proposed,20261109-082859,M00087,H10,H16,10.2.0.3,10.3.1.3,tcp,21995,1794241765.207167,4.729819,401.800,ok
proposed,20261109-082859,M00122,H10,H7,10.2.0.3,10.1.1.2,tcp,44101,1794241767.136016,6.658668,784.556,ok
proposed,20261109-082859,M00123,H10,H5,10.2.0.3,10.1.0.2,tcp,21966,1794241767.226337,6.748988,618.184,ok
proposed,20261109-082859,M00143,H10,H4,10.2.0.3,10.0.1.3,tcp,25999,1794241768.122390,7.645042,597.319,ok
proposed,20261109-082859,M00169,H10,H2,10.2.0.3,10.0.0.3,tcp,46116,1794241769.016528,8.539180,1030.518,ok
proposed,20261109-082859,M00170,H10,H16,10.2.0.3,10.3.1.3,tcp,52582,1794241769.018006,8.540658,951.596,ok
proposed,20261109-082859,M00195,H10,H5,10.2.0.3,10.1.0.2,tcp,23300,1794241770.250863,9.773515,737.164,ok
proposed,20261109-082859,M00199,H10,H2,10.2.0.3,10.0.0.3,tcp,62759,1794241770.582416,10.105068,2806.591,ok
proposed,20261109-082859,M00202,H10,H12,10.2.0.3,10.2.1.3,tcp,16467,1794241770.792080,10.314731,1139.971,ok
proposed,20261109-082859,M00204,H10,H8,10.2.0.3,10.1.1.3,tcp,39565,1794241770.844525,10.367176,2367.343,ok
proposed,20261109-082859,M00209,H10,H2,10.2.0.3,10.0.0.3,tcp,24457,1794241771.150208,10.672860,2261.139,ok
proposed,20261109-082859,M00218,H10,H1,10.2.0.3,10.0.0.2,tcp,23362,1794241771.464804,10.987455,2286.716,ok
proposed,20261109-082859,M00219,H10,H12,10.2.0.3,10.2.1.3,tcp,64527,1794241771.479932,11.002584,1555.403,ok
proposed,20261109-082859,M00224,H10,H8,10.2.0.3,10.1.1.3,tcp,16341,1794241771.803313,11.325964,1502.493,ok
proposed,20261109-082859,M00225,H10,H12,10.2.0.3,10.2.1.3,tcp,34570,1794241771.828098,11.350750,1364.762,ok
proposed,20261109-082859,M00243,H10,H8,10.2.0.3,10.1.1.3,tcp,53952,1794241772.836561,12.359213,2601.272,ok
proposed,20261109-082859,M00271,H10,H4,10.2.0.3,10.0.1.3,tcp,57132,1794241774.377992,13.900643,2727.833,ok
proposed,20261109-082859,M00275,H10,H6,10.2.0.3,10.1.0.3,tcp,32757,1794241774.580824,14.103476,3779.537,ok
proposed,20261109-082859,M00283,H10,H11,10.2.0.3,10.2.1.2,tcp,35999,1794241774.927937,14.450588,3135.804,ok
proposed,20261109-082859,M00287,H10,H1,10.2.0.3,10.0.0.2,tcp,62589,1794241775.042362,14.565014,13512.039,timeout
proposed,20261109-082859,M00290,H10,H13,10.2.0.3,10.3.0.2,tcp,60546,1794241775.265238,14.787890,3622.239,ok
proposed,20261109-082859,M00296,H10,H6,10.2.0.3,10.1.0.3,tcp,57178,1794241775.500096,15.022748,3032.762,ok
proposed,20261109-082859,M00300,H10,H12,10.2.0.3,10.2.1.3,tcp,65234,1794241775.832445,15.355096,2829.984,ok
proposed,20261109-082859,M00336,H10,H8,10.2.0.3,10.1.1.3,tcp,50353,1794241777.604851,17.127502,3770.543,ok
proposed,20261109-082859,M00353,H10,H16,10.2.0.3,10.3.1.3,tcp,28668,1794241778.332911,17.855562,2999.093,ok
proposed,20261109-082859,M00360,H10,H15,10.2.0.3,10.3.1.2,tcp,30462,1794241778.755282,18.277933,2602.397,ok
proposed,20261109-082859,M00396,H10,H13,10.2.0.3,10.3.0.2,tcp,26193,1794241780.071401,19.594053,3486.678,ok
proposed,20261109-082859,M00399,H10,H2,10.2.0.3,10.0.0.3,tcp,50440,1794241780.157618,19.680269,17091.534,timeout
proposed,20261109-082859,M00449,H10,H2,10.2.0.3,10.0.0.3,tcp,11321,1794241781.952810,21.475462,9564.363,ok
proposed,20261109-082859,M00466,H10,H9,10.2.0.3,10.2.0.2,tcp,34290,1794241782.564680,22.087331,5844.678,ok
proposed,20261109-082859,M00496,H10,H15,10.2.0.3,10.3.1.2,tcp,27849,1794241784.101031,23.623682,7457.189,ok
proposed,20261109-082859,M00509,H10,H3,10.2.0.3,10.0.1.2,tcp,18806,1794241784.829744,24.352395,8008.565,ok
proposed,20261109-082859,M00519,H10,H11,10.2.0.3,10.2.1.2,tcp,27990,1794241785.311036,24.833688,10010.534,timeout
proposed,20261109-082859,M00036,H4,H14,10.0.1.3,10.3.0.3,tcp,49171,1794241762.288753,1.811405,546.287,ok
proposed,20261109-082859,M00039,H4,H15,10.0.1.3,10.3.1.2,tcp,31980,1794241762.413154,1.935805,405.096,ok
proposed,20261109-082859,M00057,H4,H1,10.0.1.3,10.0.0.2,tcp,34578,1794241763.447642,2.970294,1366.636,ok
proposed,20261109-082859,M00058,H4,H16,10.0.1.3,10.3.1.3,tcp,20748,1794241763.505625,3.028276,570.437,ok
proposed,20261109-082859,M00062,H4,H1,10.0.1.3,10.0.0.2,tcp,50930,1794241763.750402,3.273054,2058.868,ok
proposed,20261109-082859,M00080,H4,H2,10.0.1.3,10.0.0.3,tcp,43426,1794241764.953690,4.476341,1334.809,ok
proposed,20261109-082859,M00085,H4,H15,10.0.1.3,10.3.1.2,tcp,16992,1794241765.173128,4.695780,333.584,ok
proposed,20261109-082859,M00086,H4,H12,10.0.1.3,10.2.1.3,tcp,44448,1794241765.196127,4.718779,442.001,ok
proposed,20261109-082859,M00104,H4,H7,10.0.1.3,10.1.1.2,tcp,65350,1794241766.110160,5.632812,277.012,ok
proposed,20261109-082859,M00114,H4,H7,10.0.1.3,10.1.1.2,tcp,45979,1794241766.799628,6.322280,646.019,ok
proposed,20261109-082859,M00118,H4,H12,10.0.1.3,10.2.1.3,tcp,27873,1794241766.868160,6.390811,443.836,ok
proposed,20261109-082859,M00134,H4,H13,10.0.1.3,10.3.0.2,tcp,36054,1794241767.778311,7.300963,907.450,ok
proposed,20261109-082859,M00136,H4,H8,10.0.1.3,10.1.1.3,tcp,16930,1794241767.799189,7.321841,876.475,ok
proposed,20261109-082859,M00168,H4,H13,10.0.1.3,10.3.0.2,tcp,61885,1794241769.009494,8.532146,996.127,ok
proposed,20261109-082859,M00173,H4,H13,10.0.1.3,10.3.0.2,tcp,45280,1794241769.222300,8.744951,720.227,ok
proposed,20261109-082859,M00214,H4,H13,10.0.1.3,10.3.0.2,tcp,21809,1794241771.352677,10.875329,1769.399,ok
proposed,20261109-082859,M00238,H4,H13,10.0.1.3,10.3.0.2,tcp,34680,1794241772.575892,12.098543,1582.388,ok
proposed,20261109-082859,M00247,H4,H8,10.0.1.3,10.1.1.3,tcp,15890,1794241773.186715,12.709367,2083.439,ok
proposed,20261109-082859,M00272,H4,H13,10.0.1.3,10.3.0.2,tcp,49725,1794241774.381436,13.904088,2628.671,ok
proposed,20261109-082859,M00273,H4,H8,10.0.1.3,10.1.1.3,tcp,19094,1794241774.442428,13.965080,3267.548,ok
proposed,20261109-082859,M00301,H4,H8,10.0.1.3,10.1.1.3,tcp,13126,1794241775.852641,15.375293,2222.986,ok
proposed,20261109-082859,M00305,H4,H8,10.0.1.3,10.1.1.3,tcp,32347,1794241775.965695,15.488347,2234.710,ok
proposed,20261109-082859,M00321,H4,H14,10.0.1.3,10.3.0.3,tcp,62206,1794241776.782687,16.305339,3269.615,ok
proposed,20261109-082859,M00329,H4,H11,10.0.1.3,10.2.1.2,tcp,64975,1794241777.264678,16.787329,2496.983,ok
proposed,20261109-082859,M00334,H4,H8,10.0.1.3,10.1.1.3,tcp,41211,1794241777.480524,17.003176,2252.899,ok
proposed,20261109-082859,M00338,H4,H8,10.0.1.3,10.1.1.3,tcp,54879,1794241777.802623,17.325274,3525.227,ok
proposed,20261109-082859,M00343,H4,H14,10.0.1.3,10.3.0.3,tcp,51099,1794241778.020762,17.543414,3349.531,ok
proposed,20261109-082859,M00355,H4,H6,10.0.1.3,10.1.0.3,tcp,63594,1794241778.384222,17.906874,2954.618,ok
proposed,20261109-082859,M00356,H4,H13,10.0.1.3,10.3.0.2,tcp,26448,1794241778.425453,17.948105,2956.473,ok
proposed,20261109-082859,M00381,H4,H7,10.0.1.3,10.1.1.2,tcp,9558,1794241779.396595,18.919247,2949.056,ok
proposed,20261109-082859,M00421,H4,H9,10.0.1.3,10.2.0.2,tcp,58712,1794241781.056513,20.579164,6262.823,ok
proposed,20261109-082859,M00435,H4,H12,10.0.1.3,10.2.1.3,tcp,63628,1794241781.556105,21.078756,8543.343,ok
proposed,20261109-082859,M00436,H4,H5,10.0.1.3,10.1.0.2,tcp,13688,1794241781.592252,21.114904,5400.012,ok
proposed,20261109-082859,M00442,H4,H14,10.0.1.3,10.3.0.3,tcp,55030,1794241781.799881,21.322533,4439.040,ok
proposed,20261109-082859,M00459,H4,H13,10.0.1.3,10.3.0.2,tcp,46327,1794241782.310660,21.833312,5431.668,ok
proposed,20261109-082859,M00470,H4,H13,10.0.1.3,10.3.0.2,tcp,32310,1794241782.981365,22.504017,8974.977,ok
proposed,20261109-082859,M00490,H4,H5,10.0.1.3,10.1.0.2,tcp,58980,1794241783.828941,23.351592,10009.919,timeout
proposed,20261109-082859,M00498,H4,H16,10.0.1.3,10.3.1.3,tcp,60100,1794241784.229110,23.751761,9077.937,ok
proposed,20261109-082859,M00037,H7,H3,10.1.1.2,10.0.1.2,tcp,32696,1794241762.308781,1.831433,376.156,ok
proposed,20261109-082859,M00046,H7,H12,10.1.1.2,10.2.1.3,tcp,28620,1794241762.951383,2.474035,371.412,ok
proposed,20261109-082859,M00088,H7,H11,10.1.1.2,10.2.1.2,tcp,49757,1794241765.267131,4.789782,405.095,ok
proposed,20261109-082859,M00116,H7,H6,10.1.1.2,10.1.0.3,tcp,36457,1794241766.812370,6.335021,448.622,ok
proposed,20261109-082859,M00125,H7,H9,10.1.1.2,10.2.0.2,tcp,39202,1794241767.286953,6.809605,325.426,ok
proposed,20261109-082859,M00129,H7,H3,10.1.1.2,10.0.1.2,tcp,9792,1794241767.551603,7.074255,873.706,ok
proposed,20261109-082859,M00164,H7,H14,10.1.1.2,10.3.0.3,tcp,9527,1794241768.851313,8.373965,935.072,ok
proposed,20261109-082859,M00166,H7,H8,10.1.1.2,10.1.1.3,tcp,54032,1794241768.946734,8.469386,780.323,ok
proposed,20261109-082859,M00188,H7,H2,10.1.1.2,10.0.0.3,tcp,24009,1794241770.032527,9.555178,1991.713,ok
proposed,20261109-082859,M00196,H7,H6,10.1.1.2,10.1.0.3,tcp,16027,1794241770.257886,9.780538,542.063,ok
proposed,20261109-082859,M00221,H7,H16,10.1.1.2,10.3.1.3,tcp,15713,1794241771.593288,11.115940,1014.782,ok
proposed,20261109-082859,M00242,H7,H5,10.1.1.2,10.1.0.2,tcp,44185,1794241772.826320,12.348971,2466.165,ok
proposed,20261109-082859,M00297,H7,H3,10.1.1.2,10.0.1.2,tcp,59736,1794241775.529211,15.051862,3111.828,ok
proposed,20261109-082859,M00318,H7,H8,10.1.1.2,10.1.1.3,tcp,37335,1794241776.594223,16.116874,2709.144,ok
proposed,20261109-082859,M00322,H7,H11,10.1.1.2,10.2.1.2,tcp,50101,1794241776.815008,16.337659,3411.709,ok
proposed,20261109-082859,M00325,H7,H10,10.1.1.2,10.2.0.3,tcp,22344,1794241776.885507,16.408159,3005.607,ok
proposed,20261109-082859,M00344,H7,H16,10.1.1.2,10.3.1.3,tcp,37785,1794241778.073567,17.596219,2900.927,ok
proposed,20261109-082859,M00354,H7,H3,10.1.1.2,10.0.1.2,tcp,51231,1794241778.352710,17.875361,3221.037,ok
proposed,20261109-082859,M00373,H7,H5,10.1.1.2,10.1.0.2,tcp,22035,1794241779.093544,18.616195,1944.642,ok
proposed,20261109-082859,M00388,H7,H15,10.1.1.2,10.3.1.2,tcp,59062,1794241779.812741,19.335393,3872.464,ok
proposed,20261109-082859,M00390,H7,H3,10.1.1.2,10.0.1.2,tcp,47670,1794241779.841536,19.364188,3486.494,ok
proposed,20261109-082859,M00403,H7,H6,10.1.1.2,10.1.0.3,tcp,47841,1794241780.491871,20.014522,3252.951,ok
proposed,20261109-082859,M00419,H7,H6,10.1.1.2,10.1.0.3,tcp,58402,1794241781.024575,20.547226,5138.855,ok
proposed,20261109-082859,M00465,H7,H15,10.1.1.2,10.3.1.2,tcp,10938,1794241782.542895,22.065547,7148.904,ok
proposed,20261109-082859,M00468,H7,H5,10.1.1.2,10.1.0.2,tcp,31586,1794241782.854830,22.377482,8378.634,ok
proposed,20261109-082859,M00052,H16,H4,10.3.1.3,10.0.1.3,tcp,43531,1794241763.165260,2.687912,511.225,ok
proposed,20261109-082859,M00053,H16,H4,10.3.1.3,10.0.1.3,tcp,59612,1794241763.169364,2.692016,517.552,ok
proposed,20261109-082859,M00068,H16,H3,10.3.1.3,10.0.1.2,tcp,20637,1794241764.436852,3.959504,280.352,ok
proposed,20261109-082859,M00089,H16,H5,10.3.1.3,10.1.0.2,tcp,63642,1794241765.354385,4.877036,389.594,ok
proposed,20261109-082859,M00105,H16,H4,10.3.1.3,10.0.1.3,tcp,21271,1794241766.110608,5.633260,215.184,ok
proposed,20261109-082859,M00130,H16,H10,10.3.1.3,10.2.0.3,tcp,63880,1794241767.550556,7.073208,1161.832,ok
proposed,20261109-082859,M00149,H16,H1,10.3.1.3,10.0.0.2,tcp,31244,1794241768.267341,7.789993,759.404,ok
proposed,20261109-082859,M00152,H16,H1,10.3.1.3,10.0.0.2,tcp,33676,1794241768.331765,7.854416,697.662,ok
proposed,20261109-082859,M00180,H16,H8,10.3.1.3,10.1.1.3,tcp,36737,1794241769.604431,9.127083,1226.397,ok
proposed,20261109-082859,M00189,H16,H9,10.3.1.3,10.2.0.2,tcp,57879,1794241770.095815,9.618467,2077.975,ok
proposed,20261109-082859,M00190,H16,H11,10.3.1.3,10.2.1.2,tcp,54843,1794241770.109877,9.632529,743.175,ok
proposed,20261109-082859,M00193,H16,H9,10.3.1.3,10.2.0.2,tcp,42986,1794241770.206072,9.728723,1776.688,ok
proposed,20261109-082859,M00203,H16,H2,10.3.1.3,10.0.0.3,tcp,65386,1794241770.843692,10.366344,2575.282,ok
proposed,20261109-082859,M00206,H16,H2,10.3.1.3,10.0.0.3,tcp,62070,1794241771.014180,10.536832,2988.631,ok
proposed,20261109-082859,M00207,H16,H2,10.3.1.3,10.0.0.3,tcp,46007,1794241771.017473,10.540124,2342.707,ok
proposed,20261109-082859,M00226,H16,H8,10.3.1.3,10.1.1.3,tcp,13463,1794241771.968413,11.491065,1186.489,ok
proposed,20261109-082859,M00241,H16,H11,10.3.1.3,10.2.1.2,tcp,37161,1794241772.752997,12.275649,2580.486,ok
proposed,20261109-082859,M00250,H16,H4,10.3.1.3,10.0.1.3,tcp,38127,1794241773.288685,12.811337,2017.404,ok
proposed,20261109-082859,M00255,H16,H14,10.3.1.3,10.3.0.3,tcp,17930,1794241773.433546,12.956197,1764.905,ok
proposed,20261109-082859,M00256,H16,H6,10.3.1.3,10.1.0.3,tcp,29978,1794241773.464006,12.986658,1856.931,ok
proposed,20261109-082859,M00269,H16,H14,10.3.1.3,10.3.0.3,tcp,43648,1794241774.317737,13.840389,2562.087,ok
proposed,20261109-082859,M00371,H16,H9,10.3.1.3,10.2.0.2,tcp,51913,1794241779.082599,18.605251,2738.563,ok
proposed,20261109-082859,M00375,H16,H12,10.3.1.3,10.2.1.3,tcp,36415,1794241779.171765,18.694417,2103.882,ok
proposed,20261109-082859,M00429,H16,H15,10.3.1.3,10.3.1.2,tcp,62921,1794241781.380291,20.902942,4782.839,ok
proposed,20261109-082859,M00434,H16,H14,10.3.1.3,10.3.0.3,tcp,32519,1794241781.519125,21.041777,4689.740,ok
proposed,20261109-082859,M00444,H16,H12,10.3.1.3,10.2.1.3,tcp,62659,1794241781.868245,21.390897,5388.943,ok
proposed,20261109-082859,M00445,H16,H8,10.3.1.3,10.1.1.3,tcp,48519,1794241781.877294,21.399946,4391.995,ok
proposed,20261109-082859,M00451,H16,H10,10.3.1.3,10.2.0.3,tcp,43322,1794241782.057075,21.579726,4348.012,ok
proposed,20261109-082859,M00467,H16,H5,10.3.1.3,10.1.0.2,tcp,10591,1794241782.794343,22.316994,8438.907,ok
proposed,20261109-082859,M00473,H16,H6,10.3.1.3,10.1.0.3,tcp,60443,1794241783.150490,22.673141,10010.262,timeout
proposed,20261109-082859,M00492,H16,H4,10.3.1.3,10.0.1.3,tcp,16229,1794241783.852788,23.375440,7457.733,ok
proposed,20261109-082859,M00499,H16,H1,10.3.1.3,10.0.0.2,tcp,31843,1794241784.238892,23.761543,10019.917,timeout

```

# results\plots\core_load.png

This is a binary file of the type: Image

# results\plots\fct_cdf.png

This is a binary file of the type: Image

# results\plots\throughput_vs_time.png

This is a binary file of the type: Image

# results\port_stats.csv

```csv
epoch,scheme,dpid,switch,port,tx_mbps,rx_mbps,utilization
1794241551.851,proposed,196865,e_1_0,4,0.0,0.0,0.0
1794241551.851,proposed,196865,e_1_0,1,0.0,0.0,0.0
1794241551.851,proposed,196865,e_1_0,2,0.0,0.0,0.0
1794241551.851,proposed,196865,e_1_0,3,0.0,0.0,0.0
1794241551.853,proposed,131584,a_2_0,4,0.0,0.0,0.0
1794241551.853,proposed,131584,a_2_0,1,0.0,0.0,0.0
1794241551.853,proposed,131584,a_2_0,2,0.0,0.0,0.0
1794241551.853,proposed,131584,a_2_0,3,0.0,0.0,0.0
1794241551.857,proposed,196610,e_0_1,4,0.0,0.0,0.0
1794241551.857,proposed,196610,e_0_1,1,0.0,0.0,0.0
1794241551.857,proposed,196610,e_0_1,2,0.0,0.0,0.0
1794241551.857,proposed,196610,e_0_1,3,0.0,0.0,0.0
1794241551.861,proposed,131072,a_0_0,4,0.0,0.0,0.0
1794241551.861,proposed,131072,a_0_0,1,0.0,0.0,0.0
1794241551.861,proposed,131072,a_0_0,2,0.0,0.0,0.0
1794241551.861,proposed,131072,a_0_0,3,0.0,0.0,0.0
1794241551.864,proposed,131840,a_3_0,4,0.0,0.0,0.0
1794241551.864,proposed,131840,a_3_0,1,0.0,0.0,0.0
1794241551.864,proposed,131840,a_3_0,2,0.0,0.0,0.0
1794241551.864,proposed,131840,a_3_0,3,0.0,0.0,0.0
1794241551.866,proposed,197377,e_3_0,1,0.0,0.0,0.0
1794241551.866,proposed,197377,e_3_0,4,0.0,0.0,0.0
1794241551.866,proposed,197377,e_3_0,2,0.0,0.0,0.0
1794241551.866,proposed,197377,e_3_0,3,0.0,0.0,0.0
1794241551.868,proposed,196609,e_0_0,1,0.0,0.0,0.0
1794241551.868,proposed,196609,e_0_0,4,0.0,0.0,0.0
1794241551.868,proposed,196609,e_0_0,2,0.0,0.0,0.0
1794241551.868,proposed,196609,e_0_0,3,0.0,0.0,0.0
1794241551.871,proposed,65538,c_1_0,4,0.0,0.0,0.0
1794241551.871,proposed,65538,c_1_0,1,0.0,0.0,0.0
1794241551.871,proposed,65538,c_1_0,2,0.0,0.0,0.0
1794241551.871,proposed,65538,c_1_0,3,0.0,0.0,0.0
1794241551.875,proposed,131329,a_1_1,1,0.0,0.0,0.0
1794241551.875,proposed,131329,a_1_1,4,0.0,0.0,0.0
1794241551.875,proposed,131329,a_1_1,2,0.0,0.0,0.0
1794241551.875,proposed,131329,a_1_1,3,0.0,0.0,0.0
1794241551.878,proposed,197378,e_3_1,4,0.0,0.0,0.0
1794241551.878,proposed,197378,e_3_1,1,0.0,0.0,0.0
1794241551.878,proposed,197378,e_3_1,2,0.0,0.0,0.0
1794241551.878,proposed,197378,e_3_1,3,0.0,0.0,0.0
1794241551.88,proposed,131328,a_1_0,1,0.0,0.0,0.0
1794241551.88,proposed,131328,a_1_0,4,0.0,0.0,0.0
1794241551.88,proposed,131328,a_1_0,2,0.0,0.0,0.0
1794241551.88,proposed,131328,a_1_0,3,0.0,0.0,0.0
1794241551.883,proposed,65539,c_1_1,4,0.0,0.0,0.0
1794241551.883,proposed,65539,c_1_1,1,0.0,0.0,0.0
1794241551.883,proposed,65539,c_1_1,2,0.0,0.0,0.0
1794241551.883,proposed,65539,c_1_1,3,0.0,0.0,0.0
1794241551.885,proposed,196866,e_1_1,4,0.0,0.0,0.0
1794241551.885,proposed,196866,e_1_1,1,0.0,0.0,0.0
1794241551.885,proposed,196866,e_1_1,2,0.0,0.0,0.0
1794241551.885,proposed,196866,e_1_1,3,0.0,0.0,0.0
1794241551.888,proposed,197122,e_2_1,4,0.0,0.0,0.0
1794241551.888,proposed,197122,e_2_1,1,0.0,0.0,0.0
1794241551.888,proposed,197122,e_2_1,2,0.0,0.0,0.0
1794241551.888,proposed,197122,e_2_1,3,0.0,0.0,0.0
1794241551.891,proposed,131073,a_0_1,1,0.0,0.0,0.0
1794241551.891,proposed,131073,a_0_1,4,0.0,0.0,0.0
1794241551.891,proposed,131073,a_0_1,2,0.0,0.0,0.0
1794241551.891,proposed,131073,a_0_1,3,0.0,0.0,0.0
1794241551.893,proposed,65537,c_0_1,4,0.0,0.0,0.0
1794241551.893,proposed,65537,c_0_1,1,0.0,0.0,0.0
1794241551.893,proposed,65537,c_0_1,2,0.0,0.0,0.0
1794241551.893,proposed,65537,c_0_1,3,0.0,0.0,0.0
1794241551.895,proposed,131585,a_2_1,1,0.0,0.0,0.0
1794241551.895,proposed,131585,a_2_1,4,0.0,0.0,0.0
1794241551.895,proposed,131585,a_2_1,2,0.0,0.0,0.0
1794241551.895,proposed,131585,a_2_1,3,0.0,0.0,0.0
1794241551.898,proposed,65536,c_0_0,4,0.0,0.0,0.0
1794241551.898,proposed,65536,c_0_0,1,0.0,0.0,0.0
1794241551.898,proposed,65536,c_0_0,2,0.0,0.0,0.0
1794241551.898,proposed,65536,c_0_0,3,0.0,0.0,0.0
1794241551.9,proposed,197121,e_2_0,4,0.0,0.0,0.0
1794241551.9,proposed,197121,e_2_0,1,0.0,0.0,0.0
1794241551.9,proposed,197121,e_2_0,2,0.0,0.0,0.0
1794241551.9,proposed,197121,e_2_0,3,0.0,0.0,0.0
1794241551.903,proposed,131841,a_3_1,1,0.0,0.0,0.0
1794241551.903,proposed,131841,a_3_1,4,0.0,0.0,0.0
1794241551.903,proposed,131841,a_3_1,2,0.0,0.0,0.0
1794241551.903,proposed,131841,a_3_1,3,0.0,0.0,0.0
1794241552.851,proposed,131073,a_0_1,1,0.0,0.0,0.0
1794241552.851,proposed,131073,a_0_1,4,0.0,0.0,0.0
1794241552.851,proposed,131073,a_0_1,2,0.0,0.0,0.0
1794241552.851,proposed,131073,a_0_1,3,0.0,0.0,0.0
1794241552.853,proposed,131840,a_3_0,4,0.0,0.0,0.0
1794241552.853,proposed,131840,a_3_0,1,0.0,0.0,0.0
1794241552.853,proposed,131840,a_3_0,2,0.0,0.0,0.0
1794241552.853,proposed,131840,a_3_0,3,0.0,0.0,0.0
1794241552.855,proposed,196610,e_0_1,4,0.0,0.0,0.0
1794241552.855,proposed,196610,e_0_1,1,0.0,0.0,0.0
1794241552.855,proposed,196610,e_0_1,2,0.0,0.0,0.0
1794241552.855,proposed,196610,e_0_1,3,0.0,0.0,0.0
1794241552.857,proposed,65536,c_0_0,4,0.0,0.0,0.0
1794241552.857,proposed,65536,c_0_0,1,0.0,0.0,0.0
1794241552.857,proposed,65536,c_0_0,2,0.0,0.0,0.0
1794241552.857,proposed,65536,c_0_0,3,0.0,0.0,0.0
1794241552.86,proposed,131328,a_1_0,1,0.0,0.0,0.0
1794241552.86,proposed,131328,a_1_0,4,0.0,0.0,0.0
1794241552.86,proposed,131328,a_1_0,2,0.0,0.0,0.0
1794241552.86,proposed,131328,a_1_0,3,0.0,0.0,0.0
1794241552.863,proposed,131584,a_2_0,4,0.0,0.0,0.0
1794241552.863,proposed,131584,a_2_0,1,0.0,0.0,0.0
1794241552.863,proposed,131584,a_2_0,2,0.0,0.0,0.0
1794241552.863,proposed,131584,a_2_0,3,0.0,0.0,0.0
1794241552.865,proposed,131841,a_3_1,1,0.0,0.0,0.0
1794241552.865,proposed,131841,a_3_1,4,0.0,0.0,0.0
1794241552.865,proposed,131841,a_3_1,2,0.0,0.0,0.0
1794241552.865,proposed,131841,a_3_1,3,0.0,0.0,0.0
1794241552.868,proposed,197378,e_3_1,4,0.0,0.0,0.0
1794241552.868,proposed,197378,e_3_1,1,0.0,0.0,0.0
1794241552.868,proposed,197378,e_3_1,2,0.0,0.0,0.0
1794241552.868,proposed,197378,e_3_1,3,0.0,0.0,0.0
1794241552.871,proposed,65538,c_1_0,4,0.0,0.0,0.0
1794241552.871,proposed,65538,c_1_0,1,0.0,0.0,0.0
1794241552.871,proposed,65538,c_1_0,2,0.0,0.0,0.0
1794241552.871,proposed,65538,c_1_0,3,0.0,0.0,0.0
1794241552.873,proposed,65537,c_0_1,4,0.0,0.0,0.0
1794241552.873,proposed,65537,c_0_1,1,0.0,0.0,0.0
1794241552.873,proposed,65537,c_0_1,2,0.0,0.0,0.0
1794241552.873,proposed,65537,c_0_1,3,0.0,0.0,0.0
1794241552.876,proposed,196865,e_1_0,4,0.0,0.0,0.0
1794241552.876,proposed,196865,e_1_0,1,0.0,0.0,0.0
1794241552.876,proposed,196865,e_1_0,2,0.0,0.0,0.0
1794241552.876,proposed,196865,e_1_0,3,0.0,0.0,0.0
1794241552.878,proposed,65539,c_1_1,4,0.0,0.0,0.0
1794241552.878,proposed,65539,c_1_1,1,0.0,0.0,0.0
1794241552.878,proposed,65539,c_1_1,2,0.0,0.0,0.0
1794241552.878,proposed,65539,c_1_1,3,0.0,0.0,0.0
1794241552.881,proposed,131072,a_0_0,4,0.0,0.0,0.0
1794241552.881,proposed,131072,a_0_0,1,0.0,0.0,0.0
1794241552.881,proposed,131072,a_0_0,2,0.0,0.0,0.0
1794241552.881,proposed,131072,a_0_0,3,0.0,0.0,0.0
1794241552.883,proposed,131585,a_2_1,1,0.0,0.0,0.0
1794241552.883,proposed,131585,a_2_1,4,0.0,0.0,0.0
1794241552.883,proposed,131585,a_2_1,2,0.0,0.0,0.0
1794241552.883,proposed,131585,a_2_1,3,0.0,0.0,0.0
1794241552.885,proposed,197377,e_3_0,1,0.0,0.0,0.0
1794241552.885,proposed,197377,e_3_0,4,0.0,0.0,0.0
1794241552.885,proposed,197377,e_3_0,2,0.0,0.0,0.0
1794241552.885,proposed,197377,e_3_0,3,0.0,0.0,0.0
1794241552.888,proposed,197122,e_2_1,4,0.0,0.0,0.0
1794241552.888,proposed,197122,e_2_1,1,0.0,0.0,0.0
1794241552.888,proposed,197122,e_2_1,2,0.0,0.0,0.0
1794241552.888,proposed,197122,e_2_1,3,0.0,0.0,0.0
1794241552.891,proposed,131329,a_1_1,1,0.0,0.0,0.0
1794241552.891,proposed,131329,a_1_1,4,0.0,0.0,0.0
1794241552.891,proposed,131329,a_1_1,2,0.0,0.0,0.0
1794241552.891,proposed,131329,a_1_1,3,0.0,0.0,0.0
1794241552.893,proposed,196866,e_1_1,4,0.0,0.0,0.0
1794241552.893,proposed,196866,e_1_1,1,0.0,0.0,0.0
1794241552.893,proposed,196866,e_1_1,2,0.0,0.0,0.0
1794241552.893,proposed,196866,e_1_1,3,0.0,0.0,0.0
1794241552.896,proposed,197121,e_2_0,4,0.0,0.0,0.0
1794241552.896,proposed,197121,e_2_0,1,0.0,0.0,0.0
1794241552.896,proposed,197121,e_2_0,2,0.0,0.0,0.0
1794241552.896,proposed,197121,e_2_0,3,0.0,0.0,0.0
1794241552.899,proposed,196609,e_0_0,1,0.0,0.0,0.0
1794241552.899,proposed,196609,e_0_0,4,0.0,0.0,0.0
1794241552.899,proposed,196609,e_0_0,2,0.0,0.0,0.0
1794241552.899,proposed,196609,e_0_0,3,0.0,0.0,0.0
1794241553.853,proposed,131073,a_0_1,1,0.0,0.0,0.0
1794241553.853,proposed,131073,a_0_1,4,0.0,0.0,0.0
1794241553.853,proposed,131073,a_0_1,2,0.0,0.0,0.0
1794241553.853,proposed,131073,a_0_1,3,0.0,0.0,0.0
1794241553.856,proposed,65537,c_0_1,4,0.0,0.0,0.0
1794241553.856,proposed,65537,c_0_1,1,0.0,0.0,0.0
1794241553.856,proposed,65537,c_0_1,2,0.0,0.0,0.0
1794241553.856,proposed,65537,c_0_1,3,0.0,0.0,0.0
1794241553.859,proposed,65538,c_1_0,4,0.0,0.0,0.0
1794241553.859,proposed,65538,c_1_0,1,0.0,0.0,0.0
1794241553.859,proposed,65538,c_1_0,2,0.0,0.0,0.0
1794241553.859,proposed,65538,c_1_0,3,0.0,0.0,0.0
1794241553.865,proposed,131585,a_2_1,1,0.0,0.0,0.0
1794241553.865,proposed,131585,a_2_1,4,0.0,0.0,0.0
1794241553.865,proposed,131585,a_2_1,2,0.0,0.0,0.0
1794241553.865,proposed,131585,a_2_1,3,0.0,0.0,0.0
1794241553.868,proposed,131329,a_1_1,1,0.0,0.0,0.0
1794241553.868,proposed,131329,a_1_1,4,0.0,0.0,0.0
1794241553.868,proposed,131329,a_1_1,2,0.0,0.0,0.0
1794241553.868,proposed,131329,a_1_1,3,0.0,0.0,0.0
1794241553.872,proposed,65536,c_0_0,4,0.0,0.0,0.0
1794241553.872,proposed,65536,c_0_0,1,0.0,0.0,0.0
1794241553.872,proposed,65536,c_0_0,2,0.0,0.0,0.0
1794241553.872,proposed,65536,c_0_0,3,0.0,0.0,0.0
1794241553.875,proposed,196865,e_1_0,4,0.0,0.0,0.0
1794241553.875,proposed,196865,e_1_0,1,0.0,0.0,0.0
1794241553.875,proposed,196865,e_1_0,2,0.0,0.0,0.0
1794241553.875,proposed,196865,e_1_0,3,0.0,0.0,0.0
1794241553.878,proposed,65539,c_1_1,4,0.0,0.0,0.0
1794241553.878,proposed,65539,c_1_1,1,0.0,0.0,0.0
1794241553.878,proposed,65539,c_1_1,2,0.0,0.0,0.0
1794241553.878,proposed,65539,c_1_1,3,0.0,0.0,0.0
1794241553.881,proposed,131840,a_3_0,4,0.0,0.0,0.0
1794241553.881,proposed,131840,a_3_0,1,0.0,0.0,0.0
1794241553.881,proposed,131840,a_3_0,2,0.0,0.0,0.0
1794241553.881,proposed,131840,a_3_0,3,0.0,0.0,0.0
1794241553.884,proposed,197377,e_3_0,1,0.0,0.0,0.0
1794241553.884,proposed,197377,e_3_0,4,0.0,0.0,0.0
1794241553.884,proposed,197377,e_3_0,2,0.0,0.0,0.0
1794241553.884,proposed,197377,e_3_0,3,0.0,0.0,0.0
1794241553.887,proposed,131328,a_1_0,1,0.0,0.0,0.0
1794241553.887,proposed,131328,a_1_0,4,0.0,0.0,0.0
1794241553.887,proposed,131328,a_1_0,2,0.0,0.0,0.0
1794241553.887,proposed,131328,a_1_0,3,0.0,0.0,0.0
1794241553.891,proposed,196866,e_1_1,4,0.0,0.0,0.0
1794241553.891,proposed,196866,e_1_1,1,0.0,0.0,0.0
1794241553.891,proposed,196866,e_1_1,2,0.0,0.0,0.0
1794241553.891,proposed,196866,e_1_1,3,0.0,0.0,0.0
1794241553.893,proposed,131072,a_0_0,4,0.0,0.0,0.0
1794241553.893,proposed,131072,a_0_0,1,0.0,0.0,0.0
1794241553.893,proposed,131072,a_0_0,2,0.0,0.0,0.0
1794241553.893,proposed,131072,a_0_0,3,0.0,0.0,0.0
1794241553.897,proposed,196610,e_0_1,4,0.0,0.0,0.0
1794241553.897,proposed,196610,e_0_1,1,0.0,0.0,0.0
1794241553.897,proposed,196610,e_0_1,2,0.0,0.0,0.0
1794241553.897,proposed,196610,e_0_1,3,0.0,0.0,0.0
1794241553.9,proposed,131584,a_2_0,4,0.0,0.0,0.0
1794241553.9,proposed,131584,a_2_0,1,0.0,0.0,0.0
1794241553.9,proposed,131584,a_2_0,2,0.0,0.0,0.0
1794241553.9,proposed,131584,a_2_0,3,0.0,0.0,0.0
1794241553.904,proposed,197121,e_2_0,4,0.0,0.0,0.0
1794241553.904,proposed,197121,e_2_0,1,0.0,0.0,0.0
1794241553.904,proposed,197121,e_2_0,2,0.0,0.0,0.0
1794241553.904,proposed,197121,e_2_0,3,0.0,0.0,0.0
1794241553.907,proposed,197378,e_3_1,4,0.0,0.0,0.0
1794241553.907,proposed,197378,e_3_1,1,0.0,0.0,0.0
1794241553.907,proposed,197378,e_3_1,2,0.0,0.0,0.0
1794241553.907,proposed,197378,e_3_1,3,0.0,0.0,0.0
1794241553.909,proposed,131841,a_3_1,1,0.0,0.0,0.0
1794241553.909,proposed,131841,a_3_1,4,0.0,0.0,0.0
1794241553.909,proposed,131841,a_3_1,2,0.0,0.0,0.0
1794241553.909,proposed,131841,a_3_1,3,0.0,0.0,0.0
1794241553.911,proposed,197122,e_2_1,4,0.0,0.0,0.0
1794241553.911,proposed,197122,e_2_1,1,0.0,0.0,0.0
1794241553.911,proposed,197122,e_2_1,2,0.0,0.0,0.0
1794241553.911,proposed,197122,e_2_1,3,0.0,0.0,0.0
1794241553.914,proposed,196609,e_0_0,1,0.0,0.0,0.0
1794241553.914,proposed,196609,e_0_0,4,0.0,0.0,0.0
1794241553.914,proposed,196609,e_0_0,2,0.0,0.0,0.0
1794241553.914,proposed,196609,e_0_0,3,0.0,0.0,0.0
1794241554.854,proposed,65537,c_0_1,4,0.0,0.0,0.0
1794241554.854,proposed,65537,c_0_1,1,0.0,0.0,0.0
1794241554.854,proposed,65537,c_0_1,2,0.0,0.0,0.0
1794241554.854,proposed,65537,c_0_1,3,0.0,0.0,0.0
1794241554.859,proposed,131585,a_2_1,1,0.0,0.0,0.0
1794241554.859,proposed,131585,a_2_1,4,0.0,0.0,0.0
1794241554.859,proposed,131585,a_2_1,2,0.0,0.0,0.0
1794241554.859,proposed,131585,a_2_1,3,0.0,0.0,0.0
1794241554.861,proposed,65538,c_1_0,4,0.0,0.0,0.0
1794241554.861,proposed,65538,c_1_0,1,0.0,0.0,0.0
1794241554.861,proposed,65538,c_1_0,2,0.0,0.0,0.0
1794241554.861,proposed,65538,c_1_0,3,0.0,0.0,0.0
1794241554.864,proposed,197122,e_2_1,4,0.0,0.0,0.0
1794241554.864,proposed,197122,e_2_1,1,0.0,0.0,0.0
1794241554.864,proposed,197122,e_2_1,2,0.0,0.0,0.0
1794241554.864,proposed,197122,e_2_1,3,0.0,0.0,0.0
1794241554.867,proposed,131073,a_0_1,1,0.0,0.0,0.0
1794241554.867,proposed,131073,a_0_1,4,0.0,0.0,0.0
1794241554.867,proposed,131073,a_0_1,2,0.0,0.0,0.0
1794241554.867,proposed,131073,a_0_1,3,0.0,0.0,0.0
1794241554.873,proposed,196610,e_0_1,4,0.0,0.0,0.0
1794241554.873,proposed,196610,e_0_1,1,0.0,0.0,0.0
1794241554.873,proposed,196610,e_0_1,2,0.0,0.0,0.0
1794241554.873,proposed,196610,e_0_1,3,0.0,0.0,0.0
1794241554.875,proposed,131584,a_2_0,4,0.0,0.0,0.0
1794241554.875,proposed,131584,a_2_0,1,0.0,0.0,0.0
1794241554.875,proposed,131584,a_2_0,2,0.0,0.0,0.0
1794241554.875,proposed,131584,a_2_0,3,0.0,0.0,0.0
1794241554.877,proposed,131840,a_3_0,4,0.0,0.0,0.0
1794241554.877,proposed,131840,a_3_0,1,0.0,0.0,0.0
1794241554.877,proposed,131840,a_3_0,2,0.0,0.0,0.0
1794241554.877,proposed,131840,a_3_0,3,0.0,0.0,0.0
1794241554.88,proposed,197121,e_2_0,4,0.0,0.0,0.0
1794241554.88,proposed,197121,e_2_0,1,0.0,0.0,0.0
1794241554.88,proposed,197121,e_2_0,2,0.0,0.0,0.0
1794241554.88,proposed,197121,e_2_0,3,0.0,0.0,0.0
1794241554.883,proposed,197377,e_3_0,1,0.0,0.0,0.0
1794241554.883,proposed,197377,e_3_0,4,0.0,0.0,0.0
1794241554.883,proposed,197377,e_3_0,2,0.0,0.0,0.0
1794241554.883,proposed,197377,e_3_0,3,0.0,0.0,0.0
1794241554.885,proposed,131328,a_1_0,1,0.0,0.0,0.0
1794241554.885,proposed,131328,a_1_0,4,0.0,0.0,0.0
1794241554.885,proposed,131328,a_1_0,2,0.0,0.0,0.0
1794241554.885,proposed,131328,a_1_0,3,0.0,0.0,0.0
1794241554.888,proposed,65536,c_0_0,4,0.0,0.0,0.0
1794241554.888,proposed,65536,c_0_0,1,0.0,0.0,0.0
1794241554.888,proposed,65536,c_0_0,2,0.0,0.0,0.0
1794241554.888,proposed,65536,c_0_0,3,0.0,0.0,0.0
1794241554.891,proposed,196865,e_1_0,4,0.0,0.0,0.0
1794241554.891,proposed,196865,e_1_0,1,0.0,0.0,0.0
1794241554.891,proposed,196865,e_1_0,2,0.0,0.0,0.0
1794241554.891,proposed,196865,e_1_0,3,0.0,0.0,0.0
1794241554.894,proposed,65539,c_1_1,4,0.0,0.0,0.0
1794241554.894,proposed,65539,c_1_1,1,0.0,0.0,0.0
1794241554.894,proposed,65539,c_1_1,2,0.0,0.0,0.0
1794241554.894,proposed,65539,c_1_1,3,0.0,0.0,0.0
1794241554.897,proposed,131072,a_0_0,4,0.0,0.0,0.0
1794241554.897,proposed,131072,a_0_0,1,0.0,0.0,0.0
1794241554.897,proposed,131072,a_0_0,2,0.0,0.0,0.0
1794241554.897,proposed,131072,a_0_0,3,0.0,0.0,0.0
1794241554.899,proposed,196866,e_1_1,4,0.0,0.0,0.0
1794241554.899,proposed,196866,e_1_1,1,0.0,0.0,0.0
1794241554.899,proposed,196866,e_1_1,2,0.0,0.0,0.0
1794241554.899,proposed,196866,e_1_1,3,0.0,0.0,0.0
1794241554.902,proposed,196609,e_0_0,1,0.0,0.0,0.0
1794241554.902,proposed,196609,e_0_0,4,0.0,0.0,0.0
1794241554.902,proposed,196609,e_0_0,2,0.0,0.0,0.0
1794241554.902,proposed,196609,e_0_0,3,0.0,0.0,0.0
1794241554.905,proposed,197378,e_3_1,4,0.0,0.0,0.0
1794241554.905,proposed,197378,e_3_1,1,0.0,0.0,0.0
1794241554.905,proposed,197378,e_3_1,2,0.0,0.0,0.0
1794241554.905,proposed,197378,e_3_1,3,0.0,0.0,0.0
1794241554.908,proposed,131329,a_1_1,1,0.0,0.0,0.0
1794241554.908,proposed,131329,a_1_1,4,0.0,0.0,0.0
1794241554.908,proposed,131329,a_1_1,2,0.0,0.0,0.0
1794241554.908,proposed,131329,a_1_1,3,0.0,0.0,0.0
1794241554.911,proposed,131841,a_3_1,1,0.0,0.0,0.0
1794241554.911,proposed,131841,a_3_1,4,0.0,0.0,0.0
1794241554.911,proposed,131841,a_3_1,2,0.0,0.0,0.0
1794241554.911,proposed,131841,a_3_1,3,0.0,0.0,0.0
1794241555.859,proposed,131073,a_0_1,1,0.0,0.0,0.0
1794241555.859,proposed,131073,a_0_1,4,0.0,0.0,0.0
1794241555.859,proposed,131073,a_0_1,2,0.0,0.0,0.0
1794241555.859,proposed,131073,a_0_1,3,0.0,0.0,0.0
1794241555.865,proposed,65538,c_1_0,4,0.0,0.0,0.0
1794241555.865,proposed,65538,c_1_0,1,0.0,0.0,0.0
1794241555.865,proposed,65538,c_1_0,2,0.0,0.0,0.0
1794241555.865,proposed,65538,c_1_0,3,0.0,0.0,0.0
1794241555.868,proposed,65537,c_0_1,4,0.0,0.0,0.0
1794241555.868,proposed,65537,c_0_1,1,0.0,0.0,0.0
1794241555.868,proposed,65537,c_0_1,2,0.0,0.0,0.0
1794241555.868,proposed,65537,c_0_1,3,0.0,0.0,0.0
1794241555.876,proposed,196610,e_0_1,4,0.0,0.0,0.0
1794241555.876,proposed,196610,e_0_1,1,0.0,0.0,0.0
1794241555.876,proposed,196610,e_0_1,2,0.0,0.0,0.0
1794241555.876,proposed,196610,e_0_1,3,0.0,0.0,0.0
1794241555.879,proposed,131584,a_2_0,4,0.0,0.0,0.0
1794241555.879,proposed,131584,a_2_0,1,0.0,0.0,0.0
1794241555.879,proposed,131584,a_2_0,2,0.0,0.0,0.0
1794241555.879,proposed,131584,a_2_0,3,0.0,0.0,0.0
1794241555.881,proposed,131328,a_1_0,1,0.0,0.0,0.0
1794241555.881,proposed,131328,a_1_0,4,0.0,0.0,0.0
1794241555.881,proposed,131328,a_1_0,2,0.0,0.0,0.0
1794241555.881,proposed,131328,a_1_0,3,0.0,0.0,0.0
1794241555.884,proposed,197377,e_3_0,1,0.0,0.0,0.0
1794241555.884,proposed,197377,e_3_0,4,0.0,0.0,0.0
1794241555.884,proposed,197377,e_3_0,2,0.0,0.0,0.0
1794241555.884,proposed,197377,e_3_0,3,0.0,0.0,0.0
1794241555.887,proposed,196865,e_1_0,4,0.0,0.0,0.0
1794241555.887,proposed,196865,e_1_0,1,0.0,0.0,0.0
1794241555.887,proposed,196865,e_1_0,2,0.0,0.0,0.0
1794241555.887,proposed,196865,e_1_0,3,0.0,0.0,0.0
1794241555.894,proposed,197121,e_2_0,4,0.0,0.0,0.0
1794241555.894,proposed,197121,e_2_0,1,0.0,0.0,0.0
1794241555.894,proposed,197121,e_2_0,2,0.0,0.0,0.0
1794241555.894,proposed,197121,e_2_0,3,0.0,0.0,0.0
1794241555.897,proposed,65536,c_0_0,4,0.0,0.0,0.0
1794241555.897,proposed,65536,c_0_0,1,0.0,0.0,0.0
1794241555.897,proposed,65536,c_0_0,2,0.0,0.0,0.0
1794241555.897,proposed,65536,c_0_0,3,0.0,0.0,0.0
1794241555.9,proposed,131329,a_1_1,1,0.0,0.0,0.0
1794241555.9,proposed,131329,a_1_1,4,0.0,0.0,0.0
1794241555.9,proposed,131329,a_1_1,2,0.0,0.0,0.0
1794241555.9,proposed,131329,a_1_1,3,0.0,0.0,0.0
1794241555.902,proposed,197378,e_3_1,4,0.0,0.0,0.0
1794241555.902,proposed,197378,e_3_1,1,0.0,0.0,0.0
1794241555.902,proposed,197378,e_3_1,2,0.0,0.0,0.0
1794241555.902,proposed,197378,e_3_1,3,0.0,0.0,0.0
1794241555.906,proposed,131841,a_3_1,1,0.0,0.0,0.0
1794241555.906,proposed,131841,a_3_1,4,0.0,0.0,0.0
1794241555.906,proposed,131841,a_3_1,2,0.0,0.0,0.0
1794241555.906,proposed,131841,a_3_1,3,0.0,0.0,0.0
1794241555.909,proposed,196866,e_1_1,4,0.0,0.0,0.0
1794241555.909,proposed,196866,e_1_1,1,0.0,0.0,0.0
1794241555.909,proposed,196866,e_1_1,2,0.0,0.0,0.0
1794241555.909,proposed,196866,e_1_1,3,0.0,0.0,0.0
1794241555.912,proposed,197122,e_2_1,4,0.0,0.0,0.0
1794241555.912,proposed,197122,e_2_1,1,0.0,0.0,0.0
1794241555.912,proposed,197122,e_2_1,2,0.0,0.0,0.0
1794241555.912,proposed,197122,e_2_1,3,0.0,0.0,0.0
1794241555.915,proposed,131585,a_2_1,1,0.0,0.0,0.0
1794241555.915,proposed,131585,a_2_1,4,0.0,0.0,0.0
1794241555.915,proposed,131585,a_2_1,2,0.0,0.0,0.0
1794241555.915,proposed,131585,a_2_1,3,0.0,0.0,0.0
1794241555.919,proposed,65539,c_1_1,4,0.0,0.0,0.0
1794241555.919,proposed,65539,c_1_1,1,0.0,0.0,0.0
1794241555.919,proposed,65539,c_1_1,2,0.0,0.0,0.0
1794241555.919,proposed,65539,c_1_1,3,0.0,0.0,0.0
1794241555.922,proposed,131840,a_3_0,4,0.0,0.0,0.0
1794241555.922,proposed,131840,a_3_0,1,0.0,0.0,0.0
1794241555.922,proposed,131840,a_3_0,2,0.0,0.0,0.0
1794241555.922,proposed,131840,a_3_0,3,0.0,0.0,0.0
1794241555.925,proposed,131072,a_0_0,4,0.0,0.0,0.0
1794241555.925,proposed,131072,a_0_0,1,0.0,0.0,0.0
1794241555.925,proposed,131072,a_0_0,2,0.0,0.0,0.0
1794241555.925,proposed,131072,a_0_0,3,0.0,0.0,0.0
1794241555.927,proposed,196609,e_0_0,1,0.0,0.0,0.0
1794241555.927,proposed,196609,e_0_0,4,0.0,0.0,0.0
1794241555.927,proposed,196609,e_0_0,2,0.0,0.0,0.0
1794241555.927,proposed,196609,e_0_0,3,0.0,0.0,0.0
1794241556.86,proposed,131073,a_0_1,1,0.0,0.0,0.0
1794241556.86,proposed,131073,a_0_1,4,0.0,0.0,0.0
1794241556.86,proposed,131073,a_0_1,2,0.0,0.0,0.0
1794241556.86,proposed,131073,a_0_1,3,0.0,0.0,0.0
1794241556.866,proposed,65538,c_1_0,4,0.0,0.0,0.0
1794241556.866,proposed,65538,c_1_0,1,0.0,0.0,0.0
1794241556.866,proposed,65538,c_1_0,2,0.0,0.0,0.0
1794241556.866,proposed,65538,c_1_0,3,0.0,0.0,0.0
1794241556.869,proposed,197122,e_2_1,4,0.0,0.0,0.0
1794241556.869,proposed,197122,e_2_1,1,0.0,0.0,0.0
1794241556.869,proposed,197122,e_2_1,2,0.0,0.0,0.0
1794241556.869,proposed,197122,e_2_1,3,0.0,0.0,0.0
1794241556.873,proposed,65537,c_0_1,4,0.0,0.0,0.0
1794241556.873,proposed,65537,c_0_1,1,0.0,0.0,0.0
1794241556.873,proposed,65537,c_0_1,2,0.0,0.0,0.0
1794241556.873,proposed,65537,c_0_1,3,0.0,0.0,0.0
1794241556.877,proposed,131585,a_2_1,1,0.0,0.0,0.0
1794241556.877,proposed,131585,a_2_1,4,0.0,0.0,0.0
1794241556.877,proposed,131585,a_2_1,2,0.0,0.0,0.0
1794241556.877,proposed,131585,a_2_1,3,0.0,0.0,0.0
1794241556.884,proposed,131840,a_3_0,4,0.0,0.0,0.0
1794241556.884,proposed,131840,a_3_0,1,0.0,0.0,0.0
1794241556.884,proposed,131840,a_3_0,2,0.0,0.0,0.0
1794241556.884,proposed,131840,a_3_0,3,0.0,0.0,0.0
1794241556.887,proposed,131584,a_2_0,4,0.0,0.0,0.0
1794241556.887,proposed,131584,a_2_0,1,0.0,0.0,0.0
1794241556.887,proposed,131584,a_2_0,2,0.0,0.0,0.0
1794241556.887,proposed,131584,a_2_0,3,0.0,0.0,0.0
1794241556.89,proposed,197121,e_2_0,4,0.0,0.0,0.0
1794241556.89,proposed,197121,e_2_0,1,0.0,0.0,0.0
1794241556.89,proposed,197121,e_2_0,2,0.0,0.0,0.0
1794241556.89,proposed,197121,e_2_0,3,0.0,0.0,0.0
1794241556.892,proposed,131329,a_1_1,1,0.0,0.0,0.0
1794241556.892,proposed,131329,a_1_1,4,0.0,0.0,0.0
1794241556.892,proposed,131329,a_1_1,2,0.0,0.0,0.0
1794241556.892,proposed,131329,a_1_1,3,0.0,0.0,0.0
1794241556.894,proposed,131328,a_1_0,1,0.0,0.0,0.0
1794241556.894,proposed,131328,a_1_0,4,0.0,0.0,0.0
1794241556.894,proposed,131328,a_1_0,2,0.0,0.0,0.0
1794241556.894,proposed,131328,a_1_0,3,0.0,0.0,0.0
1794241556.897,proposed,65536,c_0_0,4,0.0,0.0,0.0
1794241556.897,proposed,65536,c_0_0,1,0.0,0.0,0.0
1794241556.897,proposed,65536,c_0_0,2,0.0,0.0,0.0
1794241556.897,proposed,65536,c_0_0,3,0.0,0.0,0.0
1794241556.9,proposed,196866,e_1_1,4,0.0,0.0,0.0
1794241556.9,proposed,196866,e_1_1,1,0.0,0.0,0.0
1794241556.9,proposed,196866,e_1_1,2,0.0,0.0,0.0
1794241556.9,proposed,196866,e_1_1,3,0.0,0.0,0.0
1794241556.903,proposed,197378,e_3_1,4,0.0,0.0,0.0
1794241556.903,proposed,197378,e_3_1,1,0.0,0.0,0.0
1794241556.903,proposed,197378,e_3_1,2,0.0,0.0,0.0
1794241556.903,proposed,197378,e_3_1,3,0.0,0.0,0.0
1794241556.906,proposed,196865,e_1_0,4,0.0,0.0,0.0
1794241556.906,proposed,196865,e_1_0,1,0.0,0.0,0.0
1794241556.906,proposed,196865,e_1_0,2,0.0,0.0,0.0
1794241556.906,proposed,196865,e_1_0,3,0.0,0.0,0.0
1794241556.909,proposed,131072,a_0_0,4,0.0,0.0,0.0
1794241556.909,proposed,131072,a_0_0,1,0.0,0.0,0.0
1794241556.909,proposed,131072,a_0_0,2,0.0,0.0,0.0
1794241556.909,proposed,131072,a_0_0,3,0.0,0.0,0.0
1794241556.912,proposed,196610,e_0_1,4,0.0,0.0,0.0
1794241556.912,proposed,196610,e_0_1,1,0.0,0.0,0.0
1794241556.912,proposed,196610,e_0_1,2,0.0,0.0,0.0
1794241556.912,proposed,196610,e_0_1,3,0.0,0.0,0.0
1794241556.916,proposed,65539,c_1_1,4,0.0,0.0,0.0
1794241556.916,proposed,65539,c_1_1,1,0.0,0.0,0.0
1794241556.916,proposed,65539,c_1_1,2,0.0,0.0,0.0
1794241556.916,proposed,65539,c_1_1,3,0.0,0.0,0.0
1794241556.919,proposed,197377,e_3_0,1,0.0,0.0,0.0
1794241556.919,proposed,197377,e_3_0,4,0.0,0.0,0.0
1794241556.919,proposed,197377,e_3_0,2,0.0,0.0,0.0
1794241556.919,proposed,197377,e_3_0,3,0.0,0.0,0.0
1794241556.922,proposed,131841,a_3_1,1,0.0,0.0,0.0
1794241556.922,proposed,131841,a_3_1,4,0.0,0.0,0.0
1794241556.922,proposed,131841,a_3_1,2,0.0,0.0,0.0
1794241556.922,proposed,131841,a_3_1,3,0.0,0.0,0.0
1794241556.925,proposed,196609,e_0_0,1,0.0,0.0,0.0
1794241556.925,proposed,196609,e_0_0,4,0.0,0.0,0.0
1794241556.925,proposed,196609,e_0_0,2,0.0,0.0,0.0
1794241556.925,proposed,196609,e_0_0,3,0.0,0.0,0.0
1794241557.864,proposed,131585,a_2_1,1,0.0,0.0,0.0
1794241557.864,proposed,131585,a_2_1,4,0.0,0.0,0.0
1794241557.864,proposed,131585,a_2_1,2,0.0,0.0,0.0
1794241557.864,proposed,131585,a_2_1,3,0.0,0.0,0.0
1794241557.869,proposed,197378,e_3_1,4,0.0,0.0,0.0
1794241557.869,proposed,197378,e_3_1,1,0.0,0.0,0.0
1794241557.869,proposed,197378,e_3_1,2,0.0,0.0,0.0
1794241557.869,proposed,197378,e_3_1,3,0.0,0.0,0.0
1794241557.872,proposed,65538,c_1_0,4,0.0,0.0,0.0
1794241557.872,proposed,65538,c_1_0,1,0.0,0.0,0.0
1794241557.872,proposed,65538,c_1_0,2,0.0,0.0,0.0
1794241557.872,proposed,65538,c_1_0,3,0.0,0.0,0.0
1794241557.875,proposed,196865,e_1_0,4,0.0,0.0,0.0
1794241557.875,proposed,196865,e_1_0,1,0.0,0.0,0.0
1794241557.875,proposed,196865,e_1_0,2,0.0,0.0,0.0
1794241557.875,proposed,196865,e_1_0,3,0.0,0.0,0.0
1794241557.877,proposed,197122,e_2_1,4,0.0,0.0,0.0
1794241557.877,proposed,197122,e_2_1,1,0.0,0.0,0.0
1794241557.877,proposed,197122,e_2_1,2,0.0,0.0,0.0
1794241557.877,proposed,197122,e_2_1,3,0.0,0.0,0.0
1794241557.88,proposed,65537,c_0_1,4,0.0,0.0,0.0
1794241557.88,proposed,65537,c_0_1,1,0.0,0.0,0.0
1794241557.88,proposed,65537,c_0_1,2,0.0,0.0,0.0
1794241557.88,proposed,65537,c_0_1,3,0.0,0.0,0.0
1794241557.883,proposed,131073,a_0_1,1,0.0,0.0,0.0
1794241557.883,proposed,131073,a_0_1,4,0.0,0.0,0.0
1794241557.883,proposed,131073,a_0_1,2,0.0,0.0,0.0
1794241557.883,proposed,131073,a_0_1,3,0.0,0.0,0.0
1794241557.886,proposed,65539,c_1_1,4,0.0,0.0,0.0
1794241557.886,proposed,65539,c_1_1,1,0.0,0.0,0.0
1794241557.886,proposed,65539,c_1_1,2,0.0,0.0,0.0
1794241557.886,proposed,65539,c_1_1,3,0.0,0.0,0.0
1794241557.891,proposed,196610,e_0_1,4,0.0,0.0,0.0
1794241557.891,proposed,196610,e_0_1,1,0.0,0.0,0.0
1794241557.891,proposed,196610,e_0_1,2,0.0,0.0,0.0
1794241557.891,proposed,196610,e_0_1,3,0.0,0.0,0.0
1794241557.893,proposed,197121,e_2_0,4,0.0,0.0,0.0
1794241557.893,proposed,197121,e_2_0,1,0.0,0.0,0.0
1794241557.893,proposed,197121,e_2_0,2,0.0,0.0,0.0
1794241557.893,proposed,197121,e_2_0,3,0.0,0.0,0.0
1794241557.895,proposed,131328,a_1_0,1,0.0,0.0,0.0
1794241557.895,proposed,131328,a_1_0,4,0.0,0.0,0.0
1794241557.895,proposed,131328,a_1_0,2,0.0,0.0,0.0
1794241557.895,proposed,131328,a_1_0,3,0.0,0.0,0.0
1794241557.897,proposed,65536,c_0_0,4,0.0,0.0,0.0
1794241557.897,proposed,65536,c_0_0,1,0.0,0.0,0.0
1794241557.897,proposed,65536,c_0_0,2,0.0,0.0,0.0
1794241557.897,proposed,65536,c_0_0,3,0.0,0.0,0.0
1794241557.9,proposed,131584,a_2_0,4,0.0,0.0,0.0
1794241557.9,proposed,131584,a_2_0,1,0.0,0.0,0.0
1794241557.9,proposed,131584,a_2_0,2,0.0,0.0,0.0
1794241557.9,proposed,131584,a_2_0,3,0.0,0.0,0.0
1794241557.901,proposed,131072,a_0_0,4,0.0,0.0,0.0
1794241557.901,proposed,131072,a_0_0,1,0.0,0.0,0.0
1794241557.901,proposed,131072,a_0_0,2,0.0,0.0,0.0
1794241557.901,proposed,131072,a_0_0,3,0.0,0.0,0.0
1794241557.904,proposed,196866,e_1_1,4,0.0,0.0,0.0
1794241557.904,proposed,196866,e_1_1,1,0.0,0.0,0.0
1794241557.904,proposed,196866,e_1_1,2,0.0,0.0,0.0
1794241557.904,proposed,196866,e_1_1,3,0.0,0.0,0.0
1794241557.908,proposed,131840,a_3_0,4,0.0,0.0,0.0
1794241557.908,proposed,131840,a_3_0,1,0.0,0.0,0.0
1794241557.908,proposed,131840,a_3_0,2,0.0,0.0,0.0
1794241557.908,proposed,131840,a_3_0,3,0.0,0.0,0.0
1794241557.91,proposed,197377,e_3_0,1,0.0,0.0,0.0
1794241557.91,proposed,197377,e_3_0,4,0.0,0.0,0.0
1794241557.91,proposed,197377,e_3_0,2,0.0,0.0,0.0
1794241557.91,proposed,197377,e_3_0,3,0.0,0.0,0.0
1794241557.913,proposed,131841,a_3_1,1,0.0,0.0,0.0
1794241557.913,proposed,131841,a_3_1,4,0.0,0.0,0.0
1794241557.913,proposed,131841,a_3_1,2,0.0,0.0,0.0
1794241557.913,proposed,131841,a_3_1,3,0.0,0.0,0.0
1794241557.915,proposed,196609,e_0_0,1,0.0,0.0,0.0
1794241557.915,proposed,196609,e_0_0,4,0.0,0.0,0.0
1794241557.915,proposed,196609,e_0_0,2,0.0,0.0,0.0
1794241557.915,proposed,196609,e_0_0,3,0.0,0.0,0.0
1794241557.918,proposed,131329,a_1_1,1,0.0,0.0,0.0
1794241557.918,proposed,131329,a_1_1,4,0.0,0.0,0.0
1794241557.918,proposed,131329,a_1_1,2,0.0,0.0,0.0
1794241557.918,proposed,131329,a_1_1,3,0.0,0.0,0.0
1794241558.866,proposed,197122,e_2_1,4,0.0,0.0,0.0
1794241558.866,proposed,197122,e_2_1,1,0.0,0.0,0.0
1794241558.866,proposed,197122,e_2_1,2,0.0,0.0,0.0
1794241558.866,proposed,197122,e_2_1,3,0.0,0.0,0.0
1794241558.871,proposed,65537,c_0_1,4,0.0,0.0,0.0
1794241558.871,proposed,65537,c_0_1,1,0.0,0.0,0.0
1794241558.871,proposed,65537,c_0_1,2,0.0,0.0,0.0
1794241558.871,proposed,65537,c_0_1,3,0.0,0.0,0.0
1794241558.873,proposed,65538,c_1_0,4,0.0,0.0,0.0
1794241558.873,proposed,65538,c_1_0,1,0.0,0.0,0.0
1794241558.873,proposed,65538,c_1_0,2,0.0,0.0,0.0
1794241558.873,proposed,65538,c_1_0,3,0.0,0.0,0.0
1794241558.875,proposed,131073,a_0_1,1,0.0,0.0,0.0
1794241558.875,proposed,131073,a_0_1,4,0.0,0.0,0.0
1794241558.875,proposed,131073,a_0_1,2,0.0,0.0,0.0
1794241558.875,proposed,131073,a_0_1,3,0.0,0.0,0.0
1794241558.878,proposed,131585,a_2_1,1,0.0,0.0,0.0
1794241558.878,proposed,131585,a_2_1,4,0.0,0.0,0.0
1794241558.878,proposed,131585,a_2_1,2,0.0,0.0,0.0
1794241558.878,proposed,131585,a_2_1,3,0.0,0.0,0.0
1794241558.886,proposed,196610,e_0_1,4,0.0,0.0,0.0
1794241558.886,proposed,196610,e_0_1,1,0.0,0.0,0.0
1794241558.886,proposed,196610,e_0_1,2,0.0,0.0,0.0
1794241558.886,proposed,196610,e_0_1,3,0.0,0.0,0.0
1794241558.889,proposed,196609,e_0_0,1,0.0,0.0,0.0
1794241558.889,proposed,196609,e_0_0,4,0.0,0.0,0.0
1794241558.889,proposed,196609,e_0_0,2,0.0,0.0,0.0
1794241558.889,proposed,196609,e_0_0,3,0.0,0.0,0.0
1794241558.891,proposed,131328,a_1_0,1,0.0,0.0,0.0
1794241558.891,proposed,131328,a_1_0,4,0.0,0.0,0.0
1794241558.891,proposed,131328,a_1_0,2,0.0,0.0,0.0
1794241558.891,proposed,131328,a_1_0,3,0.0,0.0,0.0
1794241558.893,proposed,196865,e_1_0,4,0.0,0.0,0.0
1794241558.893,proposed,196865,e_1_0,1,0.0,0.0,0.0
1794241558.893,proposed,196865,e_1_0,2,0.0,0.0,0.0
1794241558.893,proposed,196865,e_1_0,3,0.0,0.0,0.0
1794241558.896,proposed,131329,a_1_1,1,0.0,0.0,0.0
1794241558.896,proposed,131329,a_1_1,4,0.0,0.0,0.0
1794241558.896,proposed,131329,a_1_1,2,0.0,0.0,0.0
1794241558.896,proposed,131329,a_1_1,3,0.0,0.0,0.0
1794241558.899,proposed,197377,e_3_0,1,0.0,0.0,0.0
1794241558.899,proposed,197377,e_3_0,4,0.0,0.0,0.0
1794241558.899,proposed,197377,e_3_0,2,0.0,0.0,0.0
1794241558.899,proposed,197377,e_3_0,3,0.0,0.0,0.0
1794241558.901,proposed,131840,a_3_0,4,0.0,0.0,0.0
1794241558.901,proposed,131840,a_3_0,1,0.0,0.0,0.0
1794241558.901,proposed,131840,a_3_0,2,0.0,0.0,0.0
1794241558.901,proposed,131840,a_3_0,3,0.0,0.0,0.0
1794241558.903,proposed,131584,a_2_0,4,0.0,0.0,0.0
1794241558.903,proposed,131584,a_2_0,1,0.0,0.0,0.0
1794241558.903,proposed,131584,a_2_0,2,0.0,0.0,0.0
1794241558.903,proposed,131584,a_2_0,3,0.0,0.0,0.0
1794241558.906,proposed,131072,a_0_0,4,0.0,0.0,0.0
1794241558.906,proposed,131072,a_0_0,1,0.0,0.0,0.0
1794241558.906,proposed,131072,a_0_0,2,0.0,0.0,0.0
1794241558.906,proposed,131072,a_0_0,3,0.0,0.0,0.0
1794241558.909,proposed,197378,e_3_1,4,0.0,0.0,0.0
1794241558.909,proposed,197378,e_3_1,1,0.0,0.0,0.0
1794241558.909,proposed,197378,e_3_1,2,0.0,0.0,0.0
1794241558.909,proposed,197378,e_3_1,3,0.0,0.0,0.0
1794241558.912,proposed,196866,e_1_1,4,0.0,0.0,0.0
1794241558.912,proposed,196866,e_1_1,1,0.0,0.0,0.0
1794241558.912,proposed,196866,e_1_1,2,0.0,0.0,0.0
1794241558.912,proposed,196866,e_1_1,3,0.0,0.0,0.0
1794241558.915,proposed,197121,e_2_0,4,0.0,0.0,0.0
1794241558.915,proposed,197121,e_2_0,1,0.0,0.0,0.0
1794241558.915,proposed,197121,e_2_0,2,0.0,0.0,0.0
1794241558.915,proposed,197121,e_2_0,3,0.0,0.0,0.0
1794241558.917,proposed,65536,c_0_0,4,0.0,0.0,0.0
1794241558.917,proposed,65536,c_0_0,1,0.0,0.0,0.0
1794241558.917,proposed,65536,c_0_0,2,0.0,0.0,0.0
1794241558.917,proposed,65536,c_0_0,3,0.0,0.0,0.0
1794241558.92,proposed,131841,a_3_1,1,0.0,0.0,0.0
1794241558.92,proposed,131841,a_3_1,4,0.0,0.0,0.0
1794241558.92,proposed,131841,a_3_1,2,0.0,0.0,0.0
1794241558.92,proposed,131841,a_3_1,3,0.0,0.0,0.0
1794241558.923,proposed,65539,c_1_1,4,0.0,0.0,0.0
1794241558.923,proposed,65539,c_1_1,1,0.0,0.0,0.0
1794241558.923,proposed,65539,c_1_1,2,0.0,0.0,0.0
1794241558.923,proposed,65539,c_1_1,3,0.0,0.0,0.0
1794241559.877,proposed,131584,a_2_0,4,0.0,0.0,0.0
1794241559.877,proposed,131584,a_2_0,1,0.0,0.0,0.0
1794241559.877,proposed,131584,a_2_0,2,0.0,0.0,0.0
1794241559.877,proposed,131584,a_2_0,3,0.0,0.0,0.0
1794241559.881,proposed,196865,e_1_0,4,0.0,0.0,0.0
1794241559.881,proposed,196865,e_1_0,1,0.0,0.0,0.0
1794241559.881,proposed,196865,e_1_0,2,0.0,0.0,0.0
1794241559.881,proposed,196865,e_1_0,3,0.0,0.0,0.0
1794241559.885,proposed,65539,c_1_1,4,0.0,0.0,0.0
1794241559.885,proposed,65539,c_1_1,1,0.0,0.0,0.0
1794241559.885,proposed,65539,c_1_1,2,0.0,0.0,0.0
1794241559.885,proposed,65539,c_1_1,3,0.0,0.0,0.0
1794241559.889,proposed,131840,a_3_0,4,0.0,0.0,0.0
1794241559.889,proposed,131840,a_3_0,1,0.0,0.0,0.0
1794241559.889,proposed,131840,a_3_0,2,0.0,0.0,0.0
1794241559.889,proposed,131840,a_3_0,3,0.0,0.0,0.0
1794241559.892,proposed,131841,a_3_1,1,0.0,0.0,0.0
1794241559.892,proposed,131841,a_3_1,4,0.0,0.0,0.0
1794241559.892,proposed,131841,a_3_1,2,0.0,0.0,0.0
1794241559.892,proposed,131841,a_3_1,3,0.0,0.0,0.0
1794241559.896,proposed,65537,c_0_1,4,0.0,0.0,0.0
1794241559.896,proposed,65537,c_0_1,1,0.0,0.0,0.0
1794241559.896,proposed,65537,c_0_1,2,0.0,0.0,0.0
1794241559.896,proposed,65537,c_0_1,3,0.0,0.0,0.0
1794241559.899,proposed,65536,c_0_0,4,0.0,0.0,0.0
1794241559.899,proposed,65536,c_0_0,1,0.0,0.0,0.0
1794241559.899,proposed,65536,c_0_0,2,0.0,0.0,0.0
1794241559.899,proposed,65536,c_0_0,3,0.0,0.0,0.0
1794241559.902,proposed,131329,a_1_1,1,0.0,0.0,0.0
1794241559.902,proposed,131329,a_1_1,4,0.0,0.0,0.0
1794241559.902,proposed,131329,a_1_1,2,0.0,0.0,0.0
1794241559.902,proposed,131329,a_1_1,3,0.0,0.0,0.0
1794241559.905,proposed,196610,e_0_1,4,0.0,0.0,0.0
1794241559.905,proposed,196610,e_0_1,1,0.0,0.0,0.0
1794241559.905,proposed,196610,e_0_1,2,0.0,0.0,0.0
1794241559.905,proposed,196610,e_0_1,3,0.0,0.0,0.0
1794241559.908,proposed,197377,e_3_0,1,0.0,0.0,0.0
1794241559.908,proposed,197377,e_3_0,4,0.0,0.0,0.0
1794241559.908,proposed,197377,e_3_0,2,0.0,0.0,0.0
1794241559.908,proposed,197377,e_3_0,3,0.0,0.0,0.0
1794241559.911,proposed,131073,a_0_1,1,0.0,0.0,0.0
1794241559.911,proposed,131073,a_0_1,4,0.0,0.0,0.0
1794241559.911,proposed,131073,a_0_1,2,0.0,0.0,0.0
1794241559.911,proposed,131073,a_0_1,3,0.0,0.0,0.0
1794241559.915,proposed,197122,e_2_1,4,0.0,0.0,0.0
1794241559.915,proposed,197122,e_2_1,1,0.0,0.0,0.0
1794241559.915,proposed,197122,e_2_1,2,0.0,0.0,0.0
1794241559.915,proposed,197122,e_2_1,3,0.0,0.0,0.0
1794241559.917,proposed,65538,c_1_0,4,0.0,0.0,0.0
1794241559.917,proposed,65538,c_1_0,1,0.0,0.0,0.0
1794241559.917,proposed,65538,c_1_0,2,0.0,0.0,0.0
1794241559.917,proposed,65538,c_1_0,3,0.0,0.0,0.0
1794241559.921,proposed,197121,e_2_0,4,0.0,0.0,0.0
1794241559.921,proposed,197121,e_2_0,1,0.0,0.0,0.0
1794241559.921,proposed,197121,e_2_0,2,0.0,0.0,0.0
1794241559.921,proposed,197121,e_2_0,3,0.0,0.0,0.0
1794241559.924,proposed,197378,e_3_1,4,0.0,0.0,0.0
1794241559.924,proposed,197378,e_3_1,1,0.0,0.0,0.0
1794241559.924,proposed,197378,e_3_1,2,0.0,0.0,0.0
1794241559.924,proposed,197378,e_3_1,3,0.0,0.0,0.0
1794241559.928,proposed,131585,a_2_1,1,0.0,0.0,0.0
1794241559.928,proposed,131585,a_2_1,4,0.0,0.0,0.0
1794241559.928,proposed,131585,a_2_1,2,0.0,0.0,0.0
1794241559.928,proposed,131585,a_2_1,3,0.0,0.0,0.0
1794241559.932,proposed,131072,a_0_0,4,0.0,0.0,0.0
1794241559.932,proposed,131072,a_0_0,1,0.0,0.0,0.0
1794241559.932,proposed,131072,a_0_0,2,0.0,0.0,0.0
1794241559.932,proposed,131072,a_0_0,3,0.0,0.0,0.0
1794241559.935,proposed,196609,e_0_0,1,0.0,0.0,0.0
1794241559.935,proposed,196609,e_0_0,4,0.0,0.0,0.0
1794241559.935,proposed,196609,e_0_0,2,0.0,0.0,0.0
1794241559.935,proposed,196609,e_0_0,3,0.0,0.0,0.0
1794241559.938,proposed,131328,a_1_0,1,0.0,0.0,0.0
1794241559.938,proposed,131328,a_1_0,4,0.0,0.0,0.0
1794241559.938,proposed,131328,a_1_0,2,0.0,0.0,0.0
1794241559.938,proposed,131328,a_1_0,3,0.0,0.0,0.0
1794241559.942,proposed,196866,e_1_1,4,0.0,0.0,0.0
1794241559.942,proposed,196866,e_1_1,1,0.0,0.0,0.0
1794241559.942,proposed,196866,e_1_1,2,0.0,0.0,0.0
1794241559.942,proposed,196866,e_1_1,3,0.0,0.0,0.0
1794241560.878,proposed,196865,e_1_0,4,0.0,0.0,0.0
1794241560.878,proposed,196865,e_1_0,1,0.0,0.0,0.0
1794241560.878,proposed,196865,e_1_0,2,0.0,0.0,0.0
1794241560.878,proposed,196865,e_1_0,3,0.0,0.0,0.0
1794241560.88,proposed,131841,a_3_1,1,0.0,0.0,0.0
1794241560.88,proposed,131841,a_3_1,4,0.0,0.0,0.0
1794241560.88,proposed,131841,a_3_1,2,0.0,0.0,0.0
1794241560.88,proposed,131841,a_3_1,3,0.0,0.0,0.0
1794241560.887,proposed,196609,e_0_0,1,0.0,0.0,0.0
1794241560.887,proposed,196609,e_0_0,4,0.0,0.0,0.0
1794241560.887,proposed,196609,e_0_0,2,0.0,0.0,0.0
1794241560.887,proposed,196609,e_0_0,3,0.0,0.0,0.0
1794241560.889,proposed,197378,e_3_1,4,0.0,0.0,0.0
1794241560.889,proposed,197378,e_3_1,1,0.0,0.0,0.0
1794241560.889,proposed,197378,e_3_1,2,0.0,0.0,0.0
1794241560.889,proposed,197378,e_3_1,3,0.0,0.0,0.0
1794241560.892,proposed,131072,a_0_0,4,0.0,0.0,0.0
1794241560.892,proposed,131072,a_0_0,1,0.0,0.0,0.0
1794241560.892,proposed,131072,a_0_0,2,0.0,0.0,0.0
1794241560.892,proposed,131072,a_0_0,3,0.0,0.0,0.0
1794241560.894,proposed,197122,e_2_1,4,0.0,0.0,0.0
1794241560.894,proposed,197122,e_2_1,1,0.0,0.0,0.0
1794241560.894,proposed,197122,e_2_1,2,0.0,0.0,0.0
1794241560.894,proposed,197122,e_2_1,3,0.0,0.0,0.0
1794241560.898,proposed,65539,c_1_1,4,0.0,0.0,0.0
1794241560.898,proposed,65539,c_1_1,1,0.0,0.0,0.0
1794241560.898,proposed,65539,c_1_1,2,0.0,0.0,0.0
1794241560.898,proposed,65539,c_1_1,3,0.0,0.0,0.0
1794241560.901,proposed,197377,e_3_0,1,0.0,0.0,0.0
1794241560.901,proposed,197377,e_3_0,4,0.0,0.0,0.0
1794241560.901,proposed,197377,e_3_0,2,0.0,0.0,0.0
1794241560.901,proposed,197377,e_3_0,3,0.0,0.0,0.0
1794241560.904,proposed,131329,a_1_1,1,0.0,0.0,0.0
1794241560.904,proposed,131329,a_1_1,4,0.0,0.0,0.0
1794241560.904,proposed,131329,a_1_1,2,0.0,0.0,0.0
1794241560.904,proposed,131329,a_1_1,3,0.0,0.0,0.0
1794241560.907,proposed,131073,a_0_1,1,0.0,0.0,0.0
1794241560.907,proposed,131073,a_0_1,4,0.0,0.0,0.0
1794241560.907,proposed,131073,a_0_1,2,0.0,0.0,0.0
1794241560.907,proposed,131073,a_0_1,3,0.0,0.0,0.0
1794241560.909,proposed,65537,c_0_1,4,0.0,0.0,0.0
1794241560.909,proposed,65537,c_0_1,1,0.0,0.0,0.0
1794241560.909,proposed,65537,c_0_1,2,0.0,0.0,0.0
1794241560.909,proposed,65537,c_0_1,3,0.0,0.0,0.0
1794241560.913,proposed,131585,a_2_1,1,0.0,0.0,0.0
1794241560.913,proposed,131585,a_2_1,4,0.0,0.0,0.0
1794241560.913,proposed,131585,a_2_1,2,0.0,0.0,0.0
1794241560.913,proposed,131585,a_2_1,3,0.0,0.0,0.0
1794241560.915,proposed,131840,a_3_0,4,0.0,0.0,0.0
1794241560.915,proposed,131840,a_3_0,1,0.0,0.0,0.0
1794241560.915,proposed,131840,a_3_0,2,0.0,0.0,0.0
1794241560.915,proposed,131840,a_3_0,3,0.0,0.0,0.0
1794241560.919,proposed,196610,e_0_1,4,0.0,0.0,0.0
1794241560.919,proposed,196610,e_0_1,1,0.0,0.0,0.0
1794241560.919,proposed,196610,e_0_1,2,0.0,0.0,0.0
1794241560.919,proposed,196610,e_0_1,3,0.0,0.0,0.0
1794241560.922,proposed,65538,c_1_0,4,0.0,0.0,0.0
1794241560.922,proposed,65538,c_1_0,1,0.0,0.0,0.0
1794241560.922,proposed,65538,c_1_0,2,0.0,0.0,0.0
1794241560.922,proposed,65538,c_1_0,3,0.0,0.0,0.0
1794241560.925,proposed,131328,a_1_0,1,0.0,0.0,0.0
1794241560.925,proposed,131328,a_1_0,4,0.0,0.0,0.0
1794241560.925,proposed,131328,a_1_0,2,0.0,0.0,0.0
1794241560.925,proposed,131328,a_1_0,3,0.0,0.0,0.0
1794241560.93,proposed,196866,e_1_1,4,0.0,0.0,0.0
1794241560.93,proposed,196866,e_1_1,1,0.0,0.0,0.0
1794241560.93,proposed,196866,e_1_1,2,0.0,0.0,0.0
1794241560.93,proposed,196866,e_1_1,3,0.0,0.0,0.0
1794241560.934,proposed,197121,e_2_0,4,0.0,0.0,0.0
1794241560.934,proposed,197121,e_2_0,1,0.0,0.0,0.0
1794241560.934,proposed,197121,e_2_0,2,0.0,0.0,0.0
1794241560.934,proposed,197121,e_2_0,3,0.0,0.0,0.0
1794241560.937,proposed,65536,c_0_0,4,0.0,0.0,0.0
1794241560.937,proposed,65536,c_0_0,1,0.0,0.0,0.0
1794241560.937,proposed,65536,c_0_0,2,0.0,0.0,0.0
1794241560.937,proposed,65536,c_0_0,3,0.0,0.0,0.0
1794241560.941,proposed,131584,a_2_0,4,0.0,0.0,0.0
1794241560.941,proposed,131584,a_2_0,1,0.0,0.0,0.0
1794241560.941,proposed,131584,a_2_0,2,0.0,0.0,0.0
1794241560.941,proposed,131584,a_2_0,3,0.0,0.0,0.0
1794241561.877,proposed,65537,c_0_1,4,0.0,0.0,0.0
1794241561.877,proposed,65537,c_0_1,1,0.0,0.0,0.0
1794241561.877,proposed,65537,c_0_1,2,0.0,0.0,0.0
1794241561.877,proposed,65537,c_0_1,3,0.0,0.0,0.0
1794241561.885,proposed,65538,c_1_0,4,0.0,0.0,0.0
1794241561.885,proposed,65538,c_1_0,1,0.0,0.0,0.0
1794241561.885,proposed,65538,c_1_0,2,0.0,0.0,0.0
1794241561.885,proposed,65538,c_1_0,3,0.0,0.0,0.0
1794241561.889,proposed,131073,a_0_1,1,0.0,0.0,0.0
1794241561.889,proposed,131073,a_0_1,4,0.0,0.0,0.0
1794241561.889,proposed,131073,a_0_1,2,0.0,0.0,0.0
1794241561.889,proposed,131073,a_0_1,3,0.0,0.0,0.0
1794241561.891,proposed,131585,a_2_1,1,0.0,0.0,0.0
1794241561.891,proposed,131585,a_2_1,4,0.0,0.0,0.0
1794241561.891,proposed,131585,a_2_1,2,0.0,0.0,0.0
1794241561.891,proposed,131585,a_2_1,3,0.0,0.0,0.0
1794241561.895,proposed,197122,e_2_1,4,0.0,0.0,0.0
1794241561.895,proposed,197122,e_2_1,1,0.0,0.0,0.0
1794241561.895,proposed,197122,e_2_1,2,0.0,0.0,0.0
1794241561.895,proposed,197122,e_2_1,3,0.0,0.0,0.0
1794241561.902,proposed,197121,e_2_0,4,0.0,0.0,0.0
1794241561.902,proposed,197121,e_2_0,1,0.0,0.0,0.0
1794241561.902,proposed,197121,e_2_0,2,0.0,0.0,0.0
1794241561.902,proposed,197121,e_2_0,3,0.0,0.0,0.0
1794241561.906,proposed,197377,e_3_0,1,0.0,0.0,0.0
1794241561.906,proposed,197377,e_3_0,4,0.0,0.0,0.0
1794241561.906,proposed,197377,e_3_0,2,0.0,0.0,0.0
1794241561.906,proposed,197377,e_3_0,3,0.0,0.0,0.0
1794241561.909,proposed,196866,e_1_1,4,0.0,0.0,0.0
1794241561.909,proposed,196866,e_1_1,1,0.0,0.0,0.0
1794241561.909,proposed,196866,e_1_1,2,0.0,0.0,0.0
1794241561.909,proposed,196866,e_1_1,3,0.0,0.0,0.0
1794241561.912,proposed,196609,e_0_0,1,0.0,0.0,0.0
1794241561.912,proposed,196609,e_0_0,4,0.0,0.0,0.0
1794241561.912,proposed,196609,e_0_0,2,0.0,0.0,0.0
1794241561.912,proposed,196609,e_0_0,3,0.0,0.0,0.0
1794241561.915,proposed,197378,e_3_1,4,0.0,0.0,0.0
1794241561.915,proposed,197378,e_3_1,1,0.0,0.0,0.0
1794241561.915,proposed,197378,e_3_1,2,0.0,0.0,0.0
1794241561.915,proposed,197378,e_3_1,3,0.0,0.0,0.0
1794241561.919,proposed,196610,e_0_1,4,0.0,0.0,0.0
1794241561.919,proposed,196610,e_0_1,1,0.0,0.0,0.0
1794241561.919,proposed,196610,e_0_1,2,0.0,0.0,0.0
1794241561.919,proposed,196610,e_0_1,3,0.0,0.0,0.0
1794241561.923,proposed,131841,a_3_1,1,0.0,0.0,0.0
1794241561.923,proposed,131841,a_3_1,4,0.0,0.0,0.0
1794241561.923,proposed,131841,a_3_1,2,0.0,0.0,0.0
1794241561.923,proposed,131841,a_3_1,3,0.0,0.0,0.0
1794241561.925,proposed,196865,e_1_0,4,0.0,0.0,0.0
1794241561.925,proposed,196865,e_1_0,1,0.0,0.0,0.0
1794241561.925,proposed,196865,e_1_0,2,0.0,0.0,0.0
1794241561.925,proposed,196865,e_1_0,3,0.0,0.0,0.0
1794241561.929,proposed,131584,a_2_0,4,0.0,0.0,0.0
1794241561.929,proposed,131584,a_2_0,1,0.0,0.0,0.0
1794241561.929,proposed,131584,a_2_0,2,0.0,0.0,0.0
1794241561.929,proposed,131584,a_2_0,3,0.0,0.0,0.0
1794241561.931,proposed,131840,a_3_0,4,0.0,0.0,0.0
1794241561.931,proposed,131840,a_3_0,1,0.0,0.0,0.0
1794241561.931,proposed,131840,a_3_0,2,0.0,0.0,0.0
1794241561.931,proposed,131840,a_3_0,3,0.0,0.0,0.0
1794241561.934,proposed,131072,a_0_0,4,0.0,0.0,0.0
1794241561.934,proposed,131072,a_0_0,1,0.0,0.0,0.0
1794241561.934,proposed,131072,a_0_0,2,0.0,0.0,0.0
1794241561.934,proposed,131072,a_0_0,3,0.0,0.0,0.0
1794241561.937,proposed,131328,a_1_0,1,0.0,0.0,0.0
1794241561.937,proposed,131328,a_1_0,4,0.0,0.0,0.0
1794241561.937,proposed,131328,a_1_0,2,0.0,0.0,0.0
1794241561.937,proposed,131328,a_1_0,3,0.0,0.0,0.0
1794241561.939,proposed,131329,a_1_1,1,0.0,0.0,0.0
1794241561.939,proposed,131329,a_1_1,4,0.0,0.0,0.0
1794241561.939,proposed,131329,a_1_1,2,0.0,0.0,0.0
1794241561.939,proposed,131329,a_1_1,3,0.0,0.0,0.0
1794241561.942,proposed,65539,c_1_1,4,0.0,0.0,0.0
1794241561.942,proposed,65539,c_1_1,1,0.0,0.0,0.0
1794241561.942,proposed,65539,c_1_1,2,0.0,0.0,0.0
1794241561.942,proposed,65539,c_1_1,3,0.0,0.0,0.0
1794241561.945,proposed,65536,c_0_0,4,0.0,0.0,0.0
1794241561.945,proposed,65536,c_0_0,1,0.0,0.0,0.0
1794241561.945,proposed,65536,c_0_0,2,0.0,0.0,0.0
1794241561.945,proposed,65536,c_0_0,3,0.0,0.0,0.0
1794241562.88,proposed,196865,e_1_0,4,0.0,0.0,0.0
1794241562.88,proposed,196865,e_1_0,1,0.0,0.0,0.0
1794241562.88,proposed,196865,e_1_0,2,0.0,0.0,0.0
1794241562.88,proposed,196865,e_1_0,3,0.0,0.0,0.0
1794241562.885,proposed,131841,a_3_1,1,0.0,0.0,0.0
1794241562.885,proposed,131841,a_3_1,4,0.0,0.0,0.0
1794241562.885,proposed,131841,a_3_1,2,0.0,0.0,0.0
1794241562.885,proposed,131841,a_3_1,3,0.0,0.0,0.0
1794241562.888,proposed,197121,e_2_0,4,0.0,0.0,0.0
1794241562.888,proposed,197121,e_2_0,1,0.0,0.0,0.0
1794241562.888,proposed,197121,e_2_0,2,0.0,0.0,0.0
1794241562.888,proposed,197121,e_2_0,3,0.0,0.0,0.0
1794241562.891,proposed,65537,c_0_1,4,0.0,0.0,0.0
1794241562.891,proposed,65537,c_0_1,1,0.0,0.0,0.0
1794241562.891,proposed,65537,c_0_1,2,0.0,0.0,0.0
1794241562.891,proposed,65537,c_0_1,3,0.0,0.0,0.0
1794241562.894,proposed,65538,c_1_0,4,0.0,0.0,0.0
1794241562.894,proposed,65538,c_1_0,1,0.0,0.0,0.0
1794241562.894,proposed,65538,c_1_0,2,0.0,0.0,0.0
1794241562.894,proposed,65538,c_1_0,3,0.0,0.0,0.0
1794241562.898,proposed,197122,e_2_1,4,0.0,0.0,0.0
1794241562.898,proposed,197122,e_2_1,1,0.0,0.0,0.0
1794241562.898,proposed,197122,e_2_1,2,0.0,0.0,0.0
1794241562.898,proposed,197122,e_2_1,3,0.0,0.0,0.0
1794241562.901,proposed,65536,c_0_0,4,0.0,0.0,0.0
1794241562.901,proposed,65536,c_0_0,1,0.0,0.0,0.0
1794241562.901,proposed,65536,c_0_0,2,0.0,0.0,0.0
1794241562.901,proposed,65536,c_0_0,3,0.0,0.0,0.0
1794241562.903,proposed,196609,e_0_0,1,0.0,0.0,0.0
1794241562.903,proposed,196609,e_0_0,4,0.0,0.0,0.0
1794241562.903,proposed,196609,e_0_0,2,0.0,0.0,0.0
1794241562.903,proposed,196609,e_0_0,3,0.0,0.0,0.0
1794241562.907,proposed,65539,c_1_1,4,0.0,0.0,0.0
1794241562.907,proposed,65539,c_1_1,1,0.0,0.0,0.0
1794241562.907,proposed,65539,c_1_1,2,0.0,0.0,0.0
1794241562.907,proposed,65539,c_1_1,3,0.0,0.0,0.0
1794241562.911,proposed,131073,a_0_1,1,0.0,0.0,0.0
1794241562.911,proposed,131073,a_0_1,4,0.0,0.0,0.0
1794241562.911,proposed,131073,a_0_1,2,0.0,0.0,0.0
1794241562.911,proposed,131073,a_0_1,3,0.0,0.0,0.0
1794241562.913,proposed,131585,a_2_1,1,0.0,0.0,0.0
1794241562.913,proposed,131585,a_2_1,4,0.0,0.0,0.0
1794241562.913,proposed,131585,a_2_1,2,0.0,0.0,0.0
1794241562.913,proposed,131585,a_2_1,3,0.0,0.0,0.0
1794241562.916,proposed,197377,e_3_0,1,0.0,0.0,0.0
1794241562.916,proposed,197377,e_3_0,4,0.0,0.0,0.0
1794241562.916,proposed,197377,e_3_0,2,0.0,0.0,0.0
1794241562.916,proposed,197377,e_3_0,3,0.0,0.0,0.0
1794241562.92,proposed,196610,e_0_1,4,0.0,0.0,0.0
1794241562.92,proposed,196610,e_0_1,1,0.0,0.0,0.0
1794241562.92,proposed,196610,e_0_1,2,0.0,0.0,0.0
1794241562.92,proposed,196610,e_0_1,3,0.0,0.0,0.0
1794241562.922,proposed,197378,e_3_1,4,0.0,0.0,0.0
1794241562.922,proposed,197378,e_3_1,1,0.0,0.0,0.0
1794241562.922,proposed,197378,e_3_1,2,0.0,0.0,0.0
1794241562.922,proposed,197378,e_3_1,3,0.0,0.0,0.0
1794241562.924,proposed,131840,a_3_0,4,0.0,0.0,0.0
1794241562.924,proposed,131840,a_3_0,1,0.0,0.0,0.0
1794241562.924,proposed,131840,a_3_0,2,0.0,0.0,0.0
1794241562.924,proposed,131840,a_3_0,3,0.0,0.0,0.0
1794241562.926,proposed,131329,a_1_1,1,0.0,0.0,0.0
1794241562.926,proposed,131329,a_1_1,4,0.0,0.0,0.0
1794241562.926,proposed,131329,a_1_1,2,0.0,0.0,0.0
1794241562.926,proposed,131329,a_1_1,3,0.0,0.0,0.0
1794241562.928,proposed,196866,e_1_1,4,0.0,0.0,0.0
1794241562.928,proposed,196866,e_1_1,1,0.0,0.0,0.0
1794241562.928,proposed,196866,e_1_1,2,0.0,0.0,0.0
1794241562.928,proposed,196866,e_1_1,3,0.0,0.0,0.0
1794241562.93,proposed,131072,a_0_0,4,0.0,0.0,0.0
1794241562.93,proposed,131072,a_0_0,1,0.0,0.0,0.0
1794241562.93,proposed,131072,a_0_0,2,0.0,0.0,0.0
1794241562.93,proposed,131072,a_0_0,3,0.0,0.0,0.0
1794241562.933,proposed,131584,a_2_0,4,0.0,0.0,0.0
1794241562.933,proposed,131584,a_2_0,1,0.0,0.0,0.0
1794241562.933,proposed,131584,a_2_0,2,0.0,0.0,0.0
1794241562.933,proposed,131584,a_2_0,3,0.0,0.0,0.0
1794241562.935,proposed,131328,a_1_0,1,0.0,0.0,0.0
1794241562.935,proposed,131328,a_1_0,4,0.0,0.0,0.0
1794241562.935,proposed,131328,a_1_0,2,0.0,0.0,0.0
1794241562.935,proposed,131328,a_1_0,3,0.0,0.0,0.0
1794241563.879,proposed,131073,a_0_1,1,0.0,0.0,0.0
1794241563.879,proposed,131073,a_0_1,4,0.0,0.0,0.0
1794241563.879,proposed,131073,a_0_1,2,0.0,0.0,0.0
1794241563.879,proposed,131073,a_0_1,3,0.0,0.0,0.0
1794241563.89,proposed,65537,c_0_1,4,0.0,0.0,0.0
1794241563.89,proposed,65537,c_0_1,1,0.0,0.0,0.0
1794241563.89,proposed,65537,c_0_1,2,0.0,0.0,0.0
1794241563.89,proposed,65537,c_0_1,3,0.0,0.0,0.0
1794241563.892,proposed,131585,a_2_1,1,0.0,0.0,0.0
1794241563.892,proposed,131585,a_2_1,4,0.0,0.0,0.0
1794241563.892,proposed,131585,a_2_1,2,0.0,0.0,0.0
1794241563.892,proposed,131585,a_2_1,3,0.0,0.0,0.0
1794241563.894,proposed,65538,c_1_0,4,0.0,0.0,0.0
1794241563.894,proposed,65538,c_1_0,1,0.0,0.0,0.0
1794241563.894,proposed,65538,c_1_0,2,0.0,0.0,0.0
1794241563.894,proposed,65538,c_1_0,3,0.0,0.0,0.0
1794241563.901,proposed,197122,e_2_1,4,0.0,0.0,0.0
1794241563.901,proposed,197122,e_2_1,1,0.0,0.0,0.0
1794241563.901,proposed,197122,e_2_1,2,0.0,0.0,0.0
1794241563.901,proposed,197122,e_2_1,3,0.0,0.0,0.0
1794241563.904,proposed,197377,e_3_0,1,0.0,0.0,0.0
1794241563.904,proposed,197377,e_3_0,4,0.0,0.0,0.0
1794241563.904,proposed,197377,e_3_0,2,0.0,0.0,0.0
1794241563.904,proposed,197377,e_3_0,3,0.0,0.0,0.0
1794241563.907,proposed,131072,a_0_0,4,0.0,0.0,0.0
1794241563.907,proposed,131072,a_0_0,1,0.0,0.0,0.0
1794241563.907,proposed,131072,a_0_0,2,0.0,0.0,0.0
1794241563.907,proposed,131072,a_0_0,3,0.0,0.0,0.0
1794241563.91,proposed,131329,a_1_1,1,0.0,0.0,0.0
1794241563.91,proposed,131329,a_1_1,4,0.0,0.0,0.0
1794241563.91,proposed,131329,a_1_1,2,0.0,0.0,0.0
1794241563.91,proposed,131329,a_1_1,3,0.0,0.0,0.0
1794241563.914,proposed,197378,e_3_1,4,0.0,0.0,0.0
1794241563.914,proposed,197378,e_3_1,1,0.0,0.0,0.0
1794241563.914,proposed,197378,e_3_1,2,0.0,0.0,0.0
1794241563.914,proposed,197378,e_3_1,3,0.0,0.0,0.0
1794241563.916,proposed,131584,a_2_0,4,0.0,0.0,0.0
1794241563.916,proposed,131584,a_2_0,1,0.0,0.0,0.0
1794241563.916,proposed,131584,a_2_0,2,0.0,0.0,0.0
1794241563.916,proposed,131584,a_2_0,3,0.0,0.0,0.0
1794241563.918,proposed,196865,e_1_0,4,0.0,0.0,0.0
1794241563.918,proposed,196865,e_1_0,1,0.0,0.0,0.0
1794241563.918,proposed,196865,e_1_0,2,0.0,0.0,0.0
1794241563.918,proposed,196865,e_1_0,3,0.0,0.0,0.0
1794241563.922,proposed,196866,e_1_1,4,0.0,0.0,0.0
1794241563.922,proposed,196866,e_1_1,1,0.0,0.0,0.0
1794241563.922,proposed,196866,e_1_1,2,0.0,0.0,0.0
1794241563.922,proposed,196866,e_1_1,3,0.0,0.0,0.0
1794241563.926,proposed,131840,a_3_0,4,0.0,0.0,0.0
1794241563.926,proposed,131840,a_3_0,1,0.0,0.0,0.0
1794241563.926,proposed,131840,a_3_0,2,0.0,0.0,0.0
1794241563.926,proposed,131840,a_3_0,3,0.0,0.0,0.0
1794241563.929,proposed,65536,c_0_0,4,0.0,0.0,0.0
1794241563.929,proposed,65536,c_0_0,1,0.0,0.0,0.0
1794241563.929,proposed,65536,c_0_0,2,0.0,0.0,0.0
1794241563.929,proposed,65536,c_0_0,3,0.0,0.0,0.0
1794241563.933,proposed,131328,a_1_0,1,0.0,0.0,0.0
1794241563.933,proposed,131328,a_1_0,4,0.0,0.0,0.0
1794241563.933,proposed,131328,a_1_0,2,0.0,0.0,0.0
1794241563.933,proposed,131328,a_1_0,3,0.0,0.0,0.0
1794241563.937,proposed,131841,a_3_1,1,0.0,0.0,0.0
1794241563.937,proposed,131841,a_3_1,4,0.0,0.0,0.0
1794241563.937,proposed,131841,a_3_1,2,0.0,0.0,0.0
1794241563.937,proposed,131841,a_3_1,3,0.0,0.0,0.0
1794241563.94,proposed,196610,e_0_1,4,0.0,0.0,0.0
1794241563.94,proposed,196610,e_0_1,1,0.0,0.0,0.0
1794241563.94,proposed,196610,e_0_1,2,0.0,0.0,0.0
1794241563.94,proposed,196610,e_0_1,3,0.0,0.0,0.0
1794241563.943,proposed,65539,c_1_1,4,0.0,0.0,0.0
1794241563.943,proposed,65539,c_1_1,1,0.0,0.0,0.0
1794241563.943,proposed,65539,c_1_1,2,0.0,0.0,0.0
1794241563.943,proposed,65539,c_1_1,3,0.0,0.0,0.0
1794241563.946,proposed,196609,e_0_0,1,0.0,0.0,0.0
1794241563.946,proposed,196609,e_0_0,4,0.0,0.0,0.0
1794241563.946,proposed,196609,e_0_0,2,0.0,0.0,0.0
1794241563.946,proposed,196609,e_0_0,3,0.0,0.0,0.0
1794241563.949,proposed,197121,e_2_0,4,0.0,0.0,0.0
1794241563.949,proposed,197121,e_2_0,1,0.0,0.0,0.0
1794241563.949,proposed,197121,e_2_0,2,0.0,0.0,0.0
1794241563.949,proposed,197121,e_2_0,3,0.0,0.0,0.0
1794241564.888,proposed,131073,a_0_1,1,0.0,0.0,0.0
1794241564.888,proposed,131073,a_0_1,4,0.0,0.0,0.0
1794241564.888,proposed,131073,a_0_1,2,0.0,0.0,0.0
1794241564.888,proposed,131073,a_0_1,3,0.0,0.0,0.0
1794241564.893,proposed,196609,e_0_0,1,0.0,0.0,0.0
1794241564.893,proposed,196609,e_0_0,4,0.0,0.0,0.0
1794241564.893,proposed,196609,e_0_0,2,0.0,0.0,0.0
1794241564.893,proposed,196609,e_0_0,3,0.0,0.0,0.0
1794241564.896,proposed,65538,c_1_0,4,0.0,0.0,0.0
1794241564.896,proposed,65538,c_1_0,1,0.0,0.0,0.0
1794241564.896,proposed,65538,c_1_0,2,0.0,0.0,0.0
1794241564.896,proposed,65538,c_1_0,3,0.0,0.0,0.0
1794241564.9,proposed,197122,e_2_1,4,0.0,0.0,0.0
1794241564.9,proposed,197122,e_2_1,1,0.0,0.0,0.0
1794241564.9,proposed,197122,e_2_1,2,0.0,0.0,0.0
1794241564.9,proposed,197122,e_2_1,3,0.0,0.0,0.0
1794241564.904,proposed,197378,e_3_1,4,0.0,0.0,0.0
1794241564.904,proposed,197378,e_3_1,1,0.0,0.0,0.0
1794241564.904,proposed,197378,e_3_1,2,0.0,0.0,0.0
1794241564.904,proposed,197378,e_3_1,3,0.0,0.0,0.0
1794241564.907,proposed,131585,a_2_1,1,0.0,0.0,0.0
1794241564.907,proposed,131585,a_2_1,4,0.0,0.0,0.0
1794241564.907,proposed,131585,a_2_1,2,0.0,0.0,0.0
1794241564.907,proposed,131585,a_2_1,3,0.0,0.0,0.0
1794241564.911,proposed,196865,e_1_0,4,0.0,0.0,0.0
1794241564.911,proposed,196865,e_1_0,1,0.0,0.0,0.0
1794241564.911,proposed,196865,e_1_0,2,0.0,0.0,0.0
1794241564.911,proposed,196865,e_1_0,3,0.0,0.0,0.0
1794241564.914,proposed,197377,e_3_0,1,0.0,0.0,0.0
1794241564.914,proposed,197377,e_3_0,4,0.0,0.0,0.0
1794241564.914,proposed,197377,e_3_0,2,0.0,0.0,0.0
1794241564.914,proposed,197377,e_3_0,3,0.0,0.0,0.0
1794241564.916,proposed,65537,c_0_1,4,0.0,0.0,0.0
1794241564.916,proposed,65537,c_0_1,1,0.0,0.0,0.0
1794241564.916,proposed,65537,c_0_1,2,0.0,0.0,0.0
1794241564.916,proposed,65537,c_0_1,3,0.0,0.0,0.0
1794241564.919,proposed,131841,a_3_1,1,0.0,0.0,0.0
1794241564.919,proposed,131841,a_3_1,4,0.0,0.0,0.0
1794241564.919,proposed,131841,a_3_1,2,0.0,0.0,0.0
1794241564.919,proposed,131841,a_3_1,3,0.0,0.0,0.0
1794241564.922,proposed,65539,c_1_1,4,0.0,0.0,0.0
1794241564.922,proposed,65539,c_1_1,1,0.0,0.0,0.0
1794241564.922,proposed,65539,c_1_1,2,0.0,0.0,0.0
1794241564.922,proposed,65539,c_1_1,3,0.0,0.0,0.0
1794241564.925,proposed,197121,e_2_0,4,0.0,0.0,0.0
1794241564.925,proposed,197121,e_2_0,1,0.0,0.0,0.0
1794241564.925,proposed,197121,e_2_0,2,0.0,0.0,0.0
1794241564.925,proposed,197121,e_2_0,3,0.0,0.0,0.0
1794241564.929,proposed,196610,e_0_1,4,0.0,0.0,0.0
1794241564.929,proposed,196610,e_0_1,1,0.0,0.0,0.0
1794241564.929,proposed,196610,e_0_1,2,0.0,0.0,0.0
1794241564.929,proposed,196610,e_0_1,3,0.0,0.0,0.0
1794241564.931,proposed,131328,a_1_0,1,0.0,0.0,0.0
1794241564.931,proposed,131328,a_1_0,4,0.0,0.0,0.0
1794241564.931,proposed,131328,a_1_0,2,0.0,0.0,0.0
1794241564.931,proposed,131328,a_1_0,3,0.0,0.0,0.0
1794241564.933,proposed,131329,a_1_1,1,0.0,0.0,0.0
1794241564.933,proposed,131329,a_1_1,4,0.0,0.0,0.0
1794241564.933,proposed,131329,a_1_1,2,0.0,0.0,0.0
1794241564.933,proposed,131329,a_1_1,3,0.0,0.0,0.0
1794241564.936,proposed,131072,a_0_0,4,0.0,0.0,0.0
1794241564.936,proposed,131072,a_0_0,1,0.0,0.0,0.0
1794241564.936,proposed,131072,a_0_0,2,0.0,0.0,0.0
1794241564.936,proposed,131072,a_0_0,3,0.0,0.0,0.0
1794241564.939,proposed,65536,c_0_0,4,0.0,0.0,0.0
1794241564.939,proposed,65536,c_0_0,1,0.0,0.0,0.0
1794241564.939,proposed,65536,c_0_0,2,0.0,0.0,0.0
1794241564.939,proposed,65536,c_0_0,3,0.0,0.0,0.0
1794241564.942,proposed,131584,a_2_0,4,0.0,0.0,0.0
1794241564.942,proposed,131584,a_2_0,1,0.0,0.0,0.0
1794241564.942,proposed,131584,a_2_0,2,0.0,0.0,0.0
1794241564.942,proposed,131584,a_2_0,3,0.0,0.0,0.0
1794241564.946,proposed,196866,e_1_1,4,0.0,0.0,0.0
1794241564.946,proposed,196866,e_1_1,1,0.0,0.0,0.0
1794241564.946,proposed,196866,e_1_1,2,0.0,0.0,0.0
1794241564.946,proposed,196866,e_1_1,3,0.0,0.0,0.0
1794241564.949,proposed,131840,a_3_0,4,0.0,0.0,0.0
1794241564.949,proposed,131840,a_3_0,1,0.0,0.0,0.0
1794241564.949,proposed,131840,a_3_0,2,0.0,0.0,0.0
1794241564.949,proposed,131840,a_3_0,3,0.0,0.0,0.0
1794241565.887,proposed,131073,a_0_1,1,0.0,0.0,0.0
1794241565.887,proposed,131073,a_0_1,4,0.0,0.0,0.0
1794241565.887,proposed,131073,a_0_1,2,0.0,0.0,0.0
1794241565.887,proposed,131073,a_0_1,3,0.0,0.0,0.0
1794241565.892,proposed,131585,a_2_1,1,0.0,0.0,0.0
1794241565.892,proposed,131585,a_2_1,4,0.0,0.0,0.0
1794241565.892,proposed,131585,a_2_1,2,0.0,0.0,0.0
1794241565.892,proposed,131585,a_2_1,3,0.0,0.0,0.0
1794241565.896,proposed,65537,c_0_1,4,0.0,0.0,0.0
1794241565.896,proposed,65537,c_0_1,1,0.0,0.0,0.0
1794241565.896,proposed,65537,c_0_1,2,0.0,0.0,0.0
1794241565.896,proposed,65537,c_0_1,3,0.0,0.0,0.0
1794241565.9,proposed,197122,e_2_1,4,0.0,0.0,0.0
1794241565.9,proposed,197122,e_2_1,1,0.0,0.0,0.0
1794241565.9,proposed,197122,e_2_1,2,0.0,0.0,0.0
1794241565.9,proposed,197122,e_2_1,3,0.0,0.0,0.0
1794241565.904,proposed,65538,c_1_0,4,0.0,0.0,0.0
1794241565.904,proposed,65538,c_1_0,1,0.0,0.0,0.0
1794241565.904,proposed,65538,c_1_0,2,0.0,0.0,0.0
1794241565.904,proposed,65538,c_1_0,3,0.0,0.0,0.0
1794241565.909,proposed,131841,a_3_1,1,0.0,0.0,0.0
1794241565.909,proposed,131841,a_3_1,4,0.0,0.0,0.0
1794241565.909,proposed,131841,a_3_1,2,0.0,0.0,0.0
1794241565.909,proposed,131841,a_3_1,3,0.0,0.0,0.0
1794241565.912,proposed,197377,e_3_0,1,0.0,0.0,0.0
1794241565.912,proposed,197377,e_3_0,4,0.0,0.0,0.0
1794241565.912,proposed,197377,e_3_0,2,0.0,0.0,0.0
1794241565.912,proposed,197377,e_3_0,3,0.0,0.0,0.0
1794241565.915,proposed,131328,a_1_0,1,0.0,0.0,0.0
1794241565.915,proposed,131328,a_1_0,4,0.0,0.0,0.0
1794241565.915,proposed,131328,a_1_0,2,0.0,0.0,0.0
1794241565.915,proposed,131328,a_1_0,3,0.0,0.0,0.0
1794241565.917,proposed,197378,e_3_1,4,0.0,0.0,0.0
1794241565.917,proposed,197378,e_3_1,1,0.0,0.0,0.0
1794241565.917,proposed,197378,e_3_1,2,0.0,0.0,0.0
1794241565.917,proposed,197378,e_3_1,3,0.0,0.0,0.0
1794241565.919,proposed,197121,e_2_0,4,0.0,0.0,0.0
1794241565.919,proposed,197121,e_2_0,1,0.0,0.0,0.0
1794241565.919,proposed,197121,e_2_0,2,0.0,0.0,0.0
1794241565.919,proposed,197121,e_2_0,3,0.0,0.0,0.0
1794241565.921,proposed,196866,e_1_1,4,0.0,0.0,0.0
1794241565.921,proposed,196866,e_1_1,1,0.0,0.0,0.0
1794241565.921,proposed,196866,e_1_1,2,0.0,0.0,0.0
1794241565.921,proposed,196866,e_1_1,3,0.0,0.0,0.0
1794241565.923,proposed,131072,a_0_0,4,0.0,0.0,0.0
1794241565.923,proposed,131072,a_0_0,1,0.0,0.0,0.0
1794241565.923,proposed,131072,a_0_0,2,0.0,0.0,0.0
1794241565.923,proposed,131072,a_0_0,3,0.0,0.0,0.0
1794241565.926,proposed,196865,e_1_0,4,0.0,0.0,0.0
1794241565.926,proposed,196865,e_1_0,1,0.0,0.0,0.0
1794241565.926,proposed,196865,e_1_0,2,0.0,0.0,0.0
1794241565.926,proposed,196865,e_1_0,3,0.0,0.0,0.0
1794241565.928,proposed,131584,a_2_0,4,0.0,0.0,0.0
1794241565.928,proposed,131584,a_2_0,1,0.0,0.0,0.0
1794241565.928,proposed,131584,a_2_0,2,0.0,0.0,0.0
1794241565.928,proposed,131584,a_2_0,3,0.0,0.0,0.0
1794241565.931,proposed,131840,a_3_0,4,0.0,0.0,0.0
1794241565.931,proposed,131840,a_3_0,1,0.0,0.0,0.0
1794241565.931,proposed,131840,a_3_0,2,0.0,0.0,0.0
1794241565.931,proposed,131840,a_3_0,3,0.0,0.0,0.0
1794241565.933,proposed,65539,c_1_1,4,0.0,0.0,0.0
1794241565.933,proposed,65539,c_1_1,1,0.0,0.0,0.0
1794241565.933,proposed,65539,c_1_1,2,0.0,0.0,0.0
1794241565.933,proposed,65539,c_1_1,3,0.0,0.0,0.0
1794241565.936,proposed,196609,e_0_0,1,0.0,0.0,0.0
1794241565.936,proposed,196609,e_0_0,4,0.0,0.0,0.0
1794241565.936,proposed,196609,e_0_0,2,0.0,0.0,0.0
1794241565.936,proposed,196609,e_0_0,3,0.0,0.0,0.0
1794241565.938,proposed,196610,e_0_1,4,0.0,0.0,0.0
1794241565.938,proposed,196610,e_0_1,1,0.0,0.0,0.0
1794241565.938,proposed,196610,e_0_1,2,0.0,0.0,0.0
1794241565.938,proposed,196610,e_0_1,3,0.0,0.0,0.0
1794241565.941,proposed,131329,a_1_1,1,0.0,0.0,0.0
1794241565.941,proposed,131329,a_1_1,4,0.0,0.0,0.0
1794241565.941,proposed,131329,a_1_1,2,0.0,0.0,0.0
1794241565.941,proposed,131329,a_1_1,3,0.0,0.0,0.0
1794241565.944,proposed,65536,c_0_0,4,0.0,0.0,0.0
1794241565.944,proposed,65536,c_0_0,1,0.0,0.0,0.0
1794241565.944,proposed,65536,c_0_0,2,0.0,0.0,0.0
1794241565.944,proposed,65536,c_0_0,3,0.0,0.0,0.0
1794241566.889,proposed,197122,e_2_1,4,0.0,0.0,0.0
1794241566.889,proposed,197122,e_2_1,1,0.0,0.0,0.0
1794241566.889,proposed,197122,e_2_1,2,0.0,0.0,0.0
1794241566.889,proposed,197122,e_2_1,3,0.0,0.0,0.0
1794241566.894,proposed,65538,c_1_0,4,0.0,0.0,0.0
1794241566.894,proposed,65538,c_1_0,1,0.0,0.0,0.0
1794241566.894,proposed,65538,c_1_0,2,0.0,0.0,0.0
1794241566.894,proposed,65538,c_1_0,3,0.0,0.0,0.0
1794241566.897,proposed,65537,c_0_1,4,0.0,0.0,0.0
1794241566.897,proposed,65537,c_0_1,1,0.0,0.0,0.0
1794241566.897,proposed,65537,c_0_1,2,0.0,0.0,0.0
1794241566.897,proposed,65537,c_0_1,3,0.0,0.0,0.0
1794241566.9,proposed,131073,a_0_1,1,0.0,0.0,0.0
1794241566.9,proposed,131073,a_0_1,4,0.0,0.0,0.0
1794241566.9,proposed,131073,a_0_1,2,0.0,0.0,0.0
1794241566.9,proposed,131073,a_0_1,3,0.0,0.0,0.0
1794241566.903,proposed,131585,a_2_1,1,0.0,0.0,0.0
1794241566.903,proposed,131585,a_2_1,4,0.0,0.0,0.0
1794241566.903,proposed,131585,a_2_1,2,0.0,0.0,0.0
1794241566.903,proposed,131585,a_2_1,3,0.0,0.0,0.0
1794241566.911,proposed,196865,e_1_0,4,0.0,0.0,0.0
1794241566.911,proposed,196865,e_1_0,1,0.0,0.0,0.0
1794241566.911,proposed,196865,e_1_0,2,0.0,0.0,0.0
1794241566.911,proposed,196865,e_1_0,3,0.0,0.0,0.0
1794241566.914,proposed,65539,c_1_1,4,0.0,0.0,0.0
1794241566.914,proposed,65539,c_1_1,1,0.0,0.0,0.0
1794241566.914,proposed,65539,c_1_1,2,0.0,0.0,0.0
1794241566.914,proposed,65539,c_1_1,3,0.0,0.0,0.0
1794241566.917,proposed,131841,a_3_1,1,0.0,0.0,0.0
1794241566.917,proposed,131841,a_3_1,4,0.0,0.0,0.0
1794241566.917,proposed,131841,a_3_1,2,0.0,0.0,0.0
1794241566.917,proposed,131841,a_3_1,3,0.0,0.0,0.0
1794241566.92,proposed,65536,c_0_0,4,0.0,0.0,0.0
1794241566.92,proposed,65536,c_0_0,1,0.0,0.0,0.0
1794241566.92,proposed,65536,c_0_0,2,0.0,0.0,0.0
1794241566.92,proposed,65536,c_0_0,3,0.0,0.0,0.0
1794241566.923,proposed,197378,e_3_1,4,0.0,0.0,0.0
1794241566.923,proposed,197378,e_3_1,1,0.0,0.0,0.0
1794241566.923,proposed,197378,e_3_1,2,0.0,0.0,0.0
1794241566.923,proposed,197378,e_3_1,3,0.0,0.0,0.0
1794241566.926,proposed,131328,a_1_0,1,0.0,0.0,0.0
1794241566.926,proposed,131328,a_1_0,4,0.0,0.0,0.0
1794241566.926,proposed,131328,a_1_0,2,0.0,0.0,0.0
1794241566.926,proposed,131328,a_1_0,3,0.0,0.0,0.0
1794241566.93,proposed,196610,e_0_1,4,0.0,0.0,0.0
1794241566.93,proposed,196610,e_0_1,1,0.0,0.0,0.0
1794241566.93,proposed,196610,e_0_1,2,0.0,0.0,0.0
1794241566.93,proposed,196610,e_0_1,3,0.0,0.0,0.0
1794241566.933,proposed,196609,e_0_0,1,0.0,0.0,0.0
1794241566.933,proposed,196609,e_0_0,4,0.0,0.0,0.0
1794241566.933,proposed,196609,e_0_0,2,0.0,0.0,0.0
1794241566.933,proposed,196609,e_0_0,3,0.0,0.0,0.0
1794241566.936,proposed,131329,a_1_1,1,0.0,0.0,0.0
1794241566.936,proposed,131329,a_1_1,4,0.0,0.0,0.0
1794241566.936,proposed,131329,a_1_1,2,0.0,0.0,0.0
1794241566.936,proposed,131329,a_1_1,3,0.0,0.0,0.0
1794241566.939,proposed,131584,a_2_0,4,0.0,0.0,0.0
1794241566.939,proposed,131584,a_2_0,1,0.0,0.0,0.0
1794241566.939,proposed,131584,a_2_0,2,0.0,0.0,0.0
1794241566.939,proposed,131584,a_2_0,3,0.0,0.0,0.0
1794241566.942,proposed,131840,a_3_0,4,0.0,0.0,0.0
1794241566.942,proposed,131840,a_3_0,1,0.0,0.0,0.0
1794241566.942,proposed,131840,a_3_0,2,0.0,0.0,0.0
1794241566.942,proposed,131840,a_3_0,3,0.0,0.0,0.0
1794241566.945,proposed,196866,e_1_1,4,0.0,0.0,0.0
1794241566.945,proposed,196866,e_1_1,1,0.0,0.0,0.0
1794241566.945,proposed,196866,e_1_1,2,0.0,0.0,0.0
1794241566.945,proposed,196866,e_1_1,3,0.0,0.0,0.0
1794241566.948,proposed,131072,a_0_0,4,0.0,0.0,0.0
1794241566.948,proposed,131072,a_0_0,1,0.0,0.0,0.0
1794241566.948,proposed,131072,a_0_0,2,0.0,0.0,0.0
1794241566.948,proposed,131072,a_0_0,3,0.0,0.0,0.0
1794241566.951,proposed,197121,e_2_0,4,0.0,0.0,0.0
1794241566.951,proposed,197121,e_2_0,1,0.0,0.0,0.0
1794241566.951,proposed,197121,e_2_0,2,0.0,0.0,0.0
1794241566.951,proposed,197121,e_2_0,3,0.0,0.0,0.0
1794241566.955,proposed,197377,e_3_0,1,0.0,0.0,0.0
1794241566.955,proposed,197377,e_3_0,4,0.0,0.0,0.0
1794241566.955,proposed,197377,e_3_0,2,0.0,0.0,0.0
1794241566.955,proposed,197377,e_3_0,3,0.0,0.0,0.0
1794241567.892,proposed,196609,e_0_0,1,0.0,0.0,0.0
1794241567.892,proposed,196609,e_0_0,4,0.0,0.0,0.0
1794241567.892,proposed,196609,e_0_0,2,0.0,0.0,0.0
1794241567.892,proposed,196609,e_0_0,3,0.0,0.0,0.0
1794241567.895,proposed,65536,c_0_0,4,0.0,0.0,0.0
1794241567.895,proposed,65536,c_0_0,1,0.0,0.0,0.0
1794241567.895,proposed,65536,c_0_0,2,0.0,0.0,0.0
1794241567.895,proposed,65536,c_0_0,3,0.0,0.0,0.0
1794241567.897,proposed,196865,e_1_0,4,0.0,0.0,0.0
1794241567.897,proposed,196865,e_1_0,1,0.0,0.0,0.0
1794241567.897,proposed,196865,e_1_0,2,0.0,0.0,0.0
1794241567.897,proposed,196865,e_1_0,3,0.0,0.0,0.0
1794241567.9,proposed,131585,a_2_1,1,0.0,0.0,0.0
1794241567.9,proposed,131585,a_2_1,4,0.0,0.0,0.0
1794241567.9,proposed,131585,a_2_1,2,0.0,0.0,0.0
1794241567.9,proposed,131585,a_2_1,3,0.0,0.0,0.0
1794241567.903,proposed,131072,a_0_0,4,0.0,0.0,0.0
1794241567.903,proposed,131072,a_0_0,1,0.0,0.0,0.0
1794241567.903,proposed,131072,a_0_0,2,0.0,0.0,0.0
1794241567.903,proposed,131072,a_0_0,3,0.0,0.0,0.0
1794241567.906,proposed,196866,e_1_1,4,0.0,0.0,0.0
1794241567.906,proposed,196866,e_1_1,1,0.0,0.0,0.0
1794241567.906,proposed,196866,e_1_1,2,0.0,0.0,0.0
1794241567.906,proposed,196866,e_1_1,3,0.0,0.0,0.0
1794241567.909,proposed,65538,c_1_0,4,0.0,0.0,0.0
1794241567.909,proposed,65538,c_1_0,1,0.0,0.0,0.0
1794241567.909,proposed,65538,c_1_0,2,0.0,0.0,0.0
1794241567.909,proposed,65538,c_1_0,3,0.0,0.0,0.0
1794241567.912,proposed,65539,c_1_1,4,0.0,0.0,0.0
1794241567.912,proposed,65539,c_1_1,1,0.0,0.0,0.0
1794241567.912,proposed,65539,c_1_1,2,0.0,0.0,0.0
1794241567.912,proposed,65539,c_1_1,3,0.0,0.0,0.0
1794241567.915,proposed,131073,a_0_1,1,0.0,0.0,0.0
1794241567.915,proposed,131073,a_0_1,4,0.0,0.0,0.0
1794241567.915,proposed,131073,a_0_1,2,0.0,0.0,0.0
1794241567.915,proposed,131073,a_0_1,3,0.0,0.0,0.0
1794241567.919,proposed,131840,a_3_0,4,0.0,0.0,0.0
1794241567.919,proposed,131840,a_3_0,1,0.0,0.0,0.0
1794241567.919,proposed,131840,a_3_0,2,0.0,0.0,0.0
1794241567.919,proposed,131840,a_3_0,3,0.0,0.0,0.0
1794241567.922,proposed,131584,a_2_0,4,0.0,0.0,0.0
1794241567.922,proposed,131584,a_2_0,1,0.0,0.0,0.0
1794241567.922,proposed,131584,a_2_0,2,0.0,0.0,0.0
1794241567.922,proposed,131584,a_2_0,3,0.0,0.0,0.0
1794241567.924,proposed,197122,e_2_1,4,0.0,0.0,0.0
1794241567.924,proposed,197122,e_2_1,1,0.0,0.0,0.0
1794241567.924,proposed,197122,e_2_1,2,0.0,0.0,0.0
1794241567.924,proposed,197122,e_2_1,3,0.0,0.0,0.0
1794241567.927,proposed,131329,a_1_1,1,0.0,0.0,0.0
1794241567.927,proposed,131329,a_1_1,4,0.0,0.0,0.0
1794241567.927,proposed,131329,a_1_1,2,0.0,0.0,0.0
1794241567.927,proposed,131329,a_1_1,3,0.0,0.0,0.0
1794241567.93,proposed,197378,e_3_1,4,0.0,0.0,0.0
1794241567.93,proposed,197378,e_3_1,1,0.0,0.0,0.0
1794241567.93,proposed,197378,e_3_1,2,0.0,0.0,0.0
1794241567.93,proposed,197378,e_3_1,3,0.0,0.0,0.0
1794241567.932,proposed,197121,e_2_0,4,0.0,0.0,0.0
1794241567.932,proposed,197121,e_2_0,1,0.0,0.0,0.0
1794241567.932,proposed,197121,e_2_0,2,0.0,0.0,0.0
1794241567.932,proposed,197121,e_2_0,3,0.0,0.0,0.0
1794241567.935,proposed,65537,c_0_1,4,0.0,0.0,0.0
1794241567.935,proposed,65537,c_0_1,1,0.0,0.0,0.0
1794241567.935,proposed,65537,c_0_1,2,0.0,0.0,0.0
1794241567.935,proposed,65537,c_0_1,3,0.0,0.0,0.0
1794241567.938,proposed,131328,a_1_0,1,0.0,0.0,0.0
1794241567.938,proposed,131328,a_1_0,4,0.0,0.0,0.0
1794241567.938,proposed,131328,a_1_0,2,0.0,0.0,0.0
1794241567.938,proposed,131328,a_1_0,3,0.0,0.0,0.0
1794241567.941,proposed,197377,e_3_0,1,0.0,0.0,0.0
1794241567.941,proposed,197377,e_3_0,4,0.0,0.0,0.0
1794241567.941,proposed,197377,e_3_0,2,0.0,0.0,0.0
1794241567.941,proposed,197377,e_3_0,3,0.0,0.0,0.0
1794241567.943,proposed,131841,a_3_1,1,0.0,0.0,0.0
1794241567.943,proposed,131841,a_3_1,4,0.0,0.0,0.0
1794241567.943,proposed,131841,a_3_1,2,0.0,0.0,0.0
1794241567.943,proposed,131841,a_3_1,3,0.0,0.0,0.0
1794241567.946,proposed,196610,e_0_1,4,0.0,0.0,0.0
1794241567.946,proposed,196610,e_0_1,1,0.0,0.0,0.0
1794241567.946,proposed,196610,e_0_1,2,0.0,0.0,0.0
1794241567.946,proposed,196610,e_0_1,3,0.0,0.0,0.0
1794241568.894,proposed,131073,a_0_1,1,0.0,0.0,0.0
1794241568.894,proposed,131073,a_0_1,4,0.0,0.0,0.0
1794241568.894,proposed,131073,a_0_1,2,0.0,0.0,0.0
1794241568.894,proposed,131073,a_0_1,3,0.0,0.0,0.0
1794241568.901,proposed,131585,a_2_1,1,0.0,0.0,0.0
1794241568.901,proposed,131585,a_2_1,4,0.0,0.0,0.0
1794241568.901,proposed,131585,a_2_1,2,0.0,0.0,0.0
1794241568.901,proposed,131585,a_2_1,3,0.0,0.0,0.0
1794241568.904,proposed,65538,c_1_0,4,0.0,0.0,0.0
1794241568.904,proposed,65538,c_1_0,1,0.0,0.0,0.0
1794241568.904,proposed,65538,c_1_0,2,0.0,0.0,0.0
1794241568.904,proposed,65538,c_1_0,3,0.0,0.0,0.0
1794241568.907,proposed,65537,c_0_1,4,0.0,0.0,0.0
1794241568.907,proposed,65537,c_0_1,1,0.0,0.0,0.0
1794241568.907,proposed,65537,c_0_1,2,0.0,0.0,0.0
1794241568.907,proposed,65537,c_0_1,3,0.0,0.0,0.0
1794241568.915,proposed,196866,e_1_1,4,0.0,0.0,0.0
1794241568.915,proposed,196866,e_1_1,1,0.0,0.0,0.0
1794241568.915,proposed,196866,e_1_1,2,0.0,0.0,0.0
1794241568.915,proposed,196866,e_1_1,3,0.0,0.0,0.0
1794241568.918,proposed,131072,a_0_0,4,0.0,0.0,0.0
1794241568.918,proposed,131072,a_0_0,1,0.0,0.0,0.0
1794241568.918,proposed,131072,a_0_0,2,0.0,0.0,0.0
1794241568.918,proposed,131072,a_0_0,3,0.0,0.0,0.0
1794241568.92,proposed,196609,e_0_0,1,0.0,0.0,0.0
1794241568.92,proposed,196609,e_0_0,4,0.0,0.0,0.0
1794241568.92,proposed,196609,e_0_0,2,0.0,0.0,0.0
1794241568.92,proposed,196609,e_0_0,3,0.0,0.0,0.0
1794241568.922,proposed,196610,e_0_1,4,0.0,0.0,0.0
1794241568.922,proposed,196610,e_0_1,1,0.0,0.0,0.0
1794241568.922,proposed,196610,e_0_1,2,0.0,0.0,0.0
1794241568.922,proposed,196610,e_0_1,3,0.0,0.0,0.0
1794241568.925,proposed,196865,e_1_0,4,0.0,0.0,0.0
1794241568.925,proposed,196865,e_1_0,1,0.0,0.0,0.0
1794241568.925,proposed,196865,e_1_0,2,0.0,0.0,0.0
1794241568.925,proposed,196865,e_1_0,3,0.0,0.0,0.0
1794241568.928,proposed,131584,a_2_0,4,0.0,0.0,0.0
1794241568.928,proposed,131584,a_2_0,1,0.0,0.0,0.0
1794241568.928,proposed,131584,a_2_0,2,0.0,0.0,0.0
1794241568.928,proposed,131584,a_2_0,3,0.0,0.0,0.0
1794241568.93,proposed,131841,a_3_1,1,0.0,0.0,0.0
1794241568.93,proposed,131841,a_3_1,4,0.0,0.0,0.0
1794241568.93,proposed,131841,a_3_1,2,0.0,0.0,0.0
1794241568.93,proposed,131841,a_3_1,3,0.0,0.0,0.0
1794241568.933,proposed,65539,c_1_1,4,0.0,0.0,0.0
1794241568.933,proposed,65539,c_1_1,1,0.0,0.0,0.0
1794241568.933,proposed,65539,c_1_1,2,0.0,0.0,0.0
1794241568.933,proposed,65539,c_1_1,3,0.0,0.0,0.0
1794241568.935,proposed,131328,a_1_0,1,0.0,0.0,0.0
1794241568.935,proposed,131328,a_1_0,4,0.0,0.0,0.0
1794241568.935,proposed,131328,a_1_0,2,0.0,0.0,0.0
1794241568.935,proposed,131328,a_1_0,3,0.0,0.0,0.0
1794241568.938,proposed,131840,a_3_0,4,0.0,0.0,0.0
1794241568.938,proposed,131840,a_3_0,1,0.0,0.0,0.0
1794241568.938,proposed,131840,a_3_0,2,0.0,0.0,0.0
1794241568.938,proposed,131840,a_3_0,3,0.0,0.0,0.0
1794241568.941,proposed,197121,e_2_0,4,0.0,0.0,0.0
1794241568.941,proposed,197121,e_2_0,1,0.0,0.0,0.0
1794241568.941,proposed,197121,e_2_0,2,0.0,0.0,0.0
1794241568.941,proposed,197121,e_2_0,3,0.0,0.0,0.0
1794241568.945,proposed,131329,a_1_1,1,0.0,0.0,0.0
1794241568.945,proposed,131329,a_1_1,4,0.0,0.0,0.0
1794241568.945,proposed,131329,a_1_1,2,0.0,0.0,0.0
1794241568.945,proposed,131329,a_1_1,3,0.0,0.0,0.0
1794241568.948,proposed,197122,e_2_1,4,0.0,0.0,0.0
1794241568.948,proposed,197122,e_2_1,1,0.0,0.0,0.0
1794241568.948,proposed,197122,e_2_1,2,0.0,0.0,0.0
1794241568.948,proposed,197122,e_2_1,3,0.0,0.0,0.0
1794241568.951,proposed,65536,c_0_0,4,0.0,0.0,0.0
1794241568.951,proposed,65536,c_0_0,1,0.0,0.0,0.0
1794241568.951,proposed,65536,c_0_0,2,0.0,0.0,0.0
1794241568.951,proposed,65536,c_0_0,3,0.0,0.0,0.0
1794241568.955,proposed,197377,e_3_0,1,0.0,0.0,0.0
1794241568.955,proposed,197377,e_3_0,4,0.0,0.0,0.0
1794241568.955,proposed,197377,e_3_0,2,0.0,0.0,0.0
1794241568.955,proposed,197377,e_3_0,3,0.0,0.0,0.0
1794241568.959,proposed,197378,e_3_1,4,0.0,0.0,0.0
1794241568.959,proposed,197378,e_3_1,1,0.0,0.0,0.0
1794241568.959,proposed,197378,e_3_1,2,0.0,0.0,0.0
1794241568.959,proposed,197378,e_3_1,3,0.0,0.0,0.0
1794241569.896,proposed,65538,c_1_0,4,0.0,0.0,0.0
1794241569.896,proposed,65538,c_1_0,1,0.0,0.0,0.0
1794241569.896,proposed,65538,c_1_0,2,0.0,0.0,0.0
1794241569.896,proposed,65538,c_1_0,3,0.0,0.0,0.0
1794241569.902,proposed,65537,c_0_1,4,0.0,0.0,0.0
1794241569.902,proposed,65537,c_0_1,1,0.0,0.0,0.0
1794241569.902,proposed,65537,c_0_1,2,0.0,0.0,0.0
1794241569.902,proposed,65537,c_0_1,3,0.0,0.0,0.0
1794241569.908,proposed,65536,c_0_0,4,0.0,0.0,0.0
1794241569.908,proposed,65536,c_0_0,1,0.0,0.0,0.0
1794241569.908,proposed,65536,c_0_0,2,0.0,0.0,0.0
1794241569.908,proposed,65536,c_0_0,3,0.0,0.0,0.0
1794241569.912,proposed,196609,e_0_0,1,0.0,0.0,0.0
1794241569.912,proposed,196609,e_0_0,4,0.0,0.0,0.0
1794241569.912,proposed,196609,e_0_0,2,0.0,0.0,0.0
1794241569.912,proposed,196609,e_0_0,3,0.0,0.0,0.0
1794241569.914,proposed,131584,a_2_0,4,0.0,0.0,0.0
1794241569.914,proposed,131584,a_2_0,1,0.0,0.0,0.0
1794241569.914,proposed,131584,a_2_0,2,0.0,0.0,0.0
1794241569.914,proposed,131584,a_2_0,3,0.0,0.0,0.0
1794241569.918,proposed,131585,a_2_1,1,0.0,0.0,0.0
1794241569.918,proposed,131585,a_2_1,4,0.0,0.0,0.0
1794241569.918,proposed,131585,a_2_1,2,0.0,0.0,0.0
1794241569.918,proposed,131585,a_2_1,3,0.0,0.0,0.0
1794241569.921,proposed,196866,e_1_1,4,0.0,0.0,0.0
1794241569.921,proposed,196866,e_1_1,1,0.0,0.0,0.0
1794241569.921,proposed,196866,e_1_1,2,0.0,0.0,0.0
1794241569.921,proposed,196866,e_1_1,3,0.0,0.0,0.0
1794241569.924,proposed,197378,e_3_1,4,0.0,0.0,0.0
1794241569.924,proposed,197378,e_3_1,1,0.0,0.0,0.0
1794241569.924,proposed,197378,e_3_1,2,0.0,0.0,0.0
1794241569.924,proposed,197378,e_3_1,3,0.0,0.0,0.0
1794241569.928,proposed,197121,e_2_0,4,0.0,0.0,0.0
1794241569.928,proposed,197121,e_2_0,1,0.0,0.0,0.0
1794241569.928,proposed,197121,e_2_0,2,0.0,0.0,0.0
1794241569.928,proposed,197121,e_2_0,3,0.0,0.0,0.0
1794241569.931,proposed,196865,e_1_0,4,0.0,0.0,0.0
1794241569.931,proposed,196865,e_1_0,1,0.0,0.0,0.0
1794241569.931,proposed,196865,e_1_0,2,0.0,0.0,0.0
1794241569.931,proposed,196865,e_1_0,3,0.0,0.0,0.0
1794241569.933,proposed,196610,e_0_1,4,0.0,0.0,0.0
1794241569.933,proposed,196610,e_0_1,1,0.0,0.0,0.0
1794241569.933,proposed,196610,e_0_1,2,0.0,0.0,0.0
1794241569.933,proposed,196610,e_0_1,3,0.0,0.0,0.0
1794241569.936,proposed,131840,a_3_0,4,0.0,0.0,0.0
1794241569.936,proposed,131840,a_3_0,1,0.0,0.0,0.0
1794241569.936,proposed,131840,a_3_0,2,0.0,0.0,0.0
1794241569.936,proposed,131840,a_3_0,3,0.0,0.0,0.0
1794241569.939,proposed,131329,a_1_1,1,0.0,0.0,0.0
1794241569.939,proposed,131329,a_1_1,4,0.0,0.0,0.0
1794241569.939,proposed,131329,a_1_1,2,0.0,0.0,0.0
1794241569.939,proposed,131329,a_1_1,3,0.0,0.0,0.0
1794241569.941,proposed,65539,c_1_1,4,0.0,0.0,0.0
1794241569.941,proposed,65539,c_1_1,1,0.0,0.0,0.0
1794241569.941,proposed,65539,c_1_1,2,0.0,0.0,0.0
1794241569.941,proposed,65539,c_1_1,3,0.0,0.0,0.0
1794241569.944,proposed,131328,a_1_0,1,0.0,0.0,0.0
1794241569.944,proposed,131328,a_1_0,4,0.0,0.0,0.0
1794241569.944,proposed,131328,a_1_0,2,0.0,0.0,0.0
1794241569.944,proposed,131328,a_1_0,3,0.0,0.0,0.0
1794241569.947,proposed,131073,a_0_1,1,0.0,0.0,0.0
1794241569.947,proposed,131073,a_0_1,4,0.0,0.0,0.0
1794241569.947,proposed,131073,a_0_1,2,0.0,0.0,0.0
1794241569.947,proposed,131073,a_0_1,3,0.0,0.0,0.0
1794241569.95,proposed,131072,a_0_0,4,0.0,0.0,0.0
1794241569.95,proposed,131072,a_0_0,1,0.0,0.0,0.0
1794241569.95,proposed,131072,a_0_0,2,0.0,0.0,0.0
1794241569.95,proposed,131072,a_0_0,3,0.0,0.0,0.0
1794241569.954,proposed,197122,e_2_1,4,0.0,0.0,0.0
1794241569.954,proposed,197122,e_2_1,1,0.0,0.0,0.0
1794241569.954,proposed,197122,e_2_1,2,0.0,0.0,0.0
1794241569.954,proposed,197122,e_2_1,3,0.0,0.0,0.0
1794241569.956,proposed,197377,e_3_0,1,0.0,0.0,0.0
1794241569.956,proposed,197377,e_3_0,4,0.0,0.0,0.0
1794241569.956,proposed,197377,e_3_0,2,0.0,0.0,0.0
1794241569.956,proposed,197377,e_3_0,3,0.0,0.0,0.0
1794241569.959,proposed,131841,a_3_1,1,0.0,0.0,0.0
1794241569.959,proposed,131841,a_3_1,4,0.0,0.0,0.0
1794241569.959,proposed,131841,a_3_1,2,0.0,0.0,0.0
1794241569.959,proposed,131841,a_3_1,3,0.0,0.0,0.0
1794241570.9,proposed,65537,c_0_1,4,0.0,0.0,0.0
1794241570.9,proposed,65537,c_0_1,1,0.0,0.0,0.0
1794241570.9,proposed,65537,c_0_1,2,0.0,0.0,0.0
1794241570.9,proposed,65537,c_0_1,3,0.0,0.0,0.0
1794241570.907,proposed,65538,c_1_0,4,0.0,0.0,0.0
1794241570.907,proposed,65538,c_1_0,1,0.0,0.0,0.0
1794241570.907,proposed,65538,c_1_0,2,0.0,0.0,0.0
1794241570.907,proposed,65538,c_1_0,3,0.0,0.0,0.0
1794241570.911,proposed,131073,a_0_1,1,0.0,0.0,0.0
1794241570.911,proposed,131073,a_0_1,4,0.0,0.0,0.0
1794241570.911,proposed,131073,a_0_1,2,0.0,0.0,0.0
1794241570.911,proposed,131073,a_0_1,3,0.0,0.0,0.0
1794241570.914,proposed,131585,a_2_1,1,0.0,0.0,0.0
1794241570.914,proposed,131585,a_2_1,4,0.0,0.0,0.0
1794241570.914,proposed,131585,a_2_1,2,0.0,0.0,0.0
1794241570.914,proposed,131585,a_2_1,3,0.0,0.0,0.0
1794241570.922,proposed,131584,a_2_0,4,0.0,0.0,0.0
1794241570.922,proposed,131584,a_2_0,1,0.0,0.0,0.0
1794241570.922,proposed,131584,a_2_0,2,0.0,0.0,0.0
1794241570.922,proposed,131584,a_2_0,3,0.0,0.0,0.0
1794241570.924,proposed,131328,a_1_0,1,0.0,0.0,0.0
1794241570.924,proposed,131328,a_1_0,4,0.0,0.0,0.0
1794241570.924,proposed,131328,a_1_0,2,0.0,0.0,0.0
1794241570.924,proposed,131328,a_1_0,3,0.0,0.0,0.0
1794241570.927,proposed,196866,e_1_1,4,0.0,0.0,0.0
1794241570.927,proposed,196866,e_1_1,1,0.0,0.0,0.0
1794241570.927,proposed,196866,e_1_1,2,0.0,0.0,0.0
1794241570.927,proposed,196866,e_1_1,3,0.0,0.0,0.0
1794241570.93,proposed,197121,e_2_0,4,0.0,0.0,0.0
1794241570.93,proposed,197121,e_2_0,1,0.0,0.0,0.0
1794241570.93,proposed,197121,e_2_0,2,0.0,0.0,0.0
1794241570.93,proposed,197121,e_2_0,3,0.0,0.0,0.0
1794241570.932,proposed,196865,e_1_0,4,0.0,0.0,0.0
1794241570.932,proposed,196865,e_1_0,1,0.0,0.0,0.0
1794241570.932,proposed,196865,e_1_0,2,0.0,0.0,0.0
1794241570.932,proposed,196865,e_1_0,3,0.0,0.0,0.0
1794241570.934,proposed,131329,a_1_1,1,0.0,0.0,0.0
1794241570.934,proposed,131329,a_1_1,4,0.0,0.0,0.0
1794241570.934,proposed,131329,a_1_1,2,0.0,0.0,0.0
1794241570.934,proposed,131329,a_1_1,3,0.0,0.0,0.0
1794241570.937,proposed,197122,e_2_1,4,0.0,0.0,0.0
1794241570.937,proposed,197122,e_2_1,1,0.0,0.0,0.0
1794241570.937,proposed,197122,e_2_1,2,0.0,0.0,0.0
1794241570.937,proposed,197122,e_2_1,3,0.0,0.0,0.0
1794241570.94,proposed,197378,e_3_1,4,0.0,0.0,0.0
1794241570.94,proposed,197378,e_3_1,1,0.0,0.0,0.0
1794241570.94,proposed,197378,e_3_1,2,0.0,0.0,0.0
1794241570.94,proposed,197378,e_3_1,3,0.0,0.0,0.0
1794241570.943,proposed,197377,e_3_0,1,0.0,0.0,0.0
1794241570.943,proposed,197377,e_3_0,4,0.0,0.0,0.0
1794241570.943,proposed,197377,e_3_0,2,0.0,0.0,0.0
1794241570.943,proposed,197377,e_3_0,3,0.0,0.0,0.0
1794241570.946,proposed,131841,a_3_1,1,0.0,0.0,0.0
1794241570.946,proposed,131841,a_3_1,4,0.0,0.0,0.0
1794241570.946,proposed,131841,a_3_1,2,0.0,0.0,0.0
1794241570.946,proposed,131841,a_3_1,3,0.0,0.0,0.0
1794241570.949,proposed,131840,a_3_0,4,0.0,0.0,0.0
1794241570.949,proposed,131840,a_3_0,1,0.0,0.0,0.0
1794241570.949,proposed,131840,a_3_0,2,0.0,0.0,0.0
1794241570.949,proposed,131840,a_3_0,3,0.0,0.0,0.0
1794241570.953,proposed,196610,e_0_1,4,0.0,0.0,0.0
1794241570.953,proposed,196610,e_0_1,1,0.0,0.0,0.0
1794241570.953,proposed,196610,e_0_1,2,0.0,0.0,0.0
1794241570.953,proposed,196610,e_0_1,3,0.0,0.0,0.0
1794241570.955,proposed,65539,c_1_1,4,0.0,0.0,0.0
1794241570.955,proposed,65539,c_1_1,1,0.0,0.0,0.0
1794241570.955,proposed,65539,c_1_1,2,0.0,0.0,0.0
1794241570.955,proposed,65539,c_1_1,3,0.0,0.0,0.0
1794241570.957,proposed,65536,c_0_0,4,0.0,0.0,0.0
1794241570.957,proposed,65536,c_0_0,1,0.0,0.0,0.0
1794241570.957,proposed,65536,c_0_0,2,0.0,0.0,0.0
1794241570.957,proposed,65536,c_0_0,3,0.0,0.0,0.0
1794241570.96,proposed,131072,a_0_0,4,0.0,0.0,0.0
1794241570.96,proposed,131072,a_0_0,1,0.0,0.0,0.0
1794241570.96,proposed,131072,a_0_0,2,0.0,0.0,0.0
1794241570.96,proposed,131072,a_0_0,3,0.0,0.0,0.0
1794241570.962,proposed,196609,e_0_0,1,0.0,0.0,0.0
1794241570.962,proposed,196609,e_0_0,4,0.0,0.0,0.0
1794241570.962,proposed,196609,e_0_0,2,0.0,0.0,0.0
1794241570.962,proposed,196609,e_0_0,3,0.0,0.0,0.0
1794241571.902,proposed,65537,c_0_1,4,0.0,0.0,0.0
1794241571.902,proposed,65537,c_0_1,1,0.0,0.0,0.0
1794241571.902,proposed,65537,c_0_1,2,0.0,0.0,0.0
1794241571.902,proposed,65537,c_0_1,3,0.0,0.0,0.0
1794241571.908,proposed,131073,a_0_1,1,0.0,0.0,0.0
1794241571.908,proposed,131073,a_0_1,4,0.0,0.0,0.0
1794241571.908,proposed,131073,a_0_1,2,0.0,0.0,0.0
1794241571.908,proposed,131073,a_0_1,3,0.0,0.0,0.0
1794241571.911,proposed,65538,c_1_0,4,0.0,0.0,0.0
1794241571.911,proposed,65538,c_1_0,1,0.0,0.0,0.0
1794241571.911,proposed,65538,c_1_0,2,0.0,0.0,0.0
1794241571.911,proposed,65538,c_1_0,3,0.0,0.0,0.0
1794241571.914,proposed,131585,a_2_1,1,0.0,0.0,0.0
1794241571.914,proposed,131585,a_2_1,4,0.0,0.0,0.0
1794241571.914,proposed,131585,a_2_1,2,0.0,0.0,0.0
1794241571.914,proposed,131585,a_2_1,3,0.0,0.0,0.0
1794241571.922,proposed,131840,a_3_0,4,0.0,0.0,0.0
1794241571.922,proposed,131840,a_3_0,1,0.0,0.0,0.0
1794241571.922,proposed,131840,a_3_0,2,0.0,0.0,0.0
1794241571.922,proposed,131840,a_3_0,3,0.0,0.0,0.0
1794241571.924,proposed,131072,a_0_0,4,0.0,0.0,0.0
1794241571.924,proposed,131072,a_0_0,1,0.0,0.0,0.0
1794241571.924,proposed,131072,a_0_0,2,0.0,0.0,0.0
1794241571.924,proposed,131072,a_0_0,3,0.0,0.0,0.0
1794241571.927,proposed,197377,e_3_0,1,0.0,0.0,0.0
1794241571.927,proposed,197377,e_3_0,4,0.0,0.0,0.0
1794241571.927,proposed,197377,e_3_0,2,0.0,0.0,0.0
1794241571.927,proposed,197377,e_3_0,3,0.0,0.0,0.0
1794241571.929,proposed,196866,e_1_1,4,0.0,0.0,0.0
1794241571.929,proposed,196866,e_1_1,1,0.0,0.0,0.0
1794241571.929,proposed,196866,e_1_1,2,0.0,0.0,0.0
1794241571.929,proposed,196866,e_1_1,3,0.0,0.0,0.0
1794241571.934,proposed,196865,e_1_0,4,0.0,0.0,0.0
1794241571.934,proposed,196865,e_1_0,1,0.0,0.0,0.0
1794241571.934,proposed,196865,e_1_0,2,0.0,0.0,0.0
1794241571.934,proposed,196865,e_1_0,3,0.0,0.0,0.0
1794241571.937,proposed,65536,c_0_0,4,0.0,0.0,0.0
1794241571.937,proposed,65536,c_0_0,1,0.0,0.0,0.0
1794241571.937,proposed,65536,c_0_0,2,0.0,0.0,0.0
1794241571.937,proposed,65536,c_0_0,3,0.0,0.0,0.0
1794241571.94,proposed,196610,e_0_1,4,0.0,0.0,0.0
1794241571.94,proposed,196610,e_0_1,1,0.0,0.0,0.0
1794241571.94,proposed,196610,e_0_1,2,0.0,0.0,0.0
1794241571.94,proposed,196610,e_0_1,3,0.0,0.0,0.0
1794241571.944,proposed,131841,a_3_1,1,0.0,0.0,0.0
1794241571.944,proposed,131841,a_3_1,4,0.0,0.0,0.0
1794241571.944,proposed,131841,a_3_1,2,0.0,0.0,0.0
1794241571.944,proposed,131841,a_3_1,3,0.0,0.0,0.0
1794241571.947,proposed,131328,a_1_0,1,0.0,0.0,0.0
1794241571.947,proposed,131328,a_1_0,4,0.0,0.0,0.0
1794241571.947,proposed,131328,a_1_0,2,0.0,0.0,0.0
1794241571.947,proposed,131328,a_1_0,3,0.0,0.0,0.0
1794241571.95,proposed,197378,e_3_1,4,0.0,0.0,0.0
1794241571.95,proposed,197378,e_3_1,1,0.0,0.0,0.0
1794241571.95,proposed,197378,e_3_1,2,0.0,0.0,0.0
1794241571.95,proposed,197378,e_3_1,3,0.0,0.0,0.0
1794241571.953,proposed,131329,a_1_1,1,0.0,0.0,0.0
1794241571.953,proposed,131329,a_1_1,4,0.0,0.0,0.0
1794241571.953,proposed,131329,a_1_1,2,0.0,0.0,0.0
1794241571.953,proposed,131329,a_1_1,3,0.0,0.0,0.0
1794241571.956,proposed,131584,a_2_0,4,0.0,0.0,0.0
1794241571.956,proposed,131584,a_2_0,1,0.0,0.0,0.0
1794241571.956,proposed,131584,a_2_0,2,0.0,0.0,0.0
1794241571.956,proposed,131584,a_2_0,3,0.0,0.0,0.0
1794241571.957,proposed,196609,e_0_0,1,0.0,0.0,0.0
1794241571.957,proposed,196609,e_0_0,4,0.0,0.0,0.0
1794241571.957,proposed,196609,e_0_0,2,0.0,0.0,0.0
1794241571.957,proposed,196609,e_0_0,3,0.0,0.0,0.0
1794241571.96,proposed,197121,e_2_0,4,0.0,0.0,0.0
1794241571.96,proposed,197121,e_2_0,1,0.0,0.0,0.0
1794241571.96,proposed,197121,e_2_0,2,0.0,0.0,0.0
1794241571.96,proposed,197121,e_2_0,3,0.0,0.0,0.0
1794241571.962,proposed,197122,e_2_1,4,0.0,0.0,0.0
1794241571.962,proposed,197122,e_2_1,1,0.0,0.0,0.0
1794241571.962,proposed,197122,e_2_1,2,0.0,0.0,0.0
1794241571.962,proposed,197122,e_2_1,3,0.0,0.0,0.0
1794241571.965,proposed,65539,c_1_1,4,0.0,0.0,0.0
1794241571.965,proposed,65539,c_1_1,1,0.0,0.0,0.0
1794241571.965,proposed,65539,c_1_1,2,0.0,0.0,0.0
1794241571.965,proposed,65539,c_1_1,3,0.0,0.0,0.0
1794241572.91,proposed,131328,a_1_0,1,0.0,0.0,0.0
1794241572.91,proposed,131328,a_1_0,4,0.0,0.0,0.0
1794241572.91,proposed,131328,a_1_0,2,0.0,0.0,0.0
1794241572.91,proposed,131328,a_1_0,3,0.0,0.0,0.0
1794241572.912,proposed,131841,a_3_1,1,0.0,0.0,0.0
1794241572.912,proposed,131841,a_3_1,4,0.0,0.0,0.0
1794241572.912,proposed,131841,a_3_1,2,0.0,0.0,0.0
1794241572.912,proposed,131841,a_3_1,3,0.0,0.0,0.0
1794241572.914,proposed,131073,a_0_1,1,0.0,0.0,0.0
1794241572.914,proposed,131073,a_0_1,4,0.0,0.0,0.0
1794241572.914,proposed,131073,a_0_1,2,0.0,0.0,0.0
1794241572.914,proposed,131073,a_0_1,3,0.0,0.0,0.0
1794241572.917,proposed,196609,e_0_0,1,0.0,0.0,0.0
1794241572.917,proposed,196609,e_0_0,4,0.0,0.0,0.0
1794241572.917,proposed,196609,e_0_0,2,0.0,0.0,0.0
1794241572.917,proposed,196609,e_0_0,3,0.0,0.0,0.0
1794241572.919,proposed,65536,c_0_0,4,0.0,0.0,0.0
1794241572.919,proposed,65536,c_0_0,1,0.0,0.0,0.0
1794241572.919,proposed,65536,c_0_0,2,0.0,0.0,0.0
1794241572.919,proposed,65536,c_0_0,3,0.0,0.0,0.0
1794241572.922,proposed,131072,a_0_0,4,0.0,0.0,0.0
1794241572.922,proposed,131072,a_0_0,1,0.0,0.0,0.0
1794241572.922,proposed,131072,a_0_0,2,0.0,0.0,0.0
1794241572.922,proposed,131072,a_0_0,3,0.0,0.0,0.0
1794241572.924,proposed,131329,a_1_1,1,0.0,0.0,0.0
1794241572.924,proposed,131329,a_1_1,4,0.0,0.0,0.0
1794241572.924,proposed,131329,a_1_1,2,0.0,0.0,0.0
1794241572.924,proposed,131329,a_1_1,3,0.0,0.0,0.0
1794241572.927,proposed,196865,e_1_0,4,0.0,0.0,0.0
1794241572.927,proposed,196865,e_1_0,1,0.0,0.0,0.0
1794241572.927,proposed,196865,e_1_0,2,0.0,0.0,0.0
1794241572.927,proposed,196865,e_1_0,3,0.0,0.0,0.0
1794241572.931,proposed,197121,e_2_0,4,0.0,0.0,0.0
1794241572.931,proposed,197121,e_2_0,1,0.0,0.0,0.0
1794241572.931,proposed,197121,e_2_0,2,0.0,0.0,0.0
1794241572.931,proposed,197121,e_2_0,3,0.0,0.0,0.0
1794241572.934,proposed,131840,a_3_0,4,0.0,0.0,0.0
1794241572.934,proposed,131840,a_3_0,1,0.0,0.0,0.0
1794241572.934,proposed,131840,a_3_0,2,0.0,0.0,0.0
1794241572.934,proposed,131840,a_3_0,3,0.0,0.0,0.0
1794241572.936,proposed,197122,e_2_1,4,0.0,0.0,0.0
1794241572.936,proposed,197122,e_2_1,1,0.0,0.0,0.0
1794241572.936,proposed,197122,e_2_1,2,0.0,0.0,0.0
1794241572.936,proposed,197122,e_2_1,3,0.0,0.0,0.0
1794241572.939,proposed,65538,c_1_0,4,0.0,0.0,0.0
1794241572.939,proposed,65538,c_1_0,1,0.0,0.0,0.0
1794241572.939,proposed,65538,c_1_0,2,0.0,0.0,0.0
1794241572.939,proposed,65538,c_1_0,3,0.0,0.0,0.0
1794241572.941,proposed,65539,c_1_1,4,0.0,0.0,0.0
1794241572.941,proposed,65539,c_1_1,1,0.0,0.0,0.0
1794241572.941,proposed,65539,c_1_1,2,0.0,0.0,0.0
1794241572.941,proposed,65539,c_1_1,3,0.0,0.0,0.0
1794241572.944,proposed,196610,e_0_1,4,0.0,0.0,0.0
1794241572.944,proposed,196610,e_0_1,1,0.0,0.0,0.0
1794241572.944,proposed,196610,e_0_1,2,0.0,0.0,0.0
1794241572.944,proposed,196610,e_0_1,3,0.0,0.0,0.0
1794241572.946,proposed,197378,e_3_1,4,0.0,0.0,0.0
1794241572.946,proposed,197378,e_3_1,1,0.0,0.0,0.0
1794241572.946,proposed,197378,e_3_1,2,0.0,0.0,0.0
1794241572.946,proposed,197378,e_3_1,3,0.0,0.0,0.0
1794241572.949,proposed,197377,e_3_0,1,0.0,0.0,0.0
1794241572.949,proposed,197377,e_3_0,4,0.0,0.0,0.0
1794241572.949,proposed,197377,e_3_0,2,0.0,0.0,0.0
1794241572.949,proposed,197377,e_3_0,3,0.0,0.0,0.0
1794241572.952,proposed,65537,c_0_1,4,0.0,0.0,0.0
1794241572.952,proposed,65537,c_0_1,1,0.0,0.0,0.0
1794241572.952,proposed,65537,c_0_1,2,0.0,0.0,0.0
1794241572.952,proposed,65537,c_0_1,3,0.0,0.0,0.0
1794241572.955,proposed,196866,e_1_1,4,0.0,0.0,0.0
1794241572.955,proposed,196866,e_1_1,1,0.0,0.0,0.0
1794241572.955,proposed,196866,e_1_1,2,0.0,0.0,0.0
1794241572.955,proposed,196866,e_1_1,3,0.0,0.0,0.0
1794241572.958,proposed,131584,a_2_0,4,0.0,0.0,0.0
1794241572.958,proposed,131584,a_2_0,1,0.0,0.0,0.0
1794241572.958,proposed,131584,a_2_0,2,0.0,0.0,0.0
1794241572.958,proposed,131584,a_2_0,3,0.0,0.0,0.0
1794241572.961,proposed,131585,a_2_1,1,0.0,0.0,0.0
1794241572.961,proposed,131585,a_2_1,4,0.0,0.0,0.0
1794241572.961,proposed,131585,a_2_1,2,0.0,0.0,0.0
1794241572.961,proposed,131585,a_2_1,3,0.0,0.0,0.0
1794241574.33,proposed,197122,e_2_1,4,0.0,0.0,0.0
1794241574.33,proposed,197122,e_2_1,1,0.0,0.0,0.0
1794241574.33,proposed,197122,e_2_1,2,0.0,0.0,0.0
1794241574.33,proposed,197122,e_2_1,3,0.0,0.0,0.0
1794241574.34,proposed,196865,e_1_0,4,0.0,0.0,0.0
1794241574.34,proposed,196865,e_1_0,1,0.0,0.0,0.0
1794241574.34,proposed,196865,e_1_0,2,0.0,0.0,0.0
1794241574.34,proposed,196865,e_1_0,3,0.0,0.0,0.0
1794241574.372,proposed,197377,e_3_0,1,0.0,0.0,0.0
1794241574.372,proposed,197377,e_3_0,4,0.0,0.0,0.0
1794241574.372,proposed,197377,e_3_0,2,0.0,0.0,0.0
1794241574.372,proposed,197377,e_3_0,3,0.0,0.0,0.0
1794241574.38,proposed,131073,a_0_1,2,0.0,0.0,0.0
1794241574.389,proposed,131841,a_3_1,1,0.0,0.0,0.0
1794241574.389,proposed,131841,a_3_1,2,0.0,0.0,0.0
1794241574.399,proposed,197121,e_2_0,4,0.0,0.0,0.0
1794241574.399,proposed,197121,e_2_0,1,0.0,0.0,0.0
1794241574.399,proposed,197121,e_2_0,2,0.0,0.0,0.0
1794241574.399,proposed,197121,e_2_0,3,0.0,0.0,0.0
1794241574.425,proposed,131585,a_2_1,1,0.0,0.0,0.0
1794241574.425,proposed,131585,a_2_1,2,0.0,0.0,0.0
1794241574.429,proposed,196609,e_0_0,4,0.0,0.0,0.0
1794241574.429,proposed,196609,e_0_0,3,0.0,0.0,0.0
1794241574.447,proposed,131328,a_1_0,1,0.0,0.0,0.0
1794241574.447,proposed,131328,a_1_0,2,0.0,0.0,0.0
1794241574.453,proposed,196610,e_0_1,4,0.0,0.0,0.0
1794241574.453,proposed,196610,e_0_1,1,97791112433737.4,97791112433737.4,9779111243373.74
1794241574.453,proposed,196610,e_0_1,2,0.0,0.0,0.0
1794241574.453,proposed,196610,e_0_1,3,0.0,0.0,0.0
1794241574.456,proposed,131329,a_1_1,1,0.0,0.0,0.0
1794241574.456,proposed,131329,a_1_1,2,0.0,0.0,0.0
1794241574.467,proposed,131072,a_0_0,2,95503469558019.72,95503469558019.72,9550346955801.973
1794241574.47,proposed,196866,e_1_1,4,0.0,0.0,0.0
1794241574.47,proposed,196866,e_1_1,1,0.0,0.0,0.0
1794241574.47,proposed,196866,e_1_1,2,0.0,0.0,0.0
1794241574.47,proposed,196866,e_1_1,3,0.0,0.0,0.0
1794241574.499,proposed,131584,a_2_0,1,0.0,0.0,0.0
1794241574.499,proposed,131584,a_2_0,2,0.0,0.0,0.0
1794241574.5,proposed,197378,e_3_1,4,0.0,0.0,0.0
1794241574.5,proposed,197378,e_3_1,1,0.0,0.0,0.0
1794241574.5,proposed,197378,e_3_1,2,0.0,0.0,0.0
1794241574.5,proposed,197378,e_3_1,3,0.0,0.0,0.0
1794241574.503,proposed,131840,a_3_0,1,0.0,0.0,0.0
1794241574.503,proposed,131840,a_3_0,2,0.0,0.0,0.0

```

# run_experiments.py

```py
# run_experiments.py
# Runs our 3 experiments one after another and then makes the plots.
#   Experiment 1: static   (every flow on the first path)
#   Experiment 2: ecmp     (hash-based path choice)
#   Experiment 3: proposed (ecmp + elephant rerouting)
# For each one we start the Ryu controller with the right SDN_SCHEME,
# run traffic_gen.py (it starts Mininet), then stop the controller.
#
# Run:  sudo python3 run_experiments.py
#       sudo python3 run_experiments.py --duration 40 --stagger 5

import argparse
import os
import shutil
import subprocess
import sys
import time

import config

SCHEMES = ['static', 'ecmp', 'proposed']
RESULT_FILES = ['mice_results.csv', 'elephant_results.csv', 'core_load.csv',
                'events.csv', 'port_stats.csv', 'flow_stats.csv']


def clean_mininet():
    subprocess.run(['mn', '-c'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def start_controller(ryu, scheme):
    env = os.environ.copy()
    env['SDN_SCHEME'] = scheme
    log = open(os.path.join(config.RESULTS_DIR, 'controller_%s.log' % scheme), 'w')
    proc = subprocess.Popen([ryu, 'main_controller.py',
                             '--ofp-tcp-listen-port', str(config.CONTROLLER_PORT)],
                            env=env, stdout=log, stderr=subprocess.STDOUT)
    time.sleep(3)  # give Ryu time to start listening
    if proc.poll() is not None:
        log.close()
        sys.exit('controller crashed, see results/controller_%s.log' % scheme)
    return proc, log


def stop_controller(proc, log):
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
    log.close()


def run_traffic(scheme, args):
    cmd = [sys.executable, 'traffic_gen.py', 'run',
           '--scenario', args.scenario,
           '--scheme', scheme,
           '--duration', str(args.duration),
           '--stagger', str(args.stagger),
           '--seed', str(args.seed)]   # same seed = same traffic for all schemes
    subprocess.run(cmd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', default='collision',
                        choices=['collision', 'mixed', 'mice', 'elephant'])
    parser.add_argument('--duration', type=float, default=30)
    parser.add_argument('--stagger', type=float, default=0,
                        help='seconds between the two collision flows')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--schemes', nargs='+', default=SCHEMES, choices=SCHEMES)
    parser.add_argument('--keep', action='store_true', help='keep old results')
    parser.add_argument('--ryu', default=shutil.which('ryu-manager') or 'ryu-manager')
    args = parser.parse_args()

    if os.geteuid() != 0:
        sys.exit('please run with sudo (Mininet needs root)')

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    if not args.keep:
        for name in RESULT_FILES:
            path = os.path.join(config.RESULTS_DIR, name)
            if os.path.exists(path):
                os.remove(path)

    for i, scheme in enumerate(args.schemes, 1):
        print('\n' + '=' * 60)
        print('Experiment %d: %s' % (i, scheme))
        print('=' * 60)
        clean_mininet()
        proc, log = start_controller(args.ryu, scheme)
        try:
            run_traffic(scheme, args)
        finally:
            stop_controller(proc, log)
            clean_mininet()
        time.sleep(2)

    print('\nMaking plots...')
    subprocess.run([sys.executable, 'plotter.py', '--results-dir', config.RESULTS_DIR])

    # quick check for the report: how fast did we reroute?
    events = os.path.join(config.RESULTS_DIR, 'events.csv')
    if os.path.exists(events):
        print('\nController events (proposed):')
        with open(events) as f:
            for line in f:
                if 'proposed' in line:
                    print('  ' + line.strip())


if __name__ == '__main__':
    main()

```

# test\test_classifier.py

```py
"""
Task 3.4 — feed dummy byte-counter dictionaries into classifier.py.
Run: python3 -m unittest discover -s tests -v
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classifier import (ELEPHANT, EXPIRED, MICE, FiveTuple,  # noqa: E402
                        FlowClassifier, RateTracker)

ELE = FiveTuple('10.0.0.2', '10.2.0.2', 40000, 5001, 6)   # H1 -> H9 iperf
MOUSE = FiveTuple('10.1.0.2', '10.3.1.3', 41000, 6000, 6)  # 32 KB request


def mbps_bytes(mbps, seconds=1.0):
    """Bytes a flow sends in `seconds` at `mbps`."""
    return int(mbps * 1e6 / 8 * seconds)


class ClassifierTest(unittest.TestCase):
    def setUp(self):
        # 10 Mbps links, 10% threshold -> 1 Mbps; N = 2 consecutive samples
        self.clf = FlowClassifier(link_bandwidth=10.0, threshold=0.1, consecutive_samples=2,
                                  demotion_samples=2, expiry=5.0)

    def feed(self, per_second_mbps, flow=ELE, start=0.0):
        """Feed cumulative counters for a flow whose rate in each 1 s interval is given."""
        total, transitions = 0, []
        transitions += self.clf.update({flow: 0}, start)
        for i, mbps in enumerate(per_second_mbps, 1):
            total += mbps_bytes(mbps)
            transitions += self.clf.update({flow: total}, start + i)
        return transitions

    def test_rate_formula(self):
        rt = RateTracker()
        self.assertIsNone(rt.update('f', 0, 0.0))
        self.assertAlmostEqual(rt.update('f', 1250000, 1.0), 10.0)   # 1.25 MB in 1 s
        self.assertAlmostEqual(rt.update('f', 1500000, 1.5), 4.0)    # 250 KB in 0.5 s

    def test_small_burst_stays_mice(self):
        self.clf.update({MOUSE: 0}, 0.0)
        self.clf.update({MOUSE: 32 * 1024}, 1.0)   # one 32 KB burst = 0.26 Mbps
        self.clf.update({MOUSE: 32 * 1024}, 2.0)
        self.assertEqual(self.clf.classify(MOUSE), MICE)
        self.assertEqual(self.clf.get_active_elephants(), {})

    def test_single_spike_is_not_promoted(self):
        transitions = self.feed([8.0, 0.2, 8.0, 0.2])   # never 2 high samples in a row
        self.assertEqual(self.clf.classify(ELE), MICE)
        self.assertEqual(transitions, [])

    def test_sustained_rate_becomes_elephant_after_n_samples(self):
        self.clf.update({ELE: 0}, 0.0)
        self.clf.update({ELE: mbps_bytes(8)}, 1.0)
        self.assertEqual(self.clf.classify(ELE), MICE, 'one sample must not be enough')
        t = self.clf.update({ELE: mbps_bytes(8) * 2}, 2.0)
        self.assertEqual(self.clf.classify(ELE), ELEPHANT)
        self.assertEqual(t, [(ELE, MICE, ELEPHANT)])

    def test_threshold_is_inclusive(self):
        self.feed([1.0, 1.0])   # exactly 10% of 10 Mbps
        self.assertEqual(self.clf.classify(ELE), ELEPHANT)
        self.clf.reset()
        self.feed([0.99, 0.99])
        self.assertEqual(self.clf.classify(ELE), MICE)

    def test_get_active_elephants_returns_tuples_and_demand(self):
        self.feed([9.0, 9.0, 9.0])
        elephants = self.clf.get_active_elephants()
        self.assertEqual(list(elephants), [ELE])
        self.assertIsInstance(next(iter(elephants)), FiveTuple)
        self.assertAlmostEqual(elephants[ELE], 9.0, places=3)

    def test_mixed_traffic(self):
        ts, e_total, m_total = 0.0, 0, 0
        self.clf.update({ELE: 0, MOUSE: 0}, ts)
        for i in range(1, 5):
            e_total += mbps_bytes(6)
            m_total += 16 * 1024 if i == 1 else 0
            self.clf.update({ELE: e_total, MOUSE: m_total}, float(i))
        self.assertEqual(self.clf.classify(ELE), ELEPHANT)
        self.assertEqual(self.clf.classify(MOUSE), MICE)
        self.assertEqual(self.clf.counts(), {ELEPHANT: 1, MICE: 1})

    def test_demotion_back_to_mice(self):
        t = self.feed([8, 8, 0.1, 0.1])
        self.assertEqual(self.clf.classify(ELE), MICE)
        self.assertIn((ELE, ELEPHANT, MICE), t)

    def test_counter_reset_does_not_produce_negative_rate(self):
        self.feed([8, 8])
        self.clf.update({ELE: 100}, 3.0)   # rule re-installed, counter restarts
        self.assertGreaterEqual(self.clf.get_rate(ELE), 0.0)

    def test_duplicate_timestamp_ignored(self):
        self.clf.update({ELE: 0}, 0.0)
        self.clf.update({ELE: mbps_bytes(8)}, 0.0)
        self.assertEqual(self.clf.flow_history[ELE], (0, 0.0))

    def test_expiry_reports_finished_elephant(self):
        self.feed([8, 8])
        t = self.clf.update({MOUSE: 0}, 10.0)   # ELE not reported for 8 s
        self.assertIn((ELE, ELEPHANT, EXPIRED), t)
        self.assertIsNone(self.clf.classify(ELE))
        self.assertNotIn(ELE, self.clf.flow_history)

    def test_uses_config_defaults(self):
        clf = FlowClassifier()
        self.assertGreater(clf.rate_threshold_mbps, 0)
        self.assertGreaterEqual(clf.consecutive_samples, 1)


if __name__ == '__main__':
    unittest.main()

```

# test\test_flow_monitor.py

```py
"""
flow_monitor.py tests using fake OpenFlow replies. Uses real Ryu if installed,
otherwise minimal stubs so the parsing / rate / classification path can be tested
on any machine. Run: python3 -m unittest discover -s tests -v
"""
import os
import sys
import tempfile
import types
import unittest
from collections import namedtuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _install_ryu_stubs():
    def mod(name, **attrs):
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        sys.modules[name] = m
        return m

    class RyuApp(object):
        def __init__(self, *a, **kw):
            import logging
            self.logger = logging.getLogger('stub')
            self.sent_events = []

        def send_event_to_observers(self, ev, state=None):
            self.sent_events.append(ev)

    class EventBase(object):
        def __init__(self):
            pass

    mod('ryu')
    mod('ryu.base')
    mod('ryu.base.app_manager', RyuApp=RyuApp)
    mod('ryu.controller')
    mod('ryu.controller.event', EventBase=EventBase)
    mod('ryu.controller.ofp_event', EventOFPStateChange=object, EventOFPFlowStatsReply=object,
        EventOFPPortStatsReply=object)
    mod('ryu.controller.handler', MAIN_DISPATCHER='main', DEAD_DISPATCHER='dead',
        set_ev_cls=lambda *a, **k: (lambda f: f))
    mod('ryu.lib')
    mod('ryu.lib.hub', spawn=lambda *a, **k: None, sleep=lambda s: None)
    mod('ryu.ofproto')
    mod('ryu.ofproto.ofproto_v1_3', OFP_VERSION=4, OFPMPF_REPLY_MORE=1,
        OFPP_MAX=0xffffff00, OFPP_ANY=0xffffffff)
    for parent, child in (('ryu', 'base'), ('ryu', 'controller'), ('ryu', 'lib'), ('ryu', 'ofproto')):
        setattr(sys.modules[parent], child, sys.modules['%s.%s' % (parent, child)])
    sys.modules['ryu.base'].app_manager = sys.modules['ryu.base.app_manager']
    sys.modules['ryu.controller'].event = sys.modules['ryu.controller.event']
    sys.modules['ryu.controller'].ofp_event = sys.modules['ryu.controller.ofp_event']
    sys.modules['ryu.lib'].hub = sys.modules['ryu.lib.hub']
    sys.modules['ryu.ofproto'].ofproto_v1_3 = sys.modules['ryu.ofproto.ofproto_v1_3']


try:
    import ryu.base.app_manager  # noqa: F401
    REAL_RYU = True
except Exception:
    _install_ryu_stubs()
    REAL_RYU = False

import config  # noqa: E402
config.MONITOR_LOG_CSV = False
import flow_monitor as fm  # noqa: E402
from classifier import ELEPHANT, MICE, FiveTuple  # noqa: E402

FlowStat = namedtuple('FlowStat', 'priority match byte_count packet_count duration_sec duration_nsec')
PortStat = namedtuple('PortStat', 'port_no tx_bytes rx_bytes')

E_0_0 = 0x030001   # edge e_0_0 (H1, H2)
A_0_0 = 0x020000
C_1_0 = 0x010002


def tcp_match(src, dst, sport, dport):
    return {'eth_type': 0x0800, 'ipv4_src': src, 'ipv4_dst': dst, 'ip_proto': 6,
            'tcp_src': sport, 'tcp_dst': dport}


@unittest.skipIf(REAL_RYU, 'stub-based test; real Ryu present (constructor spawns threads)')
class FlowMonitorTest(unittest.TestCase):
    def setUp(self):
        self.mon = fm.FlowMonitor()
        self.mon.datapaths = {E_0_0: object(), 0x030101: object(), A_0_0: object(), C_1_0: object()}

    def test_dpid_helpers_match_fattree_topo(self):
        self.assertEqual(fm.dpid_to_name(E_0_0), 'e_0_0')
        self.assertEqual(fm.dpid_to_name(0x030302), 'e_3_1')
        self.assertEqual(fm.dpid_to_name(0x020301), 'a_3_1')
        self.assertEqual(fm.dpid_to_name(C_1_0), 'c_1_0')
        self.assertEqual(fm.ip_to_ingress_edge_dpid('10.0.0.2'), E_0_0)
        self.assertEqual(fm.ip_to_ingress_edge_dpid('10.1.0.3'), 0x030101)

    def test_extract_five_tuple(self):
        self.assertEqual(fm.extract_five_tuple(tcp_match('10.0.0.2', '10.2.0.2', 40000, 5001)),
                         FiveTuple('10.0.0.2', '10.2.0.2', 40000, 5001, 6))
        udp = {'ipv4_src': '10.0.0.2', 'ipv4_dst': '10.1.0.2', 'ip_proto': 17, 'udp_src': 1, 'udp_dst': 53}
        self.assertEqual(fm.extract_five_tuple(udp).dst_port, 53)
        self.assertIsNone(fm.extract_five_tuple({'eth_dst': '00:00:00:00:00:01'}))

    def test_flow_replies_promote_elephant_and_skip_mice(self):
        ele = tcp_match('10.0.0.2', '10.2.0.2', 40000, 5001)
        mouse = tcp_match('10.0.0.3', '10.3.0.2', 41000, 6000)
        miss = FlowStat(0, {}, 999999, 1, 0, 0)
        for i, (eb, mb) in enumerate([(0, 0), (1_000_000, 40_000), (2_000_000, 40_000)]):
            self.mon.process_flow_stats(E_0_0, [miss, FlowStat(10, ele, eb, 1, i, 0),
                                                FlowStat(10, mouse, mb, 1, i, 0)], float(i))
        elephants = self.mon.get_active_elephants()
        key = FiveTuple('10.0.0.2', '10.2.0.2', 40000, 5001, 6)
        self.assertEqual(list(elephants), [key])
        self.assertAlmostEqual(elephants[key], 8.0)
        self.assertEqual(self.mon.flow_stats[FiveTuple('10.0.0.3', '10.3.0.2', 41000, 6000, 6)]['label'], MICE)
        changes = [e for e in self.mon.sent_events if isinstance(e, fm.EventFlowClassChanged)]
        self.assertEqual([(e.old, e.new) for e in changes], [(MICE, ELEPHANT)])

    def test_flow_counted_only_at_ingress_edge(self):
        ele = tcp_match('10.0.0.2', '10.2.0.2', 40000, 5001)
        for i in range(3):   # same flow seen at agg switch: must be ignored
            self.mon.process_flow_stats(A_0_0, [FlowStat(10, ele, i * 1_000_000, 1, i, 0)], float(i))
        self.assertEqual(len(self.mon.classifier), 0)

    def test_port_stats_utilization_and_core_load(self):
        self.mon.process_port_stats(C_1_0, [PortStat(1, 0, 0), PortStat(2, 0, 0),
                                            PortStat(0xfffffffe, 0, 0)], 0.0)
        self.mon.process_port_stats(C_1_0, [PortStat(1, 625_000, 0), PortStat(2, 312_500, 625_000),
                                            PortStat(0xfffffffe, 9, 9)], 1.0)
        util = self.mon.get_port_utilization(C_1_0)
        self.assertEqual(sorted(p for _, p in util), [1, 2])
        self.assertAlmostEqual(util[(C_1_0, 1)]['tx_mbps'], 5.0)
        self.assertAlmostEqual(util[(C_1_0, 2)]['utilization'], 0.5)
        self.assertAlmostEqual(self.mon.get_core_load()['c_1_0'], 7.5)

    def test_csv_logging(self):
        tmp = tempfile.mkdtemp()
        self.mon._open_logs(tmp)
        self.mon.log_event('reroute H1->H9 via c_1_0', 123.0)
        self.mon.close_logs()
        with open(os.path.join(tmp, 'events.csv')) as fh:
            self.assertIn('reroute H1->H9 via c_1_0', fh.read())


if __name__ == '__main__':
    unittest.main()

```

# test\test_main_controller.py

```py
# test_main_controller.py
# Offline test of main_controller.py with fake switches (no Mininet needed).
# Needs Ryu, or os-ken (pip install os-ken) which has the same API.
# Run: python3 -m unittest discover -s test -v
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# test_flow_monitor.py may have put fake ryu modules in sys.modules: remove them
if 'ryu' in sys.modules and getattr(sys.modules['ryu'], '__file__', None) is None:
    for name in list(sys.modules):
        if name == 'ryu' or name.startswith('ryu.') or name in ('flow_monitor', 'main_controller'):
            del sys.modules[name]

try:
    import ryu.base.app_manager  # noqa: F401
    HAVE_RYU = True
except Exception:
    try:  # os-ken is a fork of Ryu with the same API
        import importlib
        import pkgutil
        import os_ken
        sys.modules['ryu'] = os_ken
        for m in pkgutil.walk_packages(os_ken.__path__, 'os_ken.'):
            if m.name.split('.')[1] in ('base', 'controller', 'lib', 'ofproto') and 'test' not in m.name:
                try:
                    sys.modules['ryu' + m.name[6:]] = importlib.import_module(m.name)
                except Exception:
                    pass
        am = sys.modules['ryu.base.app_manager']
        am.RyuApp = getattr(am, 'RyuApp', None) or am.OSKenApp
        HAVE_RYU = True
    except ImportError:
        HAVE_RYU = False

if HAVE_RYU:
    import config
    config.MONITOR_LOG_CSV = False
    from ryu.lib import hub
    hub.spawn = lambda *a, **k: None  # don't start the polling thread in tests
    from ryu.lib.packet import arp, ethernet, ipv4, packet, tcp
    from ryu.ofproto import ofproto_v1_3, ofproto_v1_3_parser
    from classifier import FiveTuple
    import main_controller


class FakeDatapath(object):
    def __init__(self, dpid):
        self.id = dpid
        self.ofproto = ofproto_v1_3
        self.ofproto_parser = ofproto_v1_3_parser
        self.sent = []

    def send_msg(self, msg):
        self.sent.append(msg)


class FakeMsg(object):
    def __init__(self, dp, in_port, data):
        self.datapath = dp
        self.match = {'in_port': in_port}
        self.data = data
        self.buffer_id = ofproto_v1_3.OFP_NO_BUFFER


class FakeEvent(object):
    def __init__(self, msg):
        self.msg = msg


class FakeStat(object):
    def __init__(self, match, byte_count, priority=10):
        self.match, self.byte_count, self.priority = match, byte_count, priority
        self.packet_count, self.duration_sec, self.duration_nsec = 1, 0, 0


def tcp_syn(src_ip, dst_ip, sport, dport):
    p = packet.Packet()
    p.add_protocol(ethernet.ethernet(ethertype=0x0800, src='00:00:00:00:00:02', dst='00:00:00:02:00:02'))
    p.add_protocol(ipv4.ipv4(src=src_ip, dst=dst_ip, proto=6))
    p.add_protocol(tcp.tcp(src_port=sport, dst_port=dport, bits=tcp.TCP_SYN))
    p.serialize()
    return p.data


@unittest.skipUnless(HAVE_RYU, 'needs ryu or os-ken')
class MainControllerTest(unittest.TestCase):
    def make(self, scheme):
        os.environ['SDN_SCHEME'] = scheme
        ctl = main_controller.MainController()
        for dpid in ctl.engine.dpid_to_name:
            ctl.datapaths[dpid] = FakeDatapath(dpid)
        return ctl

    def dp(self, ctl, name):
        return ctl.datapaths[ctl.engine.name_to_dpid[name]]

    def count_flowmods(self, ctl, priority):
        return sum(1 for dp in ctl.datapaths.values() for m in dp.sent
                   if isinstance(m, ofproto_v1_3_parser.OFPFlowMod) and m.priority == priority)

    def flowmods(self, ctl, priority):
        out = {}
        for dpid, dp in ctl.datapaths.items():
            for m in dp.sent:
                if isinstance(m, ofproto_v1_3_parser.OFPFlowMod) and m.priority == priority:
                    out[ctl.engine.dpid_to_name[dpid]] = m.instructions[0].actions[0].port
        return out

    def follow(self, ctl, start, ports):
        # walk the real topology using the installed output ports
        node, seen = start, [start]
        topo = ctl.engine.topo
        while node in ports:
            nxt = [b if a == node else a for a, b in topo.links()
                   if node in (a, b) and topo.port(node, b if a == node else a)[0] == ports[node]]
            node = nxt[0]
            seen.append(node)
        return seen

    def test_proxy_arp(self):
        ctl = self.make('ecmp')
        e = self.dp(ctl, 'e_0_0')
        req = packet.Packet()
        req.add_protocol(ethernet.ethernet(ethertype=0x0806, src='00:00:00:00:00:02', dst='ff:ff:ff:ff:ff:ff'))
        req.add_protocol(arp.arp(opcode=arp.ARP_REQUEST, src_mac='00:00:00:00:00:02', src_ip='10.0.0.2',
                                 dst_mac='00:00:00:00:00:00', dst_ip='10.2.0.2'))
        req.serialize()
        ctl.packet_in_handler(FakeEvent(FakeMsg(e, 3, req.data)))
        out = e.sent[-1]
        reply = packet.Packet(out.data).get_protocol(arp.arp)
        self.assertEqual(reply.opcode, arp.ARP_REPLY)
        self.assertEqual(reply.src_mac, '00:00:00:02:00:02')
        self.assertEqual(out.actions[0].port, 3)

    def test_static_path_reaches_destination(self):
        ctl = self.make('static')
        h1_port = ctl.engine.port_map[('e_0_0', 'h_0_0_0')]
        ctl.packet_in_handler(FakeEvent(FakeMsg(self.dp(ctl, 'e_0_0'), h1_port,
                                                tcp_syn('10.0.0.2', '10.2.0.2', 40000, 5001))))
        hops = self.follow(ctl, 'e_0_0', self.flowmods(ctl, 10))
        self.assertEqual(hops[-1], 'h_2_0_0')
        self.assertEqual(len(hops), 6)  # e, a, c, a, e, host

    def test_static_puts_collision_flows_on_same_core(self):
        ctl = self.make('static')
        f1 = FiveTuple('10.0.0.2', '10.2.0.2', 40000, 5001, 6)
        f2 = FiveTuple('10.0.0.3', '10.2.0.3', 40001, 5001, 6)
        self.assertEqual(ctl.pick_path(f1)[2], ctl.pick_path(f2)[2])

    def test_ecmp_spreads_flows(self):
        ctl = self.make('ecmp')
        cores = {ctl.pick_path(FiveTuple('10.0.0.2', '10.2.0.2', p, 5001, 6))[2] for p in range(40000, 40040)}
        self.assertEqual(len(cores), 4)

    def test_proposed_reroutes_colliding_elephants(self):
        ctl = self.make('proposed')
        m1 = {'ipv4_src': '10.0.0.2', 'ipv4_dst': '10.2.0.2', 'ip_proto': 6, 'tcp_src': 40000, 'tcp_dst': 5001}
        m2 = {'ipv4_src': '10.0.0.3', 'ipv4_dst': '10.2.0.3', 'ip_proto': 6, 'tcp_src': 40001, 'tcp_dst': 5001}
        e00 = ctl.engine.name_to_dpid['e_0_0']
        for t in range(3):  # both flows at ~5 Mbps (collision) for 3 polls
            ctl.process_flow_stats(e00, [FakeStat(m1, t * 625000), FakeStat(m2, t * 625000)], float(t))
        self.assertEqual(len(ctl.get_active_elephants()), 2)
        cores = [p['path'][2] for p in ctl.engine.active_reservations.values()]
        self.assertNotEqual(cores[0], cores[1], 'elephants must use different cores')
        self.assertEqual(self.count_flowmods(ctl, 100), 10)  # 2 flows x 5 switches (e,a,c,a,e)

    def test_static_never_reroutes(self):
        ctl = self.make('static')
        m1 = {'ipv4_src': '10.0.0.2', 'ipv4_dst': '10.2.0.2', 'ip_proto': 6, 'tcp_src': 40000, 'tcp_dst': 5001}
        e00 = ctl.engine.name_to_dpid['e_0_0']
        for t in range(3):
            ctl.process_flow_stats(e00, [FakeStat(m1, t * 1250000)], float(t))
        self.assertEqual(len(ctl.get_active_elephants()), 1)
        self.assertEqual(self.count_flowmods(ctl, 100), 0)


if __name__ == '__main__':
    unittest.main()

```

# test\test_traffic_gen_plotter.py

```py
"""
Task 2.4 — traffic_gen.py on loopback + CSV validity + plotter end-to-end.
No Mininet or root needed. Run: python3 -m unittest discover -s tests -v
"""
import os
import random
import shutil
import sys
import tempfile
import threading
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import plotter  # noqa: E402
import traffic_gen as tg  # noqa: E402

HOSTS = [tg.HostInfo('h_%d_%d_%d' % (p, e, h), '10.%d.%d.%d' % (p, e, h + 2), 'H%d' % (i + 1))
         for i, (p, e, h) in enumerate((p, e, h) for p in range(4) for e in range(2) for h in range(2))]

IPERF_SAMPLE = """\
20260108120000,10.0.0.2,45678,10.2.0.2,5001,3,0.0-0.5,589824,9437184
20260108120000,10.0.0.2,45678,10.2.0.2,5001,3,0.5-1.0,327680,5242880
20260108120001,10.0.0.2,45678,10.2.0.2,5001,3,1.0-1.5,589824,9437184
20260108120001,10.0.0.2,45678,10.2.0.2,5001,3,0.0-1.5,1507328,8039082
"""

DUMP_PORTS_SAMPLE = """\
OFPST_PORT reply (OF1.3) (xid=0x2): 5 ports
  port LOCAL: rx pkts=0, bytes=0, drop=0, errs=0, frame=0, over=0, crc=0
           tx pkts=0, bytes=0, drop=0, errs=0, coll=0
           duration=10.0s
  port  1: rx pkts=10, bytes=1000, drop=0, errs=0, frame=0, over=0, crc=0
           tx pkts=20, bytes=2000, drop=0, errs=0, coll=0
           duration=10.0s
  port  2: rx pkts=30, bytes=3000, drop=0, errs=0, frame=0, over=0, crc=0
           tx pkts=40, bytes=4000, drop=0, errs=0, coll=0
"""


class ScheduleTest(unittest.TestCase):
    def test_poisson_schedule(self):
        flows = tg.generate_mice_schedule(HOSTS, duration=200, rate=20, seed=1)
        self.assertEqual(flows, tg.generate_mice_schedule(HOSTS, 200, 20, seed=1), 'seeded = reproducible')
        self.assertTrue(all(tg.MICE_MIN_BYTES <= f['size_bytes'] <= tg.MICE_MAX_BYTES for f in flows))
        self.assertTrue(all(0 <= f['start_offset_s'] < 200 for f in flows))
        self.assertTrue(all(f['src_ip'] != f['dst_ip'] for f in flows))
        gaps = [b['start_offset_s'] - a['start_offset_s'] for a, b in zip(flows, flows[1:])]
        mean_gap = sum(gaps) / len(gaps)
        self.assertAlmostEqual(mean_gap, 1 / 20.0, delta=0.01)        # exponential mean 1/λ
        cv = (sum((g - mean_gap) ** 2 for g in gaps) / len(gaps)) ** 0.5 / mean_gap
        self.assertAlmostEqual(cv, 1.0, delta=0.15)                   # exponential: CV ≈ 1

    def test_collision_pairs_are_h1_h9_and_h2_h10(self):
        specs = tg.collision_specs(HOSTS, 30)
        self.assertEqual([s['flow_id'] for s in specs], ['H1->H9', 'H2->H10'])
        self.assertEqual((specs[0]['src_ip'], specs[0]['dst_ip']), ('10.0.0.2', '10.2.0.2'))
        self.assertEqual((specs[1]['src_ip'], specs[1]['dst_ip']), ('10.0.0.3', '10.2.0.3'))
        self.assertEqual({s['offset'] for s in specs}, {0.0}, 'simultaneous by default')

    def test_random_elephants_are_inter_pod(self):
        for s in tg.random_elephant_specs(HOSTS, 4, 30, seed=3):
            self.assertNotEqual(s['src_ip'].split('.')[1], s['dst_ip'].split('.')[1])

    def test_iperf_cmd_uses_interval_csv_logging(self):
        cmd = tg.iperf_client_cmd('10.2.0.2', 5001, 30, 0.5)
        self.assertEqual(cmd[-4:], ['-i', '0.5', '-y', 'C'])


class ParserTest(unittest.TestCase):
    def test_parse_iperf_drops_summary(self):
        s = tg.parse_iperf_csv(IPERF_SAMPLE, 0.5)
        self.assertEqual(len(s), 3)
        self.assertAlmostEqual(s[1]['throughput_mbps'], 5.242880)

    def test_parse_dump_ports(self):
        p = tg.parse_dump_ports(DUMP_PORTS_SAMPLE)
        self.assertEqual(p, {'1': {'rx_bytes': 1000, 'tx_bytes': 2000},
                             '2': {'rx_bytes': 3000, 'tx_bytes': 4000}})
        rows = tg.core_load_rows({'c_0_0': {}}, {'c_0_0': p}, 1.0, 'x', 'r')
        self.assertEqual(rows[0]['tx_bytes'], 6000)


class LoopbackMiceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.port = tg._free_port()
        cls.stop, ready = threading.Event(), threading.Event()
        cls.th = threading.Thread(target=tg.run_server,
                                  kwargs={'port': cls.port, 'bind': '127.0.0.1',
                                          'stop_event': cls.stop, 'ready_event': ready}, daemon=True)
        cls.th.start()
        ready.wait(5)
        cls.tmp = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        cls.stop.set()
        cls.th.join(2)
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _run(self, proto):
        hosts = [tg.HostInfo('a', '127.0.0.1', 'H1'), tg.HostInfo('b', '127.0.0.1', 'H2')]
        flows = tg.generate_mice_schedule(hosts, 1.0, 15, seed=7, proto=proto)
        rows = tg.run_mice_flows(flows, tg.time.time() + 0.2, 'test', 'r1', self.port, timeout=3)
        self.assertEqual(len(rows), len(flows))
        self.assertTrue(all(r['status'] == 'ok' for r in rows), rows)
        self.assertTrue(all(float(r['fct_ms']) > 0 for r in rows))
        return rows

    def test_tcp_flows_complete_and_csv_is_valid(self):
        rows = self._run('tcp')
        path = tg.write_rows(os.path.join(self.tmp, 'mice_results.csv'), tg.MICE_FIELDS, rows, append=False)
        back = plotter.read_csv(path, plotter.MICE_REQUIRED)
        self.assertEqual(len(back), len(rows))
        self.assertEqual(list(back[0].keys()), tg.MICE_FIELDS)

    def test_udp_flows_complete(self):
        self._run('udp')

    def test_append_keeps_single_header(self):
        path = os.path.join(self.tmp, 'append.csv')
        row = {f: '1' for f in tg.MICE_FIELDS}
        tg.write_rows(path, tg.MICE_FIELDS, [row], append=False)
        tg.write_rows(path, tg.MICE_FIELDS, [row], append=True)
        with open(path) as fh:
            self.assertEqual(fh.read().count('scheme,'), 1)


class PlotterTest(unittest.TestCase):
    """Synthetic static / ecmp / proposed results -> all three plots render."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        rng = random.Random(0)
        mice, ele, core = [], [], []
        base_fct = {'static': 60, 'ecmp': 35, 'proposed': 15}
        for scheme in ('static', 'ecmp', 'proposed'):
            run = '20260101-00000%d' % len(scheme)
            for i in range(200):
                mice.append({'scheme': scheme, 'run_id': run, 'flow_id': 'M%d' % i, 'size_bytes': 32768,
                             'fct_ms': '%.3f' % (base_fct[scheme] * rng.lognormvariate(0, 0.6)),
                             'status': 'ok' if i % 50 else 'timeout'})
            spec_a = tg._elephant_spec(HOSTS[0], HOSTS[8], 0, 20)
            spec_b = tg._elephant_spec(HOSTS[1], HOSTS[9], 0, 20)
            for spec in (spec_a, spec_b):
                samples = []
                for k in range(40):
                    t = (k + 1) * 0.5
                    rerouted = scheme == 'proposed' and t > 5
                    mbps = 9.5 if rerouted else 4.8 + rng.uniform(-0.3, 0.3)
                    samples.append({'interval_start': t - 0.5, 'interval_end': t,
                                    'bytes': int(mbps * 1e6 / 16), 'throughput_mbps': mbps})
                ele += tg.elephant_rows(samples, spec, 1000.0, 1000.0, scheme, run)
            loads = {'static': [9, 0.5, 0.4, 0.3], 'ecmp': [6, 4, 1, 0.5], 'proposed': [3, 3, 2.5, 2.8]}[scheme]
            for j, load in enumerate(loads):
                core.append({'scheme': scheme, 'run_id': run, 'switch': 'c_%d_%d' % (j // 2, j % 2),
                             'duration_s': 20, 'tx_bytes': 0, 'rx_bytes': 0,
                             'avg_tx_mbps': load, 'avg_rx_mbps': load})
        self.mice = tg.write_rows(os.path.join(self.tmp, 'mice_results.csv'), tg.MICE_FIELDS, mice, False)
        self.ele = tg.write_rows(os.path.join(self.tmp, 'elephant_results.csv'), tg.ELEPHANT_FIELDS, ele, False)
        self.core = tg.write_rows(os.path.join(self.tmp, 'core_load.csv'), tg.CORE_FIELDS, core, False)
        with open(os.path.join(self.tmp, 'events.csv'), 'w') as fh:
            fh.write('epoch,scheme,event\n1003.0,proposed,ELEPHANT 10.0.0.2->10.2.0.2\n'
                     '1005.0,proposed,reroute H2->H10 via c_1_1\n')

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _ok(self, path):
        self.assertTrue(os.path.exists(path))
        self.assertGreater(os.path.getsize(path), 10000)

    def test_each_plot(self):
        self._ok(plotter.plot_throughput_vs_time(self.ele, os.path.join(self.tmp, 't.png'),
                                                 events_csv=os.path.join(self.tmp, 'events.csv')))
        self._ok(plotter.plot_fct_cdf(self.mice, os.path.join(self.tmp, 'f.png')))
        self._ok(plotter.plot_core_load(self.core, os.path.join(self.tmp, 'c.png')))

    def test_cli_main(self):
        self.assertEqual(plotter.main(['--results-dir', self.tmp]), 0)
        for name in ('throughput_vs_time.png', 'fct_cdf.png', 'core_load.png'):
            self._ok(os.path.join(self.tmp, 'plots', name))

    def test_missing_column_is_reported(self):
        bad = os.path.join(self.tmp, 'bad.csv')
        with open(bad, 'w') as fh:
            fh.write('scheme,run_id\nx,1\n')
        with self.assertRaises(ValueError):
            plotter.plot_fct_cdf(bad, os.path.join(self.tmp, 'x.png'))

    def test_latest_run_only(self):
        rows = [{'scheme': 'ecmp', 'run_id': '1'}, {'scheme': 'ecmp', 'run_id': '2'}]
        self.assertEqual(plotter.select_runs(rows), [rows[1]])
        self.assertEqual(len(plotter.select_runs(rows, all_runs=True)), 2)


if __name__ == '__main__':
    unittest.main()

```

# traffic_gen.py

```py
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
MICE_PORT = _cfg('MICE_PORT', 6000)
IPERF_PORT = _cfg('IPERF_PORT', 5001)
RESULTS_DIR = _cfg('RESULTS_DIR', 'results')
MICE_MIN_BYTES = 8 * 1024
MICE_MAX_BYTES = 64 * 1024
UDP_CHUNK = 1200
DEFAULT_COLLISION_PAIRS = [(1, 9), (2, 10)]

_HDR = struct.Struct('!Q')          # TCP request header: payload size
_UDP_HDR = struct.Struct('!IIQ')    # UDP datagram header: tag, seq, total size
_ACK = b'\x06'

MICE_FIELDS = ['scheme', 'run_id', 'flow_id', 'src', 'dst', 'src_ip', 'dst_ip', 'proto',
               'size_bytes', 'start_epoch', 'start_offset_s', 'fct_ms', 'status']
ELEPHANT_FIELDS = ['scheme', 'run_id', 'flow_id', 'src', 'dst', 'src_ip', 'dst_ip',
                   'interval_start', 'interval_end', 'time_s', 'epoch', 'bytes',
                   'throughput_mbps']
CORE_FIELDS = ['scheme', 'run_id', 'switch', 'duration_s', 'tx_bytes', 'rx_bytes',
               'avg_tx_mbps', 'avg_rx_mbps']

HostInfo = namedtuple('HostInfo', ['name', 'ip', 'alias'])


# =========================================================== CSV helpers
def write_rows(path, fields, rows, append=True):
    """Write rows to a CSV; appends (without a second header) when the file exists."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    new = not (append and os.path.exists(path) and os.path.getsize(path) > 0)
    with open(path, 'w' if new else 'a', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction='ignore')
        if new:
            writer.writeheader()
        writer.writerows(rows)
    return path


def read_rows(path):
    with open(path, newline='') as fh:
        return list(csv.DictReader(fh))


# ====================================================== mice: server side
def _recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError('peer closed early')
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


def run_server(port=MICE_PORT, bind='0.0.0.0', stop_event=None, ready_event=None):
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
                udp.sendto(_ACK + struct.pack('!I', tag), addr)
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
def send_mice_flow(dst_ip, size, port=MICE_PORT, proto='tcp', timeout=10.0):
    """
    Send one mice request of `size` bytes and wait for the server's ACK.
    Returns (fct_ms, status) with status in {'ok', 'timeout', 'error', 'bad_ack'}.
    """
    start = time.perf_counter()
    try:
        if proto == 'tcp':
            with socket.create_connection((dst_ip, port), timeout=timeout) as s:
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                s.sendall(_HDR.pack(size) + bytes(size))
                ack = _recv_exact(s, 1)
        elif proto == 'udp':
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
                    if len(data) >= 5 and data[:1] == _ACK and struct.unpack('!I', data[1:5])[0] == tag:
                        ack = _ACK
                        break
        else:
            raise ValueError('proto must be tcp or udp')
        fct = (time.perf_counter() - start) * 1000.0
        return fct, ('ok' if ack == _ACK else 'bad_ack')
    except socket.timeout:
        return (time.perf_counter() - start) * 1000.0, 'timeout'
    except OSError:
        return (time.perf_counter() - start) * 1000.0, 'error'


def generate_mice_schedule(hosts, duration, rate, min_bytes=MICE_MIN_BYTES,
                           max_bytes=MICE_MAX_BYTES, seed=None, proto='tcp'):
    """
    Poisson arrival process: aggregate `rate` flows/s, exponential inter-arrival times,
    uniformly random (src != dst) host pairs and sizes in [min_bytes, max_bytes].
    hosts: list of HostInfo.
    """
    if len(hosts) < 2:
        raise ValueError('need at least two hosts')
    if rate <= 0:
        raise ValueError('rate must be > 0')
    rng = random.Random(seed)
    flows = []
    t = rng.expovariate(rate)
    while t < duration:
        src, dst = rng.sample(hosts, 2)
        flows.append({'flow_id': 'M%05d' % len(flows), 'src': src.alias, 'dst': dst.alias,
                      'src_name': src.name, 'src_ip': src.ip, 'dst_ip': dst.ip,
                      'size_bytes': rng.randint(min_bytes, max_bytes),
                      'start_offset_s': round(t, 6), 'proto': proto})
        t += rng.expovariate(rate)
    return flows


def run_mice_flows(flows, t0, scheme, run_id, port=MICE_PORT, timeout=10.0):
    """Fire each flow at t0 + start_offset_s (own thread, so bursts overlap) and log FCTs."""
    rows, lock, threads = [], threading.Lock(), []

    def worker(f):
        start_epoch = time.time()
        fct, status = send_mice_flow(f['dst_ip'], f['size_bytes'], port, f['proto'], timeout)
        row = {'scheme': scheme, 'run_id': run_id, 'flow_id': f['flow_id'],
               'src': f['src'], 'dst': f['dst'], 'src_ip': f['src_ip'], 'dst_ip': f['dst_ip'],
               'proto': f['proto'], 'size_bytes': f['size_bytes'],
               'start_epoch': '%.6f' % start_epoch,
               'start_offset_s': '%.6f' % (start_epoch - t0),
               'fct_ms': '%.3f' % fct, 'status': status}
        with lock:
            rows.append(row)

    for f in sorted(flows, key=lambda x: x['start_offset_s']):
        delay = t0 + f['start_offset_s'] - time.time()
        if delay > 0:
            time.sleep(delay)
        th = threading.Thread(target=worker, args=(f,), daemon=True)
        th.start()
        threads.append(th)
    for th in threads:
        th.join(timeout + 5)
    rows.sort(key=lambda r: float(r['start_offset_s']))
    return rows


# ============================================================ elephants
def iperf_client_cmd(dst_ip, port=IPERF_PORT, duration=30, interval=0.5):
    return ['iperf', '-c', dst_ip, '-p', str(port), '-t', str(duration),
            '-i', str(interval), '-y', 'C']


def parse_iperf_csv(text, interval=0.5):
    """
    Parse iperf2 `-y C` output:
      timestamp,src_ip,src_port,dst_ip,dst_port,id,start-end,bytes,bits_per_second
    The final whole-run summary line is dropped.
    """
    samples = []
    for line in text.splitlines():
        parts = line.strip().split(',')
        if len(parts) < 9:
            continue
        try:
            a, b = (float(x) for x in parts[6].split('-'))
            nbytes = int(float(parts[7]))
            bps = float(parts[8])
        except ValueError:
            continue
        if b - a > interval * 1.5:
            continue  # summary line (0.0-30.0)
        samples.append({'interval_start': a, 'interval_end': b, 'bytes': nbytes,
                        'throughput_mbps': bps / 1e6})
    return samples


def elephant_rows(samples, spec, launch_epoch, t0, scheme, run_id):
    rows = []
    for s in samples:
        rows.append({'scheme': scheme, 'run_id': run_id, 'flow_id': spec['flow_id'],
                     'src': spec['src'], 'dst': spec['dst'], 'src_ip': spec['src_ip'],
                     'dst_ip': spec['dst_ip'], 'interval_start': s['interval_start'],
                     'interval_end': s['interval_end'],
                     'time_s': '%.3f' % (launch_epoch - t0 + s['interval_end']),
                     'epoch': '%.3f' % (launch_epoch + s['interval_end']),
                     'bytes': s['bytes'], 'throughput_mbps': '%.4f' % s['throughput_mbps']})
    return rows


def start_elephant(src_node, spec, port=IPERF_PORT, interval=0.5):
    """Launch one long-lived iperf TCP transfer from a Mininet host."""
    cmd = iperf_client_cmd(spec['dst_ip'], port, spec['duration'], interval)
    proc = src_node.popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return {'proc': proc, 'launch_epoch': time.time(), 'spec': spec}


def collect_elephant(handle, t0, scheme, run_id, interval=0.5):
    proc, spec = handle['proc'], handle['spec']
    try:
        out, err = proc.communicate(timeout=spec['duration'] + 20)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, err = proc.communicate()
    text = out.decode(errors='replace') if isinstance(out, bytes) else (out or '')
    samples = parse_iperf_csv(text, interval)
    if not samples:
        msg = err.decode(errors='replace') if isinstance(err, bytes) else err
        print('  [!] no iperf samples for %s: %s' % (spec['flow_id'], (msg or '').strip()[:200]))
    return elephant_rows(samples, spec, handle['launch_epoch'], t0, scheme, run_id)


def _elephant_spec(src, dst, offset, duration):
    return {'flow_id': '%s->%s' % (src.alias, dst.alias), 'src': src.alias, 'dst': dst.alias,
            'src_name': src.name, 'src_ip': src.ip, 'dst_ip': dst.ip,
            'offset': round(offset, 3), 'duration': max(1.0, round(duration, 1))}


def collision_specs(hosts, duration, pairs=None, stagger=0.0):
    """
    Collision trigger: by default H1->H9 and H2->H10 start together (stagger=0).
    Use stagger>0 to start the 2nd flow later, so the plot shows the moment of collapse.
    """
    pairs = pairs or DEFAULT_COLLISION_PAIRS
    if max(max(p) for p in pairs) > len(hosts):
        print('  [!] collision pairs %s need %d hosts; using H1->H2' % (pairs, max(max(p) for p in pairs)))
        pairs = [(1, 2)]
    specs = []
    for k, (s, d) in enumerate(pairs):
        offset = k * stagger
        specs.append(_elephant_spec(hosts[s - 1], hosts[d - 1], offset, duration - offset))
    return specs


def _pod(ip):
    parts = ip.split('.')
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
        for i, spec in sorted(enumerate(specs), key=lambda x: x[1]['offset']):
            delay = t0 + spec['offset'] - time.time()
            if delay > 0:
                time.sleep(delay)
            handles[i] = start_elephant(nodes[spec['src_name']], spec, IPERF_PORT, interval)
            print('  elephant %-10s started (%s -> %s, %.0fs)'
                  % (spec['flow_id'], spec['src_ip'], spec['dst_ip'], spec['duration']))

    th = threading.Thread(target=launcher, daemon=True)
    th.start()
    return th, handles


# ============================================================ core load
_PORT_RE = re.compile(r'port\s+"?([^:"]+?)"?:\s*rx pkts=\d+,\s*bytes=(\d+)')
_TX_RE = re.compile(r'tx pkts=\d+,\s*bytes=(\d+)')


def parse_dump_ports(text):
    """Parse `ovs-ofctl dump-ports` output -> {port: {'rx_bytes', 'tx_bytes'}} (LOCAL dropped)."""
    ports, current = {}, None
    for line in text.splitlines():
        m = _PORT_RE.search(line)
        if m:
            current = m.group(1).strip()
            ports[current] = {'rx_bytes': int(m.group(2)), 'tx_bytes': 0}
            continue
        m = _TX_RE.search(line)
        if m and current is not None:
            ports[current]['tx_bytes'] = int(m.group(1))
    ports.pop('LOCAL', None)
    return ports


def snapshot_ports(switch_names):
    snap = {}
    for sw in switch_names:
        try:
            out = subprocess.run(['ovs-ofctl', '-O', 'OpenFlow13', 'dump-ports', sw],
                                 capture_output=True, text=True, timeout=10).stdout
        except (OSError, subprocess.SubprocessError):
            out = ''
        snap[sw] = parse_dump_ports(out)
    return snap


def core_load_rows(before, after, duration, scheme, run_id):
    rows = []
    for sw in sorted(after):
        tx = rx = 0
        for port, cnt in after[sw].items():
            prev = before.get(sw, {}).get(port, {'tx_bytes': 0, 'rx_bytes': 0})
            tx += max(0, cnt['tx_bytes'] - prev['tx_bytes'])
            rx += max(0, cnt['rx_bytes'] - prev['rx_bytes'])
        rows.append({'scheme': scheme, 'run_id': run_id, 'switch': sw,
                     'duration_s': '%.2f' % duration, 'tx_bytes': tx, 'rx_bytes': rx,
                     'avg_tx_mbps': '%.4f' % (tx * 8 / duration / 1e6),
                     'avg_rx_mbps': '%.4f' % (rx * 8 / duration / 1e6)})
    return rows


# ============================================================ Mininet glue
def _natural_key(name):
    return (tuple(int(x) for x in re.findall(r'\d+', name)), name)


def mininet_hosts(net):
    nodes = sorted(net.hosts, key=lambda h: _natural_key(h.name))
    hosts = [HostInfo(h.name, h.IP(), 'H%d' % (i + 1)) for i, h in enumerate(nodes)]
    return hosts, {h.name: h for h in nodes}


def build_net(topo='fattree'):
    from mininet.link import TCLink
    from mininet.log import setLogLevel
    from mininet.net import Mininet
    from mininet.node import OVSSwitch, RemoteController
    setLogLevel('info')
    bw = _cfg('LINK_BANDWIDTH', 10.0)
    if topo == 'fattree':
        from fattree_topo import FatTreeTopo, OpenFlow13Switch
        net = Mininet(topo=FatTreeTopo(k=_cfg('K_FAT_TREE', 4)), switch=OpenFlow13Switch,
                      controller=RemoteController('c0', ip=_cfg('CONTROLLER_IP', '127.0.0.1'),
                                                  port=_cfg('CONTROLLER_PORT', 6653)),
                      link=TCLink, autoSetMacs=False, autoStaticArp=False)
    elif topo == 'two':
        # 2-host testbed; standalone OVS = learning switch, no controller needed
        from functools import partial
        from mininet.topo import SingleSwitchTopo
        net = Mininet(topo=SingleSwitchTopo(k=2, lopts={'bw': bw}),
                      switch=partial(OVSSwitch, failMode='standalone'),
                      controller=None, link=TCLink, autoSetMacs=True)
    else:
        raise ValueError('unknown topo %r' % topo)
    net.start()
    return net


def run_experiment(net, scenario='mixed', scheme='proposed', duration=30.0, mice_rate=20.0,
                   mice_proto='tcp', mice_min=MICE_MIN_BYTES, mice_max=MICE_MAX_BYTES,
                   mice_timeout=10.0, n_elephants=4, collision_pairs=None, stagger=0.0,
                   interval=0.5, seed=42, results_dir=RESULTS_DIR, append=True,
                   warmup=True, with_mice=None):
    """
    Run one benchmark on an already-started Mininet `net`.
    scenario: 'mice' | 'elephant' | 'collision' | 'mixed'
    """
    run_id = time.strftime('%Y%m%d-%H%M%S')
    hosts, nodes = mininet_hosts(net)
    tmp = os.path.join(results_dir, 'tmp', run_id)
    os.makedirs(tmp, exist_ok=True)
    if with_mice is None:
        with_mice = scenario != 'elephant'
    print('\n=== run %s | scenario=%s scheme=%s duration=%.0fs hosts=%d ==='
          % (run_id, scenario, scheme, duration, len(hosts)))

    if warmup:
        print('*** warm-up pingAll (ARP / initial rules)')
        loss = net.pingAll(timeout='1')
        if loss > 0:
            print('  [!] %.0f%% ping loss — check the controller before trusting results' % loss)

    if scenario == 'collision':
        specs = collision_specs(hosts, duration, collision_pairs, stagger)
    elif scenario in ('elephant', 'mixed'):
        specs = random_elephant_specs(hosts, n_elephants, duration, seed)
    else:
        specs = []

    servers = []
    for node in nodes.values():
        if with_mice:
            servers.append(node.popen([sys.executable, SCRIPT_PATH, 'server', '--port', str(MICE_PORT)],
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        if specs:
            servers.append(node.popen(['iperf', '-s', '-p', str(IPERF_PORT)],
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    time.sleep(1.0)

    core_names = [s.name for s in net.switches if s.name.startswith('c_')] or \
                 [s.name for s in net.switches]
    before, before_ts = snapshot_ports(core_names), time.time()
    t0 = time.time() + 2.0

    mice_procs = []
    if with_mice:
        flows = generate_mice_schedule(hosts, duration, mice_rate, mice_min, mice_max, seed, mice_proto)
        print('*** %d mice flows scheduled (%.1f flows/s, %s)' % (len(flows), mice_rate, mice_proto))
        by_src = defaultdict(list)
        for f in flows:
            by_src[f['src_name']].append(f)
        for src_name, fl in by_src.items():
            sched = os.path.join(tmp, 'mice_%s.json' % src_name)
            part = os.path.join(tmp, 'mice_%s.csv' % src_name)
            with open(sched, 'w') as fh:
                json.dump({'t0': t0, 'scheme': scheme, 'run_id': run_id, 'port': MICE_PORT,
                           'timeout': mice_timeout, 'flows': fl}, fh)
            log = open(os.path.join(tmp, 'mice_%s.log' % src_name), 'w')
            proc = nodes[src_name].popen([sys.executable, SCRIPT_PATH, 'mice-client',
                                          '--schedule', sched, '--out', part],
                                         stdout=log, stderr=subprocess.STDOUT)
            mice_procs.append((proc, part, log))

    launcher, handles = launch_elephants(specs, t0, nodes, interval)

    end = t0 + duration
    while time.time() < end:
        time.sleep(min(5.0, max(0.0, end - time.time())))
        print('  ... %.0f / %.0f s' % (min(duration, time.time() - t0), duration))
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
        out['mice'] = write_rows(os.path.join(results_dir, 'mice_results.csv'), MICE_FIELDS, m_rows, append)
    if e_rows:
        out['elephant'] = write_rows(os.path.join(results_dir, 'elephant_results.csv'),
                                     ELEPHANT_FIELDS, e_rows, append)
    if c_rows:
        out['core'] = write_rows(os.path.join(results_dir, 'core_load.csv'), CORE_FIELDS, c_rows, append)
    print_summary(m_rows, e_rows, c_rows)
    return out


def print_summary(m_rows, e_rows, c_rows):
    ok = sorted(float(r['fct_ms']) for r in m_rows if r['status'] == 'ok')
    if m_rows:
        p = (lambda q: ok[min(len(ok) - 1, int(q * len(ok)))]) if ok else (lambda q: float('nan'))
        print('mice : %d/%d ok   FCT p50=%.1f ms  p99=%.1f ms' % (len(ok), len(m_rows), p(0.5), p(0.99)))
    per_flow = defaultdict(list)
    for r in e_rows:
        per_flow[r['flow_id']].append(float(r['throughput_mbps']))
    for fid, v in sorted(per_flow.items()):
        print('eleph: %-10s mean %.2f Mbps over %d samples' % (fid, sum(v) / len(v), len(v)))
    for r in c_rows:
        print('core : %-8s %s Mbps' % (r['switch'], r['avg_tx_mbps']))


# ============================================================ local test
def _free_port():
    while True:
        t = socket.socket()
        t.bind(('127.0.0.1', 0))
        port = t.getsockname()[1]
        u = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            u.bind(('127.0.0.1', port))
            return port
        except OSError:
            continue
        finally:
            t.close()
            u.close()


def run_local(scheme='local', duration=5.0, rate=10.0, proto='tcp', seed=42,
              results_dir=RESULTS_DIR, append=True, elephant_secs=5, interval=0.5):
    """Loopback sanity test (Task 2.4) — no Mininet or root needed."""
    port = _free_port()
    stop, ready = threading.Event(), threading.Event()
    th = threading.Thread(target=run_server, kwargs={'port': port, 'bind': '127.0.0.1',
                                                     'stop_event': stop, 'ready_event': ready},
                          daemon=True)
    th.start()
    ready.wait(5)
    run_id = time.strftime('%Y%m%d-%H%M%S')
    hosts = [HostInfo('lo_a', '127.0.0.1', 'H1'), HostInfo('lo_b', '127.0.0.1', 'H2')]
    t0 = time.time() + 0.5
    flows = generate_mice_schedule(hosts, duration, rate, seed=seed, proto=proto)
    m_rows = run_mice_flows(flows, t0, scheme, run_id, port, timeout=5.0)
    stop.set()
    th.join(2)
    out = {'mice': write_rows(os.path.join(results_dir, 'mice_results.csv'), MICE_FIELDS, m_rows, append)}

    e_rows = []
    if shutil.which('iperf'):
        eport = _free_port()
        srv = subprocess.Popen(['iperf', '-s', '-p', str(eport)],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(0.5)
        spec = _elephant_spec(hosts[0], hosts[1], 0, elephant_secs)
        launch = time.time()
        res = subprocess.run(iperf_client_cmd('127.0.0.1', eport, elephant_secs, interval),
                             capture_output=True, text=True, timeout=elephant_secs + 15)
        srv.terminate()
        e_rows = elephant_rows(parse_iperf_csv(res.stdout, interval), spec, launch, launch,
                               scheme, run_id)
        out['elephant'] = write_rows(os.path.join(results_dir, 'elephant_results.csv'),
                                     ELEPHANT_FIELDS, e_rows, append)
    else:
        print('[!] iperf not installed: skipped elephant test (sudo apt install iperf)')
    print_summary(m_rows, e_rows, [])
    return out


# ============================================================ CLI
def _parse_pairs(text):
    pairs = []
    for item in text.split(','):
        s, d = item.strip().upper().replace('H', '').split('-')
        pairs.append((int(s), int(d)))
    return pairs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)

    r = sub.add_parser('run', help='run a benchmark inside Mininet (needs sudo)')
    r.add_argument('--topo', choices=['fattree', 'two'], default='fattree')
    r.add_argument('--scenario', choices=['mice', 'elephant', 'collision', 'mixed'], default='mixed')
    r.add_argument('--scheme', default=_cfg('EXPERIMENT_SCHEME', 'proposed'),
                   help='label for this run, e.g. static | ecmp | proposed')
    r.add_argument('--duration', type=float, default=30.0)
    r.add_argument('--mice-rate', type=float, default=20.0, help='aggregate mice arrivals per second')
    r.add_argument('--mice-proto', choices=['tcp', 'udp'], default='tcp')
    r.add_argument('--mice-min', type=int, default=MICE_MIN_BYTES)
    r.add_argument('--mice-max', type=int, default=MICE_MAX_BYTES)
    r.add_argument('--mice-timeout', type=float, default=10.0)
    r.add_argument('--no-mice', action='store_true')
    r.add_argument('--elephants', type=int, default=4, help='random elephants (elephant/mixed)')
    r.add_argument('--pairs', type=_parse_pairs, default=None, help='collision pairs, e.g. "1-9,2-10"')
    r.add_argument('--stagger', type=float, default=0.0, help='delay between collision flows (s)')
    r.add_argument('--interval', type=float, default=0.5)
    r.add_argument('--seed', type=int, default=42)
    r.add_argument('--results-dir', default=RESULTS_DIR)
    r.add_argument('--fresh', action='store_true', help='overwrite CSVs instead of appending')
    r.add_argument('--no-warmup', action='store_true')
    r.add_argument('--cli', action='store_true', help='open the Mininet CLI afterwards')

    lo = sub.add_parser('local', help='loopback test without Mininet')
    lo.add_argument('--scheme', default='local')
    lo.add_argument('--duration', type=float, default=5.0)
    lo.add_argument('--mice-rate', type=float, default=10.0)
    lo.add_argument('--mice-proto', choices=['tcp', 'udp'], default='tcp')
    lo.add_argument('--seed', type=int, default=42)
    lo.add_argument('--results-dir', default=RESULTS_DIR)
    lo.add_argument('--fresh', action='store_true')

    sv = sub.add_parser('server', help='mice sink (run on a host)')
    sv.add_argument('--port', type=int, default=MICE_PORT)

    mc = sub.add_parser('mice-client', help='replay a mice schedule (run on a host)')
    mc.add_argument('--schedule', required=True)
    mc.add_argument('--out', required=True)

    args = ap.parse_args(argv)

    if args.cmd == 'server':
        try:
            run_server(args.port)
        except KeyboardInterrupt:
            pass
    elif args.cmd == 'mice-client':
        with open(args.schedule) as fh:
            s = json.load(fh)
        rows = run_mice_flows(s['flows'], s['t0'], s['scheme'], s['run_id'], s['port'], s['timeout'])
        write_rows(args.out, MICE_FIELDS, rows, append=False)
    elif args.cmd == 'local':
        print(run_local(args.scheme, args.duration, args.mice_rate, args.mice_proto, args.seed,
                        args.results_dir, append=not args.fresh))
    elif args.cmd == 'run':
        if hasattr(os, 'geteuid') and os.geteuid() != 0:
            sys.exit('Mininet needs root: sudo python3 traffic_gen.py run ...')
        net = build_net(args.topo)
        try:
            out = run_experiment(net, args.scenario, args.scheme, args.duration, args.mice_rate,
                                 args.mice_proto, args.mice_min, args.mice_max, args.mice_timeout,
                                 args.elephants, args.pairs, args.stagger, args.interval, args.seed,
                                 args.results_dir, append=not args.fresh,
                                 warmup=not args.no_warmup,
                                 with_mice=False if args.no_mice else None)
            print('results:', out)
            if args.cli:
                from mininet.cli import CLI
                CLI(net)
        finally:
            net.stop()


if __name__ == '__main__':
    main()

```

