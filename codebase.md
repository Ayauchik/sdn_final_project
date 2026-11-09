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
                    
                    ip_addr = f'10.{pod_id}.{edge_id}.{host_id + 2}/24'
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
            if stat.byte_count >= byte_counts.get(ft, -1):
                byte_counts[ft] = stat.byte_count
                meta[ft] = {'dpid': dpid, 'packets': stat.packet_count,
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
Author: Member 4 (Path Routing & Graph Engine)

Core Responsibilities:
1. Extracts the Fat-Tree topology directly from `fattree_topo.py` into a NetworkX DiGraph.
2. Tracks per-link capacity, used bandwidth, and residual bandwidth.
3. Implements the Hedera-style Global First-Fit (GFF) dynamic rerouting algorithm.
4. Provides clean APIs for the Ryu controller to allocate and reroute elephant flows.
"""

import networkx as nx
from fattree_topo import FatTreeTopo
import config

try:
    DEFAULT_BW = config.LINK_BANDWIDTH
except AttributeError:
    DEFAULT_BW = 10.0


class RerouteEngine:
    def __init__(self, topo=None):
        """
        Initializes the dynamic graph directly from a Mininet Topo instance.
        """
        self.topo = topo if topo is not None else FatTreeTopo(k=config.K_FAT_TREE)
        self.graph = nx.DiGraph()
        
        self.dpid_to_name = {}
        self.name_to_dpid = {}
        self.host_to_edge = {}
        
        self.active_reservations = {}
        
        self._path_cache = {}
        
        self._build_graph_from_topo()
        self._precompute_all_shortest_paths()

    def _build_graph_from_topo(self):
        """
        Converts the Mininet FatTreeTopo instance into a NetworkX directed graph.
        Links are full-duplex, so each link is modeled as two directed edges (u->v and v->u).
        """
        for sw in self.topo.switches():
            dpid_str = self.topo.nodeInfo(sw).get('dpid')
            if dpid_str:
                dpid_int = int(dpid_str, 16)
                self.dpid_to_name[dpid_int] = sw
                self.name_to_dpid[sw] = dpid_int
            self.graph.add_node(sw, type='switch')

        for host in self.topo.hosts():
            self.graph.add_node(host, type='host')

        for src, dst in self.topo.links():
            if self.topo.isSwitch(src) and not self.topo.isSwitch(dst):
                self.host_to_edge[dst] = src
            elif self.topo.isSwitch(dst) and not self.topo.isSwitch(src):
                self.host_to_edge[src] = dst

            self.graph.add_edge(src, dst, capacity=DEFAULT_BW, used_bw=0.0, residual_bw=DEFAULT_BW)
            self.graph.add_edge(dst, src, capacity=DEFAULT_BW, used_bw=0.0, residual_bw=DEFAULT_BW)

    def _precompute_all_shortest_paths(self):
        """
        Precomputes all equal-cost shortest paths between every pair of edge switches.
        In a k=4 Fat-Tree, there are exactly 4 parallel paths through the core switches.
        """
        edge_switches = [n for n in self.graph.nodes if n.startswith('e_')]
        for src in edge_switches:
            for dst in edge_switches:
                if src == dst:
                    self._path_cache[(src, dst)] = [[src]]
                else:
                    paths = list(nx.all_shortest_paths(self.graph, source=src, target=dst))
                    self._path_cache[(src, dst)] = paths

    def get_candidate_paths(self, src_node, dst_node):
        """
        Returns all candidate equal-cost paths between any two nodes.
        Automatically resolves host names to their attached edge switches.
        """
        src_edge = self.host_to_edge.get(src_node, src_node)
        dst_edge = self.host_to_edge.get(dst_node, dst_node)
        
        return self._path_cache.get((src_edge, dst_edge), [])

    def get_path_bottleneck_residual(self, path):
        """
        Computes the minimum residual bandwidth along a given path:
        B(p) = min(residual_bw(u, v)) for all (u, v) in path
        """
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
        """
        Hedera-style Global First-Fit (GFF) / Best-Fit Rerouting:
        1. Finds all candidate equal-cost paths between src and dst.
        2. Filters paths where bottleneck residual capacity >= demand_mbps.
        3. If multiple paths qualify, picks the one with the maximum residual capacity.
        4. If no path satisfies the full demand, picks the path with the largest available bottleneck capacity.
        5. Updates reservations and link residual capacities.
        
        Returns:
            optimal_path (list of node names), bottleneck_capacity (float)
        """
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
        """
        Deducts demand_mbps from all directed links on the chosen path.
        """
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            self.graph[u][v]['used_bw'] += demand_mbps
            self.graph[u][v]['residual_bw'] = max(0.0, self.graph[u][v]['capacity'] - self.graph[u][v]['used_bw'])

        self.active_reservations[flow_id] = {
            'path': path,
            'demand': demand_mbps
        }

    def release_flow(self, flow_id):
        """
        Frees up bandwidth previously booked by an elephant flow when it completes.
        """
        if flow_id not in self.active_reservations:
            return

        info = self.active_reservations.pop(flow_id)
        path = info['path']
        demand = info['demand']

        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            self.graph[u][v]['used_bw'] = max(0.0, self.graph[u][v]['used_bw'] - demand)
            self.graph[u][v]['residual_bw'] = min(self.graph[u][v]['capacity'], self.graph[u][v]['capacity'] - self.graph[u][v]['used_bw'])

    def get_core_utilization_summary(self):
        """
        Returns a dictionary showing utilization across all 4 core switches.
        Crucial for plotting load-balancing variance in your project benchmarks!
        """
        core_switches = [n for n in self.graph.nodes if n.startswith('c_')]
        summary = {}
        for c in sorted(core_switches):
            # Sum inbound/outbound bandwidth through the core switch
            total_load = sum(self.graph[u][c]['used_bw'] for u in self.graph.predecessors(c))
            summary[c] = total_load
        return summary


if __name__ == '__main__':
    print("=" * 65)
    print("TESTING REROUTE ENGINE (HEDERA GLOBAL FIRST-FIT)")
    print("=" * 65)

    # 1. Initialize engine directly with FatTree topology
    engine = RerouteEngine()
    print(f"Graph initialized with {engine.graph.number_of_nodes()} nodes and {engine.graph.number_of_edges()} directed edges.")
    print(f"Core switches mapped: {[n for n in engine.graph.nodes if n.startswith('c_')]}\n")

    # 2. Inspect candidate paths between two hosts in different pods
    src_host = 'h_0_0_0'  # Pod 0
    dst_host = 'h_2_0_0'  # Pod 2
    paths = engine.get_candidate_paths(src_host, dst_host)
    print(f"Candidate parallel paths between {src_host} and {dst_host} ({len(paths)} core paths found):")
    for idx, p in enumerate(paths, 1):
        print(f"  Path {idx}: {' -> '.join(p)}")
    print()

    # 3. Simulate Flow 1 (Elephant: 6.0 Mbps)
    flow1_id = "flow_h0_to_h2"
    print(f"--- Allocating Flow 1 ({flow1_id}: 6.0 Mbps) ---")
    p1, cap1 = engine.compute_optimal_path(flow1_id, 6.0, src_host, dst_host)
    print(f"Flow 1 Assigned Path: {' -> '.join(p1)} (Remaining BottleNeck: {cap1:.1f} Mbps)")
    print("Core link utilization:", engine.get_core_utilization_summary())
    print()

    # 4. Simulate Flow 2 (Elephant Collision: 6.0 Mbps from h_0_0_1 to h_2_0_1)
    # Under naive ECMP, this might collide on the same core switch and drop packets.
    # The RerouteEngine should automatically divert it to an alternate core switch!
    flow2_id = "flow_h01_to_h21"
    src_host2 = 'h_0_0_1'
    dst_host2 = 'h_2_0_1'
    print(f"--- Allocating Flow 2 ({flow2_id}: 6.0 Mbps) ---")
    p2, cap2 = engine.compute_optimal_path(flow2_id, 6.0, src_host2, dst_host2)
    print(f"Flow 2 Assigned Path: {' -> '.join(p2)} (Remaining BottleNeck: {cap2:.1f} Mbps)")
    print("Core link utilization:", engine.get_core_utilization_summary())
    print()

    # 5. Verify that paths use DIFFERENT core switches!
    core_f1 = [n for n in p1 if n.startswith('c_')][0]
    core_f2 = [n for n in p2 if n.startswith('c_')][0]
    print("=" * 65)
    print(f"Flow 1 Core Switch: {core_f1}")
    print(f"Flow 2 Core Switch: {core_f2}")
    if core_f1 != core_f2:
        print("SUCCESS! Reroute engine avoided collision and balanced traffic across the core switches.")
    else:
        print("NOTICE: Both placed on same core (check capacities).")
    print("=" * 65)

    # 6. Test flow release
    print("\n--- Releasing Flow 1 ---")
    engine.release_flow(flow1_id)
    print("Core link utilization after releasing Flow 1:", engine.get_core_utilization_summary())
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

