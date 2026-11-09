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

