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
