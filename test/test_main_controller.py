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
