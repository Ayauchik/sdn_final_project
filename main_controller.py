# main_controller.py
# Ryu controller for our project. It does 3 things:
#   1. answers ARP requests itself (so we never flood in the fat-tree, it has loops)
#   2. installs normal paths for every new flow (priority 10)
#        static   -> always the first path
#        ecmp     -> path chosen by hashing the 5-tuple
#        proposed -> same as ecmp, but elephants get rerouted
#   3. (proposed only) when flow_monitor finds an elephant, ask reroute_engine
#      for the best path and install it with priority 100
#
# Run:  SDN_SCHEME=proposed ryu-manager main_controller.py --ofp-tcp-listen-port 6653
#       (SDN_SCHEME can be static, ecmp or proposed)

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

        # ip -> host name and ip -> mac, taken from the topology
        self.ip_to_host = {}
        self.ip_to_mac = {}
        for host in self.engine.topo.hosts():
            info = self.engine.topo.nodeInfo(host)
            ip = info['ip'].split('/')[0]
            self.ip_to_host[ip] = host
            self.ip_to_mac[ip] = info['mac']

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
