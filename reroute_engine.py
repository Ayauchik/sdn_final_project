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