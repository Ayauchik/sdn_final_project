#!/usr/bin/env python3
"""
reroute_engine.py

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
        self.ip_to_edge = {}
        self.port_map = {}
        
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
            ip = self.topo.nodeInfo(host).get('ip', '').split('/')[0]
            if ip:
                self.graph.nodes[host]['ip'] = ip
            
        for src, dst in self.topo.links():
            p1, p2 = self.topo.port(src, dst)
            self.port_map[(src, dst)] = p1
            self.port_map[(dst, src)] = p2
            
            if self.topo.isSwitch(src) and not self.topo.isSwitch(dst):
                self.host_to_edge[dst] = src
                ip = self.graph.nodes[dst].get('ip')
                if ip:
                    self.ip_to_edge[ip] = src
            elif self.topo.isSwitch(dst) and not self.topo.isSwitch(src):
                self.host_to_edge[src] = dst
                ip = self.graph.nodes[src].get('ip')
                if ip:
                    self.ip_to_edge[ip] = dst

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
        src_edge = self.ip_to_edge.get(src_node, self.host_to_edge.get(src_node, src_node))
        dst_edge = self.ip_to_edge.get(dst_node, self.host_to_edge.get(dst_node, dst_node))
        
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