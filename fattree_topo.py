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