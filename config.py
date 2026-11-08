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
