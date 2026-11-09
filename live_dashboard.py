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