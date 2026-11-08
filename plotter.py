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
