#!/usr/bin/env python3
"""
classifier.py
Telemetry & Flow Classifier — Tasks 3.2 (rate pipeline) and 3.3 (classification).

Pure Python: no Ryu / Mininet imports, so it can be unit-tested anywhere.

Rate calculation (per flow, per poll):

    Rate [Mbps] = (byte_count_now - byte_count_last) * 8 / (t_now - t_last) / 1e6

Classification:
    Rate >= ELEPHANT_THRESHOLD * LINK_BANDWIDTH for N consecutive samples -> ELEPHANT
    otherwise the flow stays MICE.
    An ELEPHANT is demoted back to MICE after MICE_DEMOTION_SAMPLES slow samples
    (hysteresis, so a flow does not flap between classes), and is reported as
    EXPIRED when the switch stops reporting it for FLOW_EXPIRY seconds.

Typical use (flow_monitor.py does this for you):

    clf = FlowClassifier()
    transitions = clf.update({five_tuple: byte_count, ...}, timestamp)
    for flow_id, old, new in transitions: ...
    clf.get_active_elephants()   # {FiveTuple: demand_mbps}
"""

import time
from collections import namedtuple

try:
    import config
except ImportError:  # allow use outside the project directory
    config = None


def _cfg(name, default):
    return getattr(config, name, default) if config is not None else default


ELEPHANT = 'ELEPHANT'
MICE = 'MICE'
EXPIRED = 'EXPIRED'

FiveTuple = namedtuple('FiveTuple', ['src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol'])


def bytes_to_mbps(delta_bytes, delta_t):
    """Convert a byte delta over delta_t seconds into Mbps."""
    return (delta_bytes * 8.0) / (delta_t * 1e6)


class RateTracker(object):
    """
    Task 3.2 — keeps history[key] = (last_byte_count, last_timestamp) and turns
    cumulative byte counters into instantaneous rates. Used for flows (key =
    5-tuple) and for ports (key = (dpid, port_no, 'tx'|'rx')).
    """

    def __init__(self):
        self.history = {}

    def update(self, key, byte_count, timestamp):
        """Store the new sample and return the rate in Mbps (None for the first sample)."""
        prev = self.history.get(key)
        if prev is None:
            self.history[key] = (byte_count, timestamp)
            return None
        last_bytes, last_ts = prev
        dt = timestamp - last_ts
        if dt <= 0:
            return None  # duplicate / out-of-order sample: keep the old reference point
        self.history[key] = (byte_count, timestamp)
        delta = byte_count - last_bytes
        if delta < 0:
            return None  # counter reset (flow rule re-installed): restart from this sample
        return bytes_to_mbps(delta, dt)

    def forget(self, key):
        self.history.pop(key, None)

    def __contains__(self, key):
        return key in self.history

    def __len__(self):
        return len(self.history)


class FlowRecord(object):
    __slots__ = ('flow_id', 'label', 'rate_mbps', 'peak_rate_mbps', 'above', 'below',
                 'samples', 'first_seen', 'last_seen', 'last_change', 'meta')

    def __init__(self, flow_id, now):
        self.flow_id = flow_id
        self.label = MICE
        self.rate_mbps = 0.0
        self.peak_rate_mbps = 0.0
        self.above = 0          # consecutive samples >= threshold
        self.below = 0          # consecutive samples <  threshold
        self.samples = 0
        self.first_seen = now
        self.last_seen = now
        self.last_change = now
        self.meta = {}


class FlowClassifier(object):
    """Task 3.3 — standalone Elephant/Mice classifier."""

    def __init__(self, link_bandwidth=None, threshold=None, consecutive_samples=None,
                 demotion_samples=None, expiry=None):
        self.link_bandwidth = float(link_bandwidth if link_bandwidth is not None
                                    else _cfg('LINK_BANDWIDTH', 10.0))
        self.threshold = float(threshold if threshold is not None
                               else _cfg('ELEPHANT_THRESHOLD', 0.1))
        self.consecutive_samples = int(consecutive_samples if consecutive_samples is not None
                                       else _cfg('ELEPHANT_CONSECUTIVE_SAMPLES', 2))
        self.demotion_samples = int(demotion_samples if demotion_samples is not None
                                    else _cfg('MICE_DEMOTION_SAMPLES', 3))
        self.expiry = float(expiry if expiry is not None else _cfg('FLOW_EXPIRY', 5.0))
        if self.consecutive_samples < 1 or self.demotion_samples < 1:
            raise ValueError('consecutive/demotion samples must be >= 1')

        self.rates = RateTracker()
        self.flows = {}

    # ------------------------------------------------------------------ state
    @property
    def rate_threshold_mbps(self):
        return self.threshold * self.link_bandwidth

    @property
    def flow_history(self):
        """flow_history[flow_id] = (last_byte_count, last_timestamp)"""
        return self.rates.history

    # --------------------------------------------------------------- updating
    def update(self, byte_counts, timestamp=None, meta=None):
        """
        Feed one polling round of cumulative byte counters.

        byte_counts: {flow_id: cumulative_byte_count}
        timestamp:   seconds (defaults to time.time())
        meta:        optional {flow_id: dict} stored with the flow (dpid, packets, ...)

        Returns a list of class transitions: (flow_id, old_label, new_label),
        where new_label is ELEPHANT, MICE or EXPIRED.
        """
        now = time.time() if timestamp is None else float(timestamp)
        transitions = []
        for flow_id, byte_count in byte_counts.items():
            rec = self._record(flow_id, now, meta)
            rate = self.rates.update(flow_id, byte_count, now)
            if rate is not None:
                change = self._apply_rate(rec, rate, now)
                if change:
                    transitions.append(change)
        transitions.extend(self.expire(now))
        return transitions

    def update_rate(self, flow_id, rate_mbps, timestamp=None, meta=None):
        """Feed an already-computed rate (bypasses the byte-counter pipeline)."""
        now = time.time() if timestamp is None else float(timestamp)
        rec = self._record(flow_id, now, meta)
        return self._apply_rate(rec, float(rate_mbps), now)

    def _record(self, flow_id, now, meta):
        rec = self.flows.get(flow_id)
        if rec is None:
            rec = self.flows[flow_id] = FlowRecord(flow_id, now)
        rec.last_seen = now
        if meta and flow_id in meta:
            rec.meta = meta[flow_id]
        return rec

    def _apply_rate(self, rec, rate, now):
        rec.rate_mbps = rate
        rec.peak_rate_mbps = max(rec.peak_rate_mbps, rate)
        rec.samples += 1
        if rate >= self.rate_threshold_mbps:
            rec.above += 1
            rec.below = 0
        else:
            rec.below += 1
            rec.above = 0

        if rec.label == MICE and rec.above >= self.consecutive_samples:
            rec.label, rec.last_change = ELEPHANT, now
            return (rec.flow_id, MICE, ELEPHANT)
        if rec.label == ELEPHANT and rec.below >= self.demotion_samples:
            rec.label, rec.last_change = MICE, now
            return (rec.flow_id, ELEPHANT, MICE)
        return None

    def expire(self, now=None):
        """Forget flows not reported for `expiry` seconds. Returns ELEPHANT->EXPIRED transitions."""
        now = time.time() if now is None else float(now)
        stale = [fid for fid, rec in self.flows.items() if now - rec.last_seen > self.expiry]
        transitions = []
        for fid in stale:
            rec = self.flows.pop(fid)
            self.rates.forget(fid)
            if rec.label == ELEPHANT:
                transitions.append((fid, ELEPHANT, EXPIRED))
        return transitions

    # ------------------------------------------------------------------ query
    def get_active_elephants(self, use_peak=False):
        """
        {flow 5-tuple: current bandwidth demand in Mbps} for every ELEPHANT.
        use_peak=True returns the highest observed rate instead, which is a better
        demand estimate for a flow that is currently squeezed by a collision.
        """
        return {fid: (rec.peak_rate_mbps if use_peak else rec.rate_mbps)
                for fid, rec in self.flows.items() if rec.label == ELEPHANT}

    def get_elephant_details(self):
        return {fid: {'rate_mbps': rec.rate_mbps, 'peak_rate_mbps': rec.peak_rate_mbps,
                      'since': rec.last_change, 'samples': rec.samples, 'meta': dict(rec.meta)}
                for fid, rec in self.flows.items() if rec.label == ELEPHANT}

    def classify(self, flow_id):
        rec = self.flows.get(flow_id)
        return rec.label if rec else None

    def get_rate(self, flow_id):
        rec = self.flows.get(flow_id)
        return rec.rate_mbps if rec else None

    def counts(self):
        out = {ELEPHANT: 0, MICE: 0}
        for rec in self.flows.values():
            out[rec.label] += 1
        return out

    def remove_flow(self, flow_id):
        self.flows.pop(flow_id, None)
        self.rates.forget(flow_id)

    def reset(self):
        self.flows.clear()
        self.rates = RateTracker()

    def __len__(self):
        return len(self.flows)
