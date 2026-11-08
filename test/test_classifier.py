"""
Task 3.4 — feed dummy byte-counter dictionaries into classifier.py.
Run: python3 -m unittest discover -s tests -v
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classifier import (ELEPHANT, EXPIRED, MICE, FiveTuple,  # noqa: E402
                        FlowClassifier, RateTracker)

ELE = FiveTuple('10.0.0.2', '10.2.0.2', 40000, 5001, 6)   # H1 -> H9 iperf
MOUSE = FiveTuple('10.1.0.2', '10.3.1.3', 41000, 6000, 6)  # 32 KB request


def mbps_bytes(mbps, seconds=1.0):
    """Bytes a flow sends in `seconds` at `mbps`."""
    return int(mbps * 1e6 / 8 * seconds)


class ClassifierTest(unittest.TestCase):
    def setUp(self):
        # 10 Mbps links, 10% threshold -> 1 Mbps; N = 2 consecutive samples
        self.clf = FlowClassifier(link_bandwidth=10.0, threshold=0.1, consecutive_samples=2,
                                  demotion_samples=2, expiry=5.0)

    def feed(self, per_second_mbps, flow=ELE, start=0.0):
        """Feed cumulative counters for a flow whose rate in each 1 s interval is given."""
        total, transitions = 0, []
        transitions += self.clf.update({flow: 0}, start)
        for i, mbps in enumerate(per_second_mbps, 1):
            total += mbps_bytes(mbps)
            transitions += self.clf.update({flow: total}, start + i)
        return transitions

    def test_rate_formula(self):
        rt = RateTracker()
        self.assertIsNone(rt.update('f', 0, 0.0))
        self.assertAlmostEqual(rt.update('f', 1250000, 1.0), 10.0)   # 1.25 MB in 1 s
        self.assertAlmostEqual(rt.update('f', 1500000, 1.5), 4.0)    # 250 KB in 0.5 s

    def test_small_burst_stays_mice(self):
        self.clf.update({MOUSE: 0}, 0.0)
        self.clf.update({MOUSE: 32 * 1024}, 1.0)   # one 32 KB burst = 0.26 Mbps
        self.clf.update({MOUSE: 32 * 1024}, 2.0)
        self.assertEqual(self.clf.classify(MOUSE), MICE)
        self.assertEqual(self.clf.get_active_elephants(), {})

    def test_single_spike_is_not_promoted(self):
        transitions = self.feed([8.0, 0.2, 8.0, 0.2])   # never 2 high samples in a row
        self.assertEqual(self.clf.classify(ELE), MICE)
        self.assertEqual(transitions, [])

    def test_sustained_rate_becomes_elephant_after_n_samples(self):
        self.clf.update({ELE: 0}, 0.0)
        self.clf.update({ELE: mbps_bytes(8)}, 1.0)
        self.assertEqual(self.clf.classify(ELE), MICE, 'one sample must not be enough')
        t = self.clf.update({ELE: mbps_bytes(8) * 2}, 2.0)
        self.assertEqual(self.clf.classify(ELE), ELEPHANT)
        self.assertEqual(t, [(ELE, MICE, ELEPHANT)])

    def test_threshold_is_inclusive(self):
        self.feed([1.0, 1.0])   # exactly 10% of 10 Mbps
        self.assertEqual(self.clf.classify(ELE), ELEPHANT)
        self.clf.reset()
        self.feed([0.99, 0.99])
        self.assertEqual(self.clf.classify(ELE), MICE)

    def test_get_active_elephants_returns_tuples_and_demand(self):
        self.feed([9.0, 9.0, 9.0])
        elephants = self.clf.get_active_elephants()
        self.assertEqual(list(elephants), [ELE])
        self.assertIsInstance(next(iter(elephants)), FiveTuple)
        self.assertAlmostEqual(elephants[ELE], 9.0, places=3)

    def test_mixed_traffic(self):
        ts, e_total, m_total = 0.0, 0, 0
        self.clf.update({ELE: 0, MOUSE: 0}, ts)
        for i in range(1, 5):
            e_total += mbps_bytes(6)
            m_total += 16 * 1024 if i == 1 else 0
            self.clf.update({ELE: e_total, MOUSE: m_total}, float(i))
        self.assertEqual(self.clf.classify(ELE), ELEPHANT)
        self.assertEqual(self.clf.classify(MOUSE), MICE)
        self.assertEqual(self.clf.counts(), {ELEPHANT: 1, MICE: 1})

    def test_demotion_back_to_mice(self):
        t = self.feed([8, 8, 0.1, 0.1])
        self.assertEqual(self.clf.classify(ELE), MICE)
        self.assertIn((ELE, ELEPHANT, MICE), t)

    def test_counter_reset_does_not_produce_negative_rate(self):
        self.feed([8, 8])
        self.clf.update({ELE: 100}, 3.0)   # rule re-installed, counter restarts
        self.assertGreaterEqual(self.clf.get_rate(ELE), 0.0)

    def test_duplicate_timestamp_ignored(self):
        self.clf.update({ELE: 0}, 0.0)
        self.clf.update({ELE: mbps_bytes(8)}, 0.0)
        self.assertEqual(self.clf.flow_history[ELE], (0, 0.0))

    def test_expiry_reports_finished_elephant(self):
        self.feed([8, 8])
        t = self.clf.update({MOUSE: 0}, 10.0)   # ELE not reported for 8 s
        self.assertIn((ELE, ELEPHANT, EXPIRED), t)
        self.assertIsNone(self.clf.classify(ELE))
        self.assertNotIn(ELE, self.clf.flow_history)

    def test_uses_config_defaults(self):
        clf = FlowClassifier()
        self.assertGreater(clf.rate_threshold_mbps, 0)
        self.assertGreaterEqual(clf.consecutive_samples, 1)


if __name__ == '__main__':
    unittest.main()
