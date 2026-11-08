"""
Task 2.4 — traffic_gen.py on loopback + CSV validity + plotter end-to-end.
No Mininet or root needed. Run: python3 -m unittest discover -s tests -v
"""
import os
import random
import shutil
import sys
import tempfile
import threading
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import plotter  # noqa: E402
import traffic_gen as tg  # noqa: E402

HOSTS = [tg.HostInfo('h_%d_%d_%d' % (p, e, h), '10.%d.%d.%d' % (p, e, h + 2), 'H%d' % (i + 1))
         for i, (p, e, h) in enumerate((p, e, h) for p in range(4) for e in range(2) for h in range(2))]

IPERF_SAMPLE = """\
20260108120000,10.0.0.2,45678,10.2.0.2,5001,3,0.0-0.5,589824,9437184
20260108120000,10.0.0.2,45678,10.2.0.2,5001,3,0.5-1.0,327680,5242880
20260108120001,10.0.0.2,45678,10.2.0.2,5001,3,1.0-1.5,589824,9437184
20260108120001,10.0.0.2,45678,10.2.0.2,5001,3,0.0-1.5,1507328,8039082
"""

DUMP_PORTS_SAMPLE = """\
OFPST_PORT reply (OF1.3) (xid=0x2): 5 ports
  port LOCAL: rx pkts=0, bytes=0, drop=0, errs=0, frame=0, over=0, crc=0
           tx pkts=0, bytes=0, drop=0, errs=0, coll=0
           duration=10.0s
  port  1: rx pkts=10, bytes=1000, drop=0, errs=0, frame=0, over=0, crc=0
           tx pkts=20, bytes=2000, drop=0, errs=0, coll=0
           duration=10.0s
  port  2: rx pkts=30, bytes=3000, drop=0, errs=0, frame=0, over=0, crc=0
           tx pkts=40, bytes=4000, drop=0, errs=0, coll=0
"""


class ScheduleTest(unittest.TestCase):
    def test_poisson_schedule(self):
        flows = tg.generate_mice_schedule(HOSTS, duration=200, rate=20, seed=1)
        self.assertEqual(flows, tg.generate_mice_schedule(HOSTS, 200, 20, seed=1), 'seeded = reproducible')
        self.assertTrue(all(tg.MICE_MIN_BYTES <= f['size_bytes'] <= tg.MICE_MAX_BYTES for f in flows))
        self.assertTrue(all(0 <= f['start_offset_s'] < 200 for f in flows))
        self.assertTrue(all(f['src_ip'] != f['dst_ip'] for f in flows))
        gaps = [b['start_offset_s'] - a['start_offset_s'] for a, b in zip(flows, flows[1:])]
        mean_gap = sum(gaps) / len(gaps)
        self.assertAlmostEqual(mean_gap, 1 / 20.0, delta=0.01)        # exponential mean 1/λ
        cv = (sum((g - mean_gap) ** 2 for g in gaps) / len(gaps)) ** 0.5 / mean_gap
        self.assertAlmostEqual(cv, 1.0, delta=0.15)                   # exponential: CV ≈ 1

    def test_collision_pairs_are_h1_h9_and_h2_h10(self):
        specs = tg.collision_specs(HOSTS, 30)
        self.assertEqual([s['flow_id'] for s in specs], ['H1->H9', 'H2->H10'])
        self.assertEqual((specs[0]['src_ip'], specs[0]['dst_ip']), ('10.0.0.2', '10.2.0.2'))
        self.assertEqual((specs[1]['src_ip'], specs[1]['dst_ip']), ('10.0.0.3', '10.2.0.3'))
        self.assertEqual({s['offset'] for s in specs}, {0.0}, 'simultaneous by default')

    def test_random_elephants_are_inter_pod(self):
        for s in tg.random_elephant_specs(HOSTS, 4, 30, seed=3):
            self.assertNotEqual(s['src_ip'].split('.')[1], s['dst_ip'].split('.')[1])

    def test_iperf_cmd_uses_interval_csv_logging(self):
        cmd = tg.iperf_client_cmd('10.2.0.2', 5001, 30, 0.5)
        self.assertEqual(cmd[-4:], ['-i', '0.5', '-y', 'C'])


class ParserTest(unittest.TestCase):
    def test_parse_iperf_drops_summary(self):
        s = tg.parse_iperf_csv(IPERF_SAMPLE, 0.5)
        self.assertEqual(len(s), 3)
        self.assertAlmostEqual(s[1]['throughput_mbps'], 5.242880)

    def test_parse_dump_ports(self):
        p = tg.parse_dump_ports(DUMP_PORTS_SAMPLE)
        self.assertEqual(p, {'1': {'rx_bytes': 1000, 'tx_bytes': 2000},
                             '2': {'rx_bytes': 3000, 'tx_bytes': 4000}})
        rows = tg.core_load_rows({'c_0_0': {}}, {'c_0_0': p}, 1.0, 'x', 'r')
        self.assertEqual(rows[0]['tx_bytes'], 6000)


class LoopbackMiceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.port = tg._free_port()
        cls.stop, ready = threading.Event(), threading.Event()
        cls.th = threading.Thread(target=tg.run_server,
                                  kwargs={'port': cls.port, 'bind': '127.0.0.1',
                                          'stop_event': cls.stop, 'ready_event': ready}, daemon=True)
        cls.th.start()
        ready.wait(5)
        cls.tmp = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        cls.stop.set()
        cls.th.join(2)
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _run(self, proto):
        hosts = [tg.HostInfo('a', '127.0.0.1', 'H1'), tg.HostInfo('b', '127.0.0.1', 'H2')]
        flows = tg.generate_mice_schedule(hosts, 1.0, 15, seed=7, proto=proto)
        rows = tg.run_mice_flows(flows, tg.time.time() + 0.2, 'test', 'r1', self.port, timeout=3)
        self.assertEqual(len(rows), len(flows))
        self.assertTrue(all(r['status'] == 'ok' for r in rows), rows)
        self.assertTrue(all(float(r['fct_ms']) > 0 for r in rows))
        return rows

    def test_tcp_flows_complete_and_csv_is_valid(self):
        rows = self._run('tcp')
        path = tg.write_rows(os.path.join(self.tmp, 'mice_results.csv'), tg.MICE_FIELDS, rows, append=False)
        back = plotter.read_csv(path, plotter.MICE_REQUIRED)
        self.assertEqual(len(back), len(rows))
        self.assertEqual(list(back[0].keys()), tg.MICE_FIELDS)

    def test_udp_flows_complete(self):
        self._run('udp')

    def test_append_keeps_single_header(self):
        path = os.path.join(self.tmp, 'append.csv')
        row = {f: '1' for f in tg.MICE_FIELDS}
        tg.write_rows(path, tg.MICE_FIELDS, [row], append=False)
        tg.write_rows(path, tg.MICE_FIELDS, [row], append=True)
        with open(path) as fh:
            self.assertEqual(fh.read().count('scheme,'), 1)


class PlotterTest(unittest.TestCase):
    """Synthetic static / ecmp / proposed results -> all three plots render."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        rng = random.Random(0)
        mice, ele, core = [], [], []
        base_fct = {'static': 60, 'ecmp': 35, 'proposed': 15}
        for scheme in ('static', 'ecmp', 'proposed'):
            run = '20260101-00000%d' % len(scheme)
            for i in range(200):
                mice.append({'scheme': scheme, 'run_id': run, 'flow_id': 'M%d' % i, 'size_bytes': 32768,
                             'fct_ms': '%.3f' % (base_fct[scheme] * rng.lognormvariate(0, 0.6)),
                             'status': 'ok' if i % 50 else 'timeout'})
            spec_a = tg._elephant_spec(HOSTS[0], HOSTS[8], 0, 20)
            spec_b = tg._elephant_spec(HOSTS[1], HOSTS[9], 0, 20)
            for spec in (spec_a, spec_b):
                samples = []
                for k in range(40):
                    t = (k + 1) * 0.5
                    rerouted = scheme == 'proposed' and t > 5
                    mbps = 9.5 if rerouted else 4.8 + rng.uniform(-0.3, 0.3)
                    samples.append({'interval_start': t - 0.5, 'interval_end': t,
                                    'bytes': int(mbps * 1e6 / 16), 'throughput_mbps': mbps})
                ele += tg.elephant_rows(samples, spec, 1000.0, 1000.0, scheme, run)
            loads = {'static': [9, 0.5, 0.4, 0.3], 'ecmp': [6, 4, 1, 0.5], 'proposed': [3, 3, 2.5, 2.8]}[scheme]
            for j, load in enumerate(loads):
                core.append({'scheme': scheme, 'run_id': run, 'switch': 'c_%d_%d' % (j // 2, j % 2),
                             'duration_s': 20, 'tx_bytes': 0, 'rx_bytes': 0,
                             'avg_tx_mbps': load, 'avg_rx_mbps': load})
        self.mice = tg.write_rows(os.path.join(self.tmp, 'mice_results.csv'), tg.MICE_FIELDS, mice, False)
        self.ele = tg.write_rows(os.path.join(self.tmp, 'elephant_results.csv'), tg.ELEPHANT_FIELDS, ele, False)
        self.core = tg.write_rows(os.path.join(self.tmp, 'core_load.csv'), tg.CORE_FIELDS, core, False)
        with open(os.path.join(self.tmp, 'events.csv'), 'w') as fh:
            fh.write('epoch,scheme,event\n1003.0,proposed,ELEPHANT 10.0.0.2->10.2.0.2\n'
                     '1005.0,proposed,reroute H2->H10 via c_1_1\n')

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _ok(self, path):
        self.assertTrue(os.path.exists(path))
        self.assertGreater(os.path.getsize(path), 10000)

    def test_each_plot(self):
        self._ok(plotter.plot_throughput_vs_time(self.ele, os.path.join(self.tmp, 't.png'),
                                                 events_csv=os.path.join(self.tmp, 'events.csv')))
        self._ok(plotter.plot_fct_cdf(self.mice, os.path.join(self.tmp, 'f.png')))
        self._ok(plotter.plot_core_load(self.core, os.path.join(self.tmp, 'c.png')))

    def test_cli_main(self):
        self.assertEqual(plotter.main(['--results-dir', self.tmp]), 0)
        for name in ('throughput_vs_time.png', 'fct_cdf.png', 'core_load.png'):
            self._ok(os.path.join(self.tmp, 'plots', name))

    def test_missing_column_is_reported(self):
        bad = os.path.join(self.tmp, 'bad.csv')
        with open(bad, 'w') as fh:
            fh.write('scheme,run_id\nx,1\n')
        with self.assertRaises(ValueError):
            plotter.plot_fct_cdf(bad, os.path.join(self.tmp, 'x.png'))

    def test_latest_run_only(self):
        rows = [{'scheme': 'ecmp', 'run_id': '1'}, {'scheme': 'ecmp', 'run_id': '2'}]
        self.assertEqual(plotter.select_runs(rows), [rows[1]])
        self.assertEqual(len(plotter.select_runs(rows, all_runs=True)), 2)


if __name__ == '__main__':
    unittest.main()
