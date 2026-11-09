# run_experiments.py
# Runs our 3 experiments one after another and then makes the plots.
#   Experiment 1: static   (every flow on the first path)
#   Experiment 2: ecmp     (hash-based path choice)
#   Experiment 3: proposed (ecmp + elephant rerouting)
# For each one we start the Ryu controller with the right SDN_SCHEME,
# run traffic_gen.py (it starts Mininet), then stop the controller.
#
# Run:  sudo python3 run_experiments.py
#       sudo python3 run_experiments.py --duration 40 --stagger 5

import argparse
import os
import shutil
import subprocess
import sys
import time

import config

SCHEMES = ['static', 'ecmp', 'proposed']
RESULT_FILES = ['mice_results.csv', 'elephant_results.csv', 'core_load.csv',
                'events.csv', 'port_stats.csv', 'flow_stats.csv']


def clean_mininet():
    subprocess.run(['mn', '-c'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def start_controller(ryu, scheme):
    env = os.environ.copy()
    env['SDN_SCHEME'] = scheme
    log = open(os.path.join(config.RESULTS_DIR, 'controller_%s.log' % scheme), 'w')
    proc = subprocess.Popen([ryu, 'main_controller.py',
                             '--ofp-tcp-listen-port', str(config.CONTROLLER_PORT)],
                            env=env, stdout=log, stderr=subprocess.STDOUT)
    time.sleep(3)  # give Ryu time to start listening
    if proc.poll() is not None:
        log.close()
        sys.exit('controller crashed, see results/controller_%s.log' % scheme)
    return proc, log


def stop_controller(proc, log):
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
    log.close()


def run_traffic(scheme, args):
    cmd = [sys.executable, 'traffic_gen.py', 'run',
           '--scenario', args.scenario,
           '--scheme', scheme,
           '--duration', str(args.duration),
           '--stagger', str(args.stagger),
           '--seed', str(args.seed)]   # same seed = same traffic for all schemes
    subprocess.run(cmd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', default='collision',
                        choices=['collision', 'mixed', 'mice', 'elephant'])
    parser.add_argument('--duration', type=float, default=30)
    parser.add_argument('--stagger', type=float, default=0,
                        help='seconds between the two collision flows')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--schemes', nargs='+', default=SCHEMES, choices=SCHEMES)
    parser.add_argument('--keep', action='store_true', help='keep old results')
    parser.add_argument('--ryu', default=shutil.which('ryu-manager') or 'ryu-manager')
    args = parser.parse_args()

    if os.geteuid() != 0:
        sys.exit('please run with sudo (Mininet needs root)')

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    if not args.keep:
        for name in RESULT_FILES:
            path = os.path.join(config.RESULTS_DIR, name)
            if os.path.exists(path):
                os.remove(path)

    for i, scheme in enumerate(args.schemes, 1):
        print('\n' + '=' * 60)
        print('Experiment %d: %s' % (i, scheme))
        print('=' * 60)
        clean_mininet()
        proc, log = start_controller(args.ryu, scheme)
        try:
            run_traffic(scheme, args)
        finally:
            stop_controller(proc, log)
            clean_mininet()
        time.sleep(2)

    print('\nMaking plots...')
    subprocess.run([sys.executable, 'plotter.py', '--results-dir', config.RESULTS_DIR])

    # quick check for the report: how fast did we reroute?
    events = os.path.join(config.RESULTS_DIR, 'events.csv')
    if os.path.exists(events):
        print('\nController events (proposed):')
        with open(events) as f:
            for line in f:
                if 'proposed' in line:
                    print('  ' + line.strip())


if __name__ == '__main__':
    main()
