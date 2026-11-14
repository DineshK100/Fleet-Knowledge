"""Run the experiment sets E1-E4 of the proposal (Section 4.2) over several seeds.

    ros2 run fleet_experiments run_sweep --experiment E2 --seeds 0 1 2 --num-cars 10
    ros2 run fleet_experiments run_sweep --experiment E4 --seeds 0 1 2 --dry-run

Each configuration is one ``ros2 launch fleet_experiments fleet_memory.launch.py``
call with its own launch arguments and parameter overrides (written to a
temporary parameter file). Results go to ``<results-dir>/<experiment>/<config>_s<seed>``.
Afterwards run ``plot_results <results-dir>/<experiment>``.
"""

import argparse
import copy
import os
import subprocess
import sys
import tempfile
import time

import yaml

# config name -> {'launch': {arg: value}, 'params': {node_key: {param: value}}}
R = '/**/memory_reader'
M = 'fleet_memory'

EXPERIMENTS = {
    'E1': {
        'vlm': {'launch': {'memory_mode': 'vlm'}},
    },
    'E2': {
        'none': {'launch': {'memory_mode': 'none'}},
        'oracle': {'launch': {'memory_mode': 'oracle'}},
        'template': {'launch': {'memory_mode': 'template'}},
        'vlm': {'launch': {'memory_mode': 'vlm'}},
    },
    'E3': {
        'clean': {'launch': {'memory_mode': 'vlm'}},
        'false_memories': {'launch': {'memory_mode': 'vlm', 'false_memories': '5'}},
        'stale_hazard': {'launch': {'memory_mode': 'vlm', 'scenario': '@town03_stale.yaml'}},
        'stale_no_refute': {'launch': {'memory_mode': 'vlm', 'scenario': '@town03_stale.yaml'},
                            'params': {R: {'feedback_enabled': False}}},
    },
    'E4': {
        'lookahead_75': {'launch': {'memory_mode': 'vlm'}, 'params': {R: {'lookahead_m': 75.0}}},
        'lookahead_150': {'launch': {'memory_mode': 'vlm'}, 'params': {R: {'lookahead_m': 150.0}}},
        'lookahead_250': {'launch': {'memory_mode': 'vlm'}, 'params': {R: {'lookahead_m': 250.0}}},
        'topk_1': {'launch': {'memory_mode': 'vlm'}, 'params': {R: {'top_k': 1}}},
        'no_heading': {'launch': {'memory_mode': 'vlm'}, 'params': {R: {'use_heading': False}}},
        'no_decay': {'launch': {'memory_mode': 'vlm'}, 'params': {M: {'decay_lambda': 0.0}}},
        'fast_decay': {'launch': {'memory_mode': 'vlm'}, 'params': {M: {'decay_lambda': 0.002}}},
        'llm_adapter': {'launch': {'memory_mode': 'vlm', 'adapter': 'llm'}},
    },
}


def deep_merge(base, over):
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def share_dir():
    try:
        from ament_index_python.packages import get_package_share_directory
        return get_package_share_directory('fleet_experiments')
    except Exception:
        return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--experiment', required=True, choices=sorted(EXPERIMENTS))
    ap.add_argument('--configs', nargs='*', default=None, help='subset of configs to run')
    ap.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    ap.add_argument('--num-cars', type=int, default=10)
    ap.add_argument('--results-dir', default='~/fleet_memory_results')
    ap.add_argument('--params-file', default='', help='base parameter file')
    ap.add_argument('--extra', nargs='*', default=[], help='extra launch args, e.g. vlm_model:=x')
    ap.add_argument('--timeout-min', type=float, default=180.0, help='per-run timeout')
    ap.add_argument('--force', action='store_true', help='re-run finished runs')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])

    share = share_dir()
    base_params_path = args.params_file or os.path.join(share, 'config', 'params.yaml')
    with open(os.path.expanduser(base_params_path)) as f:
        base_params = yaml.safe_load(f)
    exp = EXPERIMENTS[args.experiment]
    configs = args.configs or list(exp)
    root = os.path.join(os.path.expanduser(args.results_dir), args.experiment)
    os.makedirs(root, exist_ok=True)
    tmpdir = tempfile.mkdtemp(prefix='fleet_sweep_')

    for cfg_name in configs:
        cfg = exp[cfg_name]
        params = deep_merge(base_params, cfg.get('params', {}))
        params_path = os.path.join(tmpdir, f'{cfg_name}.yaml')
        with open(params_path, 'w') as f:
            yaml.safe_dump(params, f)
        for seed in args.seeds:
            run_name = f'{cfg_name}_s{seed}'
            run_dir = os.path.join(root, run_name)
            cars_csv = os.path.join(run_dir, 'cars.csv')
            if not args.force and os.path.exists(cars_csv):
                with open(cars_csv) as f:
                    if sum(1 for _ in f) - 1 >= args.num_cars:
                        print(f'[skip] {run_name} already finished')
                        continue
            launch = dict(cfg.get('launch', {}))
            if launch.get('scenario', '').startswith('@'):
                launch['scenario'] = os.path.join(share, 'config', 'scenarios', launch['scenario'][1:])
            launch.update({'num_cars': str(args.num_cars), 'seed': str(seed),
                           'results_dir': root, 'run_name': run_name, 'config_name': cfg_name,
                           'params_file': params_path})
            cmd = ['ros2', 'launch', 'fleet_experiments', 'fleet_memory.launch.py'] + \
                [f'{k}:={v}' for k, v in launch.items()] + list(args.extra)
            print(f'[run] {run_name}: {" ".join(cmd)}')
            if args.dry_run:
                continue
            t0 = time.time()
            try:
                subprocess.run(cmd, timeout=args.timeout_min * 60, check=False)
            except subprocess.TimeoutExpired:
                print(f'[timeout] {run_name} after {args.timeout_min} min')
            print(f'[done] {run_name} in {(time.time() - t0) / 60:.1f} min')
            time.sleep(5)                                   # let CARLA settle between runs
    print(f'\nAll runs finished. Plot with:\n  ros2 run fleet_experiments plot_results {root} --baseline none')
    return 0


if __name__ == '__main__':
    sys.exit(main())
