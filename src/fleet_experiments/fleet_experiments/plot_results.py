"""Aggregate experiment runs and draw the "fleet improves" plots.

    ros2 run fleet_experiments plot_results ~/fleet_memory_results/E2 --baseline none

Every sub-directory with ``config.json`` + ``cars.csv`` is a run. Runs are
grouped by ``config_name`` (usually the memory mode). For each config the
script plots the metrics against the car index with means and 95 % confidence
intervals over seeds. It also writes ``summary.md`` / ``summary.csv`` with the
success criteria of the proposal: at least a 50 % reduction in near-misses and
hard braking for cars >= 2 versus the baseline, with less than a 10 % increase
in arrival time.
"""

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

METRICS = [
    ('near_miss_rate', 'Near-miss rate'),
    ('hard_brakes_per_encounter', 'Hard brakes / hazard encounter'),
    ('collisions', 'Collisions'),
    ('arrival_time_s', 'Arrival time (s)'),
    ('peak_decel_near_hazards', 'Peak decel near hazards (m/s²)'),
    ('peak_jerk_near_hazards', 'Peak jerk near hazards (m/s³)'),
    ('unnecessary_slowdown_s', 'Unnecessary slow-down (s)'),
    ('retrieval_recall', 'Retrieval recall'),
]

# two-sided 95 % Student-t quantiles for small samples (df -> t)
_T95 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31, 9: 2.26,
        10: 2.23, 15: 2.13, 20: 2.09, 30: 2.04}


def t95(df):
    if df <= 0:
        return float('nan')
    keys = sorted(_T95)
    for k in keys:
        if df <= k:
            return _T95[k]
    return 1.96


def mean_ci(vals):
    v = np.array([x for x in vals if x is not None and not math.isnan(x)], dtype=float)
    if len(v) == 0:
        return float('nan'), float('nan'), 0
    if len(v) == 1:
        return float(v[0]), 0.0, 1
    return float(v.mean()), t95(len(v) - 1) * float(v.std(ddof=1)) / math.sqrt(len(v)), len(v)


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def load_runs(root):
    runs = []
    for dirpath, _dirs, files in os.walk(os.path.expanduser(root)):
        if 'config.json' in files and 'cars.csv' in files:
            with open(os.path.join(dirpath, 'config.json')) as f:
                cfg = json.load(f)
            runs.append({'dir': dirpath, 'config': cfg.get('config_name') or 'default', 'cfg': cfg,
                         'cars': read_csv(os.path.join(dirpath, 'cars.csv')),
                         'records': read_csv(os.path.join(dirpath, 'written_records.csv')),
                         'events': read_csv(os.path.join(dirpath, 'memory_events.csv'))})
    return runs


def aggregate(runs):
    """config -> metric -> car index -> list of values (one per seed)."""
    agg = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in runs:
        for row in r['cars']:
            k = int(row['car'])
            for m, _ in METRICS:
                v = _num(row.get(m))
                if v is not None:
                    agg[r['config']][m][k].append(v)
    return agg


def later_cars_mean(series, first=2):
    vals = [v for k, vs in series.items() if k >= first for v in vs]
    return mean_ci(vals)


def plot_learning_curves(agg, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    configs = sorted(agg)
    n = len(METRICS)
    cols = 4
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.8 * rows))
    axes = np.array(axes).reshape(-1)
    for ax, (m, label) in zip(axes, METRICS):
        for cfg in configs:
            series = agg[cfg][m]
            if not series:
                continue
            ks = sorted(series)
            stats = [mean_ci(series[k]) for k in ks]
            mu = np.array([s[0] for s in stats])
            ci = np.array([s[1] for s in stats])
            ax.plot(ks, mu, marker='o', label=cfg)
            ax.fill_between(ks, mu - ci, mu + ci, alpha=0.2)
        ax.set_title(label)
        ax.set_xlabel('car index')
        ax.grid(alpha=0.3)
    for ax in axes[n:]:
        ax.axis('off')
    axes[0].legend()
    fig.suptitle('Fleet learning curves (mean ± 95% CI over seeds)')
    fig.tight_layout()
    path = os.path.join(out_dir, 'learning_curves.png')
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_speed_profiles(runs, out_dir):
    """Speed vs. distance for car 1 and the last car of the first run of each config."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    done, paths = set(), []
    for r in runs:
        if r['config'] in done:
            continue
        tdir = os.path.join(r['dir'], 'trajectories')
        files = sorted(f for f in os.listdir(tdir) if f.endswith('.csv')) if os.path.isdir(tdir) else []
        if len(files) < 2:
            continue
        done.add(r['config'])
        fig, ax = plt.subplots(figsize=(11, 4))
        for f in (files[0], files[-1]):
            rows = read_csv(os.path.join(tdir, f))
            if not rows:
                continue
            xs = np.array([float(x['x']) for x in rows])
            ys = np.array([float(x['y']) for x in rows])
            s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(xs), np.diff(ys)))])
            ax.plot(s, [float(x['speed_kmh']) for x in rows], label=f.replace('.csv', ''))
            if f == files[0]:
                for h in r['cfg'].get('hazards', []):
                    i = int(np.argmin(np.hypot(xs - h['x'], ys - h['y'])))
                    ax.axvline(s[i], color='r', alpha=0.4, ls='--')
                    ax.text(s[i], ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 1 else 45,
                            h['name'], rotation=90, va='top', fontsize=8, color='r')
        ax.set_xlabel('distance along route (m)')
        ax.set_ylabel('speed (km/h)')
        ax.set_title(f'Speed profile: first vs last car ({r["config"]})')
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        p = os.path.join(out_dir, f'speed_profile_{r["config"]}.png')
        fig.savefig(p, dpi=130)
        plt.close(fig)
        paths.append(p)
    return paths


def vlm_accuracy(runs):
    out = {}
    by_cfg = defaultdict(list)
    for r in runs:
        by_cfg[r['config']].extend(r['records'])
    for cfg, recs in by_cfg.items():
        matched = [x for x in recs if x.get('gt_type') not in (None, '', 'none')]
        vlm = [x for x in matched if x.get('source') == 'vlm']
        out[cfg] = {
            'records': len(recs),
            'matched_to_gt': len(matched),
            'type_accuracy': (sum(int(x['correct']) for x in matched) / len(matched)) if matched else float('nan'),
            'vlm_records': len(vlm),
            'vlm_type_accuracy': (sum(int(x['correct']) for x in vlm) / len(vlm)) if vlm else float('nan'),
        }
    return out


def cleanup_stats(runs):
    """E3: how fast are injected (false) memories removed from the memory?"""
    out = {}
    for r in runs:
        injected = {e['id'] for e in r['events'] if e.get('event') == 'insert' and e.get('source') == 'injected'}
        if not injected:
            continue
        pruned = {e['id']: int(e['car']) for e in r['events'] if e.get('event') == 'prune' and e['id'] in injected}
        s = out.setdefault(r['config'], {'injected': 0, 'pruned': 0, 'prune_car': []})
        s['injected'] += len(injected)
        s['pruned'] += len(pruned)
        s['prune_car'] += list(pruned.values())
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('root', help='directory containing run directories')
    ap.add_argument('--out', default='', help='output directory (default: <root>/plots)')
    ap.add_argument('--baseline', default='none', help='config_name of the no-memory baseline')
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])

    runs = load_runs(args.root)
    if not runs:
        print(f'no runs found under {args.root}')
        return 1
    out_dir = os.path.expanduser(args.out or os.path.join(args.root, 'plots'))
    os.makedirs(out_dir, exist_ok=True)
    agg = aggregate(runs)
    lc = plot_learning_curves(agg, out_dir)
    sp = plot_speed_profiles(runs, out_dir)

    seeds = defaultdict(int)
    for r in runs:
        seeds[r['config']] += 1
    rows, md = [], ['# Fleet Memory results', '',
                    f'Runs: {len(runs)} ({", ".join(f"{c}: {n} seeds" for c, n in sorted(seeds.items()))})', '']
    base = agg.get(args.baseline)
    md.append('| config | metric | car 1 | cars ≥ 2 | vs baseline (cars ≥ 2) |')
    md.append('|---|---|---|---|---|')
    for cfg in sorted(agg):
        for m, label in METRICS:
            series = agg[cfg][m]
            if not series:
                continue
            c1 = mean_ci(series.get(1, []))
            later = later_cars_mean(series)
            rel = ''
            if base is not None and base[m]:
                b = later_cars_mean(base[m])[0]
                if b and not math.isnan(b) and b != 0:
                    rel = f'{100.0 * (later[0] - b) / abs(b):+.1f}%'
            rows.append({'config': cfg, 'metric': m, 'car1_mean': c1[0], 'car1_ci95': c1[1],
                         'later_mean': later[0], 'later_ci95': later[1], 'vs_baseline': rel})
            md.append(f'| {cfg} | {label} | {c1[0]:.3f} ± {c1[1]:.3f} | '
                      f'{later[0]:.3f} ± {later[1]:.3f} | {rel} |')

    if base is not None:
        md += ['', '## Success criteria (cars ≥ 2 vs. baseline)', '']
        for cfg in sorted(agg):
            if cfg == args.baseline:
                continue

            def red(m):
                b = later_cars_mean(base[m])[0]
                v = later_cars_mean(agg[cfg][m])[0]
                return float('nan') if not b else 100.0 * (b - v) / b
            nm, hb = red('near_miss_rate'), red('hard_brakes_per_encounter')
            at = -red('arrival_time_s')
            ok = nm >= 50 and hb >= 50 and at < 10
            md.append(f'- **{cfg}**: near-miss reduction {nm:.1f}%, hard-brake reduction {hb:.1f}%, '
                      f'arrival-time increase {at:.1f}% -> {"TARGET MET" if ok else "target not met"}')

    acc = vlm_accuracy(runs)
    md += ['', '## Memory records vs. ground truth', '',
           '| config | records | matched | type accuracy | VLM records | VLM type accuracy |',
           '|---|---|---|---|---|---|']
    for cfg, a in sorted(acc.items()):
        md.append(f'| {cfg} | {a["records"]} | {a["matched_to_gt"]} | {a["type_accuracy"]:.3f} | '
                  f'{a["vlm_records"]} | {a["vlm_type_accuracy"]:.3f} |')
    cl = cleanup_stats(runs)
    if cl:
        md += ['', '## False-memory clean-up (E3)', '']
        for cfg, s in sorted(cl.items()):
            mean_car = np.mean(s['prune_car']) if s['prune_car'] else float('nan')
            md.append(f'- {cfg}: {s["pruned"]}/{s["injected"]} injected memories pruned '
                      f'(mean car index at pruning: {mean_car:.1f})')
    md += ['', f'![learning curves]({os.path.basename(lc)})'] + \
        [f'![speed profile]({os.path.basename(p)})' for p in sp]

    with open(os.path.join(out_dir, 'summary.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ['config'])
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(out_dir, 'summary.md'), 'w') as f:
        f.write('\n'.join(md) + '\n')
    print('\n'.join(md))
    print(f'\nPlots and summary written to {out_dir}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
