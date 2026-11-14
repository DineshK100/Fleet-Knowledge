"""Stretch goal: run the VLM event writer on real dash-camera images.

Usage:
    ros2 run hazard_writer dashcam_eval --images DIR --labels labels.csv \
        --base-url http://localhost:11434/v1 --model qwen2.5vl:7b --out results.csv

``labels.csv`` has columns ``filename,hazard_type`` (hazard_type in
pedestrian_crossing | lane_closure | road_damage | stalled_vehicle | other).
Images can be any public dash-cam hazard dataset (e.g. a pothole dataset).
The script reports the hazard-type accuracy and a confusion matrix.
"""

import argparse
import csv
import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np

from fleet_memory.llm_client import ChatClient

from .vlm import HAZARD_TYPES, SYSTEM_PROMPT, build_user_prompt, encode_jpeg, parse_vlm_reply


def load_rgb(path):
    try:
        import cv2
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(path)
        return img[:, :, ::-1]
    except ImportError:
        from PIL import Image
        return np.asarray(Image.open(path).convert('RGB'))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--images', required=True)
    ap.add_argument('--labels', required=True)
    ap.add_argument('--base-url', default='http://localhost:11434/v1')
    ap.add_argument('--model', default='qwen2.5vl:7b')
    ap.add_argument('--api-key-env', default='OPENAI_API_KEY')
    ap.add_argument('--out', default='dashcam_eval.csv')
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])

    client = ChatClient(args.base_url, args.model, args.api_key_env, timeout_s=120)
    with open(args.labels) as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[:args.limit]

    results, confusion = [], defaultdict(Counter)
    for i, row in enumerate(rows):
        path = os.path.join(args.images, row['filename'])
        img = load_rgb(path)
        ctx = {'trigger': 'hard_brake', 'magnitude': 4.0, 'speed_kmh': 40,
               'obstacle_type': 'none', 'context': 'real dash-camera image'}
        t0 = time.time()
        try:
            reply = client.chat(SYSTEM_PROMPT, build_user_prompt(ctx), [encode_jpeg(img)])
            rec = parse_vlm_reply(reply) or {'hazard_type': 'other', 'description': ''}
        except Exception as exc:
            print(f'{row["filename"]}: VLM error {exc}', file=sys.stderr)
            rec = {'hazard_type': 'error', 'description': str(exc)}
        dt = time.time() - t0
        gt = row['hazard_type'].strip()
        confusion[gt][rec['hazard_type']] += 1
        results.append({'filename': row['filename'], 'gt': gt, 'pred': rec['hazard_type'],
                        'correct': int(gt == rec['hazard_type']), 'latency_s': f'{dt:.2f}',
                        'description': rec.get('description', '')})
        print(f'[{i + 1}/{len(rows)}] {row["filename"]}: gt={gt} pred={rec["hazard_type"]} '
              f'({dt:.1f}s) {rec.get("description", "")}')

    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()) if results else ['filename'])
        w.writeheader()
        w.writerows(results)
    acc = sum(r['correct'] for r in results) / max(1, len(results))
    print(f'\nAccuracy: {acc:.3f} over {len(results)} images')
    labels = list(HAZARD_TYPES)
    print('Confusion (rows = ground truth, cols = predicted):')
    print(' ' * 22 + ' '.join(f'{l[:10]:>10s}' for l in labels))
    for gt in labels:
        if gt in confusion:
            print(f'{gt:22s}' + ' '.join(f'{confusion[gt][p]:10d}' for p in labels))


if __name__ == '__main__':
    main()
