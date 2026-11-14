"""Small command-line tool to inspect / manipulate the running fleet memory.

    ros2 run fleet_memory memory_cli list
    ros2 run fleet_memory memory_cli query --x 10 --y -20 --radius 50 --text "pedestrian"
    ros2 run fleet_memory memory_cli inject --type lane_closure --x 10 --y 5 --desc "cones"
    ros2 run fleet_memory memory_cli reset
"""

import argparse
import math
import sys

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger

from fleet_msgs.msg import HazardMemory
from fleet_msgs.srv import QueryMemory, WriteMemory


def _call(node, cli, req, timeout=10.0):
    if not cli.wait_for_service(timeout_sec=timeout):
        print(f'service {cli.srv_name} not available', file=sys.stderr)
        sys.exit(1)
    fut = cli.call_async(req)
    rclpy.spin_until_future_complete(node, fut, timeout_sec=timeout)
    return fut.result()


def main(argv=None):
    ap = argparse.ArgumentParser(description='Fleet memory CLI')
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('list')
    q = sub.add_parser('query')
    q.add_argument('--x', type=float, default=0.0)
    q.add_argument('--y', type=float, default=0.0)
    q.add_argument('--radius', type=float, default=0.0)
    q.add_argument('--text', default='')
    i = sub.add_parser('inject')
    i.add_argument('--type', default='other')
    i.add_argument('--x', type=float, required=True)
    i.add_argument('--y', type=float, required=True)
    i.add_argument('--heading-deg', type=float, default=0.0)
    i.add_argument('--desc', default='')
    i.add_argument('--severity', type=int, default=3)
    i.add_argument('--confidence', type=float, default=0.0)
    sub.add_parser('reset')
    sub.add_parser('save')
    args = ap.parse_args(argv if argv is not None else rclpy.utilities.remove_ros_args(sys.argv)[1:])

    rclpy.init()
    node = Node('fleet_memory_cli')
    try:
        if args.cmd in ('list', 'query'):
            cli = node.create_client(QueryMemory, '/fleet_memory/query')
            req = QueryMemory.Request()
            if args.cmd == 'query':
                req.x, req.y, req.radius, req.query_text = args.x, args.y, args.radius, args.text
            res = _call(node, cli, req)
            print(f'{len(res.records)} records')
            for r, s, d in zip(res.records, res.similarities, res.distances):
                print(f'  {r.id} {r.hazard_type:20s} sev={r.severity} c={r.confidence:.2f} '
                      f'n={r.count} ref={r.refutations} src={r.source:8s} '
                      f'pos=({r.x:.1f},{r.y:.1f}) d={d:.1f} sim={s:.2f}\n      "{r.description}"')
        elif args.cmd == 'inject':
            cli = node.create_client(WriteMemory, '/fleet_memory/write')
            req = WriteMemory.Request()
            m = HazardMemory()
            m.hazard_type, m.x, m.y = args.type, args.x, args.y
            m.heading = math.radians(args.heading_deg)
            m.description, m.severity = args.desc, args.severity
            m.confidence, m.source, m.trigger = args.confidence, 'injected', 'injected'
            m.lane, m.reporter_id = 'unknown', 'cli'
            req.record, req.allow_merge = m, False
            res = _call(node, cli, req)
            print(f'{res.message} id={res.id} c={res.confidence:.2f}')
        else:
            cli = node.create_client(Trigger, f'/fleet_memory/{args.cmd}')
            res = _call(node, cli, Trigger.Request())
            print(res.message)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
