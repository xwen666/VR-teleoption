#!/usr/bin/env python3
from __future__ import annotations

import argparse
import selectors
import signal
import socket
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class FanoutTarget:
    label: str
    host: str
    port: int


def make_udp_listener(port: int) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", port))
    sock.setblocking(False)
    return sock


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Listen on one wrist/hand UDP port pair and fan out identical packets "
            "to separate RealMan and MuJoCo downstream ports."
        )
    )
    parser.add_argument("--listen-wrist-port", type=int, default=5005)
    parser.add_argument("--listen-hand-port", type=int, default=5010)
    parser.add_argument("--real-wrist-port", type=int, default=5105)
    parser.add_argument("--real-hand-port", type=int, default=5110)
    parser.add_argument("--mujoco-wrist-port", type=int, default=5205)
    parser.add_argument("--mujoco-hand-port", type=int, default=5210)
    parser.add_argument("--target-host", type=str, default="127.0.0.1")
    return parser.parse_args()


def build_targets(args: argparse.Namespace) -> Dict[int, List[FanoutTarget]]:
    targets: Dict[int, List[FanoutTarget]] = {
        int(args.listen_wrist_port): [],
        int(args.listen_hand_port): [],
    }
    if args.real_wrist_port > 0:
        targets[int(args.listen_wrist_port)].append(
            FanoutTarget("real_wrist", args.target_host, int(args.real_wrist_port))
        )
    if args.mujoco_wrist_port > 0:
        targets[int(args.listen_wrist_port)].append(
            FanoutTarget("mujoco_wrist", args.target_host, int(args.mujoco_wrist_port))
        )
    if args.real_hand_port > 0:
        targets[int(args.listen_hand_port)].append(
            FanoutTarget("real_hand", args.target_host, int(args.real_hand_port))
        )
    if args.mujoco_hand_port > 0:
        targets[int(args.listen_hand_port)].append(
            FanoutTarget("mujoco_hand", args.target_host, int(args.mujoco_hand_port))
        )
    return targets


def main() -> int:
    args = parse_args()
    port_targets = build_targets(args)

    selector = selectors.DefaultSelector()
    output_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    listeners: List[socket.socket] = []
    should_stop = False

    def request_stop(_signum: int, _frame) -> None:
        nonlocal should_stop
        should_stop = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    try:
        for listen_port in sorted(port_targets.keys()):
            listener = make_udp_listener(listen_port)
            selector.register(listener, selectors.EVENT_READ, data=listen_port)
            listeners.append(listener)

        print("v1 UDP fanout started")
        print(f"  listen wrist: 0.0.0.0:{args.listen_wrist_port}")
        print(f"  listen hand:  0.0.0.0:{args.listen_hand_port}")
        for listen_port in sorted(port_targets.keys()):
            targets = ", ".join(f"{t.label}@{t.host}:{t.port}" for t in port_targets[listen_port]) or "(none)"
            print(f"  {listen_port} -> {targets}")

        while not should_stop:
            events = selector.select(timeout=0.2)
            for key, _ in events:
                listener = key.fileobj
                listen_port = int(key.data)
                try:
                    payload, _ = listener.recvfrom(65535)
                except BlockingIOError:
                    continue
                for target in port_targets.get(listen_port, []):
                    output_sock.sendto(payload, (target.host, target.port))
        return 0
    finally:
        for listener in listeners:
            try:
                selector.unregister(listener)
            except Exception:
                pass
            listener.close()
        output_sock.close()


if __name__ == "__main__":
    raise SystemExit(main())
