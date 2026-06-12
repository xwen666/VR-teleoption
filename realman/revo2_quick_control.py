#!/usr/bin/env python3
"""
Minimal Revo2 hand controller for quick bring-up.

Examples:
  python3 revo2_quick_control.py --action open
  python3 revo2_quick_control.py --action close
  python3 revo2_quick_control.py --action grasp --hold-seconds 1.5
  python3 revo2_quick_control.py --port /dev/ttyUSB0 --slave-id 0x7e --action custom \
      --positions 400,0,1000,1000,1000,1000 --speeds 400,400,800,800,800,800
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
REVO2_DIR = os.path.join(ROOT_DIR, "stark-serialport-example", "python", "revo2")
if REVO2_DIR not in sys.path:
    sys.path.insert(0, REVO2_DIR)

from revo2_utils import libstark, logger, open_modbus_revo2  # noqa: E402


DEFAULT_BAUDRATE_MAP = {
    115200: "Baud115200",
    57600: "Baud57600",
    19200: "Baud19200",
    460800: "Baud460800",
    1000000: "Baud1Mbps",
    2000000: "Baud2Mbps",
    5000000: "Baud5Mbps",
}

PRESET_POSITIONS = {
    "open": [0, 0, 0, 0, 0, 0],
    "close": [500, 500, 1000, 1000, 1000, 1000],
    "grasp": [400, 0, 1000, 1000, 1000, 1000],
}


def parse_slave_id(value: str) -> int:
    return int(value, 0)


def parse_vector(raw: str) -> list[int]:
    values = [int(item.strip(), 0) for item in raw.split(",") if item.strip()]
    if len(values) != 6:
        raise argparse.ArgumentTypeError("Expected 6 comma-separated integers")
    return values


def baudrate_enum(baudrate: int):
    enum_name = DEFAULT_BAUDRATE_MAP.get(baudrate)
    if enum_name is None or not hasattr(libstark.Baudrate, enum_name):
        supported = ", ".join(str(item) for item in DEFAULT_BAUDRATE_MAP)
        raise ValueError(f"Unsupported baudrate {baudrate}. Supported: {supported}")
    return getattr(libstark.Baudrate, enum_name)


def resolve_positions(args: argparse.Namespace) -> list[int]:
    if args.action == "custom":
        if args.positions is None:
            raise ValueError("--positions is required when --action custom")
        return args.positions
    return PRESET_POSITIONS[args.action]


def resolve_speeds(args: argparse.Namespace) -> list[int]:
    if args.speeds is not None:
        return args.speeds
    return [args.speed] * 6


async def open_client(args: argparse.Namespace):
    if args.port is None and args.slave_id is None:
        logger.info("Auto-detecting Revo2 hand over Modbus/RS485")
        return await open_modbus_revo2(port_name=None)

    if args.port is None or args.slave_id is None:
        raise ValueError("Manual mode requires both --port and --slave-id")

    baudrate = baudrate_enum(args.baudrate)
    logger.info(
        "Opening Revo2 manually: port=%s baudrate=%s slave_id=0x%02x",
        args.port,
        args.baudrate,
        args.slave_id,
    )
    client = await libstark.modbus_open(args.port, baudrate)
    device_info = await client.get_device_info(args.slave_id)
    if not device_info:
        raise RuntimeError(
            f"Failed to read device info from port={args.port} slave_id=0x{args.slave_id:02x}"
        )
    logger.info("Device info: %s", device_info.description)
    return client, args.slave_id


async def main() -> int:
    parser = argparse.ArgumentParser(description="Quick Revo2 controller")
    parser.add_argument(
        "--action",
        choices=["open", "close", "grasp", "custom"],
        default="grasp",
        help="Preset action to send",
    )
    parser.add_argument("--port", help="Serial port, for example /dev/ttyUSB0")
    parser.add_argument(
        "--slave-id",
        type=parse_slave_id,
        help="Modbus slave id, for example 0x7e or 126",
    )
    parser.add_argument(
        "--baudrate",
        type=int,
        default=460800,
        help="RS485 baudrate for manual mode",
    )
    parser.add_argument(
        "--positions",
        type=parse_vector,
        help="6 comma-separated target positions in normalized mode",
    )
    parser.add_argument(
        "--speeds",
        type=parse_vector,
        help="6 comma-separated target speeds in normalized mode",
    )
    parser.add_argument(
        "--speed",
        type=int,
        default=600,
        help="Default speed used when --speeds is not provided",
    )
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=1.0,
        help="How long to wait after sending the command",
    )
    args = parser.parse_args()

    client = None
    try:
        client, slave_id = await open_client(args)
        await client.set_finger_unit_mode(slave_id, libstark.FingerUnitMode.Normalized)

        positions = resolve_positions(args)
        speeds = resolve_speeds(args)
        logger.info("Sending positions=%s speeds=%s", positions, speeds)
        await client.set_finger_positions_and_speeds(slave_id, positions, speeds)
        await asyncio.sleep(args.hold_seconds)

        status = await client.get_motor_status(slave_id)
        logger.info("Motor status: %s", status.description)
        return 0
    finally:
        if client is not None:
            libstark.modbus_close(client)


if __name__ == "__main__":
    if libstark is None:
        print("bc_stark_sdk is not installed.")
        print("Run: cd stark-serialport-example/python && pip3 install -r requirements.txt")
        sys.exit(1)

    try:
        raise SystemExit(asyncio.run(main()))
    except KeyboardInterrupt:
        logger.info("User interrupted")
        raise SystemExit(130)
