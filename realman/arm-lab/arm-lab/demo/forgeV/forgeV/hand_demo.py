import argparse
import time
import threading

from .arm import RobotArmController
from .utils.log import get_logger_spdlog

logger = get_logger_spdlog()


def disconnect_with_timeout(ctr: RobotArmController, timeout: float = 2.0):
    result = {'done': False}

    def worker():
        try:
            ctr.disconnect()
        finally:
            result['done'] = True

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join(timeout)
    if not result['done']:
        logger.warning('disconnect() is still blocking, exit without waiting further')


def main():
    parser = argparse.ArgumentParser(description='Revo2 hand demo through RealMan Modbus RTU')
    parser.add_argument('--ip', default='192.168.1.18')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--device-id', type=int, default=126)
    parser.add_argument(
        '--action',
        choices=['status', 'open', 'close', 'grasp'],
        default='status',
    )
    parser.add_argument('--speed', type=int, default=600)
    parser.add_argument('--hold-seconds', type=float, default=1.0)
    args = parser.parse_args()

    ctr = RobotArmController(args.ip, args.port, 3, hand_device_id=args.device_id)
    try:
        ctr.set_modbus_mode(1, 460800, 20)
        ctr.set_hand_unit_mode(0)

        if args.action == 'open':
            ret = ctr.open_hand(speed=args.speed)
            logger.info(f'open_hand ret={ret}')
            time.sleep(args.hold_seconds)
        elif args.action == 'close':
            ret = ctr.close_hand(speed=args.speed)
            logger.info(f'close_hand ret={ret}')
            time.sleep(args.hold_seconds)
        elif args.action == 'grasp':
            ret = ctr.grasp_hand(speed=args.speed)
            logger.info(f'grasp_hand ret={ret}')
            time.sleep(args.hold_seconds)

        status = ctr.read_hand_status()
        logger.info(f'hand status: {status}')
    finally:
        ctr.close_modbus_mode(1)
        disconnect_with_timeout(ctr)


if __name__ == '__main__':
    main()
