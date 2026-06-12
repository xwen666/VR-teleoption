#!/usr/bin/env python3

import logging
import sys
import signal
import threading
import time
import struct
from enum import Enum
from dataclasses import dataclass, field
from typing import List

from Robotic_Arm.rm_ctypes_wrap import rm_peripheral_read_write_params_t
from Robotic_Arm.rm_robot_interface import rm_thread_mode_e, RoboticArm


logger = logging.getLogger()

DEVICE_ID_LEFT_HAND = 126
DEVICE_ID_RIGHT_HAND = 126

LIMIT_V = 40  # 速度百分比系数，1~100

JOINTS_LEFT_AIO = {
    'O': [26.5, -7.4, -102.1, -31.7, -112.8, -109.0],
    'B': [52.78, -80.4, -60.0, -65.2, -100.4, -91.65],
    'A': [-60.28, -71.8, -96.64, -64.64, -98.83, -62.24],
    'OA': [
        [26.5, -7.4, -102.1, -31.7, -112.8, -109.0],
        [34.3, -24.28, -95.7, -58.18, -113.8, -84.51],
        [-25.53, -36.0, -95.6, -66.0, -98.8, -78.2],
        [-60.28, -71.8, -96.64, -64.64, -98.83, -62.24],
    ],
    'AO': [
        [-60.28, -71.8, -96.64, -64.64, -98.83, -62.24],
        [-25.53, -36.0, -95.6, -66.0, -98.8, -78.2],
        [34.3, -24.28, -95.7, -58.18, -113.8, -84.51],
        [26.5, -7.4, -102.1, -31.7, -112.8, -109.0],
    ],
    'BO': [
        [52.78, -80.4, -60.0, -65.2, -100.4, -91.65],
        [55.54, -77.96, -51.06, -64.4, -107.93, -105.44],
        [56.17, -63.96, -53.24, -64.98, -104.92, -92.62],
        [61.68, -37.25, -80.12, -67.70, -91.60, -96.56],
        [34.256, -5.56, -105.15, -36.17, -111.44, -72.9],
        [26.5, -7.4, -102.1, -31.7, -112.8, -109.0],
    ],
    'OB': [
        [26.5, -7.4, -102.1, -31.7, -112.8, -109.0],
        [34.256, -5.56, -105.15, -36.17, -111.44, -72.9],
        [98.11, -10.97, -98.25, -39.79, -106.47, -60.0],
        [56.17, -63.96, -53.24, -64.98, -104.92, -92.62],
        [55.54, -77.96, -51.06, -64.4, -107.93, -105.44],
        [52.78, -80.4, -60.0, -65.2, -100.4, -91.65],
    ],
    'BA': [
        [52.78, -80.4, -60.0, -65.2, -100.4, -91.65],
        [55.54, -77.96, -51.06, -64.4, -107.93, -105.44],
        [61.68, -37.25, -80.12, -67.70, -91.60, -96.56],
        [78.16, -24.38, -87.49, -56.54, -103.16, -95.37],
        [-40.7, -30.40, -71.7, -28.2, -117.8, -90.0],
        [-40.74, -1.23, -134.98, -48.96, -80.74, -70.24],
        [-60.28, -71.8, -96.64, -64.64, -98.83, -62.24],
    ],
    'AB': [
        [-60.28, -71.8, -96.64, -64.64, -98.83, -62.24],
        [-25.53, -36.0, -95.6, -66.0, -98.8, -78.2],
        [34.256, -5.56, -105.15, -36.17, -111.44, -72.9],
        [61.68, -37.25, -80.12, -67.70, -91.60, -96.56],
        [56.17, -63.96, -53.24, -64.98, -104.92, -92.62],
        [55.54, -77.96, -51.06, -64.4, -107.93, -105.44],
        [52.78, -80.4, -60.0, -65.2, -100.4, -91.65],
    ],
}

# 6d-actor, deg
JOINTS_LEFT = {
    'O': [28.0, 0.0, -100.0, -28.0, -100.0, -100.0],
    'B': [6.0, -74.0, -80.0, -60.0, -96.0, -45.0],
    'A': [-167.0, -74.0, -87.0, -103.0, -81.0, -56.0],
    'OA': [
        [28.0, 0.0, -100.0, -28.0, -100.0, -100.0],
        [-64.0, -17.0, -96.0, -65.0, -111.0, -98.0],
        [-164.0, -38.0, -108.0, -104.0, -80.0, -71.0],
        [-167.0, -74.0, -87.0, -103.0, -81.0, -56.0],
    ],
    'OB': [
        [28.0, 0.0, -100.0, -28.0, -100.0, -100.0],
        [6.0, -30.0, -86.0, -40.0, -100.0, -100.0],
        [6.0, -74.0, -80.0, -60.0, -96.0, -45.0],
        [52.78, -80.4, -60.0, -65.2, -100.4, -91.65],
    ],
    'BA': [
        [6.0, -74.0, -80.0, -60.0, -96.0, -45.0],
        [6.0, -48.0, -94.0, -58.0, -100.0, -56.0],
        [-25.0, -27.0, -113.0, -47.0, -112.0, -78.0],
        [-114.0, -34.0, -113.0, -70.0, -96.0, -78.0],
        [-164.0, -64.0, -92.0, -102.0, -81.0, -63.0],
        [-167.0, -74.0, -87.0, -103.0, -81.0, -56.0],
    ],
    'AB': [
        [-167.0, -74.0, -87.0, -103.0, -81.0, -56.0],
        [-164.0, -38.0, -108.0, -104.0, -80.0, -71.0],
        [-114.0, -34.0, -113.0, -70.0, -96.0, -78.0],
        [-25.0, -27.0, -113.0, -47.0, -112.0, -78.0],
        [6.0, -48.0, -94.0, -58.0, -100.0, -56.0],
        [6.0, -74.0, -80.0, -60.0, -96.0, -45.0],
    ],
    'AO': [
        [-167.0, -74.0, -87.0, -103.0, -81.0, -56.0],
        [-164.0, -38.0, -108.0, -104.0, -80.0, -71.0],
        [-64.0, -17.0, -96.0, -65.0, -111.0, -98.0],
        [28.0, 0.0, -100.0, -28.0, -100.0, -100.0],
    ],
    'BO': [
        [6.0, -74.0, -80.0, -60.0, -96.0, -45.0],
        [6.0, -48.0, -94.0, -58.0, -100.0, -56.0],
        [28.0, 0.0, -100.0, -28.0, -100.0, -100.0],
    ],
}

JOINTS_RIGHT = {
    'O': [68.0, 2.0, 100.0, 27.0, 120.0, -40.0],
    'A': [90.0, 72.0, 98.0, 80.0, 86.0, -76.0],
    'B': [164.0, 57.0, 91.0, 97.0, 80.0, -56.0],
    'OA': [
        [68.0, 2.0, 100.0, 27.0, 120.0, -40.0],
        [64.0, 56.0, 100.0, 52.0, 86.0, -64.0],
        [90.0, 72.0, 98.0, 80.0, 86.0, -76.0],
    ],
    'AO': [
        [90.0, 72.0, 98.0, 80.0, 86.0, -76.0],
        [64.0, 56.0, 100.0, 52.0, 86.0, -64.0],
        [68.0, 2.0, 100.0, 27.0, 120.0, -40.0],
    ],
    'AB': [
        [90.0, 72.0, 98.0, 80.0, 86.0, -76.0],
        [117.0, 28.0, 100.0, 62.0, 116.0, -76.0],
        [140.0, 50.0, 94.0, 68.0, 100.0, -78.0],
        [164.0, 57.0, 91.0, 97.0, 80.0, -56.0],
    ],
    'BA': [
        [164.0, 57.0, 91.0, 97.0, 80.0, -56.0],
        [140.0, 50.0, 94.0, 68.0, 100.0, -78.0],
        [117.0, 28.0, 100.0, 62.0, 116.0, -76.0],
        [90.0, 72.0, 98.0, 80.0, 86.0, -76.0],
    ],
    'BO': [
        [164.0, 57.0, 91.0, 97.0, 80.0, -56.0],
        [164.0, 37.0, 92.0, 88.0, 102.0, -56.0],
        [122.0, 20.0, 104.0, 56.0, 122.0, -56.0],
        [68.0, 2.0, 100.0, 27.0, 120.0, -40.0],
    ],
    'OB': [
        [68.0, 2.0, 100.0, 27.0, 120.0, -40.0],
        [122.0, 20.0, 104.0, 56.0, 122.0, -56.0],
        [164.0, 37.0, 92.0, 88.0, 102.0, -56.0],
        [164.0, 57.0, 91.0, 97.0, 80.0, -56.0],
    ],
}

JOINTS_LEFT_X90 = {
    'O': [40.0, -57.0, -120.0, -134.0, 87.0, -142.0],
    'B': [-52.36, 74.4, 92.8, -41.75, 75.7, 21.7],
    'A': [-35.3, -66.2, 94.8, 41.1, 49.25, 17.7],
    'OA': [
        [40.0, -57.0, -120.0, -134.0, 87.0, -142.0],
        [41.0, -70.0, -56.4, -100.0, 113.0, -122.0, 'j'],
        [36.0, -72.0, -28.2, -28.8, 88.0, -34.0, 'j'],
        [32.2, -80.3, 34.0, -50.8, 101.2, 26.4, 'j'],
        [2.7, -106.8, 90.8, 7.8, 66.0, 65.4],
        [-16.7, -62.8, 82.14, 28.2, 55.8, 28.2],
        [-35.3, -66.2, 94.8, 41.1, 49.25, 17.7],
    ],
    'AO': [
        [-35.3, -66.2, 94.8, 41.1, 49.25, 17.7],
        [-16.7, -62.8, 82.14, 28.2, 55.8, 28.2],
        [2.7, -106.8, 90.8, 7.8, 66.0, 65.4],
        [32.2, -80.3, 34.0, -50.8, 101.2, 26.4, 'j'],
        [36.0, -72.0, -28.2, -28.8, 88.0, -34.0, 'j'],
        [41.0, -70.0, -56.4, -100.0, 113.0, -122.0, 'j'],
        [40.0, -57.0, -120.0, -134.0, 87.0, -142.0],
    ],
    'BO': [
        [-52.36, 74.4, 92.8, -41.75, 75.7, 21.7],
        [-44.7, 63.7, 72.0, -24.1, 81.0, 10.7, 'j'],
        [-59.4, 54.5, 74.6, -16.8, 82.7, 1.7, 'j'],
        [53.7, -71.6, -64.5, -121.1, 110.2, -80.2, 'j'],
        [40.0, -57.0, -120.0, -134.0, 87.0, -142.0],
    ],
    'OB': [
        [40.0, -57.0, -120.0, -134.0, 87.0, -142.0],
        [53.7, -71.6, -64.5, -121.1, 110.2, -80.2, 'j'],
        [-59.4, 54.5, 74.6, -16.8, 82.7, 1.7, 'j'],
        [-44.7, 63.7, 72.0, -24.1, 81.0, 10.7, 'j'],
        [-52.36, 74.4, 92.8, -41.75, 75.7, 21.7],
    ],
    'BA': [
        [-52.36, 74.4, 92.8, -41.75, 75.7, 21.7],
        [-54.2, 59.0, 94.3, -34.6, 76.4, 12.0, 'j'],
        [-45.4, 67.5, 63.3, -22.8, 104.7, 4.0, 'j'],
        [-44.7, 45.6, 64.3, -5.8, 101.7, 17.6, 'j'],
        [-24.8, -16.2, 94.2, 4.2, 63.4, 17.2, 'j'],
        [-35.3, -66.2, 94.8, 41.1, 49.25, 17.7],
    ],
    'AB': [
        [-35.3, -66.2, 94.8, 41.1, 49.25, 17.7],
        [-24.8, -16.2, 94.2, 4.2, 63.4, 17.2, 'j'],
        [-44.7, 45.6, 64.3, -5.8, 101.7, 17.6, 'j'],
        [-45.4, 67.5, 63.3, -22.8, 104.7, 4.0, 'j'],
        [-54.2, 59.0, 94.3, -34.6, 76.4, 12.0, 'j'],
        [-52.36, 74.4, 92.8, -41.75, 75.7, 21.7],
    ],
}

JOINTS = JOINTS_LEFT_X90

OFFSET_TWEAK = {
    'A': [-0.017, -0.05, 0.0, 0, 0, 0],
    'B': [0.009, -0.021, 0.0, 0, 0, 0],
}


def config_logging(loglevel: int | str = logging.DEBUG):
    console_handler = logging.StreamHandler()
    console_handler.setLevel(loglevel)
    console_handler.setFormatter(
        logging.Formatter(
            fmt='{asctime}.{msecs:03.0f} {levelname[0]}: {message}',
            style='{',
            datefmt='%H%M%S',
        )
    )

    logger = logging.getLogger()
    logger.setLevel(loglevel)
    logger.addHandler(console_handler)


class Position(Enum):
    A = 1
    B = 2
    O = 3  # noqa: E741


@dataclass
class State:
    loaded: bool = False
    load_pose: Position = Position.A
    cur_pose: Position = Position.O

    paused: bool = False
    stopped: bool = False
    threads: List[threading.Thread] = field(default_factory=list)


class RobotArmController:
    def __init__(self, ip, port=0, level=3, mode=2):
        """Initialize and connect to the robotic arm.

        Args:
            ip (str): IP address of the robot arm.
            port (int): Port number.
            level (int, optional): Connection level. Defaults to 3.
            mode (int, optional): Thread mode (0: single, 1: dual, 2: triple). Defaults to 2.
        """
        self.state = State(loaded=False, load_pose=Position.A, cur_pose=Position.O)

        self.thread_mode = rm_thread_mode_e(mode)
        self.robot = RoboticArm(self.thread_mode)
        self.handle = self.robot.rm_create_robot_arm(ip, port, level)

        if self.handle.id == -1:
            logger.info('Failed to connect to the robot arm')
            sys.exit(1)
        else:
            logger.info(f'Successfully connected to the robot arm: {self.handle.id}')

        signal.signal(signal.SIGINT, self.handle_signal)
        signal.signal(signal.SIGTERM, self.handle_signal)
        signal.signal(signal.SIGUSR1, self.handle_signal)

    def handle_signal(self, signum, frame):
        # 急停
        if signum in (signal.SIGINT, signal.SIGTERM):
            logger.warning('emergency stop!')
            self.robot.rm_set_arm_stop()
            self.state.stopped = True
            for th in self.state.threads:
                th.join()
        # 暂停/恢复
        elif signum == signal.SIGUSR1:
            logger.warning(f'toggle pause: {self.state.paused} -> {not self.state.paused}')
            if self.state.paused:
                self.state.paused = False
                self.robot.rm_set_arm_continue()
            else:
                self.state.paused = True
                self.robot.rm_set_arm_pause()
        else:
            logger.warning(f'unkown signal: {signum}')

    def get_arm_software_info(self):
        software_info = self.robot.rm_get_arm_software_info()
        if software_info[0] == 0:
            print('================== Arm Software Information ==================')
            print('Arm Model: ', software_info[1]['product_version'])
            print(
                'Algorithm Library Version: ', software_info[1]['algorithm_info']['version']
            )
            print(
                'Control Layer Software Version: ', software_info[1]['ctrl_info']['version']
            )
            print('Dynamics Version: ', software_info[1]['dynamic_info']['model_version'])
            print(
                'Planning Layer Software Version: ', software_info[1]['plan_info']['version']
            )
            print('==============================================================')
        else:
            print(
                'Failed to get arm software information, Error code: ',
                software_info[0],
                '',
            )

    def get_install_pose(self) -> dict:
        return self.robot.rm_get_install_pose()

    def disconnect(self):
        handle = self.robot.rm_delete_robot_arm()
        if handle == 0:
            logger.info('Successfully disconnected from the robot arm')
        else:
            logger.info('Failed to disconnect from the robot arm')

    def set_modbus_mode(self, port=0, baudrate=115200, timeout=1):
        """Set the Modbus RTU mode.

        Args:
            port (int): Communication port.
                0 for controller RS485 port as RTU master,
                1 for end interface board RS485 port as RTU master,
                2 for controller RS485 port as RTU slave.
            baudrate (int): Baud rate. Supports 9600, 115200, 460800.
            timeout (int): Timeout duration in hundred milliseconds.
                For all read and write commands to Modbus devices,
                if no response data is returned within the specified timeout period,
                a timeout error is returned. The timeout cannot be 0;
                if set to 0, the robot arm will configure it as 1.

        Returns:
            None
        """
        set_result = self.robot.rm_set_modbus_mode(port, baudrate, timeout)
        if set_result == 0:
            logger.info('Successfully set the Modbus mode')
        else:
            logger.info('Failed to set the Modbus mode')

    def close_modbus_mode(self, port=0):
        """Close the Modbus RTU mode.

        Args:
            port (int): Communication port.
                0 for controller RS485 port,
                1 for end interface board RS485 port,
                3 for controller ModbusTCP device.

        Returns:
            None
        """
        close_result = self.robot.rm_close_modbus_mode(port)
        if close_result == 0:
            logger.info('Successfully closed the Modbus mode')
        else:
            logger.info('Failed to close the Modbus mode')

    def watch_cur_arm_state(self):
        def get_state():
            while not self.state.stopped:
                state = self.robot.rm_get_current_arm_state()
                joint = [round(x, 2) for x in state[1]['joint']]
                pose = [round(x, 4) for x in state[1]['pose']]
                logger.debug(f'joint: {joint}, pose: {pose}')
                time.sleep(2)

        th = threading.Thread(target=get_state)
        th.start()
        self.state.threads.append(th)

    def write_registers(self, addr: int, data: list[int]) -> int:
        logger.info(f'modbus write: addr:{addr}, len:{len(data)}, {data}')
        write_params = rm_peripheral_read_write_params_t(
            port=1, address=addr, device=DEVICE_ID_LEFT_HAND, num=len(data)
        )
        hex_data = [x for num in data for x in struct.pack('>h', num)]
        return self.robot.rm_write_registers(write_params, hex_data)

    def write_hand_registers_pos_time(self, data: List[int]):
        # 位置时间控制，受限于一次 10 个寄存器不太有解
        ret = self.write_registers(1010, data[:10])
        # 伸展小手指，不能用前面的写了，会影响 1010 指令
        ret += self.write_registers(1052, [5] + data[10:])
        return ret

    def set_hand_unit_mode(self, mode: int = 0):
        write_params = rm_peripheral_read_write_params_t(
            port=1, address=937, device=DEVICE_ID_LEFT_HAND
        )
        return self.robot.rm_write_single_register(write_params, mode)

    def hand_grasp(self) -> int:
        logger.info('hand grasp ..')

        # 从引导点开始算抓取动作
        offset = OFFSET_TWEAK[self.state.load_pose.name]

        # Y+ 直线向前, Z- 竖直向下
        self.movel_offset(offset, v=28)

        ret = self.write_hand_registers_pos_time(
            [160, 600, 900, 560, 480, 560, 480, 560, 480, 560, 360, 500]
        )

        if ret == 0:
            self.state.loaded = True

        time.sleep(1.2)

        # 原路返回回到引导点
        self.movel_offset([-x for x in offset], v=40)

        return ret

    def hand_release(self) -> int:
        logger.info('hand release ..')

        offset = OFFSET_TWEAK[self.state.load_pose.name]

        # 从引导点开始算抓取动作: z-
        self.movel_offset(offset, v=28)

        ret = self.write_hand_registers_pos_time(
            [50, 700, 820, 500, 110, 500, 120, 500, 120, 500, 110, 440]
        )

        if ret == 0:
            self.state.loaded = False

        time.sleep(1.2)

        # 原路返回回到引导点
        self.movel_offset([-x for x in offset], v=40)

        def delay_hand_fist():
            time.sleep(4)
            self.write_hand_registers_pos_time(
                [800, 600, 0, 500, 980, 500, 980, 500, 980, 500, 980, 500]
            )

        threading.Thread(target=delay_hand_fist).start()

        return ret

    def get_pose_from_joint(self, joint: List[float]) -> List[float]:
        return self.robot.rm_algo_forward_kinematics(joint, 1)

    def moves_traj(self, joints: List):
        for idx in range(0, len(joints)):
            if idx < len(joints) - 1 and joints[idx][6:] == []:
                pose = self.get_pose_from_joint(joints[idx][:6])
                ret = self.robot.rm_moves(pose, v=LIMIT_V, r=4, block=1, connect=1)
                if ret != 0:
                    logger.warning(f'failed to move poses[{idx}]')
            else:
                ret = self.robot.rm_movej(joints[idx][:6], v=LIMIT_V, r=0, block=1, connect=0)
                if ret != 0:
                    logger.warning(f'failed to move poses[{idx}]')

    def movel_offset(
        self, offset: List[float], v: int = 40, r: int = 0, connect: int = 0, block: int = 1
    ):
        # sdk 的接口不行，还得自己上
        pose = self.robot.rm_get_current_arm_state()[1]['pose']
        for i in range(len(offset)):
            pose[i] += offset[i]

        self.robot.rm_movel(pose, v, r, connect, block)

    def do_move_to_O(self):
        self.state.cur_pose = Position.O
        if self.state.loaded:
            self.state.load_pose = self.state.cur_pose

        self.write_hand_registers_pos_time(
            [960, 600, 60, 600, 980, 600, 980, 600, 980, 600, 980, 600]
        )

        # 初始原点
        self.robot.rm_movej(JOINTS['O'], v=LIMIT_V, r=0, connect=0, block=1)

        logger.info('init done, delay seconds ..')
        time.sleep(2)

    def do_move_AO(self):
        self.state.cur_pose = Position.O
        if self.state.loaded:
            self.state.load_pose = self.state.cur_pose

        self.moves_traj(JOINTS['AO'])

    def do_move_OA(self):
        self.state.cur_pose = Position.A
        if self.state.loaded:
            self.state.load_pose = self.state.cur_pose

        self.moves_traj(JOINTS['OA'])

    def do_move_AB(self):
        self.state.cur_pose = Position.B
        if self.state.loaded:
            self.state.load_pose = self.state.cur_pose

        self.moves_traj(JOINTS['AB'])

    def do_move_BA(self):
        self.state.cur_pose = Position.A
        if self.state.loaded:
            self.state.load_pose = self.state.cur_pose

        self.moves_traj(JOINTS['BA'])

    def do_move_BO(self):
        self.state.cur_pose = Position.O
        if self.state.loaded:
            self.state.load_pose = self.state.cur_pose

        self.moves_traj(JOINTS['BO'])

    def do_move_OB(self):
        self.state.cur_pose = Position.B
        if self.state.loaded:
            self.state.load_pose = self.state.cur_pose

        self.moves_traj(JOINTS['OB'])

    def do_move_to_target(self, target: str):
        move_map = {
            Position.A: {
                Position.B: self.do_move_AB,
                Position.O: self.do_move_AO,
            },
            Position.B: {
                Position.A: self.do_move_BA,
                Position.O: self.do_move_BO,
            },
            Position.O: {
                Position.A: self.do_move_OA,
                Position.B: self.do_move_OB,
            },
        }

        target_pose = self.state.cur_pose

        if target == 'load':
            target_pose = self.state.load_pose

            def delay_hand_open():
                time.sleep(4.6)
                self.write_hand_registers_pos_time(
                    [50, 700, 860, 500, 110, 500, 120, 500, 120, 500, 110, 440]
                )

            threading.Thread(target=delay_hand_open).start()

        elif target == 'unload':
            target_pose = Position.A if self.state.load_pose != Position.A else Position.B
        elif target == 'O':
            target_pose = Position.O

        if self.state.cur_pose != target_pose:
            logger.info(f'move: {self.state.cur_pose.name} -> {target_pose.name} ..')
            move_map[self.state.cur_pose][target_pose]()


def main():
    config_logging()

    ctr = RobotArmController('192.168.1.18', 8080, 3)
    ctr.get_arm_software_info()

    logger.info(f'install pose: {ctr.get_install_pose()}')

    ctr.set_hand_unit_mode(0)
    ctr.set_modbus_mode(1, 460800, 20)

    ctr.watch_cur_arm_state()
    ctr.do_move_to_O()

    while not ctr.state.stopped:
        time.sleep(1)

        if ctr.state.paused:
            continue

        ctr.do_move_to_target('load')

        time.sleep(0.2)
        if ctr.hand_grasp() != 0:
            logger.warning('failed to grasp')

        ctr.do_move_to_target('unload')

        time.sleep(0.5)
        if ctr.hand_release() != 0:
            logger.warning('failed to release')

        ctr.do_move_to_target('O')

    ctr.close_modbus_mode(1)
    ctr.disconnect()


if __name__ == '__main__':
    main()
