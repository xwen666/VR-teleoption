from __future__ import annotations

import sys
import signal
import threading
import time
import math
import struct

from enum import Enum
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from Robotic_Arm.rm_ctypes_wrap import (
    rm_modbus_rtu_read_params_t,
    rm_modbus_rtu_write_params_t,
    rm_peripheral_read_write_params_t,
)
from Robotic_Arm.rm_robot_interface import rm_thread_mode_e, RoboticArm

from .models import Pos
from .utils.log import get_logger_spdlog
from .utils.singleton import singleton

if TYPE_CHECKING:
    from .realsense import RealsenseCamera
    from .yolo import YoloManager

logger = get_logger_spdlog()

DEVICE_ID_LEFT_HAND = 126
DEVICE_ID_RIGHT_HAND = 127

LIMIT_V = 20  # 速度百分比系数，1~100

# 预设扫描路径
JOINTS_LEFT_X90 = {
    'O': [-131.7, 55.2, 115.7, 36.0, 110.8, 240.4],
    'A': [-100.0, 92.0, -44.0, -47.0, -109.2, 226.7],
}

JOINTS = JOINTS_LEFT_X90

"""标定参数
  旋转矩阵: ..
  平移向量: ..
  四元数：[-0.00101968 -0.00029065  0.72155198 -0.69235945]
"""

rotation_matrix = [
    [-0.0412747, 0.99914726, -0.00106903],
    [-0.99914608, -0.04127661, -0.00183141],
    [-0.00187397, 0.00099252, 0.99999775],
]

translation_vector = [-0.08214561, 0.03649837, 0.01958647]

# 偏差
arm_hand_length = 0.07
arm_hand_angle = math.radians(80)  # 0度偏差，逆时针旋转 80 度


class Position(Enum):
    A = 1
    B = 2
    O = 3  # noqa: E741


@dataclass
class State:
    loaded: bool = False

    pos: Pos = field(default_factory=Pos)
    pos_ts: float = 0

    load_pose: Position = Position.A
    cur_pose: Position = Position.O

    paused: bool = False
    stopped: bool = False
    threads: list[threading.Thread] = field(default_factory=list)


@dataclass
class HandStatus:
    timestamp: float
    positions: list[int]
    speeds: list[int]
    currents: list[int]
    states: list[int]
    button: int
    unit_mode: int | None = None


@singleton
class RobotArmController:
    def __init__(self, ip, port=0, level=3, mode=2, hand_device_id=DEVICE_ID_LEFT_HAND):
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
        self.hand_device_id = hand_device_id
        self.hand_api_mode: str | None = None

        self.cam: RealsenseCamera | None = None
        self.yolo: YoloManager | None = None

        if self.handle.id == -1:
            logger.info('Failed to connect to the robot arm')
            sys.exit(1)
        else:
            logger.info(f'Successfully connected to the robot arm: {self.handle.id}')

        # signal.signal(signal.SIGINT, self.handle_signal)
        # signal.signal(signal.SIGTERM, self.handle_signal)
        # signal.signal(signal.SIGUSR1, self.handle_signal)

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

    def fetch_cur_arm_pos(self):
        state = self.robot.rm_get_current_arm_state()
        self.state.pos_ts = time.time()

        # joint = [round(x, 2) for x in state[1]['joint']]
        pose = [round(x, 5) for x in state[1]['pose']]

        self.state.pos = Pos(*pose)
        return self.state.pos

    def watch_cur_arm_state(self):
        def get_state():
            while not self.state.stopped:
                ts = time.time()
                # TODO: 实际的位资更新由相机线程持续同步，目前应该是 30hz,
                # 如果 50ms 后姿态还没更新，这里强制更新，但这样会导致相机数据与姿态不同步
                if ts > self.state.pos_ts + 0.05:
                    logger.warn(
                        'force fetch pos since last fetched pos ts delta: '
                        f'{(ts - self.state.pos_ts) * 1000:.3f}ms > 40ms ago'
                    )
                    self.fetch_cur_arm_pos()

                if int(ts) % 2 == 0:
                    logger.debug(f'pos: {self.state.pos}')

                time.sleep(1)

        th = threading.Thread(target=get_state)
        th.start()

        self.state.threads.append(th)

    def write_registers(self, addr: int, data: list[int]) -> int:
        logger.info(f'modbus write: addr:{addr}, len:{len(data)}, {data}')
        write_params = rm_peripheral_read_write_params_t(
            port=1, address=addr, device=self.hand_device_id, num=len(data)
        )
        hex_data = [x for num in data for x in struct.pack('>h', num)]
        return self.robot.rm_write_registers(write_params, hex_data)

    @staticmethod
    def _decode_int16(value: int) -> int:
        return value - 65536 if value > 32767 else value

    def write_hand_holding_registers(self, addr: int, data: list[int]) -> int:
        if self.hand_api_mode == 'legacy':
            return self._legacy_write_hand_holding_registers(addr, data)

        logger.info(f'hand write holding regs: addr:{addr}, len:{len(data)}, data:{data}')
        write_params = rm_modbus_rtu_write_params_t(
            address=addr,
            device=self.hand_device_id,
            type=1,
            num=len(data),
            data=data,
        )
        ret = self.robot.rm_write_modbus_rtu_registers(write_params)
        if ret == -4:
            logger.warning(
                'Fourth-gen Modbus RTU write API is unsupported on this controller, '
                'falling back to legacy peripheral API'
            )
            self.hand_api_mode = 'legacy'
            return self._legacy_write_hand_holding_registers(addr, data)

        if ret == 0:
            self.hand_api_mode = 'rtu4'
        return ret

    def read_hand_holding_registers(self, addr: int, num: int) -> tuple[int, list[int]]:
        if self.hand_api_mode == 'legacy':
            return self._legacy_read_hand_holding_registers(addr, num)

        read_params = rm_modbus_rtu_read_params_t(
            address=addr,
            device=self.hand_device_id,
            type=1,
            num=num,
        )
        ret, data = self.robot.rm_read_modbus_rtu_holding_registers(read_params)
        if ret == -4:
            logger.warning(
                'Fourth-gen Modbus RTU holding-read API is unsupported on this controller, '
                'falling back to legacy peripheral API'
            )
            self.hand_api_mode = 'legacy'
            return self._legacy_read_hand_holding_registers(addr, num)

        if ret == 0:
            self.hand_api_mode = 'rtu4'
        logger.info(f'hand read holding regs: addr:{addr}, len:{num}, ret:{ret}, data:{data}')
        return ret, data

    def read_hand_input_registers(self, addr: int, num: int) -> tuple[int, list[int]]:
        if self.hand_api_mode == 'legacy':
            return self._legacy_read_hand_input_registers(addr, num)

        read_params = rm_modbus_rtu_read_params_t(
            address=addr,
            device=self.hand_device_id,
            type=1,
            num=num,
        )
        ret, data = self.robot.rm_read_modbus_rtu_input_registers(read_params)
        if ret == -4:
            logger.warning(
                'Fourth-gen Modbus RTU input-read API is unsupported on this controller, '
                'falling back to legacy peripheral API'
            )
            self.hand_api_mode = 'legacy'
            return self._legacy_read_hand_input_registers(addr, num)

        if ret == 0:
            self.hand_api_mode = 'rtu4'
        logger.info(f'hand read input regs: addr:{addr}, len:{num}, ret:{ret}, data:{data}')
        return ret, data

    def _legacy_write_hand_holding_registers(self, addr: int, data: list[int]) -> int:
        if len(data) == 1:
            write_params = rm_peripheral_read_write_params_t(
                port=1,
                address=addr,
                device=self.hand_device_id,
            )
            ret = self.robot.rm_write_single_register(write_params, data[0])
            logger.info(
                f'legacy hand write single reg: addr:{addr}, ret:{ret}, data:{data[0]}'
            )
            return ret

        if len(data) > 10:
            raise ValueError(
                f'legacy peripheral API supports at most 10 registers per write, got {len(data)}'
            )

        ret = self.write_registers(addr, data)
        logger.info(f'legacy hand write regs ret:{ret}')
        return ret

    def _legacy_read_hand_holding_registers(self, addr: int, num: int) -> tuple[int, list[int]]:
        values: list[int] = []
        for offset in range(num):
            read_params = rm_peripheral_read_write_params_t(
                port=1,
                address=addr + offset,
                device=self.hand_device_id,
            )
            ret, value = self.robot.rm_read_holding_registers(read_params)
            if ret != 0:
                logger.info(
                    f'legacy hand read holding reg failed: addr:{addr + offset}, ret:{ret}'
                )
                return ret, values + [0] * (num - len(values))
            values.append(value)

        logger.info(f'legacy hand read holding regs: addr:{addr}, len:{num}, data:{values}')
        return 0, values

    def _legacy_read_hand_input_registers(self, addr: int, num: int) -> tuple[int, list[int]]:
        values: list[int] = []
        for offset in range(num):
            read_params = rm_peripheral_read_write_params_t(
                port=1,
                address=addr + offset,
                device=self.hand_device_id,
            )
            ret, value = self.robot.rm_read_input_registers(read_params)
            if ret != 0:
                logger.info(
                    f'legacy hand read input reg failed: addr:{addr + offset}, ret:{ret}'
                )
                return ret, values + [0] * (num - len(values))
            values.append(value)

        logger.info(f'legacy hand read input regs: addr:{addr}, len:{num}, data:{values}')
        return 0, values

    def write_hand_registers_pos_time(self, data: list[int]):
        # 位置时间控制，受限于一次 10 个寄存器不太有解
        ret = self.write_hand_holding_registers(1010, data[:10])
        # 伸展小手指，不能用前面的写了，会影响 1010 指令
        ret += self.write_hand_holding_registers(1052, [5] + data[10:])
        return ret

    def set_hand_unit_mode(self, mode: int = 0):
        return self.write_hand_holding_registers(937, [mode])

    def get_hand_unit_mode(self) -> int:
        ret, data = self.read_hand_holding_registers(937, 1)
        if ret != 0:
            raise RuntimeError(f'failed to read hand unit mode, ret={ret}')
        return data[0]

    def set_hand_positions_and_speeds(self, positions: list[int], speeds: list[int]) -> int:
        if len(positions) != 6 or len(speeds) != 6:
            raise ValueError('positions and speeds must both have length 6')

        if self.hand_api_mode == 'legacy':
            total = 0
            for finger_id, (position, speed) in enumerate(zip(positions, speeds)):
                ret = self.set_single_finger_position_and_speed(finger_id, position, speed)
                if ret != 0:
                    return ret
                total += ret
            return total

        ret = self.write_hand_holding_registers(1022, positions + speeds)
        if ret == -4:
            self.hand_api_mode = 'legacy'
            return self.set_hand_positions_and_speeds(positions, speeds)
        return ret

    def set_hand_positions(self, positions: list[int], speed: int = 600) -> int:
        return self.set_hand_positions_and_speeds(positions, [speed] * 6)

    def set_single_finger_position_and_speed(
        self, finger_id: int, position: int, speed: int
    ) -> int:
        return self.write_hand_holding_registers(1055, [finger_id, position, speed])

    def open_hand(self, speed: int = 600) -> int:
        return self.set_hand_positions([0, 0, 0, 0, 0, 0], speed=speed)

    def close_hand(self, speed: int = 600) -> int:
        return self.set_hand_positions([500, 500, 1000, 1000, 1000, 1000], speed=speed)

    def grasp_hand(self, speed: int = 600) -> int:
        return self.set_hand_positions([400, 0, 1000, 1000, 1000, 1000], speed=speed)

    def read_hand_status(self) -> HandStatus:
        ret, data = self.read_hand_input_registers(2000, 24)
        if ret != 0:
            raise RuntimeError(f'failed to read hand status block, ret={ret}')

        positions = data[0:6]
        speeds = [self._decode_int16(v) for v in data[6:12]]
        currents = [self._decode_int16(v) for v in data[12:18]]
        states = data[18:24]

        ret, button_data = self.read_hand_input_registers(2025, 1)
        if ret != 0:
            raise RuntimeError(f'failed to read hand button state, ret={ret}')

        unit_mode = None
        ret, mode_data = self.read_hand_holding_registers(937, 1)
        if ret == 0 and mode_data:
            unit_mode = mode_data[0]

        return HandStatus(
            timestamp=time.time(),
            positions=positions,
            speeds=speeds,
            currents=currents,
            states=states,
            button=button_data[0],
            unit_mode=unit_mode,
        )

    def read_hand_positions(self) -> list[int]:
        return self.read_hand_status().positions

    def read_hand_currents(self) -> list[int]:
        return self.read_hand_status().currents

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

    def get_pose_from_joint(self, joint: list[float]) -> list[float]:
        return self.robot.rm_algo_forward_kinematics(joint, 1)

    def moves_traj(self, joints: list):
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
        self, offset: list[float], v: int = 40, r: int = 0, connect: int = 0, block: int = 1
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

    def do_move_on_scan_begin(self):
        logger.info('do move to scan begin point')
        pass

    def do_move_to_scan_begin(self):
        logger.info('do move to scan begin point')
        pass

    def get_detected_target_pos_with_cam_pos(self, target_pos_xyz: tuple, arm_end_pos: Pos):
        logger.info(f'get target pos in cam: {target_pos_xyz}, arm_pos: {arm_end_pos}')

        from .convert import convert

        pos_in_world = convert(
            *target_pos_xyz,
            *arm_end_pos,
            rotation_matrix=rotation_matrix,
            translation_vector=translation_vector,
        )
        x, y, z, rx, ry, rz = (float(x) for x in pos_in_world)
        pos_in_world = Pos(x, y, z, rx, ry, rz)

        logger.info(f'pos_w: {pos_in_world}, pos_cam: {target_pos_xyz}')

        return pos_in_world

    def get_target_grasp_pos(self, target_pos: Pos) -> Pos:
        logger.info(f'move to target pos: {target_pos}')
        return target_pos

    def move_to_target_grasp_pos(self, target_grasp_pos: Pos):
        logger.info(f'move to target pos: {target_grasp_pos}')

    def put_on_some_where(self):
        pass

    def run_main_routine(self, cam: RealsenseCamera, yolo: YoloManager):
        self.get_arm_software_info()

        self.cam = cam
        self.yolo = yolo

        logger.debug('attach hooks')
        cam.set_fetch_pos_method(self.fetch_cur_arm_pos)

        install_pose = self.get_install_pose()
        logger.info(f'install pose: {install_pose}')

        self.set_hand_unit_mode(0)
        self.set_modbus_mode(1, 460800, 20)

        self.watch_cur_arm_state()
        # self.do_move_to_O()

        while not self.state.stopped:
            # keep watch with yolo till target detected
            ret = self.yolo.watch_on_frame_for_target(self.cam)

            if not ret:
                continue

            logger.info('stop move on detect target')
            self.robot.rm_set_arm_stop()
            time.sleep(0.1)

            # TODO: how to move the target to the center of the arm end pose

            # rescan on stopped state only, need to wait?
            ret = self.yolo.watch_on_frame_for_target(self.cam)
            if not ret:
                logger.warn('no detected target after stop scan, rescan')
                continue

            frame, xyxy = ret
            ret = self.yolo.get_frame_center_pos_in_segment_of_detected_target(
                frame, xyxy, self.cam
            )

            if not ret:
                logger.warn('failed to do segment after detect target, rescan')
                continue

            frame, target_pos_xyz = ret
            arm_end_pos = frame.pos

            if target_pos_xyz is None or arm_end_pos is None:
                logger.error(
                    f'unexpected None Pose: target:{target_pos_xyz}, arm_end:{arm_end_pos}'
                )
                continue

            target_pos_world = self.get_detected_target_pos_with_cam_pos(
                target_pos_xyz, arm_end_pos
            )

            target_distance_to_arm_end = target_pos_world.distance(arm_end_pos)
            logger.info(f'target distance to arm end: {target_distance_to_arm_end:.4f}m')

            target_grasp_pos = self.get_target_grasp_pos(target_pos_world)
            logger.info(f'target grasp pos: {target_grasp_pos}')

        self.close_modbus_mode(1)
        self.disconnect()
