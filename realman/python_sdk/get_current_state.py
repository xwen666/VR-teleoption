import time
from Robotic_Arm.rm_robot_interface import *

# 实例化 RoboticArm 类
arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)

# 创建机械臂连接
handle = arm.rm_create_robot_arm("192.168.1.18", 8080)
print("handle id:", handle.id)

# 100 Hz
dt = 2

try:
    while True:
        t0 = time.perf_counter()

        # 获取当前机械臂状态
        state = arm.rm_get_current_arm_state()
        print(state)

        # 控制循环周期为 0.01 s
        elapsed = time.perf_counter() - t0
        sleep_time = dt - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

except KeyboardInterrupt:
    print("停止读取关节状态")

finally:
    arm.rm_delete_robot_arm()
    print("机械臂连接已断开")