adb reverse --list
adb reverse tcp:8010 tcp:8010
adb reverse --list


# RM65 灵巧手 Docker 工作区

这个工作区对应的是 `/home/xwen/vr/RM65B_V_DexHand_Quest3_MoveIt2_VLA_Docker.md` 这份方案。

目前有意先不包含 VLA 训练部分，当前脚手架主要聚焦在：

- ROS 2 Humble + MoveIt 2 + RViz 的 Docker 环境
- RM65 + 灵巧手的 URDF / MoveIt 配置占位
- Quest / Revo2 UDP -> 共享 DLS IK + Ruckig 控制器 -> MuJoCo 或 RealMan SDK follow
- Quest wrist UDP -> 本地 DLS IK + Ruckig 平滑 -> ROS 2 机械臂关节命令
- Quest / Revo2 hand qpos UDP -> ROS 2 灵巧手位置控制器
- 用于检查 state / action / image 的录制与 smoke-data 骨架

adb reverse tcp:8010 tcp:8010
adb reverse --list
## 快速开始

```bash
cd /home/xwen/vr/rm65_dex_docker
./scripts/allow_x11.sh
./scripts/build_image.sh
./scripts/up.sh
./scripts/enter.sh
```

进入容器后：

```bash
cd /workspace/rm65_dex_ws
./src/rm65_dex_bringup/scripts/build_ws.sh
```

真实的 RM65 描述包已经从下面这个路径复制过来：

```text
/home/xwen/vr/assets/robots/arms/rm_description
```

复制到这里：

```text
/home/xwen/vr/rm65_dex_docker/workspace/rm65_dex_ws/src/rm_description
```

组合模型使用的是真实 RM65 的连杆和关节：

- 基座 link：`base_link`
- 机械臂关节：`joint1` ... `joint6`
- 腕部 link：`Link6`
- 为 MoveIt / tool 挂接额外添加的适配 link：`rm65_tool0`

MoveIt 配置在 `workspace/rm65_dex_ws/src/rm65_dex_moveit_config`。

注意：当前机器的 ROS apt 源里没有 `ros-humble-trac-ik-kinematics-plugin`，所以 Dockerfile 里先没有装它。现阶段先使用 MoveIt 默认的 KDL 求解器；如果后续需要，再从源码补 TRAC-IK。

## 主线命令

现在日常最常用的主线命令就是这三条：

先启动 Quest / Revo2 到宿主机 UDP 转发：

```bash
cd /home/xwen/vr
./rm65_dex_docker/scripts/run_revo2_retargeting_to_ros.sh
```

启动 MuJoCo 正方体左侧安装主场景：

```bash
cd /home/xwen/vr/rm65_dex_docker
bash ./scripts/run_mujoco_cube_side_mount_validation.sh
```

启动真机 cube-side-mount 正式遥操主线：

```bash
cd /home/xwen/vr/rm65_dex_docker
bash ./scripts/run_realman_cube_side_mount_follow.sh
```

如果你要边遥操边录制真机数据，可以直接把录制参数一起带上：

```bash
cd /home/xwen/vr/rm65_dex_docker
bash ./scripts/run_realman_cube_side_mount_follow.sh \
  --record-dir /home/xwen/vr/data/teleop_pick_cube \
  --task "pick up the cube"
```

如果你希望真机启动时不要自动读取 MuJoCo 快照做 pre-sync，可以这样：

```bash
cd /home/xwen/vr/rm65_dex_docker
REALMAN_AUTO_PRESYNC=0 bash ./scripts/run_realman_cube_side_mount_follow.sh
```

如果你想先把 **真机当前关节角同步到 MuJoCo 初始姿态**，再开始仿真/遥操，可以先抓一份 SDK 快照：

```bash
cd /home/xwen/vr/rm65_dex_docker
/home/xwen/anaconda3/envs/sdk/bin/python ./scripts/read_realman_joint_state.py \
  --snapshot /tmp/rm65_real_sdk_arm_snapshot.json
```

然后把这份快照作为 MuJoCo 初始关节输入：

```bash
cd /home/xwen/vr/rm65_dex_docker
MUJOCO_INITIAL_SNAPSHOT_PATH=/tmp/rm65_real_sdk_arm_snapshot.json \
  bash ./scripts/run_mujoco_cube_side_mount_validation.sh
```

这样 MuJoCo 会先从真机当前 `arm_qpos` 起步，再进入你现在这条 wrist/hand 遥操主线。

## Quest / Revo2 接真机 SDK 主线

现在推荐的真机正式遥操主线，已经和 MuJoCo 共用同一套上游控制器：

```text
Quest / Revo2 -> 宿主机 retargeting -> wrist_pose / hand_qpos UDP
                -> 共享 DLS + 正则 + Ruckig controller
                -> RealMan Python SDK follow
```

这条链路不再依赖 Docker 里的 ROS wrist bridge 做主控制回路。ROS 仍然可以保留给建图、可视化、录包或其它辅助节点，但正式遥操的 arm backend 现在可以直接走 SDK。

宿主机上先保持 Quest / Revo2 retargeting 转发：

```bash
cd /home/xwen/vr
./rm65_dex_docker/scripts/run_revo2_retargeting_to_ros.sh
```

然后启动真机 SDK teleop。现在它默认就是 cube side mount 这条主线配置：

```bash
cd /home/xwen/vr/rm65_dex_docker
bash ./scripts/run_realman_cube_side_mount_follow.sh
```

如果你想直接调用底层的通用 SDK launcher，也可以这样：

```bash
cd /home/xwen/vr/rm65_dex_docker
bash ./scripts/run_realman_sdk_follow_teleop.sh
```

这个 launcher 会：

- 默认复用 `wrist_cube_side_mount.yaml` 里的 DLS / 正则 / Ruckig 参数
- 直接监听 `wrist_pose:5005` 和 `hand_qpos:5010`
- 用 `rm_movej_follow` 连续下发机械臂关节目标
- 默认把灵巧手也切到 SDK follow-pos
- 在检测到 `/tmp/rm65_mujoco_arm_snapshot.json` 时，先自动做一次低速 pre-sync

如果这次只想接 arm，把 hand 暂时交给别的链路，可以这样关掉 SDK 手部跟随：

```bash
REALMAN_DISABLE_HAND_FOLLOW=1 bash ./scripts/run_realman_cube_side_mount_follow.sh
```

如果你想跳过 pre-sync，或者强制要求 pre-sync：

```bash
REALMAN_AUTO_PRESYNC=0 bash ./scripts/run_realman_cube_side_mount_follow.sh
REALMAN_AUTO_PRESYNC=1 bash ./scripts/run_realman_cube_side_mount_follow.sh
```

默认的 `REALMAN_AUTO_PRESYNC=auto` 会在找到 MuJoCo 快照时自动对齐；找不到时直接进入遥操。

如果你启用了 `--record-dir`，录制发生在宿主机这条 SDK 主线里，直接读取：

- RealMan Python SDK 当前机械臂状态
- 当前下发给机械臂/灵巧手的命令
- RealMan 灵巧手 Modbus 实际状态回读
- front / wrist 两路 RealSense 图像

也就是说，这条录制链不依赖 ROS 话题订阅。

每条记录会包含：

- `observation.images.front`：front 相机当前帧
- `observation.images.wrist`：wrist 相机当前帧
- `observation.state`：当前真实状态向量，默认是 `arm q_now(6) + hand q_now_modbus(6)`
- `action`：当前动作向量，默认是 `arm q_cmd_after_ruckig(6) + hand q_cmd(6)`
- `task`：启动时通过 `--task` 传入的语言任务
- `episode_index`
- `frame_index`
- `timestamp`
- `episode_done`

录制结果默认会写成一个 episode 目录，里面包含：

- `images/front/*.jpg`
- `images/wrist/*.jpg`
- `steps.jsonl`
- `meta.json`

如果宿主机上接了多台 RealSense，也可以显式指定序列号：

```bash
cd /home/xwen/vr/rm65_dex_docker
bash ./scripts/run_realman_cube_side_mount_follow.sh \
  --record-dir /home/xwen/vr/data/teleop_pick_cube \
  --task "pick up the cube" \
  --front-camera-serial FRONT_SERIAL \
  --wrist-camera-serial WRIST_SERIAL
```

同一时刻最好只保留一种真机控制通道。用这条 SDK 主线时，不要再同时运行会主动发 arm 命令的 ROS wrist bridge、`rm_driver` 话题遥操节点，或者另一个 SDK teleop 进程。

## Quest / Revo2 接 Docker ROS（对照 / 调试）

在容器里启动完整的 RViz + ros2_control fake hardware + DLS wrist bridge：

```bash
cd /workspace/rm65_dex_ws
./src/rm65_dex_bringup/scripts/build_ws.sh
source install/setup.bash
ros2 launch rm65_dex_bringup quest_control_acceptance.launch.py
```

在宿主机上启动 Quest 手部重定向：

```bash
cd /home/xwen/vr
./rm65_dex_docker/scripts/run_revo2_retargeting_to_ros.sh
```

Quest App 走无线 TCP 时的设置：

```text
Host: 电脑的局域网 IP，例如 192.168.4.227
Port: 8010
Transport: TCP
```

Quest App 会通过 TCP `8010` 把 landmarks 发到宿主机。随后 retargeting 脚本会：

- 通过 UDP `5010` 把 hand qpos 转发给 ROS
- 通过 UDP `5005` 把 wrist pose 转发给 DLS wrist bridge

IK bridge 会直接向 `/arm_controller/commands` 发布机械臂位置命令，因此这条链路不依赖 MoveIt Servo。默认求解链是本地 DLS，Ruckig 可用时会自动做 jerk-limited 平滑；如果当前环境没有 Python `ruckig`，会退回到加速度限幅平滑。

如果只想单独启动 DLS wrist bridge：

```bash
cd /workspace/rm65_dex_ws
source install/setup.bash
ros2 launch quest_bridge wrist_ik_bridge.launch.py
```

这个 launch 现在默认就会加载 `wrist_cube_side_mount.yaml`，也就是你当前 MuJoCo 和真机共用的主线参数。它会在 wrist 标定前先把机械臂复位到固定初始关节姿态 `[0.0, -1.25, 1.95, 0.0, 0.95, 0.0]`。复位完成后的第一帧新 wrist 数据会被当作零参考，因此无论你开始时手放在哪，机器人都会从同一个初始姿态起步。

如果你想在真机遥操前，先让真实 RM65 对齐到 MuJoCo 当前机械臂姿态，现在主 MuJoCo 启动脚本已经默认会自动做这件事：

1. 默认 SDK 路线下，不需要先启动 `rm_driver`
2. 运行主场景：

```bash
cd /home/xwen/vr/rm65_dex_docker
bash ./scripts/run_mujoco_cube_side_mount_validation.sh
```

脚本会先启动 MuJoCo，把当前 arm 关节角持续写到 `/tmp/rm65_mujoco_arm_snapshot.json`，然后自动调用 `scripts/sync_realman_to_mujoco_pose.py`。默认优先走宿主机上的 RealMan Python SDK，直接连控制器 IP 做一个低速 `rm_movej` 预同步；如果你手动切到 `REALMAN_SYNC_BACKEND=ros`，它也还能退回到 `rm_driver` 的 `/rm_driver/movej_cmd` 话题模式。

同一时刻最好只保留一种真机控制通道：默认 SDK 预同步时，不要让另一个会主动下发机械臂运动命令的 `rm_driver`/遥操节点同时控制机械臂；如果后续要切到 SDK 正式遥操，直接运行 `run_realman_cube_side_mount_follow.sh` 即可。

如果这次只想跑纯仿真，可以临时关闭自动预同步：

```bash
REALMAN_AUTO_PRESYNC=0 bash ./scripts/run_mujoco_cube_side_mount_validation.sh
```

如果你明确想走 Docker 里的 ROS 话题链，而不是宿主机 SDK，那么先启动 `rm_driver`，再这样切：

```bash
REALMAN_SYNC_BACKEND=ros bash ./scripts/run_mujoco_cube_side_mount_validation.sh
```

如果你想单独手动执行一次预同步，也还能直接调用：

```bash
cd /home/xwen/vr/rm65_dex_docker
/home/xwen/anaconda3/envs/sdk/bin/python ./scripts/sync_realman_to_mujoco_pose.py --backend sdk --speed 5
```

## Quest / Revo2 接 MuJoCo

当前主要仿真入口是正方体左侧安装场景。它会复用同一套 DLS + Ruckig wrist 控制链：

```bash
cd /home/xwen/vr/rm65_dex_docker
bash ./scripts/run_mujoco_cube_side_mount_validation.sh
```

首次运行时，脚本会检查 `/home/xwen/anaconda3/envs/VR/bin/python` 里是否能 `import ruckig`；如果不能，会从 `/home/xwen/vr/ruckig` 自动编译安装 Python wrapper。需要换 Python 时可设置 `MUJOCO_PYTHON` 或 `MUJOCO_PYTHON_BIN`。

在 RViz 里验证完以后，也可以在宿主机上继续用通用 MuJoCo validator 验证同一套 hybrid 控制链。

MuJoCo 验证器会直接读取当前主线 hybrid 参数，配置文件在：

```text
/home/xwen/vr/rm65_dex_docker/workspace/rm65_dex_ws/src/quest_bridge/config/wrist_cube_side_mount.yaml
```

它监听的也是同一组转发 UDP 端口：

- `wrist_pose`：`5005`
- `hand_qpos`：`5010`

所以整条数据链会变成：

```text
Quest app -> 宿主机 retargeting TCP -> wrist_pose / hand_qpos UDP -> MuJoCo hybrid validator
```

宿主机侧的 retargeting 还是用这条命令：

```bash
cd /home/xwen/vr
./rm65_dex_docker/scripts/run_revo2_retargeting_to_ros.sh
```

然后在第二个终端里启动 MuJoCo。推荐先跑主场景：

```bash
cd /home/xwen/vr/rm65_dex_docker
bash ./scripts/run_mujoco_cube_side_mount_validation.sh
```

注意：

- 先停止 `wrist_ik_bridge`，或者完整的 `quest_control_acceptance.launch.py`，否则 ROS 会先占用 UDP `5005/5010`
- MuJoCo 验证器会动态加载组合后的 RM65 + Revo2 模型，并复用当前 RViz 使用的同一套 hybrid 机械臂逻辑
- 如果想换端口，可以在启动前设置 `MUJOCO_WRIST_PORT` 和 `MUJOCO_HAND_PORT`

## USB 有线 TCP 转发

如果你不走无线，而是想通过 USB 做 TCP 转发：

```bash
cd /home/xwen/vr/rm65_dex_docker
./scripts/adb_quest_reverse.sh
```

然后把 Quest App 里的主机地址设成 `127.0.0.1`，端口设成 `8010`。
## 正方体左侧安装 MuJoCo 场景

当前主线仿真入口是 `./scripts/run_mujoco_cube_side_mount_validation.sh`。

正方体左侧安装场景对应这一套文件：

- 场景脚本：`/home/xwen/vr/rm65_dex_docker/scripts/mujoco_cube_side_mount_validate.py`
- 启动脚本：`/home/xwen/vr/rm65_dex_docker/scripts/run_mujoco_cube_side_mount_validation.sh`
- 参数文件：`/home/xwen/vr/rm65_dex_docker/workspace/rm65_dex_ws/src/quest_bridge/config/wrist_cube_side_mount.yaml`

## 坐标系约定

这个正方体场景里我们固定约定：

- 正方体前方：`+X`
- 正方体左侧：`+Y`
- 竖直向上：`+Z`

机械臂安装关系：

- `base` 的 `z` 轴垂直于正方体左侧面，也就是 `base z -> +Y`
- `base` 的 `x` 轴对准正方体前进方向，也就是 `base x -> +X`

因此这套场景的目标是：

- 在 VR 中你的手往前伸，机械臂末端沿着正方体前方（`+X`）方向前伸
- 你的手往左/右平移，机械臂末端也沿着正方体左右方向平移

当前场景配置继续使用：

- `axis_mapping: quest3_teleop_flip_forward_up_y_left_z`
- `position_target_mode: absolute`
- `arm_solver_mode: local_dls`
- `trajectory_smoother: auto`

这套映射的平移约定是：

- 手往前 -> 机械臂沿 `base +x`
- 手往后 -> 机械臂沿 `base -x`
- 手往左 -> 机械臂沿 `base +z`
- 手往右 -> 机械臂沿 `base -z`
- 手往上 -> 机械臂沿 `base +y`
- 手往下 -> 机械臂沿 `base -y`

MuJoCo 当前机械臂关节角快照默认写到：

```text
/tmp/rm65_mujoco_arm_snapshot.json
```
## 读取机械臂关节角度
cd /home/xwen/vr/rm65_dex_docker
/home/xwen/anaconda3/envs/sdk/bin/python ./scripts/read_realman_joint_state.py \
  --snapshot /tmp/rm65_real_sdk_arm_snapshot.json
## 推荐执行顺序

### 1. 启动 Quest -> 主机 retargeting

```bash
cd /home/xwen/vr
./rm65_dex_docker/scripts/run_revo2_retargeting_to_ros.sh
```

### 2. 启动正方体左侧安装 MuJoCo 场景

如果脚本没有执行权限，直接用 `bash` 启动：

```bash
cd /home/xwen/vr/rm65_dex_docker
bash ./scripts/run_mujoco_cube_side_mount_validation.sh
```

也可以显式指定参数文件：

```bash
cd /home/xwen/vr/rm65_dex_docker
MUJOCO_CONFIG=/home/xwen/vr/rm65_dex_docker/workspace/rm65_dex_ws/src/quest_bridge/config/wrist_cube_side_mount.yaml \
bash ./scripts/run_mujoco_cube_side_mount_validation.sh
```

## 端口提醒

正方体场景和原来的 MuJoCo / ROS wrist bridge 一样，默认监听：

- `wrist_pose`: `5005`
- `hand_qpos`: `5010`

所以在运行这个场景前，请先停掉其它占用 `5005/5010` 的进程。
