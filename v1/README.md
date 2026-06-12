# v1 真机启动说明

这个 `v1/` 目录现在主要有两条线：

- `v1` MuJoCo 无桌面场景：RealMan 机械臂 + 末端 dex-hand
- `v1` 真机桌面任务：Quest / Revo2 UDP -> DLS IK + 正则 + Ruckig -> RealMan Python SDK follow

## 1. Quest 端口映射

先确认 Quest 已经通过 `adb` 连上：

```bash
adb devices
```

建立 Quest 到宿主机的反向端口映射：

```bash
adb reverse tcp:8010 tcp:8010
adb reverse --list
```

## 2. 读取真机当前关节角

先抓一份当前 RealMan 真机关节角快照：

```bash
cd /home/xwen/vr/v1
/home/xwen/anaconda3/envs/sdk/bin/python ./read_realman_joint_state.py
```

默认会写到：

```text
/tmp/rm65_real_sdk_arm_snapshot.json
```

## 3. 启动 VR retargeting

启动 Quest / Revo2 到宿主机 UDP 转发：

```bash
cd /home/xwen/vr/v1
bash ./run_revo2_retargeting_to_ros.sh
```

这条脚本会继续复用主线 retargeting，但优先走 `VR` conda 环境的 Python。

## 4. 启动 v1 真机桌面任务

最常用的真机启动命令：

```bash
cd /home/xwen/vr/v1
bash ./run_realman_desktop_follow.sh
```

默认行为：

- 真机配置：`/home/xwen/vr/v1/config/wrist_realman_arm_only.yaml`
- 真机关节快照：`/tmp/v1_real_sdk_arm_snapshot.json`
- MuJoCo 快照：`/tmp/v1_mujoco_realman_arm_snapshot.json`
- 共享启动姿态快照：`/tmp/rm65_real_sdk_arm_snapshot.json`
- 自动启动 `v1` UDP fanout：把 Quest 送到 `5005/5010` 的 wrist/hand 数据复制给真机和 MuJoCo
- 自动启动 `v1` MuJoCo 对照场景，方便同时观察仿真和真机差异
- arm backend：`movej_follow`
- hand backend：默认先关闭，避免末端 Modbus 阻塞机械臂 IK/发命令循环；确认机械臂能跟随后再显式打开

组合启动时的默认内部端口分配：

- Quest / Revo2 -> fanout listen：`5005 / 5010`
- 真机 teleop 下游：`5105 / 5110`
- MuJoCo 下游：`5205 / 5210`

## 5. 推荐的首次上真机命令

第一次上真机，建议先关闭自动 pre-sync，先确认方向和缩放：

```bash
cd /home/xwen/vr/v1
REALMAN_AUTO_PRESYNC=0 bash ./run_realman_desktop_follow.sh
```

如果你的目标就是：

- 真机启动时先运动到 `/tmp/rm65_real_sdk_arm_snapshot.json`
- MuJoCo 启动时也从 `/tmp/rm65_real_sdk_arm_snapshot.json` 对齐
- DLS/IK 的 `initial_joint_positions` 和 `nominal_joint_positions` 也一起对齐到这份 snapshot

那就不要手动关闭 pre-sync，直接运行：

```bash
cd /home/xwen/vr/v1
bash ./run_realman_desktop_follow.sh
```

如果你这次只想启动真机，不想顺带拉起 MuJoCo：

```bash
cd /home/xwen/vr/v1
V1_LAUNCH_MUJOCO_WITH_REAL=0 bash ./run_realman_desktop_follow.sh
```

如果你看到类似下面这种日志：

```text
[rm_set_hand_follow_pos] Failed to get robotic plus base info, ret=1
```

说明当前机器上的 `rm_set_hand_follow_pos` 通路不稳定。由于你的 Revo2 是安装在 RealMan 机械臂末端、并且和机械臂共用同一根网线到电脑，手对电脑来说不是一个独立的本地 EtherCAT 设备，所以如果要打开手控，优先用 `realman_modbus`，也就是通过 RealMan 控制器侧的末端手部通路来控手。

## 6. 如果要先和 v1 MuJoCo 对齐

如果你已经跑过 `v1` 仿真，并且想让真机先对齐到 `v1` MuJoCo 当前姿态，再开始遥操：

```bash
cd /home/xwen/vr/v1
bash ./run_realman_desktop_follow.sh
```

默认 `REALMAN_AUTO_PRESYNC=auto`，如果检测到：

```text
/tmp/rm65_real_sdk_arm_snapshot.json
```

就会自动做 pre-sync。

如果你想临时切成“跟随 MuJoCo 当前姿态”而不是“跟随共享启动姿态”，可以显式指定：

```bash
cd /home/xwen/vr/v1
REALMAN_PRESYNC_SNAPSHOT_PATH=/tmp/v1_mujoco_realman_arm_snapshot.json \
bash ./run_realman_desktop_follow.sh
```

如果你想强制要求 pre-sync：

```bash
cd /home/xwen/vr/v1
REALMAN_AUTO_PRESYNC=1 bash ./run_realman_desktop_follow.sh
```

## 7. 如果要临时切回更保守的真机参数

现在默认真机 launcher 已经直接走 `v1` 仿真同一份配置：

```text
/home/xwen/vr/v1/config/wrist_realman_arm_only.yaml
```

如果你想临时切回更保守的旧真机参数：

```bash
cd /home/xwen/vr/v1
REALMAN_SDK_TELEOP_CONFIG=/home/xwen/vr/v1/config/wrist_ik_bridge_real_safe.yaml \
bash ./run_realman_desktop_follow.sh
```

## 8. 如果你要打开或切换灵巧手后端

`v1` 真机桌面任务默认先只跑机械臂。如果你要打开手部控制，并且走 `realman_modbus`：

```bash
cd /home/xwen/vr/v1
V1_ENABLE_HAND_FOLLOW=1 bash ./run_realman_desktop_follow.sh
```

如果你想强制先试一次 RealMan 自带 hand follow-pos，再让程序自己失败后回退：

```bash
cd /home/xwen/vr/v1
V1_ENABLE_HAND_FOLLOW=1 REALMAN_HAND_BACKEND=auto bash ./run_realman_desktop_follow.sh
```

如果你想显式指定走 RealMan 工具口 Modbus：

```bash
cd /home/xwen/vr/v1
V1_ENABLE_HAND_FOLLOW=1 REALMAN_HAND_BACKEND=realman_modbus bash ./run_realman_desktop_follow.sh
```

如果以后你把手从机械臂末端拆下来，改成**单独一块网卡直连电脑**，那时再考虑 `brainco_ethercat` 才是对的。

## 9. 常用排查

查看 Quest 是否连上：

```bash
adb devices
```

查看 reverse 是否生效：

```bash
adb reverse --list
```

如果 retargeting 端口被占用，检查相关进程：

```bash
pgrep -af 'vr_realtime_retargeting.py|run_revo2_retargeting_to_ros.sh'
```

## 10. 最短启动顺序

如果只看“共享 snapshot 对齐真机和 MuJoCo”的最短操作顺序，就是这四步：

```bash
cd /home/xwen/vr/v1
/home/xwen/anaconda3/envs/sdk/bin/python ./read_realman_joint_state.py
bash ./run_mujoco_realman_arm_only_validation.sh
bash ./run_revo2_retargeting_to_ros.sh
bash ./run_realman_desktop_follow.sh
```
