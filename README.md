# v1 RealMan 机械臂控制命令

这个文档只记录当前推荐的 v1 控制链路。v1 的核心控制器是：

```text
Quest/Revo2 wrist UDP
-> v1 config/wrist_realman_arm_only.yaml
-> HybridArmController
-> IK 前末端 pose 平滑 + local DLS IK + 正则 + Ruckig
-> MuJoCo 或 RealMan SDK
```

## 1. 先启动 VR wrist 转发

```bash
cd /home/xwen/vr/v1
bash ./run_revo2_retargeting_to_ros.sh
```

默认 wrist UDP 发到：

```text
5005
```

## 2. 只在 MuJoCo 验证 v1 控制

```bash
cd /home/xwen/vr/v1
bash ./run_mujoco_realman_arm_only_validation.sh
```

这条是纯 v1 MuJoCo 验证：

```text
Quest wrist UDP :5005
-> HybridArmController
-> IK 前末端 pose 平滑 + DLS + regularization + Ruckig
-> MuJoCo RM65 + dex-hand
```

如果 MuJoCo 只是用来快速检查方向、四元数和坐标映射，可以用命令行临时放开真机安全限制。这个命令不会改 YAML，也不会影响真机：

```bash
cd /home/xwen/vr/v1
bash ./run_mujoco_realman_arm_only_validation.sh \
  --pre-ik-position-gain 0.4 \
  --pre-ik-rotation-gain 0.4 \
  --dls-max-delta-per-joint none \
  --trajectory-smoother none
```

如果想完全关闭 wrist 输入 EMA，也可以再加：

```bash
  --position-lpf-alpha 1.0 \
  --rotation-lpf-alpha 1.0
```

注意：上面是 MuJoCo 快速验证参数，不建议直接用于真机。

如果你正在用 `run_realman_desktop_follow.sh` 同时启动真机和 MuJoCo，fanout 会把 MuJoCo wrist 改发到 `5205`。这种情况下单独启动 MuJoCo 要用：

```bash
cd /home/xwen/vr/v1
V1_MUJOCO_WRIST_PORT=5205 V1_MUJOCO_HAND_PORT=5210 \
bash ./run_mujoco_realman_arm_only_validation.sh
```

## 3. 推荐的 v1 真机控制命令

推荐真机先用这个：

```bash
cd /home/xwen/vr/v1
bash ./run_realman_vrteleop_style_follow.sh
```

注意：这条不是改成 `vr_teleop` 的 IK。它仍然使用 v1 的：

```text
config/wrist_realman_arm_only.yaml
HybridArmController
local DLS IK
pre-IK end-effector pose smoothing
regularization
Ruckig
```

它只是把真机发送层对齐到 `vr_teleop` 做得比较稳的方式：

```text
50Hz control loop
rm_movej_canfd(..., follow=True)
feedback_source=command
startup current real-arm pose as nominal
no automatic pre-sync by default
```

也就是说，它更接近：

```text
上一帧 command -> 下一帧 IK seed
```

这和 `vr_teleop` 里的：

```python
q_init[:6] = q_sol[:6]
q_sol = solve_pose_ik(...)
```

是同一种连续迭代思路。

v1 现在也加了和 `vr_teleop` 类似的 IK 前末端平滑，参数在：

```yaml
pre_ik_position_gain: 0.08
pre_ik_rotation_gain: 0.08
```

含义是每一帧 IK 目标只追 raw wrist 目标的一部分。`0.08` 更稳、更像 `vr_teleop`；`1.0` 等于关闭这层平滑。

v1 的平移也按 `vr_teleop` 的方式处理：

```yaml
axis_mapping: wrist_left_to_base_x_right_y_forward_z_up
rotation_axis_mapping: wrist_left_rotation_xz_unswap_flip_blue
position_target_mode: absolute
orientation_target_mode: relative
```

这里的 `absolute` 不是把 VR 世界坐标直接当机器人目标，而是：

```text
target_position = 初始末端位置 + 映射后的 wrist 相对初始位移 * position_scale
```

当前轴关系按截图配置为：

```text
wrist +X 红轴 右 -> base +X 红轴 右
wrist +Y 绿轴 上 -> base +Z 蓝轴 上
wrist +Z 蓝轴 前 -> base +Y 绿轴 前
```

旋转不直接复用上面的左手系反射矩阵，而是单独使用
`wrist_left_rotation_xz_unswap_flip_blue`，避免出现“绕 wrist 红轴转，末端按 wrist 蓝轴转”的 X/Z 旋转轴交换，并翻转 wrist 蓝轴旋转方向。

这样机械臂末端换方向以后，手往右/上/前的平移方向不会被当前末端姿态重新解释；旋转仍然使用 wrist 相对初始姿态去控制末端。

### Ruckig 要不要关

`vr_teleop` 没有 Ruckig，它主要靠 IK 前 pose smoothing、wrist deadband/EMA、50Hz CANFD 跟随变稳。

v1 默认仍保留 Ruckig，因为 v1 现在是 full-pose 四元数控制，靠近奇异位姿时 DLS 可能出现关节跳动，Ruckig 可以再限制关节速度、加速度和 jerk。真机第一次测试建议先保留默认：

```bash
cd /home/xwen/vr/v1
bash ./run_realman_vrteleop_style_follow.sh
```

如果你想更接近 `vr_teleop` 的直接 CANFD 手感，可以单独关掉 Ruckig 做对比：

```bash
cd /home/xwen/vr/v1
REALMAN_TRAJECTORY_SMOOTHER=none bash ./run_realman_vrteleop_style_follow.sh
```

## 4. 标准旧 v1 真机入口

如果你想用旧的 v1 真机入口：

```bash
cd /home/xwen/vr/v1
bash ./run_realman_desktop_follow.sh
```

默认行为：

```text
arm command mode = movej_follow
feedback source = SDK real joint feedback
auto launch MuJoCo side-by-side
auto launch UDP fanout
```

这个入口更完整，但真机手感可能比 `run_realman_vrteleop_style_follow.sh` 更抖，因为它默认使用 SDK 读到的真实关节作为每帧 seed，而不是上一帧 command。

## 5. 真机只跑机械臂，不启动 MuJoCo

```bash
cd /home/xwen/vr/v1
V1_LAUNCH_MUJOCO_WITH_REAL=0 bash ./run_realman_desktop_follow.sh
```

如果用推荐入口，默认已经是不启动 MuJoCo：

```bash
cd /home/xwen/vr/v1
bash ./run_realman_vrteleop_style_follow.sh
```

## 6. 真机同时启动 MuJoCo 对照

```bash
cd /home/xwen/vr/v1
V1_LAUNCH_MUJOCO_WITH_REAL=1 bash ./run_realman_vrteleop_style_follow.sh
```

这时端口分配是：

```text
Quest/Revo2 -> fanout listen: 5005 / 5010
real arm    -> downstream:     5105 / 5110
MuJoCo      -> downstream:     5205 / 5210
```

## 7. 读取当前真机关节快照

如果需要让 MuJoCo 或 nominal pose 从当前真机姿态开始：

```bash
cd /home/xwen/vr/v1
/home/xwen/anaconda3/envs/sdk/bin/python ./read_realman_joint_state.py
```

默认写入：

```text
/tmp/rm65_real_sdk_arm_snapshot.json
```

## 8. 当前建议

调方向、调四元数、看 MuJoCo：

```bash
cd /home/xwen/vr/v1
bash ./run_mujoco_realman_arm_only_validation.sh
```

上真机跟手：

```bash
cd /home/xwen/vr/v1
bash ./run_realman_vrteleop_style_follow.sh
```

需要完整 side-by-side：

```bash
cd /home/xwen/vr/v1
V1_LAUNCH_MUJOCO_WITH_REAL=1 bash ./run_realman_vrteleop_style_follow.sh
```
