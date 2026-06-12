# RM65 Dex Docker Workspace

这个目录保留 Docker/ROS2 工作区、RealMan SDK 辅助脚本和 v1 控制链路依赖的底层模块。

当前推荐入口已经迁移到 `/home/xwen/vr/v1`。日常调试和真机部署优先看：

```text
/home/xwen/vr/v1/README.md
/home/xwen/vr/v1/v1_realman_arm_control_commands.md
/home/xwen/vr/v1/modularization_notes.md
```

## Current v1 Commands

启动 Quest/Revo2 wrist/hand 转发：

```bash
cd /home/xwen/vr/v1
bash ./run_revo2_retargeting_to_ros.sh
```

MuJoCo 验证：

```bash
cd /home/xwen/vr/v1
bash ./run_mujoco_realman_arm_only_validation.sh
```

推荐真机跟随：

```bash
cd /home/xwen/vr/v1
bash ./run_realman_vrteleop_style_follow.sh
```

旧的 cube-side-mount MuJoCo/真机入口已经清理掉，避免和 v1 主线混用。

## Kept Scripts

`rm65_dex_docker/scripts/` 中仍保留 v1 需要的公共脚本：

- `hybrid_teleop_common.py`: v1 arm controller 兼容入口。
- `teleop_core/`: 坐标映射、配置、UDP 解析、数学工具等共享模块。
- `realman_sdk_common.py`: RealMan SDK 基础封装。
- `realman_sdk_hybrid_teleop.py`: 真机 SDK teleop runner。
- `run_realman_sdk_follow_teleop.sh`: 真机 SDK launcher。
- `run_revo2_retargeting_to_ros.sh`: Quest/Revo2 retargeting 转发入口。
- `read_realman_joint_state.py`: 读取真机关节快照。
- `sync_realman_to_mujoco_pose.py`: 可选的真机/MuJoCo 姿态预同步工具。
- `brainco_hand_ethercat.py`: Revo2/BrainCo hand 连接辅助。
- Docker 辅助脚本：`build_image.sh`, `up.sh`, `down.sh`, `enter.sh`, `allow_x11.sh`, `adb_quest_reverse.sh`。

## Generated Files

不要提交 colcon 生成目录。它们会在本地 build 后重新生成：

```text
rm65_dex_docker/workspace/rm65_dex_ws/build/
rm65_dex_docker/workspace/rm65_dex_ws/install/
rm65_dex_docker/workspace/rm65_dex_ws/log/
```

重新构建 Docker 工作区：

```bash
cd /home/xwen/vr/rm65_dex_docker
./scripts/build_image.sh
./scripts/up.sh
./scripts/enter.sh

cd /workspace/rm65_dex_ws
./src/rm65_dex_bringup/scripts/build_ws.sh
```
