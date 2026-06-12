# v1 Modularization Notes

This project keeps the existing v1 launch paths stable while moving shared
teleoperation logic into small reusable modules.

## Stable entry points

These commands keep using the same control chain as before:

```bash
cd /home/xwen/vr/v1
bash ./run_mujoco_realman_arm_only_validation.sh
bash ./run_realman_vrteleop_style_follow.sh
bash ./run_realman_desktop_follow.sh
```

The legacy import path is also preserved:

```python
from hybrid_teleop_common import HybridArmController, load_hybrid_config
```

## Shared modules

The common pieces now live in:

```text
rm65_dex_docker/scripts/teleop_core/
```

- `constants.py`: joint names, hand mimic rules, RM65 defaults.
- `config.py`: `HybridConfig`, YAML loading, joint snapshot seed override.
- `math_utils.py`: quaternion, rotation vector, low-pass, pose smoothing helpers.
- `mapping.py`: wrist/base/tool axis mappings and Quest quaternion transforms.
- `udp_packets.py`: wrist and hand UDP packet parsing.
- `robot_adapter.py`: lightweight adapter protocol for future robot backends.

## Current control boundary

The current v1 controller remains in:

```text
rm65_dex_docker/scripts/hybrid_teleop_common.py
```

That file now focuses on `HybridArmController` and re-exports the split modules
for compatibility. MuJoCo and RealMan scripts can continue importing from
`hybrid_teleop_common.py` without behavior changes.

## Future robot migration

To add a non-RealMan arm, keep the VR input and control flow and add a new robot
adapter around the vendor SDK or simulator:

```text
VR wrist UDP
-> teleop_core.mapping
-> HybridArmController / IK
-> RobotAdapter implementation
-> robot SDK or simulator
```

The parts most likely to change are the robot model, end-effector frame, joint
limits, FK/IK backend, and SDK command sender.

