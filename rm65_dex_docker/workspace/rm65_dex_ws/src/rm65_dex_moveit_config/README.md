# RM65 + Revo2 MoveIt Config

MoveIt 2 configuration for the combined RM65-B-V arm and Revo2 left hand.

The config uses:

- `arm`: chain from `base_link` to `rm65_tool0`
- `hand`: the six active Revo2 joints
- `arm_hand`: convenience group combining arm and hand

Launch the demo inside the container:

```bash
cd /workspace/rm65_dex_ws
source install/setup.bash
ros2 launch rm65_dex_moveit_config demo.launch.py
```
