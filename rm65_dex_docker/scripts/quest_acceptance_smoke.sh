#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if docker info >/dev/null 2>&1; then
  DOCKER=(docker)
else
  DOCKER=(sudo docker)
fi

ros_exec() {
  "${DOCKER[@]}" exec rm65_dex_ros2 bash -lc \
    "source /opt/ros/humble/setup.bash && source /workspace/rm65_dex_ws/install/setup.bash && $*"
}

cleanup() {
  "${DOCKER[@]}" exec rm65_dex_ros2 bash -lc \
    "pkill -f 'ros2 launch rm65_dex_bringup quest_control_acceptance.launch.py' || true; \
     pkill -f 'move_group|ros2_control_node|servo_node_main|wrist_twist_bridge|hand_qpos_node|robot_state_publisher|static_transform_publisher|controller_manager/spawner' || true" \
    >/dev/null 2>&1 || true
}

trap cleanup EXIT

cd "$ROOT"
cleanup

ros_exec "cd /workspace/rm65_dex_ws && ./src/rm65_dex_bringup/scripts/build_ws.sh"

"${DOCKER[@]}" exec -d rm65_dex_ros2 bash -lc \
  "source /opt/ros/humble/setup.bash && \
   source /workspace/rm65_dex_ws/install/setup.bash && \
   ros2 launch rm65_dex_bringup quest_control_acceptance.launch.py use_rviz:=false \
   >/tmp/quest_control_acceptance.log 2>&1"

sleep 6

ros_exec "ros2 service call /servo_node/start_servo std_srvs/srv/Trigger '{}'" >/dev/null || true

echo "Controllers:"
ros_exec "ros2 control list_controllers"

echo
echo "Checking /servo_node/delta_twist_cmds ..."
python3 scripts/smoke_send_wrist_pose.py &
wrist_sender=$!
ros_exec "timeout 8s ros2 topic echo --once /servo_node/delta_twist_cmds"
wait "$wrist_sender" || true

echo
echo "Checking /arm_controller/joint_trajectory from MoveIt Servo ..."
python3 scripts/smoke_send_wrist_pose.py &
wrist_sender=$!
ros_exec "timeout 8s ros2 topic echo --once /arm_controller/joint_trajectory"
wait "$wrist_sender" || true

echo
echo "Checking /hand_position_controller/commands ..."
python3 scripts/smoke_send_hand_qpos.py &
hand_sender=$!
ros_exec "timeout 8s ros2 topic echo --once /hand_position_controller/commands"
wait "$hand_sender" || true

echo
echo "Smoke test passed. See container log: /tmp/quest_control_acceptance.log"
