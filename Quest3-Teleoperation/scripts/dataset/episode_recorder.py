#!/usr/bin/env python3

import os
import sys
# Add project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import String
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from rclpy.qos import QoSPresetProfiles, HistoryPolicy
SENSOR_QOS = QoSPresetProfiles.SENSOR_DATA.value
from rclpy.exceptions import ROSInterruptException
from rclpy.executors import ExternalShutdownException

import numpy as np
import time
import argparse
from threading import Lock
import signal
from rclpy.executors import SingleThreadedExecutor

import cv2

from xrobotoolkit_teleop.utils.dataset.load_data_utils import (
    DATASET_DIR,
    DATASET_PKL,
    NEXT_ID_PATH,
    INSTR_CSV,
    EXPECTED_STATE_LEN,
    SYNC_TOL_SEC,
    get_next_episode_id,
)

from xrobotoolkit_teleop.utils.dataset.data_save_utils import (
    is_numeric_state,
    append_pickle,
    append_instruction_csv,
    raw_to_bgr,
    raw_to_jpg,
)

exit_flag = False
def handle_shutdown(signum, frame):
    global exit_flag
    exit_flag = True
    print("\n[Shutdown] Ctrl+C pressed. Cleaning up...")

signal.signal(signal.SIGINT, handle_shutdown)

class Recorder(Node):
    def __init__(self, save_format: str = "array_bgr", jpeg_quality: int = 90):
        super().__init__('recorder')
        # self.bridge = CvBridge()
        self.lock = Lock()
        self.save_format = (save_format or "array_bgr").lower()
        self.jpeg_quality = int(jpeg_quality)

        self.episode_id = get_next_episode_id()
        self.recording = False  # Initially not recording, wait for START

        # Only collect required data (no longer maintain preview frame)
        self.latest_image_side = None
        self.latest_image_wrist = None
        self.latest_image_head = None
        self.latest_state = None
        self.latest_action = None
        self.episode = []
        self.recording = False

        # Task language instruction
        self.language_instruction = "pick up the blue cube"

        # Subscribe to cameras and state (using your current topic names)
        self.image1_sub = self.create_subscription(Image, '/cam_head', self.image1_callback, SENSOR_QOS)
        self.image2_sub = self.create_subscription(Image, '/cam_wrist', self.image2_callback, SENSOR_QOS)
        self.image3_sub = self.create_subscription(Image, '/cam_side', self.image3_callback, SENSOR_QOS)
        self.state_sub = self.create_subscription(
            JointState, '/robot_state', self.state_callback,
            QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        )
        self.episode_cmd_sub = self.create_subscription(
            String, '/episode_control', self.episode_cmd_callback,
            QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE,
                       history=HistoryPolicy.KEEP_LAST, depth=1)
        )
        self.ts_side = None
        self.ts_wrist = None
        self.ts_head = None
        self.ts_state = None

        self.get_logger().info("Recorder node initialized (no preview)")
        self.get_logger().info(f"Ready for episode {self.episode_id}. Waiting START on /episode_control ...")

    # Convert YUYV raw data to BGR (called only during preview/save)
    def _yuyv_to_bgr(self, data_bytes: bytes, w: int, h: int):
        try:
            arr = np.frombuffer(data_bytes, dtype=np.uint8).reshape(h, w, 2)
            bgr = cv2.cvtColor(arr, cv2.COLOR_YUV2BGR_YUY2)
            return bgr
        except Exception as e:
            self.get_logger().error(f"YUYV->BGR conversion failed: {e}")
            return None

    # Camera callback: only cache raw bytes and timestamp
    def image1_callback(self, msg):
        try:
            raw = (memoryview(msg.data), msg.width, msg.height, msg.encoding)
        except Exception as e:
            self.get_logger().error(f"Failed to copy side_cam raw: {e}")
            return
        with self.lock:
            self.latest_image_side = raw
            self.ts_side = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            self.try_record()

    def image2_callback(self, msg):
        try:
            raw = (memoryview(msg.data), msg.width, msg.height, msg.encoding)
        except Exception as e:
            self.get_logger().error(f"Failed to copy wrist_cam raw: {e}")
            return
        with self.lock:
            self.latest_image_wrist = raw
            self.ts_wrist = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            self.try_record()

    def image3_callback(self, msg):
        try:
            raw = (memoryview(msg.data), msg.width, msg.height, msg.encoding)
        except Exception as e:
            self.get_logger().error(f"Failed to copy head_cam raw: {e}")
            return
        with self.lock:
            self.latest_image_head = raw
            self.ts_head = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            self.try_record()

    def state_callback(self, msg):
        with self.lock:
            arr = np.asarray(msg.position, dtype=np.float64).ravel()
            if not is_numeric_state(arr, EXPECTED_STATE_LEN):
                print(f"[recorder] drop state: invalid len={arr.size}")
                return
            self.latest_state = arr
            self.ts_state = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            self.try_record()

    def episode_cmd_callback(self, msg: String):
        cmd = (msg.data or "").strip().upper()
        if cmd == "START":
            self.start_episode()
        elif cmd == "STOP":
            self.stop_and_save_episode(reason=cmd)
        else:
            self.get_logger().warn(f"Unknown episode command: {cmd}")

    def _clear_cache(self):
        self.latest_image_side = None
        self.latest_image_wrist = None
        self.latest_image_head = None
        self.latest_state = None
        self.ts_side = self.ts_wrist = self.ts_head = self.ts_state = None
        self.episode = []

    def start_episode(self):
        with self.lock:
            if self.recording:
                self.get_logger().warn("Already recording. START ignored.")
                return
            self._clear_cache()
            self.recording = True
        self.get_logger().info(f"🔴 Start recording episode {self.episode_id}")

    def stop_and_save_episode(self, reason: str = "STOP"):
        with self.lock:
            was_recording = self.recording
            self.recording = False
            T = len(self.episode)
        if not was_recording and T == 0:
            self.get_logger().warn("Not recording or empty episode. Skip save.")
            return
        try:
            self.get_logger().info(f"⏹️ Saving episode {self.episode_id} (reason={reason}) with {T} steps ...")
            self.save_episode(self.episode_id)
            self.get_logger().info(f"✅ Episode {self.episode_id} saved.")
            self.episode_id += 1
        except Exception as e:
            self.get_logger().error(f"Failed to save episode: {e}")
        finally:
            with self.lock:
                self._clear_cache()
        self.get_logger().info(f"Ready for next episode {self.episode_id}. Waiting START...")

    def try_record(self):
        if not self.recording:
            return
        if (self.latest_image_side is not None and
            self.latest_image_wrist is not None and
            self.latest_image_head is not None and
            self.latest_state is not None and
            self.ts_side is not None and self.ts_wrist is not None and
            self.ts_head is not None and self.ts_state is not None):

            tmin = min(self.ts_side, self.ts_wrist, self.ts_head, self.ts_state)
            tmax = max(self.ts_side, self.ts_wrist, self.ts_head, self.ts_state)
            if (tmax - tmin) > SYNC_TOL_SEC:
                return

            if not is_numeric_state(self.latest_state, EXPECTED_STATE_LEN):
                print(f"[recorder] skip non-numeric state at step {len(self.episode)+1}")
                self.latest_state = None
                self.ts_state = None
                return

            self.episode.append({
                "image_side_raw": self.latest_image_side,
                "image_wrist_raw": self.latest_image_wrist,
                "image_head_raw": self.latest_image_head,
                "state": self.latest_state.copy(),
            })
            self.get_logger().info(f"Recorded step {len(self.episode)}")

            # Clear cache
            self.latest_image_side = None
            self.latest_image_wrist = None
            self.latest_image_head = None
            self.latest_state = None
            self.ts_side = self.ts_wrist = self.ts_head = self.ts_state = None

    def save_episode(self, episode_id: int):
        if not self.episode:
            print("[recorder] No data recorded, skipping save.")
            return

        os.makedirs(DATASET_DIR, exist_ok=True)

        try:
            episode_obj = {
                "metadata": {
                    "episode_id": episode_id,
                    "language_instruction": self.language_instruction,
                    "camera_ids": ["cam_side", "cam_wrist", "cam_head"],
                    "raw_encoding": "yuv422_yuy2",   # Record raw encoding
                    "stored_format": self.save_format,         # New: storage format
                    "jpeg_quality": (self.jpeg_quality if self.save_format == "jpg" else None),
                },
                "steps": []
            }

            bad = 0
            for i, ts in enumerate(self.episode):
                side_raw  = ts["image_side_raw"]
                wrist_raw = ts["image_wrist_raw"]
                head_raw  = ts["image_head_raw"]

                # Raw -> JPG or BGR (unified using image_* variables)
                if self.save_format == "jpg":
                    image_side  = raw_to_jpg(side_raw,  quality=self.jpeg_quality)
                    image_wrist = raw_to_jpg(wrist_raw, quality=self.jpeg_quality)
                    image_head  = raw_to_jpg(head_raw,  quality=self.jpeg_quality)
                else:
                    image_side  = raw_to_bgr(side_raw)
                    image_wrist = raw_to_bgr(wrist_raw)
                    image_head  = raw_to_bgr(head_raw)

                valid_imgs = (image_side is not None and image_wrist is not None and image_head is not None)

                state_vec = np.asarray(ts["state"], dtype=np.float64).ravel()
                if (state_vec.size != EXPECTED_STATE_LEN or
                    not np.all(np.isfinite(state_vec)) or
                    not valid_imgs):
                    bad += 1
                    continue

                episode_obj["steps"].append({
                    "observation": {
                        "images": {
                            "image_head":  image_head,
                            "image_side":  image_side,
                            "image_wrist": image_wrist,
                        },
                        "qpos": state_vec[:7],
                        "qvel": state_vec[13:20],
                        "ee_pose": state_vec[7:13],
                    },
                    "action": (np.asarray(self.episode[i + 1]['state'][:13], dtype=np.float64).ravel()
                               if i + 1 < len(self.episode) else state_vec[:13]),
                    "is_first": i == 0,
                    "is_last": i == len(self.episode) - 1,
                    "is_terminal": i == len(self.episode) - 1,
                })

            if bad:
                print(f"[recorder] warning: skipped {bad} invalid step(s) while saving")

            append_pickle(DATASET_PKL, episode_obj)
            append_instruction_csv(INSTR_CSV, episode_id, self.language_instruction, len(self.episode))
            with open(NEXT_ID_PATH, "w") as f:
                f.write(str(episode_id + 1))
            print(f"[recorder] ✅ Appended episode {episode_id} to {DATASET_PKL}")
        except Exception as e:
            print(f"[recorder] Failed to append episode: {e}")
        finally:
            self.episode = []

def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-format", choices=["array_bgr", "jpg"],
                        default=os.getenv("SAVE_FORMAT", "array_bgr"),
                        help="Image format: array_bgr (lossless) or jpg (lossy compression, space-saving)")
    parser.add_argument("--jpeg-quality", type=int, default=int(os.getenv("JPEG_QUALITY", "100")),
                        help="JPG quality (1-100), effective only when --save-format=jpg")
    parsed, _ = parser.parse_known_args(args=args)

    rclpy.init(args=args)

    recorder = None
    executor = None
    try:
        recorder = Recorder(save_format=parsed.save_format, jpeg_quality=parsed.jpeg_quality)
        executor = SingleThreadedExecutor()
        executor.add_node(recorder)

        recorder.get_logger().info("Waiting START command to begin recording (topic: /episode_control)")
 
        start_time = time.time()
        while True:
            if exit_flag or (not rclpy.ok()):
                break
            try:
                executor.spin_once(timeout_sec=0.02)
            except (KeyboardInterrupt, ExternalShutdownException):
                print("[recorder] Stop requested by user (SIGINT).")
                break
            except Exception as e:
                if not rclpy.ok():
                    break
                raise

    except (KeyboardInterrupt, ExternalShutdownException):
        print("[recorder] Stop requested by user (SIGINT).")
    except ROSInterruptException:
        print("ROS Interrupt received. Shutdown.")
    except Exception as e:
        print(f"[recorder] Unexpected error: {e}")
    finally:
        try:
            if recorder:
                if len(recorder.episode) > 0:
                    print("[recorder] ⏹️ Stopped recording. Saving current episode before exit...")
                    recorder.stop_and_save_episode(reason="SHUTDOWN")
                if 'start_time' in locals():
                    print(f"[recorder] 🕒 Total run time: {time.time() - start_time:.2f}s")
        except Exception as e:
            print(f"[recorder] Failed to save on exit: {e}")
        finally:
            if executor is not None:
                try:
                    if recorder:
                        executor.remove_node(recorder)
                    executor.shutdown()
                except Exception:
                    pass
            if recorder:
                recorder.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()

if __name__ == '__main__':
    main()