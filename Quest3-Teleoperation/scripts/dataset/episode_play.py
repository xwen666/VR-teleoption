#!/usr/bin/env python3
import os
import sys
# Add project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xrobotoolkit_teleop.utils.dataset.data_play_utils import (
    play_mujoco_with_preview,
    play_opencv_with_controls,
    build_traj_image,
)

from xrobotoolkit_teleop.utils.dataset.load_data_utils import load_episode

def parse_size(s: str, default=(1280, 720)):
    try:
        w, h = s.lower().split("x")
        return int(w), int(h)
    except Exception:
        return default

def parse_args(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Episode playback (no ROS)")
    ap.add_argument("pkl", help="path to episode .pkl or dataset.pkl")
    ap.add_argument("--index", type=int, default=-1, help="episode index in dataset.pkl (default: last)")
    ap.add_argument("--fps", type=float, default=10.0, help="playback FPS")
    ap.add_argument("--show-traj", action="store_true", help="show state trajectory window")

    # MuJoCo related arguments
    ap.add_argument("--mujoco-xml", type=str, default='/home/lyh/RealMan-Quest3-teleoperation/assets/realman/eco_63b/scene_eco_63b.xml', help="path to MuJoCo XML; if set, use MuJoCo to visualize joints")
    ap.add_argument("--joint-start", type=int, default=0, help="start index of joint vector in state")
    ap.add_argument("--joint-dims", type=int, default=6, help="number of joint dims to use")
    ap.add_argument("--state-in-deg", action="store_true", help="state joint angles are in degree (convert to rad)")

    # Video recording
    ap.add_argument("--save-img-video", type=str, default='outputs/episode_video.mp4', help="path to save OpenCV mosaic video (mp4)")
    ap.add_argument("--save-mj-video", type=str, default='outputs/episode_mj.mp4', help="path to save MuJoCo render video (mp4)")
    ap.add_argument("--mj-size", type=str, default="960x540", help="size for MuJoCo offscreen video, e.g. 1280x720")
    ap.add_argument("--mj-camera", type=str, default='side', help="MuJoCo camera name for recording (None=free)")
    args = ap.parse_args(argv or sys.argv[1:])

    return args

def main(argv=None):
    args = parse_args(argv)

    if not os.path.exists(args.pkl):
        print(f"[player] File not found: {args.pkl}")
        sys.exit(1)
    ep, steps = load_episode(args.pkl, index=args.index)
    instruction = ep.get("metadata", {}).get("language_instruction", None)

    if len(steps) == 0:
        print("[player] Empty episode."); sys.exit(0)

    # If MuJoCo XML is provided, enter synchronized visualization: MuJoCo + images
    if args.mujoco_xml:
        play_mujoco_with_preview(
            steps,
            xml_path=args.mujoco_xml,
            joint_start=args.joint_start,
            joint_dims=args.joint_dims,
            fps=max(args.fps, 1.0),
            deg=args.state_in_deg,
            save_img_video=args.save_img_video,
            save_mj_video=args.save_mj_video,
            mj_video_size=parse_size(args.mj_size),
            mj_camera=args.mj_camera,
            instruction=instruction
        )
        return
    # Otherwise: build trajectory image + OpenCV playback
    traj_img = build_traj_image(steps, args.show_traj, window_name="State Trajectory")
    play_opencv_with_controls(
        steps, 
        args.fps, 
        traj_img, 
        window_name="Episode Playback", 
        save_img_video=args.save_img_video, 
        instruction=instruction
        )

if __name__ == "__main__":
    main()