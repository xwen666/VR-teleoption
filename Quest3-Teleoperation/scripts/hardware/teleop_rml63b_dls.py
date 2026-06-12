#!/usr/bin/env python3
"""
Launch script for RealMan DLS IK Controller.

This controller uses Local Damped Least Squares IK instead of exact IK,
providing smoother motion and avoiding getting stuck near singularities.

Usage:
    python scripts/hardware/teleop_rml63b_dls.py
"""

import signal
import sys
import threading

from xrobotoolkit_teleop.common.xr_client import XrClient
from xrobotoolkit_teleop.hardware.realman_dls_controller import ArmRealManDLSController


def main():
    print("[DLS Teleop] Starting RealMan DLS IK teleoperation...")
    
    xr_client = XrClient()
    
    controller = ArmRealManDLSController(
        xr_client=xr_client,
        dls_damping=0.05,
        dls_gain=0.5,
        centering_gain=0.01,
        singularity_threshold=0.1,
        max_joint_delta_deg=2.0,
        scale_factor=1.2,
    )
    
    stop_event = threading.Event()
    
    def signal_handler(sig, frame):
        print(f"\n[DLS Teleop] Received signal {sig}, stopping...")
        stop_event.set()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        controller.run(stop_event)
    except Exception as e:
        print(f"[DLS Teleop] Error: {e}")
        stop_event.set()
    
    print("[DLS Teleop] Controller stopped.")
    controller.close()
    
    sys.exit(0)


if __name__ == "__main__":
    main()