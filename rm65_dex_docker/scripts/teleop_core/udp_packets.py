from __future__ import annotations

import json
import socket
from typing import Optional

import numpy as np


def parse_landmark_array(landmarks_value: object) -> Optional[np.ndarray]:
    if landmarks_value is None:
        return None
    try:
        landmarks = np.array(landmarks_value, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if landmarks.shape != (21, 3):
        return None
    return landmarks


def parse_wrist_packet(data: bytes) -> Optional[tuple[np.ndarray, Optional[np.ndarray]]]:
    text = data.decode("utf-8", errors="ignore").strip()
    if not text:
        return None
    try:
        packet = json.loads(text)
        wrist = np.array(packet["wrist_pose"], dtype=np.float32)
        landmarks = parse_landmark_array(packet.get("landmarks"))
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        try:
            values = [float(v.strip()) for v in text.split(":", 1)[1].split(",") if v.strip()]
            wrist = np.array(values, dtype=np.float32)
            landmarks = None
        except (IndexError, ValueError):
            return None
    return (wrist, landmarks) if wrist.shape[0] >= 7 else None


def parse_hand_qpos_packet(data: bytes) -> Optional[np.ndarray]:
    text = data.decode("utf-8", errors="ignore").strip()
    if not text:
        return None
    try:
        packet = json.loads(text)
        qpos = np.array(packet["hand_qpos"], dtype=np.float32)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None
    return qpos if qpos.shape[0] >= 6 else None


def poll_latest_packet(sock: socket.socket, parser):
    latest = None
    try:
        while True:
            packet, _ = sock.recvfrom(65535)
            parsed = parser(packet)
            if parsed is not None:
                latest = parsed
    except BlockingIOError:
        pass
    return latest

