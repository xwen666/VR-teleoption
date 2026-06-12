<div align="center">

# Quest3-Teleoperation
### VLA Data Collection Framework for Embodied AI

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)](https://www.python.org/)
[![ROS2](https://img.shields.io/badge/ROS2-Humble-green?logo=ros&logoColor=white)](https://docs.ros.org/en/humble/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

[**Overview**](#overview) | [**Installation**](#installation) | [**Usage**](#usage) | [**Data Workflow**](#efficient-data-collection-workflow)

</div>

---

## 📖 Overview

This project provides a robust, high-frequency teleoperation and data collection framework using the **Meta Quest 3** mixed reality headset. 

While our real-world data collection and validation were conducted exclusively on **Realman Robotic Arms (Eco63b / RML63b)**, the core teleoperation end-effector control and data collection pipeline are **highly universal and applicable to any commercially available robotic arm** (e.g., Franka, UR5). 

To adapt this framework for your own robot, you simply need to replace the Realman-specific end-effector control logic in the `realman_increamental_controller.py` file with the control API of your target robotic arm.

Forked and evolved from [XRoboToolkit-Teleop](XROBO.md), this framework introduces significant architectural improvements designed for high-precision **Imitation Learning** and **Embodied AI** research. It effectively bridges the gap between VR-based human intent capture and industrial-grade robot control via ROS 2 middleware.

### ✨ Key Features

* **🕹️ Incremental End-Effector Control (Delta Pose)**
    * Implements a relative control scheme where the robot follows the *delta* (changes) of the VR controller.
    * **Benefit:** Solves workspace mismatch issues and enables sub-millimeter precision for fine manipulation tasks. Includes an intuitive "Clutch" mechanism for re-centering.

* **📡 High-Fidelity ROS 2 Data Pipeline**
    * Replaces legacy VR return data with a direct **ROS 2 subscription pipeline**.
    * **Benefit:** Captures Joint States ($q, \dot{q}$) and End-Effector Poses directly from the robot controller, ensuring "Ground Truth" data quality essential for stable VLA (Vision-Language-Action) model training.

* **🦾 Native Realman Support**
    * Full kinematic adaptation and hardware interface support for **Realman Eco63b** and **RML63b** series.

---

## ⚙️ Installation

### Prerequisites

| Component | Requirement | Note |
| :--- | :--- | :--- |
| **OS** | 22.04 | Linux environment required |
| **ROS 2** | Humble | Ensure `rclpy` is accessible |
| **Python** | 3.10 | Conda env recommended (other versions not yet verified) |
| **Hardware** | Intel RealSense | SDK 2.0 required |

### Setup Guide

#### 1. Dependencies Configuration
Ensure your environment can bridge Conda and ROS 2 system packages.
```bash
# Install RealSense Python bindings
pip install pyrealsense2

# Note: Ensure 'import rclpy' works in your virtual environment.
# You may need to source your ROS 2 setup script (e.g., source /opt/ros/humble/setup.bash)
```

#### 2. RealSense Camera Dependencies
The framework uses Intel RealSense cameras (e.g., D435i, D405) for visual observation.

* **System Driver:** Please install the **Intel RealSense SDK 2.0 (`librealsense2`)** following the [Official Installation Guide](https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md).
* **Python Binding:** The Python wrapper `pyrealsense2` is required.
    ```bash
    # If not installed via the setup script:
    pip install pyrealsense2
    ```

#### 3. Project Installation
* **Clone This Repo:**

```bash
git clone https://github.com/is-aHuan/Quest3-Teleoperation.git
cd Quest3-Teleoperation
```

Refer to [XRoboToolkit-Teleop-Sample-Python](XROBO.md), install the required basic dependencies.

## 🚀 Usage

### 1. Launch Camera Streams
Initialize the multi-view camera nodes (Head, Side, Wrist) to broadcast visual observations to ROS 2.

**Run in 3 separate terminals:**
```bash
# Terminal 1: Head Camera (Global View)
python scripts/dataset/cam_pub.py --camera /cam_head

# Terminal 2: Side Camera (Third-person View)
python scripts/dataset/cam_pub.py --camera /cam_side

# Terminal 3: Wrist Camera (Egocentric View)
python scripts/dataset/cam_pub.py --camera /cam_wrist
```
This script is located in [came_pub](scripts/dataset/cam_pub.py). You can find more info in this file.

### 2. Launch Data Record Node

Start the episode logging node. This listens to all topics and buffers data.

```bash
# Terminal 4
python scripts/dataset/episode_recorder.py
```
Refer to [episode-recorder](scripts/dataset/episode_recorder.py) for more info.


### 3. Launch Robot Controller

Start the teleoperation interface. We recommend the **Incremental Control** mode for high-precision tasks.

```bash
# Terminal 5 (Example for Realman RML63b)
python scripts/hardware/teleop_rml63b_hardware.py
```

---

## 🎮 Teleoperation Guide

### Controller Mapping
The system separates **Data State Management** (Left Hand) from **Robot Actuation** (Right Hand) to prevent accidental movements.

<div align="center">
  <img src="media/controller.png" width="80%" alt="Quest 3 Controller Mapping with Labels">
  <br>
  <em>Figure: Quest 3 Controller Button Layout</em>
</div>

### Button Functions

| Hand | Component | Label in Fig | Action | Function |
| :--- | :--- | :---: | :--- | :--- |
| **Left** | **Grip Button** | **1** | Press | 🔴 **Start Recording**<br>System begins logging frame data. |
| **Left** | **Index Trigger** | **4** | Press | 💾 **Stop & Save**<br>Ends episode and flushes data to disk. |
| **Right** | **Grip Button** | **1** | **Hold** | 🟢 **Engage Clutch (Move Robot)**<br>Robot follows hand movement (Incremental). |
| **Right** | **Grip Button** | **1** | **Release** | ⏸️ **Disengage Clutch (Stop)**<br>Robot freezes. Allows operator to re-center hand. |
| **Right** | **Index Trigger** | **4** | Analog | 🤏 **Gripper Control**<br>Linear mapping to gripper width. |

---

### ♻️ Efficient Data Collection Workflow

Follow this **"Start-Execute-Stop-Reset"** loop to rapidly collect high-quality demonstrations:

1.  **Start Episode (Left Grip):**
    * Console confirms: `[INFO] Recording Started`.
2.  **Execute Task (Right Hand):**
    * **Hold Right Grip** to move robot; **Release** to adjust your posture (Clutch).
    * Use **Right Trigger** to interact with objects.
3.  **Finish & Save (Left Trigger):**
    * System saves the trajectory to a `.pkl` file.
4.  **Reset Scene (Inter-Episode):**
    * *Note: Robot is still controllable.* Use the robot to reset objects or clear the workspace.
    * **Loop:** Return to Step 1 immediately.


## 📊 Dataset Visualization & Management

### Episode Replay
Visualize your collected data (video + robot state) to verify quality.

```bash
python scripts/dataset/episode_play.py [path_to_pkl] --index [episode_idx]
```
<div align="center"> <img src="media/episode_video.gif" height="200" alt="Global View Demo" style="margin-right: 10px;"/> <img src="media/episode_mj.gif" height="200" alt="Simulation View Demo"/>

<em>Visualization: Real-world Camera Stream (Top) vs. Recorded State Replay (Bottom)</em> </div>

### Data Filtering

Remove bad episodes (human error, failures) and re-index the dataset.

```bash
# 1. Delete specific episode
python xrobotoolkit_teleop/utils/dataset/episode_delete_utils.py --pkl [path_to_pkl] --index [idx_to_delete]

# 2. Re-index dataset (Update metadata)
python xrobotoolkit_teleop/utils/dataset/reindex_dataset_utils.py --pkl [path_to_pkl]
```
More info is shown in [Episode Delete](xrobotoolkit_teleop/utils/dataset/episode_delete_utils.py) and [Reindex](xrobotoolkit_teleop/utils/dataset/reindex_dataset_utils.py)


## 🧩 Advanced Configuration

* **Robot State Publishing:** Configurable standalone ROS 2 node for state broadcasting.

* **PKL Structure:** See `scripts/dataset/episode_recorder.py` for the detailed data schema definition.

* **Lerobot Format:** 🤗 Refer to **[LeRobot Dataset](https://github.com/huggingface/lerobot)**, convert `.pkl` file to Lerobot format. Additionally, we have open-sourced our Realman teleoperation dataset in LeRobot format! You can explore and download it on Hugging Face: **[yonghuanli/realman_all](https://huggingface.co/datasets/yonghuanli/realman_all)**.