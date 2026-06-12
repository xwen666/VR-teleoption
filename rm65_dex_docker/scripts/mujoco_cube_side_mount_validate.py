#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import math
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_VALIDATOR = ROOT / "scripts" / "mujoco_dls_hybrid_validate.py"
DEFAULT_CONFIG = (
    ROOT
    / "workspace"
    / "rm65_dex_ws"
    / "src"
    / "quest_bridge"
    / "config"
    / "wrist_cube_side_mount.yaml"
)


# World frame for the cube scene:
#   +X: cube front
#   +Y: cube left
#   +Z: up
#
# Robot installation goal:
#   - robot base z axis perpendicular to the cube left face, i.e. base z -> world +Y
#   - robot "forward" direction should face the red front marker on the cube
# In practice the RM65 model's natural forward direction is flipped relative to our first
# side-mount attempt, so we keep base z -> +Y and rotate an extra 180 deg about base z.
CUBE_SIZE = 0.42
CUBE_CENTER_XYZ = (0.0, 0.0, CUBE_SIZE / 2.0)
ROBOT_BASE_IN_CUBE_FRAME_XYZ = (0.0, CUBE_SIZE / 2.0 + 0.03, 0.0)
ROBOT_BASE_IN_CUBE_FRAME_RPY = (math.pi / 2.0, 0.0, math.pi)


def load_base_module():
    spec = importlib.util.spec_from_file_location("mujoco_hybrid_validate_base", BASE_VALIDATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load base validator from {BASE_VALIDATOR}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def xyz_text(values: tuple[float, float, float]) -> str:
    return " ".join(f"{value:.6f}" for value in values)


def rpy_text(values: tuple[float, float, float]) -> str:
    return " ".join(f"{value:.6f}" for value in values)


def should_strip_scene_name(name: str) -> bool:
    lowered = name.lower()
    tokens = (
        "task_",
        "scene_world_to_table",
        "table",
        "bin",
    )
    return any(token in lowered for token in tokens)


def find_robot_root_link(root: ET.Element) -> str:
    link_names = [link.attrib["name"] for link in root.findall("link") if "name" in link.attrib]
    child_links = {
        child.attrib["link"]
        for joint in root.findall("joint")
        for child in joint.findall("child")
        if "link" in child.attrib
    }
    if "base_link" in link_names:
        return "base_link"
    root_candidates = [name for name in link_names if name not in child_links]
    if not root_candidates:
        raise RuntimeError("Could not determine robot root link in generated URDF.")
    return root_candidates[0]


def add_origin(parent: ET.Element, xyz: tuple[float, float, float], rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> None:
    ET.SubElement(parent, "origin", xyz=xyz_text(xyz), rpy=rpy_text(rpy))


def add_box_visual_and_collision(parent: ET.Element, size_xyz: tuple[float, float, float], color_rgba: str) -> None:
    visual = ET.SubElement(parent, "visual")
    ET.SubElement(visual, "origin", xyz="0 0 0", rpy="0 0 0")
    visual_geometry = ET.SubElement(visual, "geometry")
    ET.SubElement(visual_geometry, "box", size=xyz_text(size_xyz))
    material = ET.SubElement(visual, "material", name=f"{parent.attrib['name']}_material")
    ET.SubElement(material, "color", rgba=color_rgba)

    collision = ET.SubElement(parent, "collision")
    ET.SubElement(collision, "origin", xyz="0 0 0", rpy="0 0 0")
    collision_geometry = ET.SubElement(collision, "geometry")
    ET.SubElement(collision_geometry, "box", size=xyz_text(size_xyz))


def add_inertial_box(parent: ET.Element, mass: float, size_xyz: tuple[float, float, float]) -> None:
    sx, sy, sz = size_xyz
    ixx = mass * (sy * sy + sz * sz) / 12.0
    iyy = mass * (sx * sx + sz * sz) / 12.0
    izz = mass * (sx * sx + sy * sy) / 12.0
    inertial = ET.SubElement(parent, "inertial")
    ET.SubElement(inertial, "origin", xyz="0 0 0", rpy="0 0 0")
    ET.SubElement(inertial, "mass", value=f"{mass:.6f}")
    ET.SubElement(
        inertial,
        "inertia",
        ixx=f"{ixx:.6f}",
        ixy="0.0",
        ixz="0.0",
        iyy=f"{iyy:.6f}",
        iyz="0.0",
        izz=f"{izz:.6f}",
    )


def rewrite_generated_urdf_for_cube_scene(urdf_path: str) -> str:
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    removable_links = []
    removable_link_names = set()
    for link in root.findall("link"):
        name = link.attrib.get("name", "")
        if should_strip_scene_name(name):
            removable_links.append(link)
            removable_link_names.add(name)

    removable_joints = []
    for joint in root.findall("joint"):
        joint_name = joint.attrib.get("name", "")
        parent_link = joint.find("parent").attrib.get("link", "") if joint.find("parent") is not None else ""
        child_link = joint.find("child").attrib.get("link", "") if joint.find("child") is not None else ""
        if (
            should_strip_scene_name(joint_name)
            or parent_link in removable_link_names
            or child_link in removable_link_names
            or should_strip_scene_name(parent_link)
            or should_strip_scene_name(child_link)
        ):
            removable_joints.append(joint)

    for joint in removable_joints:
        root.remove(joint)
    for link in removable_links:
        root.remove(link)

    robot_root_link = find_robot_root_link(root)

    # The generated combined URDF already mounts the robot root under a world link.
    # For the cube-side-mount scene we must detach that original parent first,
    # otherwise MuJoCo sees two parents for the same body.
    robot_parent_link_name: str | None = None
    robot_parent_joints = []
    for joint in root.findall("joint"):
        child = joint.find("child")
        if child is None:
            continue
        if child.attrib.get("link") != robot_root_link:
            continue
        robot_parent_joints.append(joint)
        parent = joint.find("parent")
        if parent is not None and robot_parent_link_name is None:
            robot_parent_link_name = parent.attrib.get("link")

    for joint in robot_parent_joints:
        root.remove(joint)

    if robot_parent_link_name is None:
        robot_parent_link_name = "cube_scene_world"
        ET.SubElement(root, "link", name=robot_parent_link_name)

    cube_link = ET.SubElement(root, "link", name="cube_side_mount_block")
    add_inertial_box(cube_link, mass=12.0, size_xyz=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE))
    add_box_visual_and_collision(cube_link, size_xyz=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE), color_rgba="0.20 0.45 0.90 1.0")

    cube_joint = ET.SubElement(root, "joint", name="cube_scene_world_to_block", type="fixed")
    add_origin(cube_joint, xyz=CUBE_CENTER_XYZ)
    ET.SubElement(cube_joint, "parent", link=robot_parent_link_name)
    ET.SubElement(cube_joint, "child", link=cube_link.attrib["name"])

    front_marker_link = ET.SubElement(root, "link", name="cube_front_marker")
    add_inertial_box(front_marker_link, mass=0.01, size_xyz=(0.08, 0.02, 0.02))
    add_box_visual_and_collision(front_marker_link, size_xyz=(0.08, 0.02, 0.02), color_rgba="0.95 0.25 0.25 1.0")
    front_marker_joint = ET.SubElement(root, "joint", name="cube_block_to_front_marker", type="fixed")
    add_origin(front_marker_joint, xyz=(CUBE_SIZE / 2.0 + 0.05, 0.0, 0.0))
    ET.SubElement(front_marker_joint, "parent", link=cube_link.attrib["name"])
    ET.SubElement(front_marker_joint, "child", link=front_marker_link.attrib["name"])

    left_marker_link = ET.SubElement(root, "link", name="cube_left_marker")
    add_inertial_box(left_marker_link, mass=0.01, size_xyz=(0.02, 0.08, 0.02))
    add_box_visual_and_collision(left_marker_link, size_xyz=(0.02, 0.08, 0.02), color_rgba="0.25 0.85 0.35 1.0")
    left_marker_joint = ET.SubElement(root, "joint", name="cube_block_to_left_marker", type="fixed")
    add_origin(left_marker_joint, xyz=(0.0, CUBE_SIZE / 2.0 + 0.05, 0.0))
    ET.SubElement(left_marker_joint, "parent", link=cube_link.attrib["name"])
    ET.SubElement(left_marker_joint, "child", link=left_marker_link.attrib["name"])

    robot_mount_joint = ET.SubElement(root, "joint", name="cube_block_to_robot_base", type="fixed")
    add_origin(
        robot_mount_joint,
        xyz=ROBOT_BASE_IN_CUBE_FRAME_XYZ,
        rpy=ROBOT_BASE_IN_CUBE_FRAME_RPY,
    )
    ET.SubElement(robot_mount_joint, "parent", link=cube_link.attrib["name"])
    ET.SubElement(robot_mount_joint, "child", link=robot_root_link)

    output_dir = Path(tempfile.mkdtemp(prefix="rm65_cube_side_mount_"))
    output_path = output_dir / "rm65_cube_side_mount.urdf"
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    return str(output_path)


def main() -> None:
    base_module = load_base_module()
    if not hasattr(base_module, "build_combined_urdf"):
        raise RuntimeError("Base validator does not expose build_combined_urdf; wrapper cannot patch the scene safely.")
    if not hasattr(base_module, "main"):
        raise RuntimeError("Base validator does not expose main(); wrapper cannot launch it.")

    original_build_combined_urdf = base_module.build_combined_urdf

    def patched_build_combined_urdf(*args, **kwargs):
        urdf_path = original_build_combined_urdf(*args, **kwargs)
        patched_path = rewrite_generated_urdf_for_cube_scene(urdf_path)
        print(
            "Patched cube-side-mount scene:"
            f" cube_size={CUBE_SIZE:.3f}m,"
            f" base_mount_xyz={ROBOT_BASE_IN_CUBE_FRAME_XYZ},"
            f" base_mount_rpy={ROBOT_BASE_IN_CUBE_FRAME_RPY},"
            f" output={patched_path}"
        )
        return patched_path

    base_module.build_combined_urdf = patched_build_combined_urdf
    if hasattr(base_module, "DEFAULT_CONFIG_PATH"):
        base_module.DEFAULT_CONFIG_PATH = DEFAULT_CONFIG

    forwarded_args = sys.argv[1:]
    if "--config" not in forwarded_args:
        forwarded_args.extend(["--config", str(DEFAULT_CONFIG)])
    sys.argv = [str(BASE_VALIDATOR)] + forwarded_args
    base_module.main()


if __name__ == "__main__":
    main()
