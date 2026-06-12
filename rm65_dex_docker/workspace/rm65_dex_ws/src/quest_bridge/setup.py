import os
from glob import glob

from setuptools import find_packages, setup

package_name = "quest_bridge"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    package_data={package_name: ["libapi_cpp.so"]},
    include_package_data=True,
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools", "numpy"],
    zip_safe=True,
    maintainer="xwen",
    maintainer_email="xwen@example.com",
    description="Quest wrist/controller socket bridge for MoveIt Servo.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "wrist_twist_bridge = quest_bridge.wrist_twist_bridge:main",
            "wrist_ik_bridge = quest_bridge.wrist_ik_bridge:main",
        ],
    },
)
