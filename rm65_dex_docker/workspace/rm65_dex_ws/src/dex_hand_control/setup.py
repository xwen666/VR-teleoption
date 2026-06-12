import os
from glob import glob

from setuptools import find_packages, setup

package_name = "dex_hand_control"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
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
    description="Dex hand qpos socket controller bridge.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "hand_qpos_node = dex_hand_control.hand_qpos_node:main",
        ],
    },
)
