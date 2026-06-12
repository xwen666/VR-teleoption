import os
from glob import glob

from setuptools import find_packages, setup

package_name = "rm65_dex_bringup"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="xwen",
    maintainer_email="xwen@example.com",
    description="Launch and config helpers for RM65 + dex hand MVP.",
    license="MIT",
)
