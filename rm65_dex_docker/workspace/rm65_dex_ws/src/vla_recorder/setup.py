from setuptools import find_packages, setup

package_name = "vla_recorder"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
    ],
    install_requires=["setuptools", "numpy"],
    zip_safe=True,
    maintainer="xwen",
    maintainer_email="xwen@example.com",
    description="Smoke recorder for robot state/action/image synchronization checks.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "smoke_recorder = vla_recorder.smoke_recorder:main",
        ],
    },
)
