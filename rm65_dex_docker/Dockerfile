FROM osrf/ros:humble-desktop-full

ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=humble
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

SHELL ["/bin/bash", "-c"]

RUN apt update && apt install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    vim \
    nano \
    htop \
    tmux \
    net-tools \
    iputils-ping \
    usbutils \
    udev \
    sudo \
    lsb-release \
    software-properties-common \
    ninja-build \
    python3-pip \
    python3-dev \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-vcstool \
    python3-argcomplete \
    && rm -rf /var/lib/apt/lists/*

RUN apt update && apt install -y \
    ros-humble-moveit \
    ros-humble-moveit-servo \
    ros-humble-moveit-setup-assistant \
    ros-humble-ros2-control \
    ros-humble-ros2-controllers \
    ros-humble-joint-state-publisher \
    ros-humble-joint-state-publisher-gui \
    ros-humble-robot-state-publisher \
    ros-humble-xacro \
    ros-humble-tf2-tools \
    ros-humble-rviz2 \
    ros-humble-controller-manager \
    ros-humble-joint-trajectory-controller \
    ros-humble-forward-command-controller \
    ros-humble-image-transport \
    ros-humble-cv-bridge \
    ros-humble-camera-info-manager \
    liburdfdom-tools \
    android-tools-adb \
    android-tools-fastboot \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir \
    numpy \
    scipy \
    opencv-python \
    pandas \
    pyarrow \
    pyyaml \
    matplotlib \
    scikit-build-core \
    nanobind \
    tqdm

ARG USERNAME=ros
ARG USER_UID=1000
ARG USER_GID=1000

RUN groupadd --gid ${USER_GID} ${USERNAME} || true && \
    useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME} || true && \
    echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

RUN rosdep init || true
RUN rosdep update || true

RUN echo "source /opt/ros/humble/setup.bash" >> /home/${USERNAME}/.bashrc && \
    echo "if [ -f /workspace/rm65_dex_ws/install/setup.bash ]; then source /workspace/rm65_dex_ws/install/setup.bash; fi" >> /home/${USERNAME}/.bashrc

WORKDIR /workspace
USER ${USERNAME}

CMD ["/bin/bash"]
