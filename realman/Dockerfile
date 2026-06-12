FROM scratch

ADD ubuntu-base.tar.gz /

ENV DEBIAN_FRONTEND=noninteractive
ENV SUDO_USER=root
ENV TZ=Asia/Shanghai

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
 && echo $TZ > /etc/timezone \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
    apt-utils \
    ca-certificates \
    curl \
    dirmngr \
    feh \
    ffmpeg \
    gnupg \
    iputils-ping \
    libdbus-1-3 \
    libegl1 \
    libfontconfig1 \
    libgl1 \
    libglib2.0-0 \
    libopengl0 \
    libsm6 \
    libx11-6 \
    libxext6 \
    libxkbcommon-x11-0 \
    libxcb-cursor0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-shape0 \
    libxcb-shm0 \
    libxcb-sync1 \
    libxcb-xfixes0 \
    libxcb-xinerama0 \
    libxcb-xinput0 \
    libxcb-xkb1 \
    libxcb1 \
    libxrender1 \
    lsb-release \
    python3-pip \
    python3-tk \
    software-properties-common \
    sudo \
    tzdata \
    usbutils \
    x11-apps \
    xauth \
 && curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    | gpg --dearmor -o /etc/apt/trusted.gpg.d/ros-archive-keyring.gpg \
 && printf 'Defaults env_keep += "DEBIAN_FRONTEND TZ"\n' > /etc/sudoers.d/noninteractive-env \
 && chmod 0440 /etc/sudoers.d/noninteractive-env \
 && ln -sfn /root /home/root \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

CMD ["/bin/bash"]
