FROM registry.dev.oo/library/ros-dev:v0.1

RUN apt update && apt install -y python3-tk usbutils && \
	ssh_pubkey="ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIDHwkNejQjqMDKvVZTksK6lxGJxFIeydXf/tSn+wYUnz o" && \
	mkdir -p /root/.ssh && echo "$ssh_pubkey" > /root/.ssh/authorized_keys
