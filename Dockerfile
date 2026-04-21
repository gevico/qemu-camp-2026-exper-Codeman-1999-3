FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# 1. 替换 Ubuntu APT 为清华源，并启用 deb-src
RUN sed -i 's@//.*archive.ubuntu.com@//mirrors.tuna.tsinghua.edu.cn@g' /etc/apt/sources.list.d/ubuntu.sources && \
	sed -i 's@//.*security.ubuntu.com@//mirrors.tuna.tsinghua.edu.cn@g' /etc/apt/sources.list.d/ubuntu.sources && \
	sed -i 's/^Types: deb$/Types: deb deb-src/' /etc/apt/sources.list.d/ubuntu.sources && \
	apt-get update && \
	# 2. 移除 build-dep qemu，改为安装核心的编译依赖，大幅减少下载和安装时间
	apt-get install -y --no-install-recommends \
	curl git gdb clang libclang-dev build-essential pkg-config ninja-build \
	libglib2.0-dev libpixman-1-dev python3-venv python3-setuptools flex bison && \
	rm -rf /var/lib/apt/lists/*

# 3. 设置 Rustup 国内加速镜像
ENV RUSTUP_DIST_SERVER="https://mirrors.tuna.tsinghua.edu.cn/rustup"
ENV RUSTUP_UPDATE_ROOT="https://mirrors.tuna.tsinghua.edu.cn/rustup/rustup"

# 安装 Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# 4. 配置 Cargo 的清华源加速，再安装 bindgen-cli
RUN mkdir -p $HOME/.cargo && echo '[source.crates-io]\n\
	replace-with = "tuna"\n\
	[source.tuna]\n\
	registry = "https://mirrors.tuna.tsinghua.edu.cn/git/crates.io-index.git"\n\
	[net]\n\
	git-fetch-with-cli = true' > $HOME/.cargo/config.toml && \
	cargo install bindgen-cli && \
	git config --global --add safe.directory '*'

WORKDIR /src
