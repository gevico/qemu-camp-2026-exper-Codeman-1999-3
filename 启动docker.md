# 启动 Docker 环境

## 1. 构建镜像

在项目根目录下执行：

```bash
docker build -t qemu-camp .
```

## 2. 运行容器

```bash
docker run -it --rm \
  -v $(pwd):/src \
  qemu-camp \
  bash
```

这样会将当前项目目录挂载到容器的 `/src` 工作目录，你可以在容器内编译和测试。

## 3. 容器内操作

进入容器后，按 `README_zh.md` 的流程操作：

```bash
# 配置
make -f Makefile.camp configure

# 编译
make -f Makefile.camp build

# 运行测试（根据你的实验方向选择）
make -f Makefile.camp test-gpgpu
```

## 补充说明

- **`.dockerignore`** 中 `WORKDIR` 设为 `/src`，构建时会将源码 COPY 进去（因为 `.dockerignore` 只排除了一部分），但更推荐用 `-v` 挂载方式，这样容器内修改的文件会同步到宿主机。
- 如果你需要**保留容器内的构建产物**（如 `build/` 目录），建议用挂载方式而不是重新 `docker build`。
- 如果需要 RISC-V 交叉编译工具链（CPU 实验必需），`Dockerfile` 中未安装，需自行在容器内按 README 步骤安装，或修改 Dockerfile 添加。
