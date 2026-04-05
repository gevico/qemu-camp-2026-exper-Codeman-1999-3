<p align="center"><strong>QEMU 训练营 2026 — 实验仓库</strong></p>
<p align="center"><a href="README.md">English</a> | <a href="README_zh.md">中文</a></p>

本仓库是 QEMU 训练营 2026 专业阶段的实验仓库，涵盖四个实验方向，均基于 RISC-V 架构。

## 在线讲义

| 方向 | 实验手册 | 硬件手册 / 编程指南 |
|------|---------|-------------------|
| **CPU** | [CPU 实验手册](https://qemu.gevico.online/exercise/2026/stage1/cpu/cpu-exper-manual/) | [CPU Datasheet](https://qemu.gevico.online/exercise/2026/stage1/cpu/cpu-datasheet/) |
| **SoC** | [SoC 实验手册](https://qemu.gevico.online/exercise/2026/stage1/soc/g233-exper-manual/) | [G233 SoC 硬件手册](https://qemu.gevico.online/exercise/2026/stage1/soc/g233-datasheet/) |
| **GPGPU** | [GPU 实验手册](https://qemu.gevico.online/exercise/2026/stage1/gpu/gpu-exper-manual/) | [GPU 硬件手册](https://qemu.gevico.online/exercise/2026/stage1/gpu/gpu-datasheet/) |
| **Rust** | [Rust 实验手册](https://qemu.gevico.online/exercise/2026/stage1/rust/rust-exper-manual/) | [Rust 编程指南](https://qemu.gevico.online/exercise/2026/stage1/rust/rust-lang-manual/) |

完整讲义网站：<https://qemu.gevico.online/>

## 实验方向

| 方向 | 测试框架 | 测试位置 | 评分 |
|------|---------|---------|------|
| **CPU** | TCG 测题 | `tests/gevico/tcg/` | 10 题 x 10 分 = 100 分 |
| **SoC** | QTest | `tests/gevico/qtest/` | 10 题 x 10 分 = 100 分 |
| **GPGPU** | QTest (QOS) | `tests/qtest/gpgpu-test.c` | 17 题 -> 100 分 |
| **Rust** | QTest + 单元测试 | `tests/gevico/qtest/` + `rust/hw/i2c/` | 10 题 x 10 分 = 100 分 |

## 快速开始

### 第一步：安装依赖

```bash
# Ubuntu 24.04
sudo sed -i 's/^Types: deb$/Types: deb deb-src/' /etc/apt/sources.list.d/ubuntu.sources
sudo apt-get update
sudo apt-get build-dep -y qemu

# 安装 RISC-V 裸机交叉编译器（CPU 实验必需）
sudo mkdir -p /opt/riscv
wget -q https://github.com/riscv-collab/riscv-gnu-toolchain/releases/download/2025.09.28/riscv64-elf-ubuntu-24.04-gcc-nightly-2025.09.28-nightly.tar.xz -O riscv-toolchain.tar.xz
sudo tar -xJf riscv-toolchain.tar.xz -C /opt/riscv --strip-components=1
sudo chown -R $USER:$USER /opt/riscv
export PATH="/opt/riscv/bin:$PATH"
echo 'export PATH="/opt/riscv/bin:$PATH"' >> ~/.bashrc
riscv64-unknown-elf-gcc --version

# 安装 Rust 工具链（Rust 实验及构建必需）
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
. "$HOME/.cargo/env"
cargo install bindgen-cli
```

### 第二步：配置

```bash
make -f Makefile.camp configure
```

### 第三步：编译

```bash
make -f Makefile.camp build
```

### 第四步：运行测试

```bash
# 运行指定方向的实验测试
make -f Makefile.camp test-cpu
make -f Makefile.camp test-soc
make -f Makefile.camp test-gpgpu
make -f Makefile.camp test-rust

# 运行所有实验测试
make -f Makefile.camp test
```

### 第五步：提交代码

```bash
git add .
git commit -m "feat: implement ..."
git push origin main
```

推送到 `main` 后，CI 会自动编译、运行测试、计算得分并上传到排行榜平台。得分为 0 时不会上传。

## 实验详情

### CPU 实验（TCG 测题）

为 G233 虚拟机实现自定义 RISC-V 指令。测试通过 semihosting 裸机程序验证指令行为。

- 虚拟机：`g233`（`hw/riscv/g233.c`）
- 测试：`tests/gevico/tcg/riscv64/test-insn-*.c`
- 运行：`make -f Makefile.camp test-cpu`
- 文档：[实验手册](https://qemu.gevico.online/exercise/2026/stage1/cpu/cpu-exper-manual/) | [CPU Datasheet](https://qemu.gevico.online/exercise/2026/stage1/cpu/cpu-datasheet/)

### SoC 实验（QTest 测题）

为 G233 SoC 实现外设设备模型（GPIO、PWM、WDT、SPI、Flash）。测试通过 QTest 的 MMIO 读写验证寄存器行为和中断连接。

- 外设地址：GPIO（`0x10012000`）、PWM（`0x10015000`）、WDT（`0x10010000`）、SPI（`0x10018000`）
- 测试：`tests/gevico/qtest/test-*.c`
- 运行：`make -f Makefile.camp test-soc`
- 文档：[实验手册](https://qemu.gevico.online/exercise/2026/stage1/soc/g233-exper-manual/) | [G233 SoC 硬件手册](https://qemu.gevico.online/exercise/2026/stage1/soc/g233-datasheet/)

### GPGPU 实验（QTest 测题）

实现 PCIe GPGPU 设备，包含 SIMT 执行引擎、DMA 和低精度浮点支持。测试验证设备寄存器、显存、内核执行和 FP8/FP4 转换。

- 设备：`hw/gpgpu/`（PCI 设备 `gpgpu`）
- 测试：`tests/qtest/gpgpu-test.c`（17 个子测试）
- 运行：`make -f Makefile.camp test-gpgpu`
- 文档：[实验手册](https://qemu.gevico.online/exercise/2026/stage1/gpu/gpu-exper-manual/) | [GPU 硬件手册](https://qemu.gevico.online/exercise/2026/stage1/gpu/gpu-datasheet/)

### Rust 实验（QTest + 单元测试）

使用 Rust 为 G233 SoC 实现 I2C 总线、GPIO I2C 控制器和 SPI 控制器。单元测试验证 Rust 核心逻辑；QTest 测试验证设备寄存器行为和外设通信（I2C 连接 AT24C02 EEPROM，SPI 连接 AT25 EEPROM）。

- I2C 总线：`rust/hw/i2c/src/lib.rs`（3 道单元测试）
- GPIO I2C 控制器：基地址 `0x10013000`，连接 AT24C02 EEPROM（地址 `0x50`）
- SPI 控制器：基地址 `0x10019000`，连接 AT25 EEPROM
- 测试：`tests/gevico/qtest/test-i2c-*.c`、`tests/gevico/qtest/test-spi-rust-*.c`
- 运行：`make -f Makefile.camp test-rust`
- 文档：[实验手册](https://qemu.gevico.online/exercise/2026/stage1/rust/rust-exper-manual/) | [Rust 编程指南](https://qemu.gevico.online/exercise/2026/stage1/rust/rust-lang-manual/)

## Make 命令一览

```
make -f Makefile.camp help       # 查看所有命令
make -f Makefile.camp configure  # 配置 QEMU
make -f Makefile.camp build      # 编译 QEMU
make -f Makefile.camp test-cpu   # CPU 实验测试
make -f Makefile.camp test-soc   # SoC 实验测试
make -f Makefile.camp test-gpgpu # GPGPU 实验测试
make -f Makefile.camp test-rust  # Rust 实验测试
make -f Makefile.camp test       # 所有测试
make -f Makefile.camp clean      # 清理构建
make -f Makefile.camp distclean  # 删除构建目录
```

## 评分规则

- 测试**失败不会**导致 CI 报错，只会降低得分。
- 得分为 **0** 时不会上传到排行榜平台。
- 每次推送到 `main` 都会触发完整的 CI 流程。
