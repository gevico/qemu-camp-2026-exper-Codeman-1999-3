# GPGPU 设备的 Meson 构建配置
#
# 这个文件告诉 Meson 构建系统如何编译 GPGPU 设备
# system_ss 是系统模拟器的源文件集合

# 当 CONFIG_GPGPU 配置选项被启用时，编译 gpgpu.c 和 gpgpu_core.c
system_ss.add(when: 'CONFIG_GPGPU', if_true: files(
    'gpgpu.c',
    'gpgpu_core.c',
))
