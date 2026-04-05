_libi2c_rs = static_library(
  'i2c_rs',
  files('src/lib.rs', 'src/bus.rs'),
  override_options: ['rust_std=2021', 'build.rust_std=2021'],
  rust_abi: 'rust',
  dependencies: [common_rs],
)

rust_devices_ss.add(when: 'CONFIG_X_I2C_RUST', if_true: [declare_dependency(
  link_whole: [_libi2c_rs],
  variables: {'crate': 'i2c_rs'},
)])

test('rust-i2c-unit',
    executable(
        'rust-i2c-unit',
        files('src/lib.rs', 'src/bus.rs'),
        rust_args: ['--test'],
        install: false,
        dependencies: [common_rs]),
    args: [
        '--test', '--test-threads', '1',
        '--format', 'pretty',
    ],
    protocol: 'rust',
    suite: ['unit', 'rust'])
