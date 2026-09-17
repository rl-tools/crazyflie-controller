For STM32-only flashing, see the radio firmware notes below.

Install dependencies according to the [official docs](https://www.bitcraze.io/documentation/repository/crazyflie-firmware/master/building-and-flashing/build/)

```
git submodule update --init --recursive -- external/firmware
git submodule update --init -- external/rl_tools
git submodule update --init -- external/blob external/nrf-firmware
```

### macOS

```
brew install libusb
```


### build

Build ZIPs with `./build_firmware.sh cf2` or `./build_firmware.sh cf21bl` (omit the argument for both).
Outputs: `build/firmware-<platform>.zip`, containing only local STM32/nRF firmware and a manifest. Requires S130 already installed.
Flash with `cfloader flash build/firmware-<platform>.zip -w <URI>`.

<!-- ```
cd external/firmware
make cf2_defconfig
cd ../../
make
``` -->

### CF2
```
make cf2_defconfig
make
cfloader flash build/cf2.bin stm32-fw -w radio://0/80/2M/E7E7E7E7E7
```

### CF2 Brushless

```
git clean -dfx
make cf21bl_defconfig
make
cfloader flash build/cf21bl.bin stm32-fw -w radio://0/80/2M/E7E7E7E7E9 # Note: use the correct id here, we assigned different ones for cf2 and cfbl
```

# Usage
```
git submodule update --init --recursive external/cfclient
deactivate
python3 -m venv .venv
. .venv/bin/activate
pip install -e external/cfclient[joystream]
JOYSTREAM=1 cfclient
```
### Figure Eight Tracking
Set these Crazyflie parameters (from the `cfclient` UI)
```sh
rlt.fes = 0.2 # Scale of the figure eight in [m]
rlt.fei = 3 # 3 s interval
rlt.wn = 4 # Figure eight tracking mode (0 = position hold)
rlt.target_z_fe = 0.3 # 0.3 m target height (take-off)
```


### "Too many packets lost" Issue
For STM32-only flashes, use `2023.02` radio firmware for CF2 or `2024.10.2` for Brushless. The ZIP builds include the local nRF firmware with the boot delay restored and require S130 already installed.
### flash
```
cfloader flash build/cf2.bin stm32-fw -w radio://0/80/2M
```
#### macOS
```
DYLD_LIBRARY_PATH="/opt/homebrew/lib:$DYLD_LIBRARY_PATH" cfloader flash build/cf2.bin stm32-fw -w radio://0/80/2M
DYLD_LIBRARY_PATH="/opt/homebrew/lib:$DYLD_LIBRARY_PATH" cfclient
```


## Cleanup

Note: Changes to tracked files need to be cleaned up manually

```
git submodule foreach --recursive 'git clean -dffx -e .venv/'
git clean -dffx -e .venv/

git submodule update --init --recursive -- external/firmware
git submodule update --init -- external/rl_tools
```


# MOCAP

Please configure the `locSrv.ExtQuatStdDev` such that the yaw estimate is stable

For a local WebSocket pose stream usable from `https://rc.rl.tools`, see the
[poseproxy instructions and API](poseproxy/README.md). It includes a Vicon
adapter, a simulated source, and a JSON input format for other mocap systems.


# Facing Connection Issues?
Flash the modified NRF firmware (re-adding a boot delay that was lost somewhere between `2023.02` and `2025.02`):

```
sudo apt-get install gcc-arm-none-eabi gdb-arm-none-eabi binutils-arm-none-eabi
cd external/nrf-firmware
./tools/fetch-dependencies.sh
make PLATFORM=cf21bl
make PLATFORM=cf21bl cload
```
While `cload` is running, restart the Crazyflie in bootloader mode by keeping the button pressed for a few seconds until the LEDs start blinking
