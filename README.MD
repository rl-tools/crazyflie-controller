> **⚠️ Note**: This was tested with RLtools `d7be0c225fe9a6123c841e0ef01fe1fe4df3de63`
>

For good measure please install the `2023.2` firmware using `cfclient` first (to update all the decks and communication firmwares)

Install dependencies according to the [official docs](https://www.bitcraze.io/documentation/repository/crazyflie-firmware/master/building-and-flashing/build/)

```
git submodule update --init --recursive -- external/firmware
git submodule update --init -- external/rl_tools
```

### macOS

```
brew install libusb
```


### build
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
cfloader flash build/cf21bl.bin stm32-fw -w radio://0/80/2M/E7E7E7E7E9
```


### "Too many packets lost" Issue
To prevent the "Too many packets lost" issue after startup please use the cfclient UI to flash `2023.02` first. This will flash an older firmware for the radio module which has a longer boot delay for better stability. When flashing the modified firmware with cfloader, the radio firmware is not overwritten (you can confirm this in the cfclient console). For the Crazyflie Brushless use `2024.10.2` (not perfect, but more stable than `2025.02`)
### flash
```
cfloader flash build/cf2.bin stm32-fw -w radio://0/80/2M
```
#### macOS
```
DYLD_LIBRARY_PATH="/opt/homebrew/lib:$DYLD_LIBRARY_PATH" cfloader flash build/cf2.bin stm32-fw -w radio://0/80/2M
DYLD_LIBRARY_PATH="/opt/homebrew/lib:$DYLD_LIBRARY_PATH" cfclient
```


# MOCAP

Please configure the `locSrv.ExtQuatStdDev` such that the yaw estimate is stable


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
