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
```
cd external/firmware
make cf2_defconfig
cd ../../
make
```

### flash
```
cfloader flash build/cf2.bin stm32-fw -w radio://0/80/2M
```
#### macOS
```
DYLD_LIBRARY_PATH="/opt/homebrew/lib:$DYLD_LIBRARY_PATH" cfloader flash build/cf2.bin stm32-fw -w radio://0/80/2M
DYLD_LIBRARY_PATH="/opt/homebrew/lib:$DYLD_LIBRARY_PATH" cfclient
```