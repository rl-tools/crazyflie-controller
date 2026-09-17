# Web Bluetooth

```sh
python3 -m http.server 8000 --bind 127.0.0.1 --directory web/dist
```

Open <http://localhost:8000> in Chrome. Linux: enable `chrome://flags/#enable-experimental-web-platform-features`.

Connect, enter `group.name type value`, then **Write configuration**. Send the learned-controller packet once or hold for 20 Hz. Releasing stops transmission; it does not land.

With bundled nRF firmware, the app blocks long names such as `rlt.motor_warmup`.

**Firmware console** shows the Crazyflie's CRTP console output, including startup messages still queued when connected. Capture continues while viewing Parameters or while paused. Use the arrows to page through the last 1,000 lines and **Live** to follow new output. Power-cycle and reconnect to capture a fresh boot; reconnecting alone does not replay messages already consumed.

**Pose** connects to the [local pose server](../pose/README.md), independently of Bluetooth. Start `pose --demo` (or the Vicon pipeline), enter `ws://127.0.0.1:8765/pose`, and click **Connect** in the Pose panel. Allow Chrome's local network permission if prompted. Position is shown in meters and orientation as an X/Y/Z/W quaternion. After one second without a sample, the display is marked **STALE**. **Disconnect** also cancels a pending connection.

The current legacy nRF BLE transmitter loses a byte in long packets and clears some early packets. The console marks missing data with `⟦…⟧`; it cannot recover the original text. This recovery is display-only and does not relax parameter acknowledgement validation. Idle null-packet polling keeps console output arriving without changing parameters or sending controller commands.

Tests: `node --test web/tests/*.test.mjs`.
