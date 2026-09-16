# Web Bluetooth

```sh
python3 -m http.server 8000 --bind 127.0.0.1 --directory web/dist
```

Open <http://localhost:8000> in Chrome. Linux: enable `chrome://flags/#enable-experimental-web-platform-features`.

Connect, enter `group.name type value`, then **Write configuration**. Send the learned-controller packet once or hold for 20 Hz. Releasing stops transmission; it does not land.

With bundled nRF firmware, the app blocks long names such as `rlt.motor_warmup`.

Tests: `node --test web/tests/*.test.mjs`.
