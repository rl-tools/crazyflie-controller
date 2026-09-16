# Pose WebSocket server

Streams one rigid body from any mocap system to a browser. Requires Python 3.10+.

Run from the repository root:

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install ./pose
pose --demo
```

For Vicon:

```sh
pip install './pose[vicon]'
pose.vicon --host 192.154.4.124 --list  # List current object names
pose.vicon --host 192.154.4.124 --subject crazyflie | pose
```

Vicon uses the root segment; override with `--segment NAME`. Server defaults:
`127.0.0.1:8765`; change with `--host`/`--port`. Stop with Ctrl-C.

## Pose API

Connect to `ws://127.0.0.1:8765/pose`. Each text message contains:

```json
{"timestamp":1789600000.125,"position":[0.1,-0.2,0.5],"quaternion":[0,0,0,1]}
```

- `timestamp`: Unix seconds when the producer received the sample.
- `position`: meters, right-handed world frame: X forward, Y left, Z up.
- `quaternion`: normalized `[x,y,z,w]`, rotating body-local vectors into world coordinates.

All values must be finite numbers. The Vicon adapter converts millimeters to meters
and reorders its SDK quaternion to this shared convention.

The stream is read-only. Invalid/occluded poses are skipped; tracking loss means
silence. Clients should detect stale data. Slow clients skip pending samples;
new connections receive only new samples. Input EOF stops the server.

Other adapters must emit this format as flushed, newline-delimited JSON to
stdout, with diagnostics on stderr:

```sh
python -u my_mocap_adapter.py | pose
```

## Browser

Run Python on the browser's computer. From a Connect button handler:

```js
const response = await fetch("http://127.0.0.1:8765/health");
if (!response.ok) throw new Error("Pose server unavailable");
const socket = new WebSocket("ws://127.0.0.1:8765/pose");
socket.onmessage = ({data}) => console.log(JSON.parse(data));
```

Grant Chrome's local/loopback permission. Allowed origins: `https://rc.rl.tools`,
`http://localhost:8000`, `http://127.0.0.1:8000`; extend with `--allow-origin URL`.
`/health` returns `last_pose_age_s` (`null` before any sample).

Tests: `python -B -m unittest discover -s pose/tests -v`.
