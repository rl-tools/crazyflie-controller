import asyncio
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import re
import signal
import sys
import unittest
from unittest.mock import Mock, patch

from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed, InvalidStatus

POSE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(POSE_DIR))
from pose_server import PoseServer, encode_pose
from vicon_poses import main as vicon_main, poses


POSE = {"timestamp": 1789600000.125, "position": [1, 2, 3], "quaternion": [0, 0, 0, 1]}
ORIGIN = "https://rc.rl.tools"


class FormatTests(unittest.TestCase):
    def test_normalizes_quaternion_and_preserves_pose(self):
        result = json.loads(encode_pose({**POSE, "quaternion": [0, 0, 2, 2]}))
        self.assertEqual(result["position"], POSE["position"])
        self.assertEqual(result["timestamp"], POSE["timestamp"])
        self.assertAlmostEqual(result["quaternion"][2], 2 ** -0.5)
        self.assertAlmostEqual(result["quaternion"][3], 2 ** -0.5)

    def test_rejects_invalid_measurements(self):
        invalid = [
            None, [], {}, {**POSE, "position": [1, 2]},
            {**POSE, "position": [1, float("nan"), 3]},
            {**POSE, "quaternion": [0, 0, 0, 0]},
            {**POSE, "quaternion": [0, 0, 0, float("inf")]},
            {**POSE, "timestamp": True}, {**POSE, "timestamp": "123"},
        ]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                encode_pose(value)


class ViconTests(unittest.TestCase):
    def make_sdk(self):
        sdk = Mock()
        client = sdk.PyViconDatastream.return_value
        for method in ("connect", "enable_segment_data", "set_stream_mode", "set_axis_mapping", "get_frame"):
            getattr(client, method).return_value = sdk.Result.Success
        client.get_subject_root_segment_name.return_value = "root"
        client.get_segment_global_translation.side_effect = [None, [1000, -2000, 500]]
        client.get_segment_global_quaternion.return_value = [0.5, 0.5, -0.5, 0.5]
        return sdk, client

    def test_occlusion_units_order_and_cleanup(self):
        sdk, client = self.make_sdk()
        stream = poses(sdk, "vicon:801", "drone")
        pose = next(stream)
        self.assertEqual(pose["position"], [1, -2, 0.5])
        self.assertEqual(pose["quaternion"], [0.5, -0.5, 0.5, 0.5])
        self.assertGreater(pose["timestamp"], 0)
        self.assertEqual(client.get_frame.call_count, 2)
        client.set_buffer_size.assert_called_once_with(1)
        client.set_axis_mapping.assert_called_once_with(
            sdk.Direction.Forward, sdk.Direction.Left, sdk.Direction.Up)
        client.get_segment_global_translation.assert_called_with("drone", "root")
        stream.close()
        client.disconnect.assert_called_once()

    def test_source_failure_is_reported_and_disconnected(self):
        sdk, client = self.make_sdk()
        client.get_frame.return_value = sdk.Result.NotConnected
        with self.assertRaisesRegex(RuntimeError, "get frame failed"):
            next(poses(sdk, "vicon", "drone"))
        client.disconnect.assert_called_once()

    def run_list_command(self, sdk):
        stdout, stderr = StringIO(), StringIO()
        with patch.dict(sys.modules, {"pyvicon_datastream": sdk}), \
                patch.object(sys, "argv", ["pose.vicon", "--host", "vicon", "--list"]), \
                redirect_stdout(stdout), redirect_stderr(stderr):
            vicon_main()
        return stdout.getvalue(), stderr.getvalue()

    def test_list_waits_for_frame_and_prints_names_without_streaming(self):
        sdk, client = self.make_sdk()
        client.get_frame.side_effect = [sdk.Result.NoFrame, sdk.Result.Success]
        client.get_subject_count.return_value = 2
        client.get_subject_name.side_effect = ["crazyflie", "Calibration Wand"]
        stdout, stderr = self.run_list_command(sdk)
        self.assertEqual(stdout, "crazyflie\nCalibration Wand\n")
        self.assertEqual(stderr, "")
        self.assertEqual(client.get_frame.call_count, 2)
        client.get_subject_root_segment_name.assert_not_called()
        client.get_segment_global_translation.assert_not_called()
        client.disconnect.assert_called_once()

    def test_list_handles_empty_frame(self):
        sdk, client = self.make_sdk()
        client.get_subject_count.return_value = 0
        stdout, stderr = self.run_list_command(sdk)
        self.assertEqual(stdout, "")
        self.assertEqual(stderr, "No objects found.\n")
        client.disconnect.assert_called_once()

    def test_list_connection_failure_exits_with_error_and_disconnects(self):
        sdk, client = self.make_sdk()
        client.connect.return_value = sdk.Result.ClientConnectionFailed
        with self.assertRaises(SystemExit) as caught:
            self.run_list_command(sdk)
        self.assertEqual(caught.exception.code, 1)
        client.get_subject_count.assert_not_called()
        client.disconnect.assert_called_once()


async def http_get(port, path="/health", origin=ORIGIN):
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    origin_header = "" if origin is None else f"Origin: {origin}\r\n"
    writer.write((f"GET {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n"
                  f"{origin_header}Connection: close\r\n\r\n").encode())
    await writer.drain()
    response = await asyncio.wait_for(reader.read(), 2)
    writer.close()
    await writer.wait_closed()
    header, body = response.decode().split("\r\n\r\n", 1)
    return header, body


class ServerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.server = PoseServer()
        self.listener = await self.server.listen(port=0)
        self.port = self.listener.sockets[0].getsockname()[1]
        self.url = f"ws://127.0.0.1:{self.port}/pose"

    async def asyncTearDown(self):
        self.listener.close()
        await self.listener.wait_closed()

    def connect(self, **kwargs):
        return connect(self.url, origin=ORIGIN, proxy=None, **kwargs)

    async def test_fanout_without_replaying_stale_poses(self):
        await self.server.publish(encode_pose(POSE))
        async with self.connect() as first, self.connect() as second:
            with self.assertRaises(asyncio.TimeoutError):
                await asyncio.wait_for(first.recv(), 0.05)
            next_pose = {**POSE, "timestamp": POSE["timestamp"] + 1}
            await self.server.publish(encode_pose(next_pose))
            for client in (first, second):
                received = json.loads(await asyncio.wait_for(client.recv(), 1))
                self.assertEqual(received, next_pose)

    async def test_health_cors_and_origin_rejection(self):
        headers, body = await http_get(self.port)
        self.assertIn("200 OK", headers)
        self.assertIn(f"Access-Control-Allow-Origin: {ORIGIN}", headers)
        self.assertEqual(headers.lower().count("content-type:"), 1)
        self.assertEqual(json.loads(body), {"last_pose_age_s": None})
        await self.server.publish(encode_pose(POSE))
        _, body = await http_get(self.port)
        self.assertGreaterEqual(json.loads(body)["last_pose_age_s"], 0)
        headers, _ = await http_get(self.port, origin="https://unapproved.example")
        self.assertIn("403 Forbidden", headers)
        self.assertNotIn("Access-Control-Allow-Origin", headers)
        with self.assertRaises(InvalidStatus) as caught:
            async with connect(self.url, origin="https://unapproved.example", proxy=None):
                pass
        self.assertEqual(caught.exception.response.status_code, 403)
        headers, _ = await http_get(self.port, path="/unknown")
        self.assertIn("404 Not Found", headers)

    async def test_client_cannot_inject_poses(self):
        async with self.connect() as client:
            await client.send(json.dumps(POSE))
            with self.assertRaises(ConnectionClosed):
                await asyncio.wait_for(client.recv(), 1)
            self.assertEqual(client.close_code, 1008)

    async def test_slow_client_skips_pending_frames_without_blocking_others(self):
        blocked = asyncio.Event()
        release = asyncio.Event()
        received = asyncio.Queue()

        class SlowClient:
            async def send(self, message):
                blocked.set()
                await release.wait()
                await received.put(json.loads(message))

            def __aiter__(self):
                return self

            async def __anext__(self):
                await asyncio.Future()

        task = asyncio.create_task(self.server.handle(SlowClient()))
        try:
            async with self.connect() as fast:
                await self.server.publish(encode_pose(POSE))
                await asyncio.wait_for(blocked.wait(), 1)
                await asyncio.wait_for(fast.recv(), 1)
                for timestamp in (2, 3, 4):
                    await self.server.publish(encode_pose({**POSE, "timestamp": timestamp}))
                self.assertEqual(json.loads(await asyncio.wait_for(fast.recv(), 1))["timestamp"], 4)
                release.set()
                self.assertEqual((await asyncio.wait_for(received.get(), 1))["timestamp"], POSE["timestamp"])
                self.assertEqual((await asyncio.wait_for(received.get(), 1))["timestamp"], 4)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


class CommandTests(unittest.IsolatedAsyncioTestCase):
    async def launch(self, *args):
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-B", str(POSE_DIR / "pose_server.py"), "--port", "0", *args,
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        async def cleanup():
            if process.returncode is None:
                process.terminate()
            await asyncio.wait_for(process.communicate(), 5)
        self.addAsyncCleanup(cleanup)
        while line := await asyncio.wait_for(process.stderr.readline(), 5):
            if match := re.search(rb"Pose stream: ws://127.0.0.1:(\d+)/pose", line):
                return process, int(match[1])
        self.fail("server did not start")

    async def test_stdin_validation_streaming_and_eof(self):
        process, port = await self.launch()
        async with connect(f"ws://127.0.0.1:{port}/pose", origin=ORIGIN, proxy=None) as client:
            process.stdin.write(b"not json\n" + b"x" * 5000 + b"\n" + json.dumps(POSE).encode() + b"\n")
            await process.stdin.drain()
            self.assertEqual(json.loads(await asyncio.wait_for(client.recv(), 2)), POSE)
            process.stdin.close()
            await asyncio.wait_for(process.wait(), 3)
            self.assertEqual(process.returncode, 0)
            with self.assertRaises(ConnectionClosed):
                await asyncio.wait_for(client.recv(), 1)

    async def test_demo_and_shutdown_with_open_stdin(self):
        process, port = await self.launch("--demo", "--rate", "50")
        async with connect(f"ws://127.0.0.1:{port}/pose", origin=ORIGIN, proxy=None) as client:
            first = json.loads(await asyncio.wait_for(client.recv(), 2))
            second = json.loads(await asyncio.wait_for(client.recv(), 2))
            self.assertGreater(second["timestamp"], first["timestamp"])
            self.assertEqual(first["position"][2], 1)
            self.assertAlmostEqual(sum(q*q for q in first["quaternion"]), 1)
        process.terminate()
        await asyncio.wait_for(process.wait(), 3)
        self.assertEqual(process.returncode, 0)

    async def test_shutdown_while_waiting_for_stdin(self):
        process, _ = await self.launch()
        process.terminate()
        await asyncio.wait_for(process.wait(), 3)
        self.assertEqual(process.returncode, 0)

    @unittest.skipIf(sys.platform == "win32", "POSIX Ctrl-C signal")
    async def test_ctrl_c_while_waiting_for_stdin(self):
        process, _ = await self.launch()
        process.send_signal(signal.SIGINT)
        await asyncio.wait_for(process.wait(), 3)
        self.assertEqual(process.returncode, 0)


if __name__ == "__main__":
    unittest.main()
