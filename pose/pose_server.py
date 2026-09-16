#!/usr/bin/env python3
"""Forward newline-delimited JSON poses from stdin to ws://127.0.0.1:8765/pose."""

import argparse
import asyncio
import concurrent.futures
import json
import logging
import math
import signal
import sys
import threading
import time

from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed


LOG = logging.getLogger("pose_server")
DEFAULT_ORIGINS = (
    "https://rc.rl.tools",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
)


def encode_pose(pose):
    """Validate the common pose format and normalize its xyzw quaternion."""
    def number(value):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("pose values must be numbers")
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("pose values must be finite")
        return value

    def vector(name, size):
        value = pose.get(name)
        if not isinstance(value, list) or len(value) != size:
            raise ValueError(f"{name} must be an array of {size} numbers")
        return [number(item) for item in value]

    if not isinstance(pose, dict):
        raise ValueError("pose must be a JSON object")
    timestamp = number(pose.get("timestamp"))
    position = vector("position", 3)
    quaternion = vector("quaternion", 4)
    norm = math.hypot(*quaternion)
    if norm == 0 or not math.isfinite(norm):
        raise ValueError("quaternion must have a finite, nonzero norm")
    return json.dumps({
        "timestamp": timestamp,
        "position": position,
        "quaternion": [value / norm for value in quaternion],
    }, separators=(",", ":"), allow_nan=False)


class PoseServer:
    def __init__(self, origins=DEFAULT_ORIGINS):
        self.origins = set(origins)
        self.clients = set()
        self.last_received = None

    async def publish(self, message):
        """Publish an encoded pose on the event loop; retain one pending pose/client."""
        self.last_received = time.monotonic()
        for queue in self.clients:
            if queue.full():
                queue.get_nowait()
            queue.put_nowait(message)

    def process_request(self, connection, request):
        origins = request.headers.get_all("Origin")
        if len(origins) > 1 or (origins and origins[0] not in self.origins):
            return connection.respond(403, "Origin not allowed\n")
        if request.path == "/health":
            age = (None if self.last_received is None else
                   time.monotonic() - self.last_received)
            response = connection.respond(200, json.dumps({
                "last_pose_age_s": age,
            }) + "\n")
            del response.headers["Content-Type"]
            response.headers["Content-Type"] = "application/json"
            response.headers["Cache-Control"] = "no-store"
            response.headers["Vary"] = "Origin"
            if origins:
                response.headers["Access-Control-Allow-Origin"] = origins[0]
            return response
        if request.path != "/pose":
            return connection.respond(404, "Use /pose or /health\n")
        return None

    async def handle(self, websocket):
        queue = asyncio.Queue(maxsize=1)
        self.clients.add(queue)

        async def send_poses():
            while True:
                message = await queue.get()
                try:
                    await asyncio.wait_for(websocket.send(message), timeout=1)
                except asyncio.TimeoutError:
                    await websocket.close(code=1013, reason="Client is too slow")
                    return

        async def receive():
            # Reading also detects disconnects when the mocap source is silent.
            async for _ in websocket:
                await websocket.close(code=1008, reason="Pose stream is read-only")
                return

        tasks = [asyncio.create_task(send_poses()), asyncio.create_task(receive())]
        try:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
        except ConnectionClosed:
            pass
        finally:
            self.clients.discard(queue)
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    def listen(self, host="127.0.0.1", port=8765):
        return serve(
            self.handle, host, port, process_request=self.process_request,
            origins=[None, *self.origins], compression=None,
            max_size=1024, max_queue=1, write_limit=4096, close_timeout=1,
        )


def read_stdin(fd, loop, server, finished):
    """A daemon reader keeps stdin blocking and SDK work out of the event loop."""
    try:
        # Unbuffered IO avoids a blocked buffered-reader lock during Ctrl-C exit.
        with open(fd, "rb", buffering=0, closefd=False) as stream:
            while line := stream.readline(4097):
                if len(line) > 4096:
                    LOG.warning("Skipping input line longer than 4096 bytes")
                    while line and not line.endswith(b"\n"):
                        line = stream.readline(4097)
                    continue
                if not line.strip():
                    continue
                try:
                    message = encode_pose(json.loads(line))
                except (ValueError, OverflowError, RecursionError) as error:
                    LOG.warning("Skipping invalid pose: %s", error)
                    continue
                # At most one callback is outstanding, even for a fast producer.
                publication = server.publish(message)
                try:
                    future = asyncio.run_coroutine_threadsafe(publication, loop)
                except RuntimeError:
                    publication.close()
                    return
                future.result()
    except (RuntimeError, concurrent.futures.CancelledError):
        pass  # Event loop shut down while stdin was still open.
    except OSError as error:
        LOG.error("Reading poses failed: %s", error)
    finally:
        try:
            loop.call_soon_threadsafe(finished.set)
        except RuntimeError:
            pass


async def demo(server, rate):
    start = time.monotonic()
    while True:
        angle = (time.monotonic() - start) * 0.5
        await server.publish(encode_pose({
            "timestamp": time.time(),
            "position": [0.5 * math.cos(angle), 0.5 * math.sin(angle), 1.0],
            "quaternion": [0.0, 0.0, math.sin(angle / 2), math.cos(angle / 2)],
        }))
        await asyncio.sleep(1 / rate)


async def run(args):
    loop = asyncio.get_running_loop()
    finished = asyncio.Event()
    # asyncio.run handles Ctrl-C; also shut down cleanly on POSIX SIGTERM.
    try:
        loop.add_signal_handler(signal.SIGTERM, finished.set)
    except NotImplementedError:
        pass
    server = PoseServer((*DEFAULT_ORIGINS, *args.allow_origin))
    async with server.listen(args.host, args.port) as listener:
        port = listener.sockets[0].getsockname()[1]
        LOG.info("Pose stream: ws://%s:%s/pose", args.host, port)
        if args.demo:
            producer = asyncio.create_task(demo(server, args.rate))
        else:
            producer = None
            threading.Thread(
                target=read_stdin,
                args=(sys.stdin.fileno(), loop, server, finished), daemon=True,
            ).start()
        try:
            await finished.wait()
        finally:
            if producer is not None:
                producer.cancel()
                await asyncio.gather(producer, return_exceptions=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Bind address")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--allow-origin", action="append", default=[],
                        help="Additional browser origin (repeatable)")
    parser.add_argument("--demo", action="store_true", help="Generate poses instead of reading stdin")
    parser.add_argument("--rate", type=float, default=100, help="Demo rate in Hz (default: 100)")
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error("port must be between 0 and 65535")
    if not math.isfinite(args.rate) or args.rate <= 0:
        parser.error("rate must be finite and positive")
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
