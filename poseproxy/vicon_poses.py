#!/usr/bin/env python3
"""Emit one Vicon body's poses as newline-delimited JSON for pose_server.py."""

import argparse
from contextlib import closing
import json
import sys
import time


def frames(sdk, host):
    client = sdk.PyViconDatastream()

    def check(result, action):
        if result != sdk.Result.Success:
            raise RuntimeError(f"Vicon {action} failed: {result}")

    try:
        check(client.connect(host), f"connection to {host}")
        check(client.enable_segment_data(), "enable segment data")
        client.set_buffer_size(1)
        check(client.set_stream_mode(sdk.StreamMode.ServerPush), "set stream mode")
        check(client.set_axis_mapping(
            sdk.Direction.Forward, sdk.Direction.Left, sdk.Direction.Up,
        ), "set axis mapping")
        while True:
            result = client.get_frame()
            if result == sdk.Result.NoFrame:
                time.sleep(0.001)
                continue
            check(result, "get frame")
            yield client, time.time()
    finally:
        client.disconnect()


def list_objects(sdk, host):
    with closing(frames(sdk, host)) as stream:
        client, _ = next(stream)
        return [client.get_subject_name(index)
                for index in range(client.get_subject_count())]


def poses(sdk, host, subject, segment=None):
    with closing(frames(sdk, host)) as stream:
        for client, timestamp in stream:
            if segment is None:
                segment = client.get_subject_root_segment_name(subject)
                if not segment:
                    raise RuntimeError(f"Vicon subject {subject!r} has no root segment")
            position = client.get_segment_global_translation(subject, segment)
            rotation = client.get_segment_global_quaternion(subject, segment)
            # The wrapper returns None for occluded or unavailable segment data.
            if position is None or rotation is None:
                continue
            w, x, y, z = map(float, rotation)
            yield {
                "timestamp": timestamp,
                "position": [float(value) * 0.001 for value in position],
                "quaternion": [x, y, z, w],
            }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True, help="Vicon host[:port] (default port: 801)")
    parser.add_argument("--subject", default="crazyflie")
    parser.add_argument("--segment", help="Default: subject's root segment")
    parser.add_argument("--list", action="store_true", dest="list_objects",
                        help="List current object (subject) names and exit")
    args = parser.parse_args()
    try:
        import pyvicon_datastream as sdk
    except ImportError:
        parser.exit(1, "Install the Vicon adapter: pip install 'poseproxy[vicon]'\n")
    stream = None
    try:
        if args.list_objects:
            names = list_objects(sdk, args.host)
            for name in names:
                print(name)
            if not names:
                print("No objects found.", file=sys.stderr)
            return
        stream = poses(sdk, args.host, args.subject, args.segment)
        for pose in stream:
            print(json.dumps(pose, allow_nan=False), flush=True)
    except (KeyboardInterrupt, BrokenPipeError):
        pass
    except (RuntimeError, ValueError) as error:
        parser.exit(1, f"{error}\n")
    finally:
        if stream is not None:
            stream.close()
        # Avoid retrying a failed stdout flush during interpreter shutdown.
        if sys.stdout is not None:
            try:
                sys.stdout.flush()
            except BrokenPipeError:
                sys.stdout = None


if __name__ == "__main__":
    main()
