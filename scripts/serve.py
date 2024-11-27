import asyncio
import json
import argparse
import time
import struct
import mimetypes
from dataclasses import dataclass
from typing import Optional, Dict, Set
from pathlib import Path
import websockets
from cflib.crtp import init_drivers
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crtp.crtpstack import CRTPPacket, CRTPPort
from cflib.crazyflie.commander import SET_SETPOINT_CHANNEL, META_COMMAND_CHANNEL, TYPE_HOVER
from cflib.utils import uri_helper

@dataclass
class DroneController:
    cf: Crazyflie
    current_task: Optional[asyncio.Task] = None
    
    async def send_hover_packet(self, height: float, vx: float = 0, vy: float = 0, yawrate: float = 0) -> None:
        """Send a hover command packet to the drone."""
        pk = CRTPPacket()
        pk.port = CRTPPort.COMMANDER_GENERIC
        pk.channel = SET_SETPOINT_CHANNEL
        pk.data = struct.pack('<Bffff', TYPE_HOVER, vx, vy, yawrate, height)
        self.cf.send_packet(pk)

    async def set_param(self, name: str, target: float) -> None:
        """Set a parameter value and wait for confirmation."""
        print(f"Parameter {name} was {self.cf.param.get_value(name)}, setting to {target}")
        while abs(float(self.cf.param.get_value(name)) - float(target)) > 1e-5:
            self.cf.param.set_value(name, target)
            await asyncio.sleep(0.1)
        print(f"Parameter {name} is {self.cf.param.get_value(name)} now")

    async def hover_learned(self, height: float = 0.5) -> None:
        """Execute hover in learned mode."""
        try:
            await self.set_param("rlt.trigger", 0)
            await self.set_param("rlt.wn", 1)
            await self.set_param("rlt.motor_warmup", 1)
            await self.set_param("rlt.target_z", height)
            
            while True:
                await self.send_hover_packet(height)
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            await self.send_hover_packet(0)  # Land the drone
            raise

class CombinedServer:
    def __init__(self, controller: DroneController, static_dir: Path):
        self.controller = controller
        self.static_dir = static_dir
        self.active_websockets: Set[websockets.WebSocketServerProtocol] = set()

    async def handle_websocket(self, websocket):
        """Handle WebSocket connections."""
        try:
            self.active_websockets.add(websocket)
            async for message in websocket:
                await self.process_message(websocket, message)
        finally:
            self.active_websockets.remove(websocket)
            if len(self.active_websockets) == 0 and self.controller.current_task:
                self.controller.current_task.cancel()
                try:
                    await self.controller.current_task
                except asyncio.CancelledError:
                    pass

    async def process_message(self, websocket, message: str):
        """Process incoming WebSocket messages."""
        if message == "hover_learned":
            if self.controller.current_task:
                self.controller.current_task.cancel()
                try:
                    await self.controller.current_task
                except asyncio.CancelledError:
                    pass
            
            await websocket.send("Starting hover learned mode...")
            self.controller.current_task = asyncio.create_task(
                self.controller.hover_learned(0.5)
            )
        
        elif message == "stop":
            if self.controller.current_task:
                self.controller.current_task.cancel()
                try:
                    await self.controller.current_task
                except asyncio.CancelledError:
                    pass
            await websocket.send("Stopping...")

    async def handle_http(self, path: str, headers: Dict) -> tuple:
        """Handle HTTP requests for static files."""
        if path == "/ws":
            return None  # Let the WebSocket handler take over
            
        # Convert URL path to filesystem path
        if path == "/":
            path = "/index.html"
        file_path = self.static_dir / path.lstrip('/')
        
        try:
            if not file_path.is_file() or not file_path.resolve().is_relative_to(self.static_dir.resolve()):
                return (404, [], b"404 Not Found")
            
            content_type, _ = mimetypes.guess_type(str(file_path))
            response_headers = [
                ('Content-Type', content_type or 'application/octet-stream'),
                ('Cache-Control', 'no-cache')
            ]
            
            with open(file_path, 'rb') as f:
                content = f.read()
            
            return (200, response_headers, content)
        
        except Exception as e:
            print(f"Error serving {path}: {e}")
            return (500, [], b"500 Internal Server Error")

    async def serve(self, host: str, port: int):
        """Start the combined HTTP and WebSocket server."""
        async def handler(websocket, path):
            if path == "/ws":
                await self.handle_websocket(websocket)
        
        async with websockets.serve(
            handler,
            host,
            port,
            process_request=self.handle_http
        ):
            print(f"Server started at http://{host}:{port}")
            print(f"WebSocket endpoint available at ws://{host}:{port}/ws")
            await asyncio.Future()  # run forever

async def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    default_uri = 'radio://0/80/2M/E7E7E7E7E7'
    parser.add_argument('--uri', default=default_uri)
    parser.add_argument('--static-dir', default='.', help='Directory to serve static files from')
    parser.add_argument('--host', default='localhost', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8080, help='Port to bind to')
    args = parser.parse_args()
    
    # Initialize drivers
    init_drivers()
    
    # Setup Crazyflie
    uri = uri_helper.uri_from_env(default=default_uri)
    cf = Crazyflie(rw_cache='/tmp/cf_cache')
    scf = SyncCrazyflie(uri, cf=cf)
    
    try:
        # Connect to the Crazyflie
        print("Connecting to Crazyflie...")
        scf.open_link()
        
        # Initialize controller and server
        controller = DroneController(scf.cf)
        static_dir = Path(args.static_dir)
        server = CombinedServer(controller, static_dir)
        
        # Start server
        await server.serve(args.host, args.port)
    
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        # Close the Crazyflie connection
        if scf:
            try:
                scf.close_link()
            except Exception as e:
                print(f"Error closing Crazyflie connection: {e}")

if __name__ == "__main__":
    asyncio.run(main())

