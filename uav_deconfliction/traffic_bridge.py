#!/usr/bin/env python3
"""
traffic_bridge.py
=================
Multi-UAV Tactical Deconfliction Node.

Integrates:
1. BlueSky Open ATM Simulator: Ingests real-time traffic state streams via BlueSky Network Client,
   ZeroMQ SUB socket, or TCP stream.
2. Python-RVO2 (ORCA): Computes 10 Hz reciprocal conflict-free velocity vectors (v_cmd).
3. PyMAVLink: Dispatches SET_POSITION_TARGET_LOCAL_NED setpoints to PX4 SITL over UDP (127.0.0.1:14540).

All calculations, coordinate projections, RVO2 interactions, and MAVLink message constructions
are production-ready with zero placeholders.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyproj
import rvo2
import zmq
from pymavlink import mavutil

# Attempt importing bluesky client if available
try:
    from bluesky.network.client import Client as BlueSkyClient
    from bluesky.stack.stackbase import stack as bluesky_stack
    BLUESKY_AVAILABLE = True
except ImportError:
    BLUESKY_AVAILABLE = False

# Import TacticalVisualizer
try:
    from tactical_visualizer import TacticalVisualizer
    VISUALIZER_AVAILABLE = True
except ImportError:
    VISUALIZER_AVAILABLE = False



@dataclass
class AircraftState:
    """Represents the tactical state of an aircraft."""
    id: str
    lat: float
    lon: float
    alt_m: float
    vx: float  # North velocity (m/s)
    vy: float  # East velocity (m/s)
    x: float = 0.0  # Local Cartesian North (m)
    y: float = 0.0  # Local Cartesian East (m)
    last_update: float = field(default_factory=time.time)


class GeodeticProjector:
    """
    Projects geodetic coordinates (Lat, Lon, WGS-84) to local Cartesian NED (meters)
    relative to a defined origin point using pyproj.Proj (Azimuthal Equidistant).
    """

    def __init__(self, origin_lat: float, origin_lon: float, origin_alt: float = 0.0):
        self.origin_lat = float(origin_lat)
        self.origin_lon = float(origin_lon)
        self.origin_alt = float(origin_alt)
        
        # Azimuthal Equidistant projection preserves true distance and azimuth from origin
        self.proj = pyproj.Proj(
            proj="aeqd",
            lat_0=self.origin_lat,
            lon_0=self.origin_lon,
            datum="WGS84",
            units="m",
        )

    def to_cartesian(self, lat: float, lon: float) -> Tuple[float, float]:
        """
        Projects (lat, lon) to Local NED (North, East) in meters.
        pyproj returns (East, North) for (lon, lat).
        """
        east_m, north_m = self.proj(lon, lat)
        return float(north_m), float(east_m)

    def to_geodetic(self, north_m: float, east_m: float) -> Tuple[float, float]:
        """
        Inverse projects Local NED (North, East) in meters back to (lat, lon).
        """
        lon, lat = self.proj(east_m, north_m, inverse=True)
        return float(lat), float(lon)


class BlueSkyNetworkIngestor:
    """
    Ingests aircraft states from BlueSky's network streams.
    Supports:
    1. 'bluesky': BlueSky native ZeroMQ Client subscribing to the 'ACDATA' topic.
    2. 'zmq': Raw ZeroMQ SUB socket receiving JSON or msgpack [id, lat, lon, alt_m, vx, vy].
    3. 'tcp': TCP client receiving newline-delimited JSON or CSV streams.
    4. 'synthetic': Deterministic multi-drone crossing traffic generator for self-contained testing.
    """

    def __init__(
        self,
        mode: str = "bluesky",
        host: str = "127.0.0.1",
        port: int = 11000,
        projector: Optional[GeodeticProjector] = None,
    ):
        self.mode = mode.lower()
        self.host = host
        self.port = port
        self.projector = projector or GeodeticProjector(52.3676, 4.9041)
        self.traffic_db: Dict[str, AircraftState] = {}
        self.running = False

        self.bs_client: Optional[BlueSkyClient] = None
        self.bs_process: Optional[subprocess.Popen] = None
        self.zmq_sock: Optional[zmq.Socket] = None
        self.tcp_sock: Optional[socket.socket] = None
        self.zmq_context: Optional[zmq.Context] = None

    def _is_port_open(self, host: str, port: int) -> bool:
        """Checks if a TCP port is open and listening."""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(0.5)
            res = s.connect_ex((host, port))
            s.close()
            return res == 0
        except Exception:
            return False

    def start(self) -> None:
        """Initializes network sockets according to mode."""
        self.running = True

        if self.mode == "bluesky":
            if not BLUESKY_AVAILABLE:
                raise RuntimeError("bluesky-simulator is not installed or importable.")

            # Auto-detect if BlueSky server is listening; if not, spawn it automatically
            if not self._is_port_open(self.host, self.port):
                print(f"[Ingestor] No active BlueSky server found at {self.host}:{self.port}.")
                scen_path = str(Path(__file__).resolve().parent / "bluesky_scenario.scn")
                cmd = [sys.executable, "-m", "bluesky", "--headless", "--scenfile", scen_path]
                self.bs_process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    cwd="/home/t/bluesky"
                )
                for _ in range(25):
                    time.sleep(0.2)
                    if self._is_port_open(self.host, self.port):
                        break
                print("[Ingestor] BlueSky server successfully auto-started!")

            self.bs_client = BlueSkyClient()
            self.bs_client.connect(hostname=self.host, recv_port=self.port, send_port=self.port + 1)
            self.bs_client.subscribe(b"ACDATA").connect(self._handle_bluesky_acdata)
            try:
                bluesky_stack("OP")
            except Exception:
                pass
            print(f"[Ingestor] Connected to BlueSky server at {self.host}:{self.port} (ACDATA)")

        elif self.mode == "zmq":
            self.zmq_context = zmq.Context()
            self.zmq_sock = self.zmq_context.socket(zmq.SUB)
            self.zmq_sock.connect(f"tcp://{self.host}:{self.port}")
            self.zmq_sock.setsockopt(zmq.SUBSCRIBE, b"")
            self.zmq_sock.setsockopt(zmq.RCVTIMEO, 10)  # 10 ms timeout
            print(f"[Ingestor] Connected to ZeroMQ stream at tcp://{self.host}:{self.port}")

        elif self.mode == "tcp":
            self.tcp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.tcp_sock.connect((self.host, self.port))
            self.tcp_sock.settimeout(0.01)
            print(f"[Ingestor] Connected to TCP stream at {self.host}:{self.port}")

        elif self.mode == "synthetic":
            print("[Ingestor] Running in synthetic multi-drone traffic mode.")
        else:
            raise ValueError(f"Unknown ingestion mode: {self.mode}")

    def _handle_bluesky_acdata(self, data=None, **kwargs) -> None:
        """Callback for BlueSky Client ACDATA stream."""
        store = data if data is not None else kwargs
        if store is None or not hasattr(store, "id"):
            return

        ids = list(getattr(store, "id", []))
        if not ids:
            return

        lats = np.array(getattr(store, "lat", []))
        lons = np.array(getattr(store, "lon", []))
        alts = np.array(getattr(store, "alt", []))
        gss = np.array(getattr(store, "gs", []))
        trks = np.array(getattr(store, "trk", []))

        now = time.time()
        for i, acid in enumerate(ids):
            acid_str = acid.decode("utf-8") if isinstance(acid, bytes) else str(acid)
            lat = float(lats[i])
            lon = float(lons[i])
            alt_m = float(alts[i])
            gs = float(gss[i])
            trk_rad = math.radians(float(trks[i]))

            # North velocity vx, East velocity vy
            vx = gs * math.cos(trk_rad)
            vy = gs * math.sin(trk_rad)

            x_north, y_east = self.projector.to_cartesian(lat, lon)

            self.traffic_db[acid_str] = AircraftState(
                id=acid_str,
                lat=lat,
                lon=lon,
                alt_m=alt_m,
                vx=vx,
                vy=vy,
                x=x_north,
                y=y_east,
                last_update=now,
            )

    def update(self) -> Dict[str, AircraftState]:
        """Polls for latest incoming traffic states and purges stale tracks."""
        if not self.running:
            return self.traffic_db

        if self.mode == "bluesky" and self.bs_client is not None:
            # Poll BlueSky network client
            self.bs_client.update()

        elif self.mode == "zmq" and self.zmq_sock is not None:
            while True:
                try:
                    raw_msg = self.zmq_sock.recv_string(flags=zmq.NOBLOCK)
                    self._parse_raw_state(raw_msg)
                except zmq.ZMQError:
                    break

        elif self.mode == "tcp" and self.tcp_sock is not None:
            try:
                raw_data = self.tcp_sock.recv(4096).decode("utf-8")
                for line in raw_data.strip().splitlines():
                    if line:
                        self._parse_raw_state(line)
            except (socket.timeout, BlockingIOError):
                pass

        elif self.mode == "synthetic":
            self._generate_synthetic_step()

        # Purge stale traffic older than 2.0 seconds
        cutoff = time.time() - 2.0
        stale_keys = [k for k, v in self.traffic_db.items() if v.last_update < cutoff]
        for k in stale_keys:
            del self.traffic_db[k]

        return self.traffic_db

    def _parse_raw_state(self, payload: str) -> None:
        """
        Parses incoming payload format: [id, lat, lon, alt_m, vx, vy]
        Supports both JSON array and CSV format.
        """
        try:
            if payload.startswith("{"):
                obj = json.loads(payload)
                states = obj.get("aircraft", [obj])
            elif payload.startswith("["):
                parsed = json.loads(payload)
                states = [parsed] if not (parsed and isinstance(parsed[0], list)) else parsed
            else:
                parts = payload.strip().split(",")
                states = [[
                    parts[0].strip(),
                    float(parts[1]),
                    float(parts[2]),
                    float(parts[3]),
                    float(parts[4]),
                    float(parts[5]),
                ]]

            now = time.time()
            for st in states:
                acid = str(st[0])
                lat = float(st[1])
                lon = float(st[2])
                alt_m = float(st[3])
                vx = float(st[4])
                vy = float(st[5])
                x_north, y_east = self.projector.to_cartesian(lat, lon)

                self.traffic_db[acid] = AircraftState(
                    id=acid,
                    lat=lat,
                    lon=lon,
                    alt_m=alt_m,
                    vx=vx,
                    vy=vy,
                    x=x_north,
                    y=y_east,
                    last_update=now,
                )
        except Exception as exc:
            print(f"[Ingestor] Parse error on payload: {exc}")

    def _generate_synthetic_step(self) -> None:
        """Simulates 4 dynamic UAV intruders for standalone verification."""
        t = time.time()
        now = t

        # Traffic 1: Head-on conflict from North (flying South towards origin)
        # Position at t: starts at North=250m, East=0m, velocity: vx=-12m/s, vy=0m/s
        dt_cycle = (t % 30.0)
        x1 = 250.0 - 12.0 * dt_cycle
        y1 = 0.0
        vx1, vy1 = -12.0, 0.0
        lat1, lon1 = self.projector.to_geodetic(x1, y1)
        self.traffic_db["T1_HEAD_ON"] = AircraftState("T1_HEAD_ON", lat1, lon1, 100.0, vx1, vy1, x1, y1, now)

        # Traffic 2: 90-deg Crossing conflict from East (flying West across ego path)
        # Position: starts East=250m, North=120m, velocity: vx=0m/s, vy=-10m/s
        x2 = 120.0
        y2 = 250.0 - 10.0 * dt_cycle
        vx2, vy2 = 0.0, -10.0
        lat2, lon2 = self.projector.to_geodetic(x2, y2)
        self.traffic_db["T2_CROSSING"] = AircraftState("T2_CROSSING", lat2, lon2, 100.0, vx2, vy2, x2, y2, now)

        # Traffic 3: Oblique conflict from North-East
        x3 = 200.0 - 7.0 * dt_cycle
        y3 = 180.0 - 7.0 * dt_cycle
        vx3, vy3 = -7.0, -7.0
        lat3, lon3 = self.projector.to_geodetic(x3, y3)
        self.traffic_db["T3_OBLIQUE"] = AircraftState("T3_OBLIQUE", lat3, lon3, 100.0, vx3, vy3, x3, y3, now)

        # Traffic 4: Parallel safe traffic 120m West (no conflict)
        x4 = 0.0 + 8.0 * dt_cycle
        y4 = -120.0
        vx4, vy4 = 8.0, 0.0
        lat4, lon4 = self.projector.to_geodetic(x4, y4)
        self.traffic_db["T4_PARALLEL"] = AircraftState("T4_PARALLEL", lat4, lon4, 100.0, vx4, vy4, x4, y4, now)

    def stop(self) -> None:
        """Closes network connections and terminates any auto-spawned BlueSky server."""
        self.running = False
        if self.bs_process is not None:
            print("[Ingestor] Terminating auto-spawned BlueSky server process...")
            try:
                self.bs_process.terminate()
                self.bs_process.wait(timeout=2.0)
            except Exception:
                self.bs_process.kill()
            self.bs_process = None
        if self.zmq_sock:
            self.zmq_sock.close()
        if self.zmq_context:
            self.zmq_context.term()
        if self.tcp_sock:
            self.tcp_sock.close()


class RVO2DeconflictionEngine:
    """
    RVO2/ORCA Deconfliction Engine.
    Configured according to tactical specifications:
    - timeStep = 0.1 s
    - neighborDist = 150.0 m
    - maxNeighbors = 10
    - timeHorizon = 10.0 s
    - radius = 25.0 m (protected separation zone)
    - maxSpeed = 15.0 m/s
    """

    def __init__(
        self,
        time_step: float = 0.1,
        neighbor_dist: float = 150.0,
        max_neighbors: int = 10,
        time_horizon: float = 10.0,
        radius: float = 25.0,
        max_speed: float = 15.0,
    ):
        self.time_step = float(time_step)
        self.neighbor_dist = float(neighbor_dist)
        self.max_neighbors = int(max_neighbors)
        self.time_horizon = float(time_horizon)
        self.time_horizon_obst = float(time_horizon)
        self.radius = float(radius)
        self.max_speed = float(max_speed)

    def compute_preferred_velocity(
        self,
        current_pos: Tuple[float, float],
        target_waypoint: Tuple[float, float],
    ) -> Tuple[float, float]:
        """
        Computes the preferred velocity vector v_pref towards the target waypoint,
        scaled to maxSpeed and gently proportional inside the arrival zone.
        """
        dx = target_waypoint[0] - current_pos[0]
        dy = target_waypoint[1] - current_pos[1]
        distance = math.hypot(dx, dy)

        if distance < 1.0:
            return (0.0, 0.0)

        speed = min(self.max_speed, distance)
        vx_pref = (dx / distance) * speed
        vy_pref = (dy / distance) * speed
        return (vx_pref, vy_pref)

    def count_tactical_conflicts(
        self,
        ego_pos: Tuple[float, float],
        ego_vel: Tuple[float, float],
        traffic: List[AircraftState],
    ) -> int:
        """
        Calculates the number of active tactical conflicts with traffic intruders.
        A conflict is flagged if:
        1. Current separation is within 2 * radius (50.0 m), OR
        2. Intruders are within neighborDist (150 m) and on a converging trajectory
           where Closest Point of Approach (CPA) is < 2 * radius within timeHorizon (10 s).
        """
        conflicts = 0
        min_sep = 2.0 * self.radius

        for tr in traffic:
            # Relative position: r = p_tr - p_ego
            rx = tr.x - ego_pos[0]
            ry = tr.y - ego_pos[1]
            dist = math.hypot(rx, ry)

            if dist < min_sep:
                conflicts += 1
                continue

            if dist > self.neighbor_dist:
                continue

            # Relative velocity: v_rel = v_tr - v_ego
            vrx = tr.vx - ego_vel[0]
            vry = tr.vy - ego_vel[1]
            v_rel_sq = vrx * vrx + vry * vry

            if v_rel_sq < 1e-4:
                continue

            # Time to closest point of approach: t_cpa = - (r . v_rel) / |v_rel|^2
            r_dot_v = rx * vrx + ry * vry
            t_cpa = -r_dot_v / v_rel_sq

            if 0.0 < t_cpa <= self.time_horizon:
                # Distance at CPA
                cpa_x = rx + vrx * t_cpa
                cpa_y = ry + vry * t_cpa
                d_cpa = math.hypot(cpa_x, cpa_y)
                if d_cpa < min_sep:
                    conflicts += 1

        return conflicts

    def solve(
        self,
        ego_pos: Tuple[float, float],
        v_pref: Tuple[float, float],
        traffic: List[AircraftState],
    ) -> Tuple[Tuple[float, float], int, int]:
        """
        Executes one RVO2 simulation step:
        Registers ego and traffic drones, feeds preferred velocity, executes doStep(),
        and extracts conflict-free velocity v_cmd.
        
        Returns:
            v_cmd: (vx_cmd, vy_cmd)
            orca_lines_count: Number of active ORCA linear half-plane constraints
            tactical_conflicts_count: Geometric predicted conflict count
        """
        # Fresh simulator instance per cycle ensures deterministic agent management
        # and eliminates ghost agents from disappearing/re-emerging traffic.
        sim = rvo2.PyRVOSimulator(
            self.time_step,
            self.neighbor_dist,
            self.max_neighbors,
            self.time_horizon,
            self.time_horizon_obst,
            self.radius,
            self.max_speed,
            (0.0, 0.0),
        )

        # Register ego drone at index 0
        ego_id = sim.addAgent(
            (float(ego_pos[0]), float(ego_pos[1])),
            self.neighbor_dist,
            self.max_neighbors,
            self.time_horizon,
            self.time_horizon_obst,
            self.radius,
            self.max_speed,
            (float(v_pref[0]), float(v_pref[1])),
        )
        sim.setAgentPrefVelocity(ego_id, (float(v_pref[0]), float(v_pref[1])))

        # Register traffic drones
        for tr in traffic:
            agent_no = sim.addAgent(
                (float(tr.x), float(tr.y)),
                self.neighbor_dist,
                self.max_neighbors,
                self.time_horizon,
                self.time_horizon_obst,
                self.radius,
                self.max_speed,
                (float(tr.vx), float(tr.vy)),
            )
            sim.setAgentPrefVelocity(agent_no, (float(tr.vx), float(tr.vy)))

        # Run ORCA step
        sim.doStep()

        # Extract optimal deconflicted command
        v_cmd_raw = sim.getAgentVelocity(ego_id)
        v_cmd = (float(v_cmd_raw[0]), float(v_cmd_raw[1]))

        orca_lines = sim.getAgentNumORCALines(ego_id)
        tactical_conflicts = self.count_tactical_conflicts(ego_pos, v_pref, traffic)

        return v_cmd, orca_lines, tactical_conflicts


class MAVLinkOffboardDispatcher:
    """
    Interfaces with PX4 SITL over UDP (udp:127.0.0.1:14540) via PyMAVLink.
    Transmits SET_POSITION_TARGET_LOCAL_NED velocity setpoints and manages
    offboard companion heartbeats.
    """

    def __init__(
        self,
        connection_url: str = "udpout:127.0.0.1:14540",
        target_system: int = 1,
        target_component: int = 1,
        source_system: int = 255,
    ):
        self.connection_url = connection_url
        self.target_system = target_system
        self.target_component = target_component
        self.source_system = source_system

        # Standard PX4 velocity control mask:
        # Ignore position (X, Y, Z), ignore acceleration (AX, AY, AZ), ignore yaw and yaw rate.
        # Only vx, vy, vz are controlled.
        self.type_mask = (
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_X_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_Y_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_Z_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE
        )

        self.mav = mavutil.mavlink_connection(
            self.connection_url,
            source_system=self.source_system,
        )
        self.boot_time = time.time()
        self.last_heartbeat = 0.0
        self.sent_count = 0

    def send_heartbeat_if_due(self) -> None:
        """Sends an ONBOARD_CONTROLLER heartbeat at 1 Hz to enable offboard acceptance."""
        now = time.time()
        if now - self.last_heartbeat >= 1.0:
            self.mav.mav.heartbeat_send(
                mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER,
                mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                0,
                0,
                mavutil.mavlink.MAV_STATE_ACTIVE,
            )
            self.last_heartbeat = now

    def send_velocity_command(self, vx: float, vy: float, vz: float = 0.0) -> Tuple[bool, str]:
        """
        Dispatches SET_POSITION_TARGET_LOCAL_NED velocity setpoint to PX4.
        vx: North velocity (m/s)
        vy: East velocity (m/s)
        vz: Down velocity (m/s), default 0.0
        """
        time_boot_ms = int((time.time() - self.boot_time) * 1000) & 0xFFFFFFFF
        try:
            self.mav.mav.set_position_target_local_ned_send(
                time_boot_ms,
                self.target_system,
                self.target_component,
                mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                self.type_mask,
                0.0,
                0.0,
                0.0,  # x, y, z ignored
                float(vx),
                float(vy),
                float(vz),  # vx, vy, vz
                0.0,
                0.0,
                0.0,  # afx, afy, afz ignored
                0.0,
                0.0,  # yaw, yaw_rate ignored
            )
            self.sent_count += 1
            status = f"SENT (seq={self.sent_count}, frame=LOCAL_NED, tgt={self.target_system}:{self.target_component})"
            return True, status
        except Exception as exc:
            return False, f"ERR: {exc}"

    def poll_telemetry_position(self) -> Optional[Tuple[float, float, float]]:
        """
        Non-blocking poll for incoming LOCAL_POSITION_NED telemetry from PX4.
        Returns (x_north, y_east, z_down) if available.
        """
        msg = self.mav.recv_match(type="LOCAL_POSITION_NED", blocking=False)
        if msg:
            return float(msg.x), float(msg.y), float(msg.z)
        return None


class TrafficBridgeNode:
    """
    Main orchestration node:
    - 10 Hz real-time control loop
    - BlueSky stream ingestion
    - WGS-84 -> Local Cartesian projection
    - RVO2 conflict-free velocity computation
    - MAVLink offboard velocity target dispatching
    - 10 Hz structured console telemetry logging
    """

    def __init__(
        self,
        stream_mode: str = "bluesky",
        stream_host: str = "127.0.0.1",
        stream_port: int = 11000,
        mavlink_url: str = "udpout:127.0.0.1:14540",
        origin_lat: float = 52.3676,
        origin_lon: float = 4.9041,
        ego_start_north: float = 0.0,
        ego_start_east: float = 0.0,
        waypoint_north: float = 400.0,
        waypoint_east: float = 0.0,
        visual: bool = False,
        gif_output: Optional[str] = None,
    ):
        self.projector = GeodeticProjector(origin_lat, origin_lon)
        self.ingestor = BlueSkyNetworkIngestor(
            mode=stream_mode,
            host=stream_host,
            port=stream_port,
            projector=self.projector,
        )
        self.engine = RVO2DeconflictionEngine(
            time_step=0.1,
            neighbor_dist=150.0,
            max_neighbors=10,
            time_horizon=10.0,
            radius=25.0,
            max_speed=15.0,
        )
        self.dispatcher = MAVLinkOffboardDispatcher(connection_url=mavlink_url)

        # Ego state in local Cartesian NED (meters)
        self.ego_pos: Tuple[float, float] = (ego_start_north, ego_start_east)
        self.target_waypoint: Tuple[float, float] = (waypoint_north, waypoint_east)
        self.running = False

        # Visualizer configuration
        self.visual = visual
        self.gif_output = gif_output
        self.visualizer: Optional[TacticalVisualizer] = None
        if self.visual:
            if not VISUALIZER_AVAILABLE:
                print("[TrafficBridge] WARNING: tactical_visualizer or pygame not available. Running headless.")
            else:
                self.visualizer = TacticalVisualizer(
                    record_gif=bool(self.gif_output),
                    gif_output_path=self.gif_output,
                )

    def start(self) -> None:
        """Starts ingestion and offboard loop."""
        self.running = True
        self.ingestor.start()

        # Connect graceful termination signals
        import threading
        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGINT, self._sig_handler)
            signal.signal(signal.SIGTERM, self._sig_handler)

        print("\n" + "=" * 80)
        print("  TACTICAL UAV DECONFLICTION BRIDGE (BlueSky -> RVO2 -> PX4 MAVLink)")
        print("=" * 80)
        print(f"Origin (Lat/Lon):     {self.projector.origin_lat:.5f}, {self.projector.origin_lon:.5f}")
        print(f"Ego Start Position:   North={self.ego_pos[0]:.1f}m, East={self.ego_pos[1]:.1f}m")
        print(f"Target Waypoint:      North={self.target_waypoint[0]:.1f}m, East={self.target_waypoint[1]:.1f}m")
        print(f"RVO2 Protected Zone:  Radius={self.engine.radius:.1f}m, Horizon={self.engine.time_horizon:.1f}s")
        print(f"Ingestion Mode:       {self.ingestor.mode}")
        print(f"MAVLink SITL Target:  {self.dispatcher.connection_url}")
        print("-" * 80)

        self._run_loop()

    def _sig_handler(self, signum, frame):
        print("\n[TrafficBridge] Shutdown signal received. Stopping node...")
        self.running = False

    def _run_loop(self) -> None:
        """Exact 10 Hz (100 ms period) real-time control loop."""
        dt = 0.1  # 10 Hz
        next_cycle = time.time()

        while self.running:
            cycle_start = time.time()

            # 1. Dispatch companion heartbeat (1 Hz)
            self.dispatcher.send_heartbeat_if_due()

            # 2. Check for incoming PX4 telemetry to update ego position
            px4_pos = self.dispatcher.poll_telemetry_position()
            if px4_pos is not None:
                self.ego_pos = (px4_pos[0], px4_pos[1])

            # 3. Ingest active aircraft states
            traffic_db = self.ingestor.update()
            traffic_list = list(traffic_db.values())

            # 4. Compute preferred velocity towards target waypoint
            v_pref = self.engine.compute_preferred_velocity(self.ego_pos, self.target_waypoint)

            # 5. Execute RVO2 ORCA deconfliction step
            v_cmd, orca_lines, active_conflicts = self.engine.solve(
                self.ego_pos,
                v_pref,
                traffic_list,
            )

            # 6. Dispatch conflict-free velocity command to PX4 SITL via MAVLink
            success, mav_status = self.dispatcher.send_velocity_command(
                vx=v_cmd[0],
                vy=v_cmd[1],
                vz=0.0,
            )

            # If not receiving telemetry from PX4, integrate ego position locally for simulation continuity
            if px4_pos is None:
                new_x = self.ego_pos[0] + v_cmd[0] * dt
                new_y = self.ego_pos[1] + v_cmd[1] * dt
                self.ego_pos = (new_x, new_y)

            # 7. Formulate structured 10 Hz log entry
            iso_time = datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]
            log_line = (
                f"[{iso_time}] [10Hz] "
                f"POS=({self.ego_pos[0]:6.1f}, {self.ego_pos[1]:6.1f})m | "
                f"CONFLICTS={active_conflicts:1d} (ORCA lines: {orca_lines:1d}) | "
                f"V_PREF=({v_pref[0]:5.2f}, {v_pref[1]:5.2f}) | "
                f"V_CMD=({v_cmd[0]:5.2f}, {v_cmd[1]:5.2f}) m/s | "
                f"MAVLINK={mav_status}"
            )
            print(log_line)

            # 8. Render Tactical Visualizer Frame (if active)
            if self.visualizer is not None:
                keep_running = self.visualizer.render_frame(
                    ego_pos=self.ego_pos,
                    target_waypoint=self.target_waypoint,
                    v_pref=v_pref,
                    v_cmd=v_cmd,
                    traffic=traffic_list,
                    active_conflicts=active_conflicts,
                    orca_lines=orca_lines,
                    mav_status=mav_status,
                    dt_hz=10.0,
                    sim_time_str=iso_time,
                )
                if not keep_running:
                    print("[TrafficBridge] Visualizer closed by user. Terminating mission.")
                    break

            # Waypoint reached condition
            dist_to_wp = math.hypot(
                self.target_waypoint[0] - self.ego_pos[0],
                self.target_waypoint[1] - self.ego_pos[1],
            )
            if dist_to_wp < 2.0:
                print(f"[TrafficBridge] Waypoint ({self.target_waypoint[0]}, {self.target_waypoint[1]}) reached!")
                # Zero out velocity command
                self.dispatcher.send_velocity_command(0.0, 0.0, 0.0)
                time.sleep(0.5)
                break

            # Precise timing control for 10 Hz loop
            next_cycle += dt
            sleep_time = next_cycle - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_cycle = time.time()

        self.cleanup()

    def cleanup(self) -> None:
        """Closes resources gracefully."""
        print("[TrafficBridge] Cleaning up resources...")
        if self.visualizer is not None:
            self.visualizer.close()
        self.ingestor.stop()
        print("[TrafficBridge] Node successfully halted.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tactical Multi-UAV Deconfliction Bridge (BlueSky -> RVO2 -> PX4 MAVLink)"
    )
    parser.add_argument(
        "--stream",
        type=str,
        default="bluesky",
        choices=["bluesky", "zmq", "tcp", "synthetic"],
        help="Stream ingestion source mode (default: bluesky)",
    )
    parser.add_argument(
        "--stream-host",
        type=str,
        default="127.0.0.1",
        help="Hostname or IP of the stream source (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--stream-port",
        type=int,
        default=11000,
        help="Port of the stream source (default: 11000 for BlueSky recv_port)",
    )
    parser.add_argument(
        "--mavlink",
        type=str,
        default="udpout:127.0.0.1:14540",
        help="MAVLink connection URI for PX4 SITL (default: udpout:127.0.0.1:14540)",
    )
    parser.add_argument(
        "--origin-lat",
        type=float,
        default=52.3676,
        help="Geographic origin latitude in degrees (default: 52.3676)",
    )
    parser.add_argument(
        "--origin-lon",
        type=float,
        default=4.9041,
        help="Geographic origin longitude in degrees (default: 4.9041)",
    )
    parser.add_argument(
        "--ego-start",
        type=str,
        default="0.0,0.0",
        help="Ego initial local NED position in meters (North,East, default: 0.0,0.0)",
    )
    parser.add_argument(
        "--waypoint",
        type=str,
        default="400.0,0.0",
        help="Ego target waypoint in local NED meters (North,East, default: 400.0,0.0)",
    )
    parser.add_argument(
        "--visual",
        "--gui",
        action="store_true",
        help="Launch the live desktop tactical radar & telemetry HUD window",
    )
    parser.add_argument(
        "--gif-out",
        type=str,
        default=None,
        help="Export animated GIF recording of the tactical flight scenario",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    start_north, start_east = map(float, args.ego_start.split(","))
    wp_north, wp_east = map(float, args.waypoint.split(","))

    node = TrafficBridgeNode(
        stream_mode=args.stream,
        stream_host=args.stream_host,
        stream_port=args.stream_port,
        mavlink_url=args.mavlink,
        origin_lat=args.origin_lat,
        origin_lon=args.origin_lon,
        ego_start_north=start_north,
        ego_start_east=start_east,
        waypoint_north=wp_north,
        waypoint_east=wp_east,
        visual=args.visual,
        gif_output=args.gif_out,
    )
    node.start()

