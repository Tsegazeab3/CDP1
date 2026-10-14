#!/usr/bin/env python3
"""
sim3d_server.py - 3D Multi-UAV Physics Flight Deck & Tactical Deconfliction Server
Combines:
- 6-DOF non-ideal multirotor physical dynamics (motor lag, acceleration limits, wind gusts, attitude banking)
- Closed-loop RVO2 (ORCA) tactical deconfliction on actual perturbed states
- BlueSky ACDATA network stream ingestion
- MAVLink PX4 SITL offboard velocity dispatching (UDP 14540)
- High-performance 3D WebGL WebSocket streaming server on http://localhost:8085
"""

import json
import math
import os
import signal
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from flask import Flask, jsonify, request, send_from_directory
from simple_websocket import ConnectionClosed, Server

import rvo2

# Local modules
from quadcopter_dynamics import DynamicUAVState, QuadcopterPhysicsSimulator
from traffic_bridge import (
    BlueSkyNetworkIngestor,
    GeodeticProjector,
    MAVLinkOffboardDispatcher,
    RVO2DeconflictionEngine,
)
from strategic_planner import MissionType, StrategicController

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "web3d"

app = Flask(__name__, static_folder=str(STATIC_DIR / "static"))


class Simulation3DCore:
    def __init__(
        self,
        origin_lat: float = 52.3676,
        origin_lon: float = 4.9041,
        stream_mode: str = "bluesky",
        stream_host: str = "127.0.0.1",
        stream_port: int = 11000,
        mavlink_url: str = "udpout:127.0.0.1:14540",
    ):
        self.projector = GeodeticProjector(origin_lat, origin_lon)
        self.ingestor = BlueSkyNetworkIngestor(
            mode=stream_mode,
            host=stream_host,
            port=stream_port,
            projector=self.projector,
        )
        self.physics = QuadcopterPhysicsSimulator(
            tau=0.35,
            max_accel=5.0,
            max_climb_rate=3.0,
            max_tilt_deg=35.0,
            enable_wind=True,
            steady_wind=(1.2, -0.8, 0.0),
            gust_intensity=0.8,
        )
        self.deconflict_engine = RVO2DeconflictionEngine(
            time_step=0.1,
            neighbor_dist=200.0,
            max_neighbors=15,
            time_horizon=10.0,
            radius=25.0,
            max_speed=15.0,
        )
        self.dispatcher = MAVLinkOffboardDispatcher(connection_url=mavlink_url)

        # Strategic 4D SIPP planner — Layer 1 above ORCA tactical layer
        self.strategic = StrategicController(num_drones=10, wind_vector=self.physics.steady_wind)

        # UAV State Store & Multi-Waypoint Flight Plans
        self.uavs: Dict[str, DynamicUAVState] = {}
        self.uav_waypoint_routes: Dict[str, List[Tuple[float, float, float]]] = {}

        # Sim control
        self.paused = False
        self.sim_time = 0.0
        self.scenario_mode = "benchmark"
        self.lock = threading.Lock()
        self.init_scenario(mode="benchmark")

    def init_scenario(
        self,
        num_drones: int = 10,
        radius: float = 260.0,
        mode: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialises 10 drones with SIPP-planned 4D spacetime corridor routes.
        Supports:
          - 'benchmark': Fixed symmetric circular antipodal hub-crossing loop.
          - 'random': Dynamic randomized origin-to-destination missions across stations.
        """
        if mode is not None:
            self.scenario_mode = mode
        elif not hasattr(self, "scenario_mode"):
            self.scenario_mode = "benchmark"

        colors = [
            "#00e5ff",  # 1: Electric Cyan
            "#f97316",  # 2: Bright Orange
            "#10b981",  # 3: Emerald Green
            "#ef4444",  # 4: Crimson Red
            "#a855f7",  # 5: Royal Purple
            "#eab308",  # 6: Golden Yellow
            "#ec4899",  # 7: Hot Pink
            "#06b6d4",  # 8: Sky Cyan
            "#84cc16",  # 9: Lime Green
            "#6366f1",  # 10: Indigo
        ]

        drone_ids = [f"UAV-{i+1:02d}" for i in range(num_drones)]
        self.strategic.reset(num_drones=num_drones, sim_time=self.sim_time)

        if self.scenario_mode == "random":
            print("\n" + "=" * 70)
            print("  STRATEGIC SIPP CORRIDOR PLANNER — Random Station-to-Station Missions")
            print("=" * 70)
            od_pairs = self.strategic.initialize_random_scenario(
                drone_ids, sim_time=self.sim_time, seed=seed
            )
            print("=" * 70 + "\n")

            with self.lock:
                self.uavs.clear()
                self.uav_waypoint_routes.clear()

                for i, uid in enumerate(drone_ids):
                    orig, dest = od_pairs.get(uid, (f"PERIM_{i}", f"PERIM_{(i+5)%10}"))
                    start_node = self.strategic.graph.nodes[orig]
                    plan = self.strategic.plans.get(uid)

                    if plan and plan.legs:
                        leg0 = plan.legs[0]
                        e0 = self.strategic.graph.edges[leg0.edge_id]
                        tgt0 = self.strategic.graph.nodes[e0.to_id]
                        tx, ty, tz = tgt0.x, tgt0.y, -leg0.alt_m
                    else:
                        tx, ty, tz = start_node.x, start_node.y, start_node.z

                    dx = tx - start_node.x
                    dy = ty - start_node.y
                    dist = math.hypot(dx, dy)
                    init_vx = 0.0
                    init_vy = 0.0

                    self.uavs[uid] = DynamicUAVState(
                        id=uid,
                        x=start_node.x, y=start_node.y, z=start_node.z,
                        vx=init_vx, vy=init_vy, vz=0.0,
                        color=colors[i % len(colors)],
                        target_x=tx, target_y=ty, target_z=tz,
                        wp_index=0,
                    )
        else:
            with self.lock:
                self.uavs.clear()
                self.uav_waypoint_routes.clear()

                for i in range(num_drones):
                    uav_id = f"UAV-{i+1:02d}"
                    angle  = i * (2.0 * math.pi / num_drones)

                    alt_m    = 90.0 if (i % 2 == 0) else 105.0
                    altitude = -alt_m  # NED z (negative altitude)

                    start_x = radius * math.cos(angle)
                    start_y = radius * math.sin(angle)

                    # Hub entry gate — first SIPP corridor node target
                    entry_angle = angle + 0.35
                    wp1 = (110.0 * math.cos(entry_angle), 110.0 * math.sin(entry_angle), altitude)

                    exit_angle = angle + math.pi - 0.25
                    wp2 = (110.0 * math.cos(exit_angle), 110.0 * math.sin(exit_angle), altitude)
                    wp3 = (-start_x, -start_y, altitude)
                    wp4 = (start_x, start_y, altitude)
                    self.uav_waypoint_routes[uav_id] = [wp1, wp2, wp3, wp4]

                    # Initial velocity at rest (strategic layer commands acceleration after scheduled departure)
                    init_vx = 0.0
                    init_vy = 0.0

                    self.uavs[uav_id] = DynamicUAVState(
                        id=uav_id,
                        x=start_x, y=start_y, z=altitude,
                        vx=init_vx, vy=init_vy, vz=0.0,
                        color=colors[i % len(colors)],
                        target_x=wp1[0], target_y=wp1[1], target_z=wp1[2],
                        wp_index=0,
                    )

            print("\n" + "=" * 70)
            print("  STRATEGIC SIPP CORRIDOR PLANNER — 4D Mission Assignment")
            print("=" * 70)
            self.strategic.initialize_scenario(drone_ids, sim_time=self.sim_time)
            print("=" * 70 + "\n")



    def step(self, dt: float) -> List[dict]:
        """
        Executes one physics + tactical deconfliction cycle over timestep dt.
        Returns serialized 3D telemetry for all active drones.
        """
        with self.lock:
            if self.paused:
                return [asdict(u) for u in self.uavs.values()]

            self.sim_time += dt

            # Update stochastic wind gust
            wind = self.physics.update_gust(dt)

            # Ingest external BlueSky traffic if streaming
            try:
                bs_traffic = self.ingestor.update()
                for acid, tr in bs_traffic.items():
                    if acid not in self.uavs and acid != "Ego":
                        self.uavs[acid] = DynamicUAVState(
                            id=acid,
                            x=tr.x, y=tr.y, z=-tr.alt_m,
                            vx=tr.vx, vy=tr.vy, vz=0.0,
                        )
            except Exception:
                pass

            uav_keys = list(self.uavs.keys())

            # ------------------------------------------------------------------
            # 1. Layer 1 (Strategic): Query SIPP controller for each drone's
            #    current 4D corridor target (target_x, target_y, target_z, speed).
            #    The controller checks the 3 replanning triggers internally and
            #    returns the up-to-date strategic objective.
            # ------------------------------------------------------------------
            strategic_targets: dict = {}
            for uid in uav_keys:
                u = self.uavs[uid]
                spd = math.hypot(u.vx, u.vy)
                tx, ty, tz, tspeed = self.strategic.get_current_target(
                    uid, (u.x, u.y, u.z), spd, self.sim_time
                )
                strategic_targets[uid] = (tx, ty, tz, tspeed)

                # Update UAV display target for WebGL waypoint beacon
                u.target_x = tx
                u.target_y = ty
                u.target_z = tz

            # ------------------------------------------------------------------
            # 2. Layer 1→2 Handoff: Preferred Velocities from Strategic Targets
            #    Preferred velocity points toward the active corridor target (tx, ty),
            #    scaled to the SIPP-assigned cruise speed for this edge.
            # ------------------------------------------------------------------
            for uid in uav_keys:
                u = self.uavs[uid]
                tx, ty, tz, tspeed = strategic_targets[uid]

                dx = tx - u.x
                dy = ty - u.y
                dz = tz - u.z
                dist_2d = math.hypot(dx, dy)

                if tspeed <= 0.0:
                    speed = 0.0
                else:
                    speed = min(tspeed, max(4.0, dist_2d))

                if dist_2d > 1.0 and speed > 0.0:
                    u.vx_pref = (dx / dist_2d) * speed
                    u.vy_pref = (dy / dist_2d) * speed
                else:
                    u.vx_pref = 0.0
                    u.vy_pref = 0.0

                # 3D climb/descent rate toward target altitude (hold on ground if speed <= 0.0)
                if speed <= 0.0:
                    u.vz_pref = 0.0
                else:
                    u.vz_pref = max(-2.5, min(2.5, dz * 0.8))

            # ------------------------------------------------------------------
            # 3. Layer 2 (Tactical): Altitude-Aware 3D ORCA Deconfliction
            #    To achieve true 3D deconfliction without ghost conflicts across
            #    different vertical flight levels, ORCA deconfliction is filtered
            #    by vertical separation (TCAS ALIM principle):
            #    Aircraft separated by >= 12m vertically operate in separate flight
            #    levels and cross freely over/under each other at the hub.
            #    For co-altitude threats (|dz| < 12m), 2D ORCA in (x, y) provides
            #    guaranteed reactive horizontal collision avoidance.
            #    Vertical velocity vz is locked to the corridor altitude controller,
            #    preventing RVO2 from pushing drones out of their flight levels.
            # ------------------------------------------------------------------
            # Group agents by co-altitude layer (15m nominal spacing, >=12m vertical clearance)
            layer_groups: Dict[int, List[str]] = {}
            for uid in uav_keys:
                u = self.uavs[uid]
                alt = abs(u.z)
                layer_key = int(round(alt / 15.0)) * 15
                layer_groups.setdefault(layer_key, []).append(uid)

            commands = {}
            for layer_key, group_uids in layer_groups.items():
                if len(group_uids) <= 1:
                    uid = group_uids[0]
                    u = self.uavs[uid]
                    commands[uid] = (float(u.vx_pref), float(u.vy_pref), float(u.vz_pref))
                    continue

                sim = rvo2.PyRVOSimulator(
                    dt,
                    self.deconflict_engine.neighbor_dist,
                    self.deconflict_engine.max_neighbors,
                    self.deconflict_engine.time_horizon,
                    self.deconflict_engine.time_horizon_obst,
                    self.deconflict_engine.radius,
                    self.deconflict_engine.max_speed,
                    (0.0, 0.0),
                )

                agent_map = {}
                for uid in group_uids:
                    u = self.uavs[uid]
                    aid = sim.addAgent(
                        (float(u.x), float(u.y)),
                        self.deconflict_engine.neighbor_dist,
                        self.deconflict_engine.max_neighbors,
                        self.deconflict_engine.time_horizon,
                        self.deconflict_engine.time_horizon_obst,
                        self.deconflict_engine.radius,
                        self.deconflict_engine.max_speed,
                        (float(u.vx), float(u.vy)),
                    )
                    sim.setAgentPrefVelocity(aid, (float(u.vx_pref), float(u.vy_pref)))
                    agent_map[uid] = aid

                sim.doStep()

                for uid, aid in agent_map.items():
                    u = self.uavs[uid]
                    v_cmd_2d = sim.getAgentVelocity(aid)
                    commands[uid] = (float(v_cmd_2d[0]), float(v_cmd_2d[1]), float(u.vz_pref))

            # ------------------------------------------------------------------
            # 5. Layer 3 (Physical): Integrate non-ideal multirotor dynamics.
            # ------------------------------------------------------------------
            for uid in uav_keys:
                u = self.uavs[uid]
                self.physics.step_uav(u, commands[uid], dt, wind)

            # ------------------------------------------------------------------
            # 6. Conflict status & minimum separation distances (3D).
            # ------------------------------------------------------------------
            for i, uid1 in enumerate(uav_keys):
                u1 = self.uavs[uid1]
                u1.in_conflict = False
                u1.min_dist_to_traffic = 999.0

                for j, uid2 in enumerate(uav_keys):
                    if i == j:
                        continue
                    u2 = self.uavs[uid2]
                    dist_3d = math.sqrt(
                        (u1.x - u2.x) ** 2 + (u1.y - u2.y) ** 2 + (u1.z - u2.z) ** 2
                    )
                    if dist_3d < u1.min_dist_to_traffic:
                        u1.min_dist_to_traffic = dist_3d
                    if dist_3d < 2.0 * self.deconflict_engine.radius:
                        u1.in_conflict = True

            # 7. MAVLink setpoints for PX4 SITL ego drone
            if "Ego" in self.uavs:
                ego = self.uavs["Ego"]
                self.dispatcher.send_heartbeat_if_due()
                self.dispatcher.send_velocity_command(vx=ego.vx_cmd, vy=ego.vy_cmd, vz=ego.vz_cmd)

            # ------------------------------------------------------------------
            # 8. Serialize drone telemetry + corridor telemetry for SSE clients.
            #    The corridor_telemetry dict is broadcast once per 30 Hz tick
            #    alongside the drone physics state, allowing the WebGL frontend
            #    to render corridor tubes and SIPP slot labels in real-time.
            # ------------------------------------------------------------------
            telemetry = []
            for u in self.uavs.values():
                d = asdict(u)
                d["min_dist"] = round(u.min_dist_to_traffic, 1)
                # Attach SIPP plan info for this drone (for UI telemetry card)
                plan_info = self.strategic.plans.get(u.id)
                if plan_info and plan_info.legs:
                    li = plan_info.current_leg % len(plan_info.legs)
                    leg = plan_info.legs[li]
                    d["sipp_edge"]   = leg.edge_id
                    d["sipp_speed"]  = round(leg.speed, 2)
                    d["sipp_alt_m"]  = leg.alt_m
                    d["sipp_leg"]    = f"{li + 1}/{len(plan_info.legs)}"
                    d["mission_type"] = MissionType.name(plan_info.mission_type)
                    d["origin"]      = getattr(plan_info, "origin_node", "")
                    d["destination"] = getattr(plan_info, "destination_node", "")
                    d["scenario_mode"] = getattr(self, "scenario_mode", "benchmark")
                telemetry.append(d)

            return telemetry

        # Return empty if paused (lock already acquired at top)




import queue

# Global simulation core and connected telemetry queues
core = Simulation3DCore()
client_queues: List[queue.Queue] = []
clients_lock = threading.Lock()


# --- HTTP Routes ---

@app.route("/")
def index():
    return send_from_directory(str(STATIC_DIR), "index.html")


@app.route("/api/params", methods=["POST"])
def set_params():
    data = request.get_json(silent=True) or {}
    with core.lock:
        if "tau" in data:
            core.physics.tau = float(data["tau"])
        if "wind" in data:
            core.physics.gust_intensity = float(data["wind"])
        if "max_accel" in data:
            core.physics.max_accel = float(data["max_accel"])
    return jsonify({"success": True})


@app.route("/api/reset", methods=["POST"])
def reset():
    core.init_scenario()
    return jsonify({"success": True})


@app.route("/api/scenario_mode", methods=["POST"])
def set_scenario_mode():
    data = request.get_json(silent=True) or {}
    mode = data.get("mode", "benchmark")
    core.init_scenario(mode=mode)
    return jsonify({"success": True, "mode": core.scenario_mode})


@app.route("/api/randomize", methods=["POST"])
def randomize_missions():
    core.init_scenario(mode="random")
    return jsonify({"success": True, "mode": "random"})


@app.route("/api/pause", methods=["POST"])
def pause():
    data = request.get_json(silent=True) or {}
    with core.lock:
        core.paused = bool(data.get("paused", not core.paused))
    return jsonify({"paused": core.paused})


@app.route("/ws_telemetry")
def ws_telemetry_stub():
    return ("", 204)


@app.route("/telemetry_stream")
def telemetry_stream():
    """Server-Sent Events (SSE) 30 Hz real-time telemetry stream."""
    def event_stream():
        q = queue.Queue(maxsize=10)
        with clients_lock:
            client_queues.append(q)
        try:
            while True:
                payload = q.get()
                yield f"data: {payload}\n\n"
        except GeneratorExit:
            pass
        finally:
            with clients_lock:
                if q in client_queues:
                    client_queues.remove(q)

    from flask import Response
    return Response(
        event_stream(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.route("/api/corridor_graph")
def corridor_graph():
    """Returns the static corridor graph (nodes + edges) as JSON for initial WebGL render."""
    with core.lock:
        data = core.strategic.get_corridor_telemetry(sim_time=core.sim_time)
    return jsonify(data)


def background_physics_loop():
    """30 Hz physical integration, SIPP strategic update, and SSE broadcast thread."""
    dt = 1.0 / 30.0  # 30 Hz
    corridor_broadcast_counter = 0
    print("[3D Flight Deck] Background physics & deconfliction loop started (30 Hz).")

    while True:
        t0 = time.time()
        telemetry = core.step(dt)

        # Broadcast to all active browser SSE queues
        with clients_lock:
            active_qs = list(client_queues)

        if active_qs:
            # Always broadcast drone physics telemetry at 30 Hz
            payload = json.dumps({"type": "telemetry", "uavs": telemetry})

            # Broadcast corridor/SIPP telemetry at 2 Hz (every 15 frames) to save bandwidth
            corridor_broadcast_counter += 1
            corridor_payload = None
            if corridor_broadcast_counter >= 15:
                corridor_broadcast_counter = 0
                with core.lock:
                    corridor_data = core.strategic.get_corridor_telemetry(sim_time=core.sim_time)
                corridor_payload = json.dumps(corridor_data)

            for q in active_qs:
                try:
                    q.put_nowait(payload)
                    if corridor_payload is not None:
                        q.put_nowait(corridor_payload)
                except queue.Full:
                    try:
                        _ = q.get_nowait()
                        q.put_nowait(payload)
                    except Exception:
                        pass

        elapsed = time.time() - t0
        sleep_time = max(0.001, dt - elapsed)
        time.sleep(sleep_time)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8085))

    # Start background physics thread
    sim_thread = threading.Thread(target=background_physics_loop, daemon=True)
    sim_thread.start()

    print("\n" + "=" * 80)
    print("  AUTONOMOUS UAV 3D FLIGHT DECK  ·  STRATEGIC 4D SIPP + TACTICAL ORCA")
    print("=" * 80)
    print(f"  URL:              http://localhost:{port}")
    print(f"  Layer 1 (Strat):  SIPP Corridor Planner  (4D spacetime paths)")
    print(f"  Layer 2 (Tact):   Python-RVO2 ORCA       (corridor-projected, 10 Hz)")
    print(f"  Layer 3 (Phys):   6-DOF Quadcopter       (motor lag, wind, attitude)")
    print(f"  Autopilot:        PyMAVLink UDP 14540     (PX4 SITL offboard)")
    print("=" * 80 + "\n")

    from werkzeug.serving import run_simple
    run_simple("0.0.0.0", port, app, threaded=True)

