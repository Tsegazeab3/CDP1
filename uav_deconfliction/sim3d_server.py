#!/usr/bin/env python3
"""
sim3d_server.py - 3D Multi-UAV Physics Flight Deck & Strategic/Tactical UTM Server

Features:
- Layer 1 Strategic Optimization: D4orm (Diffusion Denoising), SIPP Corridors, MPPI, CEM, Straight-Line
- Sensor Profiles: UWB (Decimeter, 2.5m bubble), RTK-GPS (6.0m), Standard GPS (25.0m)
- Heterogeneous Fleet: Scout, Courier, Heavy-Lift with distinct dynamics
- Layer 2 Tactical Deconfliction: 3D Altitude-Aware RVO2 (ORCA)
- Layer 3 Physical Aerodynamics: 6-DOF non-ideal multirotor physics with motor lag and wind gusts
- High-Performance WebGL 3D Flight Deck streaming on http://localhost:8085
"""

import json
import math
import os
import queue
import signal
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from flask import Flask, jsonify, request, send_from_directory, Response
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
from strategic_optimizer import (
    DRONE_CLASSES,
    SENSOR_PROFILES,
    StrategicPlanResult,
    StrategicTrajectoryOptimizer,
)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "web3d"

app = Flask(__name__, static_folder=str(STATIC_DIR / "static"))


class Simulation3DCore:
    def __init__(
        self,
        origin_lat: float = 25.2048,
        origin_lon: float = 55.2708,
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
            steady_wind=(1.0, -0.6, 0.0),
            gust_intensity=0.5,
        )

        # Strategic Method & Sensor Profile (Default: UWB + D4orm)
        self.sensor_profile_id = "UWB"
        self.strategic_method = "d4orm"
        self.current_sensor_profile = SENSOR_PROFILES[self.sensor_profile_id]

        # Tactical RVO2 Engine
        self.deconflict_engine = RVO2DeconflictionEngine(
            time_step=0.1,
            neighbor_dist=80.0,
            max_neighbors=15,
            time_horizon=8.0,
            radius=self.current_sensor_profile.safe_separation_radius_m,
            max_speed=15.0,
        )
        self.dispatcher = MAVLinkOffboardDispatcher(connection_url=mavlink_url)

        # Strategic Layer 1 Controllers
        self.strategic_sipp = StrategicController(num_drones=10, wind_vector=self.physics.steady_wind)
        self.strategic_optimizer = StrategicTrajectoryOptimizer(
            sensor_profile_id=self.sensor_profile_id,
            dt=0.35,
            horizon_steps=70,
            num_samples=64,
            num_diffuse_steps=20,
            max_iterations=5,
        )
        self.strategic_plan_result: Optional[StrategicPlanResult] = None

        # UAV State Store & Trajectories
        self.uavs: Dict[str, DynamicUAVState] = {}
        self.uav_classes: Dict[str, str] = {}
        self.uav_waypoint_routes: Dict[str, List[Tuple[float, float, float]]] = {}
        self.uav_waypoints_4d: Dict[str, List[Tuple[float, float, float, float, float, float]]] = {}

        # Sim control & metrics
        self.paused = False
        self.sim_time = 0.0
        self.scenario_mode = "benchmark"
        self.lock = threading.Lock()

        # Initialize default scenario
        self.init_scenario(mode="benchmark")

    def set_sensor_profile(self, profile_id: str):
        """Switches active sensor profile and updates physical safety parameters."""
        if profile_id not in SENSOR_PROFILES:
            return
        with self.lock:
            self.sensor_profile_id = profile_id
            self.current_sensor_profile = SENSOR_PROFILES[profile_id]
            self.deconflict_engine.radius = self.current_sensor_profile.safe_separation_radius_m
            self.strategic_optimizer = StrategicTrajectoryOptimizer(
                sensor_profile_id=self.sensor_profile_id,
                dt=0.35,
                horizon_steps=70,
                num_samples=64,
                num_diffuse_steps=20,
                max_iterations=5,
            )
            print(f"[UTM Config] Sensor profile set to: {self.current_sensor_profile.name} (Safety Radius: {self.deconflict_engine.radius:.1f}m)")
        self.init_scenario(mode=self.scenario_mode)

    def set_strategic_method(self, method: str):
        """Switches active strategic planning algorithm."""
        allowed = ["d4orm", "d4orm-d", "sipp", "mppi", "cem", "straight"]
        if method not in allowed:
            return
        with self.lock:
            self.strategic_method = method
            print(f"[UTM Config] Strategic algorithm set to: {self.strategic_method.upper()}")
        self.init_scenario(mode=self.scenario_mode)

    def init_scenario(
        self,
        num_drones: int = 10,
        radius: float = 260.0,
        mode: Optional[str] = None,
        seed: Optional[int] = 42,
    ):
        """Initializes multi-UAV scenario with assigned strategic trajectories along corridors."""
        if mode is not None:
            self.scenario_mode = mode

        drone_ids = [f"UAV-{i+1:02d}" for i in range(num_drones)]
        class_keys = ["SCOUT", "COURIER", "HEAVY_LIFT"]
        self.uav_classes = {
            uid: class_keys[i % len(class_keys)] for i, uid in enumerate(drone_ids)
        }

        # Build corridor waypoints for each drone matching CorridorGraph:
        # W0: PERIM_i (station pad at 260m)
        # W1: HUB_IN_i (hub entry gate at 110m, angle + 0.35)
        # W2: HUB_OUT_i (hub exit gate at 110m, angle + pi - 0.25)
        # W3: PERIM_{(i+5)%10} (opposite destination station at 260m)
        # W4: PERIM_i (return station at 260m)
        hub_r = 110.0
        routes = {}
        hub_ins = []
        hub_outs = []

        for i, uid in enumerate(drone_ids):
            angle = i * (2.0 * math.pi / num_drones)
            alt_m = 90.0 if (i % 2 == 0) else 105.0
            altitude = -alt_m

            start_x = radius * math.cos(angle)
            start_y = radius * math.sin(angle)

            in_angle = angle + 0.35
            out_angle = angle + math.pi - 0.25

            wp1 = (hub_r * math.cos(in_angle), hub_r * math.sin(in_angle), altitude)
            wp2 = (hub_r * math.cos(out_angle), hub_r * math.sin(out_angle), altitude)
            wp3 = (-start_x, -start_y, altitude)
            wp4 = (start_x, start_y, altitude)
            routes[uid] = [wp1, wp2, wp3, wp4]

            hub_ins.append([wp1[0], wp1[1], alt_m, 0.0, 0.0, 0.0])
            hub_outs.append([wp2[0], wp2[1], alt_m, 0.0, 0.0, 0.0])

        hub_ins_arr = np.array(hub_ins)
        hub_outs_arr = np.array(hub_outs)

        # Strategic planning for Corridor and Hub Crossing
        if self.strategic_method == "sipp":
            self.strategic_sipp.reset(num_drones=num_drones, sim_time=0.0)
            if self.scenario_mode == "random":
                self.strategic_sipp.initialize_random_scenario(drone_ids, sim_time=0.0, seed=seed)
            else:
                self.strategic_sipp.initialize_scenario(drone_ids, sim_time=0.0)
            self.strategic_plan_result = None
        else:
            opt_method = self.strategic_method if self.strategic_method in ["d4orm", "d4orm-d", "mppi", "cem", "straight"] else "d4orm"
            agent_classes = [self.uav_classes[uid] for uid in drone_ids]
            self.strategic_plan_result = self.strategic_optimizer.plan(
                starts=hub_ins_arr,
                goals=hub_outs_arr,
                method=opt_method,
                agent_classes=agent_classes,
                seed=seed if seed is not None else 42,
            )

        with self.lock:
            self.sim_time = 0.0
            self.uavs.clear()
            self.uav_waypoint_routes = routes
            self.uav_waypoints_4d.clear()

            if self.strategic_plan_result is not None:
                for i, uid in enumerate(drone_ids):
                    wpts = self.strategic_plan_result.get_agent_waypoints(i)
                    self.uav_waypoints_4d[uid] = [
                        (w.t, w.x, w.y, -w.z, w.vx, w.vy) for w in wpts
                    ]

            if self.scenario_mode == "random" and self.strategic_method == "sipp":
                for uid in drone_ids:
                    plan = self.strategic_sipp.plans.get(uid)
                    d_class = DRONE_CLASSES[self.uav_classes[uid]]
                    if plan and plan.legs:
                        start_edge = self.strategic_sipp.graph.edges.get(plan.legs[0].edge_id)
                        start_node = self.strategic_sipp.graph.nodes[start_edge.from_id]
                        dest_node = self.strategic_sipp.graph.nodes[plan.destination_node]
                        sx, sy, sz = start_node.x, start_node.y, start_node.z
                        gx, gy, gz = dest_node.x, dest_node.y, dest_node.z
                        tgt_node = self.strategic_sipp.graph.nodes[start_edge.to_id]
                        tx, ty, tz = tgt_node.x, tgt_node.y, tgt_node.z
                    else:
                        sx, sy, sz = 0.0, 0.0, -90.0
                        gx, gy, gz = 0.0, 0.0, -90.0
                        tx, ty, tz = 0.0, 0.0, -90.0
                    self.uavs[uid] = DynamicUAVState(
                        id=uid,
                        x=sx, y=sy, z=sz,
                        vx=0.0, vy=0.0, vz=0.0,
                        color=d_class.color_hex,
                        target_x=tx, target_y=ty, target_z=tz,
                        dest_x=gx, dest_y=gy, dest_z=gz,
                        wp_index=0,
                    )
            else:
                for i, uid in enumerate(drone_ids):
                    d_class = DRONE_CLASSES[self.uav_classes[uid]]
                    angle = i * (2.0 * math.pi / num_drones)
                    alt_m = 90.0 if (i % 2 == 0) else 105.0
                    altitude = -alt_m
                    sx = radius * math.cos(angle)
                    sy = radius * math.sin(angle)
                    r = self.uav_waypoint_routes[uid]
                    tx, ty, tz = r[0]
                    gx, gy, gz = r[2]

                    self.uavs[uid] = DynamicUAVState(
                        id=uid,
                        x=sx, y=sy, z=altitude,
                        vx=0.0, vy=0.0, vz=0.0,
                        color=d_class.color_hex,
                        target_x=tx, target_y=ty, target_z=tz,
                        dest_x=gx, dest_y=gy, dest_z=gz,
                        wp_index=0,
                    )

    def step(self, dt: float) -> List[dict]:
        """Executes one simulation step across strategic tracking, tactical RVO2, and physics."""
        with self.lock:
            if self.paused:
                return [asdict(u) for u in self.uavs.values()]

            self.sim_time += dt
            wind = self.physics.update_gust(dt)

            uav_keys = list(self.uavs.keys())
            strategic_targets: Dict[str, Tuple[float, float, float, float]] = {}

            # ------------------------------------------------------------------
            # 1. Strategic Layer Target Tracking (Corridors + Hub Crossing)
            # ------------------------------------------------------------------
            for uid in uav_keys:
                u = self.uavs[uid]
                d_class = DRONE_CLASSES.get(self.uav_classes.get(uid, "COURIER"), DRONE_CLASSES["COURIER"])

                if self.strategic_method == "sipp":
                    spd = math.hypot(u.vx, u.vy)
                    tx, ty, tz, tspeed = self.strategic_sipp.get_current_target(
                        uid, (u.x, u.y, u.z), spd, self.sim_time
                    )
                    strategic_targets[uid] = (tx, ty, tz, min(d_class.max_speed_ms, max(3.0, tspeed)))
                    plan = self.strategic_sipp.plans.get(uid)
                    if plan and plan.destination_node in self.strategic_sipp.graph.nodes:
                        dn = self.strategic_sipp.graph.nodes[plan.destination_node]
                        u.dest_x, u.dest_y, u.dest_z = dn.x, dn.y, dn.z
                else:
                    routes = self.uav_waypoint_routes.get(uid, [])
                    if not routes:
                        strategic_targets[uid] = (u.x, u.y, u.z, 0.0)
                        continue

                    wp_idx = u.wp_index % len(routes)
                    curr_wp = routes[wp_idx]
                    dist_to_wp = math.hypot(curr_wp[0] - u.x, curr_wp[1] - u.y, curr_wp[2] - u.z)

                    if dist_to_wp < 14.0:
                        u.wp_index = (u.wp_index + 1) % len(routes)
                        wp_idx = u.wp_index % len(routes)
                        curr_wp = routes[wp_idx]

                    if wp_idx == 1 and uid in self.uav_waypoints_4d:
                        # Central hub crossing: track D4ORM/MPPI/CEM optimized 4D path
                        hub_wpts = self.uav_waypoints_4d[uid]
                        in_pt = routes[0]
                        out_pt = routes[1]
                        tot_hub_dist = math.hypot(out_pt[0] - in_pt[0], out_pt[1] - in_pt[1])
                        done_dist = math.hypot(u.x - in_pt[0], u.y - in_pt[1])
                        frac = max(0.0, min(1.0, done_dist / max(1.0, tot_hub_dist)))
                        wpt_idx = min(len(hub_wpts) - 1, int(frac * len(hub_wpts)) + 2)
                        hw = hub_wpts[wpt_idx]
                        tx, ty, tz = hw[1], hw[2], hw[3]
                        strategic_targets[uid] = (tx, ty, tz, d_class.max_speed_ms)
                    else:
                        tx, ty, tz = curr_wp
                        strategic_targets[uid] = (tx, ty, tz, d_class.max_speed_ms)

                    u.dest_x, u.dest_y, u.dest_z = routes[2]

                u.target_x, u.target_y, u.target_z = strategic_targets[uid][:3]

            # ------------------------------------------------------------------
            # 2. Preferred Velocities with Cross-Track Corridor Centering
            # ------------------------------------------------------------------
            for uid in uav_keys:
                u = self.uavs[uid]
                tx, ty, tz, tspeed = strategic_targets[uid]
                dx = tx - u.x
                dy = ty - u.y
                dz = tz - u.z
                dist_2d = math.hypot(dx, dy)

                if tspeed <= 0.0 or dist_2d < 0.5:
                    u.vx_pref, u.vy_pref = 0.0, 0.0
                else:
                    spd = min(tspeed, max(2.5, dist_2d * 1.5))
                    u.vx_pref = (dx / dist_2d) * spd
                    u.vy_pref = (dy / dist_2d) * spd

                u.vz_pref = max(-2.5, min(2.5, dz * 1.2))

            # ------------------------------------------------------------------
            # 3. Layer 2 Tactical Deconfliction (Altitude-Aware ORCA)
            # ------------------------------------------------------------------
            # Group by altitude band (separation >= 8m considered vertical clearance)
            vert_threshold = 8.0
            layer_groups: Dict[int, List[str]] = {}
            for uid in uav_keys:
                alt = abs(self.uavs[uid].z)
                layer_key = int(round(alt / vert_threshold)) * int(vert_threshold)
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
                    d_class = DRONE_CLASSES.get(self.uav_classes.get(uid, "COURIER"), DRONE_CLASSES["COURIER"])
                    aid = sim.addAgent(
                        (float(u.x), float(u.y)),
                        self.deconflict_engine.neighbor_dist,
                        self.deconflict_engine.max_neighbors,
                        self.deconflict_engine.time_horizon,
                        self.deconflict_engine.time_horizon_obst,
                        self.deconflict_engine.radius,
                        d_class.max_speed_ms,
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
            # 4. Physical Dynamics Integration (6-DOF with Lag & Wind)
            # ------------------------------------------------------------------
            for uid in uav_keys:
                u = self.uavs[uid]
                self.physics.step_uav(u, commands[uid], dt, wind)

            # ------------------------------------------------------------------
            # 5. Separation & Conflict Evaluation
            # ------------------------------------------------------------------
            safe_dist = self.current_sensor_profile.safe_separation_radius_m
            for i, uid1 in enumerate(uav_keys):
                u1 = self.uavs[uid1]
                u1.in_conflict = False
                u1.min_dist_to_traffic = 999.0

                for j, uid2 in enumerate(uav_keys):
                    if i == j:
                        continue
                    u2 = self.uavs[uid2]
                    dist_3d = math.hypot(u1.x - u2.x, u1.y - u2.y, u1.z - u2.z)
                    if dist_3d < u1.min_dist_to_traffic:
                        u1.min_dist_to_traffic = dist_3d
                    if dist_3d < safe_dist:
                        u1.in_conflict = True

            # Format Telemetry
            telemetry = []
            for uid in uav_keys:
                u = self.uavs[uid]
                d = asdict(u)
                d["drone_class"] = self.uav_classes.get(uid, "COURIER")
                d["sensor_profile"] = self.sensor_profile_id
                d["strategic_method"] = self.strategic_method
                telemetry.append(d)

            return telemetry


# Global simulation core and connected client queues
core = Simulation3DCore()
client_queues: List[queue.Queue] = []
clients_lock = threading.Lock()


# --- HTTP API Routes ---

@app.route("/")
def index():
    return send_from_directory(str(STATIC_DIR), "index.html")


@app.route("/api/config", methods=["GET"])
def get_config():
    """Returns available strategic algorithms, sensor profiles, and drone classes."""
    return jsonify({
        "current_method": core.strategic_method,
        "current_sensor_profile": core.sensor_profile_id,
        "sensor_profiles": {k: asdict(v) for k, v in SENSOR_PROFILES.items()},
        "drone_classes": {k: asdict(v) for k, v in DRONE_CLASSES.items()},
    })


@app.route("/api/set_strategic_method", methods=["POST"])
def set_strategic_method():
    data = request.get_json(silent=True) or {}
    method = data.get("method", "d4orm")
    core.set_strategic_method(method)
    return jsonify({"success": True, "method": core.strategic_method})


@app.route("/api/set_sensor_profile", methods=["POST"])
def set_sensor_profile():
    data = request.get_json(silent=True) or {}
    profile_id = data.get("profile", "UWB")
    core.set_sensor_profile(profile_id)
    return jsonify({"success": True, "profile": core.sensor_profile_id})


@app.route("/api/strategic_trajectories", methods=["GET"])
def get_strategic_trajectories():
    """Returns 3D waypoints of the strategic trajectory plan along corridors for WebGL visualization."""
    with core.lock:
        trajs = []
        uav_keys = list(core.uavs.keys())
        num_uavs = len(uav_keys)

        for i, uid in enumerate(uav_keys):
            pts = []
            if core.strategic_method == "sipp":
                plan = core.strategic_sipp.plans.get(uid)
                if plan and plan.legs:
                    for leg in plan.legs:
                        pts.append({
                            "three_x": float(leg.from_pos[1]),
                            "three_y": float(abs(leg.from_pos[2])),
                            "three_z": float(-leg.from_pos[0]),
                        })
                    pts.append({
                        "three_x": float(plan.legs[-1].to_pos[1]),
                        "three_y": float(abs(plan.legs[-1].to_pos[2])),
                        "three_z": float(-plan.legs[-1].to_pos[0]),
                    })
            else:
                routes = core.uav_waypoint_routes.get(uid, [])
                if routes:
                    angle = i * (2.0 * math.pi / max(1, num_uavs))
                    alt_m = 90.0 if (i % 2 == 0) else 105.0
                    sx = 260.0 * math.cos(angle)
                    sy = 260.0 * math.sin(angle)
                    # 1. Start Station
                    pts.append({"three_x": float(sy), "three_y": float(alt_m), "three_z": float(-sx)})
                    # 2. HUB_IN Gate
                    pts.append({"three_x": float(routes[0][1]), "three_y": float(alt_m), "three_z": float(-routes[0][0])})
                    # 3. Hub Crossing Trajectory (D4ORM / MPPI / CEM)
                    if uid in core.uav_waypoints_4d:
                        for w in core.uav_waypoints_4d[uid]:
                            pts.append({
                                "three_x": float(w[2]),
                                "three_y": float(abs(w[3])),
                                "three_z": float(-w[1]),
                            })
                    # 4. HUB_OUT Gate
                    pts.append({"three_x": float(routes[1][1]), "three_y": float(alt_m), "three_z": float(-routes[1][0])})
                    # 5. Destination Station
                    pts.append({"three_x": float(routes[2][1]), "three_y": float(alt_m), "three_z": float(-routes[2][0])})

            trajs.append({
                "drone_id": uid,
                "class": core.uav_classes.get(uid, "COURIER"),
                "points": pts,
            })

        min_sep = core.strategic_plan_result.min_separation_m if core.strategic_plan_result else 25.0
        conflicts = core.strategic_plan_result.conflict_count if core.strategic_plan_result else 0
        compute_time = core.strategic_plan_result.compute_time_s if core.strategic_plan_result else 0.01

        return jsonify({
            "available": True,
            "method": core.strategic_method,
            "sensor_profile": core.sensor_profile_id,
            "min_separation": min_sep,
            "conflicts": conflicts,
            "compute_time": compute_time,
            "trajectories": trajs,
        })


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
    core.init_scenario(mode=core.scenario_mode)
    return jsonify({"success": True})


@app.route("/api/pause", methods=["POST"])
def pause():
    data = request.get_json(silent=True) or {}
    with core.lock:
        core.paused = bool(data.get("paused", not core.paused))
    return jsonify({"paused": core.paused})


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
    with core.lock:
        data = core.strategic_sipp.get_corridor_telemetry(sim_time=core.sim_time)
        # Adapt pipe radius to active sensor profile
        data["pipe_radius"] = core.current_sensor_profile.corridor_pipe_radius_m
    return jsonify(data)


def background_physics_loop():
    """30 Hz physical integration, strategic update, and SSE broadcast thread."""
    dt = 1.0 / 30.0
    corridor_counter = 0

    while True:
        t0 = time.time()
        telemetry = core.step(dt)

        with clients_lock:
            active_qs = list(client_queues)

        if active_qs:
            payload = json.dumps({"type": "telemetry", "uavs": telemetry})
            corridor_counter += 1
            corridor_payload = None

            if corridor_counter >= 15:
                corridor_counter = 0
                with core.lock:
                    corridor_data = core.strategic_sipp.get_corridor_telemetry(sim_time=core.sim_time)
                    corridor_data["pipe_radius"] = core.current_sensor_profile.corridor_pipe_radius_m
                    corridor_data["sensor_profile"] = core.sensor_profile_id
                    corridor_data["strategic_method"] = core.strategic_method
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
        time.sleep(max(0.001, dt - elapsed))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8085))

    sim_thread = threading.Thread(target=background_physics_loop, daemon=True)
    sim_thread.start()

    print("\n" + "=" * 80)
    print("  AUTONOMOUS UAV 3D FLIGHT DECK  ·  HETEROGENEOUS FLEETS & SENSOR PROFILES")
    print("=" * 80)
    print(f"  URL:              http://localhost:{port}")
    print(f"  Strategic Layer:  D4orm (Diffusion), SIPP, MPPI, CEM, Straight-Line")
    print(f"  Sensor Profiles:  UWB (2.5m bubble), RTK-GPS (6.0m), Standard GPS (25.0m)")
    print(f"  Fleet Classes:    Scout (Cyan), Courier (Emerald), Heavy-Lift (Orange)")
    print("=" * 80 + "\n")

    from werkzeug.serving import run_simple
    run_simple("0.0.0.0", port, app, threaded=True)
