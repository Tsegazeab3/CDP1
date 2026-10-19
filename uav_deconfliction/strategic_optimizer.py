"""Strategic 4D Multi-UAV Trajectory Deconfliction Engine with Heterogeneous Fleets and Sensor Profiles.

Supports:
- Model-Based Diffusion Denoising (D4orm, D4orm-D) from arXiv:2609.35651
- Sampling-based Trajectory Optimization (MPPI, CEM)
- Safe Interval Path Planning (SIPP) and Uncoordinated Straight-Line Baselines
- Sensor Profiles: UWB (High Precision, Decimeter), RTK-GPS (Centimeter), Standard GPS (Wide-Area)
- Heterogeneous Drone Classes: Scout, Courier, Heavy-Lift
"""

from __future__ import annotations

import dataclasses
import math
import time
from typing import Dict, List, Literal, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from d4orm.envs.holonomic import HolonomicEnvironment
from d4orm.planners import core
from d4orm.planners.cli import PLANNERS


# ---------------------------------------------------------------------------
# Drone Class & Sensor Profile Specifications
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class DroneClass:
    """Kinodynamic specifications for a drone category."""
    name: str
    max_speed_ms: float
    max_accel_ms2: float
    physical_radius_m: float
    color_hex: str
    payload_capacity_kg: float


# Heterogeneous Drone Fleet Classes
DRONE_CLASSES: Dict[str, DroneClass] = {
    "SCOUT": DroneClass(
        name="SCOUT",
        max_speed_ms=16.0,
        max_accel_ms2=6.0,
        physical_radius_m=0.35,
        color_hex="#00e5ff",  # Cyan
        payload_capacity_kg=0.5,
    ),
    "COURIER": DroneClass(
        name="COURIER",
        max_speed_ms=12.0,
        max_accel_ms2=4.0,
        physical_radius_m=0.60,
        color_hex="#10b981",  # Emerald
        payload_capacity_kg=2.5,
    ),
    "HEAVY_LIFT": DroneClass(
        name="HEAVY_LIFT",
        max_speed_ms=8.0,
        max_accel_ms2=2.5,
        physical_radius_m=1.10,
        color_hex="#f97316",  # Orange
        payload_capacity_kg=8.0,
    ),
}


@dataclasses.dataclass(frozen=True)
class SensorProfile:
    """Sensor navigation precision and airspace constraint specifications."""
    id: str
    name: str
    description: str
    pos_noise_sigma_m: float
    update_rate_hz: float
    effective_range_m: float
    safe_separation_radius_m: float
    corridor_pipe_radius_m: float
    nominal_arena_scale: float


SENSOR_PROFILES: Dict[str, SensorProfile] = {
    "UWB": SensorProfile(
        id="UWB",
        name="UWB (Ultra-Wideband / Local Hub)",
        description="High-precision decimeter positioning for GPS-denied vertiports and lab arenas.",
        pos_noise_sigma_m=0.15,
        update_rate_hz=40.0,
        effective_range_m=50.0,
        safe_separation_radius_m=2.5,
        corridor_pipe_radius_m=3.5,
        nominal_arena_scale=0.45,
    ),
    "RTK_GPS": SensorProfile(
        id="RTK_GPS",
        name="RTK-GPS (Centimeter Urban Airway)",
        description="Dual-frequency carrier phase differential GNSS for structured urban corridors.",
        pos_noise_sigma_m=0.08,
        update_rate_hz=10.0,
        effective_range_m=5000.0,
        safe_separation_radius_m=6.0,
        corridor_pipe_radius_m=8.0,
        nominal_arena_scale=0.85,
    ),
    "STD_GPS": SensorProfile(
        id="STD_GPS",
        name="Standard GPS (Wide-Area Regional)",
        description="Standard satellite positioning subject to atmospheric delay and urban multi-path.",
        pos_noise_sigma_m=2.80,
        update_rate_hz=5.0,
        effective_range_m=50000.0,
        safe_separation_radius_m=25.0,
        corridor_pipe_radius_m=25.0,
        nominal_arena_scale=1.00,
    ),
}


@dataclasses.dataclass(frozen=True)
class Waypoint4D:
    """4D space-time waypoint for a single UAV."""
    t: float
    x: float
    y: float
    z: float
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0


@dataclasses.dataclass
class StrategicPlanResult:
    """Standardized output from a strategic trajectory optimization run."""
    method: str
    sensor_profile_id: str
    num_agents: int
    horizon_steps: int
    dt: float
    total_time_s: float
    compute_time_s: float
    trajectories: np.ndarray  # Shape: (T, N, 6) -> [x, y, z, vx, vy, vz]
    min_separation_m: float
    conflict_count: int
    total_distance_m: float
    mean_speed_ms: float
    mean_accel_norm: float
    agent_classes: List[str]

    def get_agent_waypoints(self, agent_idx: int) -> List[Waypoint4D]:
        """Returns 4D waypoints for a specific UAV."""
        wpts = []
        for step in range(self.horizon_steps):
            t = step * self.dt
            s = self.trajectories[step, agent_idx]
            wpts.append(
                Waypoint4D(
                    t=t,
                    x=float(s[0]),
                    y=float(s[1]),
                    z=float(s[2]),
                    vx=float(s[3]),
                    vy=float(s[4]),
                    vz=float(s[5]),
                )
            )
        return wpts


class HeterogeneousUAVAirspace3D(HolonomicEnvironment):
    """3D Holonomic Environment supporting heterogeneous speeds and safety envelopes."""

    def __init__(
        self,
        starts: jnp.ndarray,
        goals: jnp.ndarray,
        separation_radius: float = 2.5,
        safe_margin: float = 1.0,
        max_speed: float = 12.0,
        max_accel: float = 4.0,
        obstacles: Optional[jnp.ndarray] = None,
    ):
        self.num_agents = starts.shape[0]
        self._custom_starts = starts
        self._custom_goals = goals
        super().__init__(self.num_agents, position_dim=3)
        self.agent_radius = separation_radius
        self.safe_margin = safe_margin
        self.max_speed = max_speed
        self.max_acceleration = max_accel
        self.x0 = self._custom_starts.flatten()
        self.xg = self._custom_goals.flatten()
        if obstacles is not None and len(obstacles) > 0:
            self.obstacle_centers = obstacles
        else:
            self.obstacle_centers = jnp.empty((0, 3))

    def generate_positions(
        self, diameter: float, num_agents: int
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return self._custom_starts, self._custom_goals


class StrategicTrajectoryOptimizer:
    """Unified Strategic Deconfliction Planner for Multi-UAV UTM Operations."""

    def __init__(
        self,
        sensor_profile_id: str = "UWB",
        dt: float = 0.2,
        horizon_steps: int = 50,
        num_samples: int = 64,
        num_diffuse_steps: int = 20,
        max_iterations: int = 5,
    ):
        self.sensor_profile = SENSOR_PROFILES.get(sensor_profile_id, SENSOR_PROFILES["UWB"])
        self.dt = dt
        self.horizon_steps = horizon_steps
        self.num_samples = num_samples
        self.num_diffuse_steps = num_diffuse_steps
        self.max_iterations = max_iterations

    def plan(
        self,
        starts: np.ndarray,
        goals: np.ndarray,
        method: Literal["d4orm", "d4orm-d", "mppi", "cem", "straight"] = "d4orm",
        agent_classes: Optional[List[str]] = None,
        seed: int = 42,
    ) -> StrategicPlanResult:
        """Optimizes collision-free 4D trajectories for heterogeneous UAVs."""
        num_agents = starts.shape[0]
        if agent_classes is None:
            # Default heterogeneous class assignment
            class_keys = list(DRONE_CLASSES.keys())
            agent_classes = [class_keys[i % len(class_keys)] for i in range(num_agents)]

        max_speed = max(DRONE_CLASSES[cls].max_speed_ms for cls in agent_classes)
        max_accel = max(DRONE_CLASSES[cls].max_accel_ms2 for cls in agent_classes)
        safe_radius = self.sensor_profile.safe_separation_radius_m

        t_start = time.perf_counter()

        if method == "straight":
            trajectories = self._plan_straight_line(starts, goals, max_speed)
            compute_time = time.perf_counter() - t_start
        else:
            starts_jnp = jnp.array(starts, dtype=jnp.float32)
            goals_jnp = jnp.array(goals, dtype=jnp.float32)

            env = HeterogeneousUAVAirspace3D(
                starts=starts_jnp,
                goals=goals_jnp,
                separation_radius=safe_radius,
                safe_margin=0.5,
                max_speed=max_speed,
                max_accel=max_accel,
            )

            planner_cfg = core.PlannerConfig(
                horizon=self.horizon_steps,
                num_samples=self.num_samples,
                num_steps=self.num_diffuse_steps,
                max_iterations=self.max_iterations,
                direct_path_init=True,
            )
            rollout_cfg = core.RolloutConfig(dt=self.dt)
            rollout = core.Rollout(env, rollout_cfg)

            planner_cls = PLANNERS[method]
            planner = planner_cls(rollout, planner_cfg)

            rng = jax.random.PRNGKey(seed)
            init_state = env.reset(rng)
            res = planner.plan(rng, init_state, env.xg)

            jax.block_until_ready(res.states)
            compute_time = time.perf_counter() - t_start

            raw_states = np.array(res.states)
            trajectories = raw_states.reshape(self.horizon_steps, num_agents, 6)

        metrics = self._compute_metrics(trajectories, safe_radius)

        return StrategicPlanResult(
            method=method,
            sensor_profile_id=self.sensor_profile.id,
            num_agents=num_agents,
            horizon_steps=self.horizon_steps,
            dt=self.dt,
            total_time_s=self.horizon_steps * self.dt,
            compute_time_s=compute_time,
            trajectories=trajectories,
            min_separation_m=metrics["min_separation_m"],
            conflict_count=metrics["conflict_count"],
            total_distance_m=metrics["total_distance_m"],
            mean_speed_ms=metrics["mean_speed_ms"],
            mean_accel_norm=metrics["mean_accel_norm"],
            agent_classes=agent_classes,
        )

    def _plan_straight_line(
        self, starts: np.ndarray, goals: np.ndarray, max_speed_ms: float
    ) -> np.ndarray:
        """Direct interpolation from start to goal at constant cruise velocity."""
        num_agents = starts.shape[0]
        trajectories = np.zeros((self.horizon_steps, num_agents, 6))

        for i in range(num_agents):
            p0 = starts[i, :3]
            pg = goals[i, :3]
            diff = pg - p0
            dist = np.linalg.norm(diff)
            speed = min(max_speed_ms, dist / max(1.0, self.horizon_steps * self.dt))
            unit_dir = diff / max(dist, 1e-6)
            vel = unit_dir * speed

            for step in range(self.horizon_steps):
                progress = min(1.0, (step * self.dt * speed) / max(dist, 1e-6))
                pos = p0 + progress * diff
                trajectories[step, i, :3] = pos
                trajectories[step, i, 3:] = vel if progress < 1.0 else 0.0

        return trajectories

    def _compute_metrics(
        self, trajectories: np.ndarray, safe_radius_m: float
    ) -> Dict[str, float]:
        """Calculates safety, separation distance, and kinetic metrics."""
        T, N, _ = trajectories.shape
        min_sep = float("inf")
        conflicts = 0

        for t in range(T):
            for i in range(N):
                for j in range(i + 1, N):
                    p_i = trajectories[t, i, :3]
                    p_j = trajectories[t, j, :3]
                    dist = float(np.linalg.norm(p_i - p_j))
                    if dist < min_sep:
                        min_sep = dist
                    if dist < safe_radius_m:
                        conflicts += 1

        tot_dist = 0.0
        speeds = []
        accels = []
        for i in range(N):
            pts = trajectories[:, i, :3]
            diffs = np.diff(pts, axis=0)
            tot_dist += float(np.sum(np.linalg.norm(diffs, axis=-1)))
            vels = trajectories[:, i, 3:]
            speeds.append(np.linalg.norm(vels, axis=-1))
            if T > 1:
                accel = np.diff(vels, axis=0) / self.dt
                accels.append(np.linalg.norm(accel, axis=-1))

        return {
            "min_separation_m": min_sep if min_sep != float("inf") else 0.0,
            "conflict_count": conflicts,
            "total_distance_m": tot_dist,
            "mean_speed_ms": float(np.mean(speeds)) if speeds else 0.0,
            "mean_accel_norm": float(np.mean(accels)) if accels else 0.0,
        }

    def export_bluesky_scenario(
        self,
        result: StrategicPlanResult,
        filepath: str,
        origin_lat: float = 25.2048,
        origin_lon: float = 55.2708,
        callsign_prefix: str = "UAV",
    ) -> None:
        """Exports the optimized 4D plan to a BlueSky simulation scenario (.scn) file."""
        import pyproj

        proj = pyproj.Proj(proj="aeqd", lat_0=origin_lat, lon_0=origin_lon, datum="WGS84")

        lines = [
            f"# BlueSky Strategic 4D Trajectory Scenario",
            f"# Generated by StrategicTrajectoryOptimizer (Method: {result.method})",
            f"# Sensor Profile: {result.sensor_profile_id} | Agents: {result.num_agents}",
            "00:00:00.00>HOLD",
            f"00:00:00.00>PAN {origin_lat:.4f} {origin_lon:.4f}",
            "00:00:00.00>ZOOM 1.0",
        ]

        step_interval = max(1, result.horizon_steps // 10)

        for i in range(result.num_agents):
            callsign = f"{callsign_prefix}{i+1:02d}"
            p0 = result.trajectories[0, i, :3]
            lon0, lat0 = proj(float(p0[0]), float(p0[1]), inverse=True)
            alt0_ft = max(50.0, float(p0[2]) * 3.28084)
            speed_kts = max(6.0, float(np.linalg.norm(result.trajectories[0, i, 3:])) * 1.94384)

            lines.append(
                f"00:00:00.00>CRE {callsign} MAVIC {lat0:.6f} {lon0:.6f} 0 {alt0_ft:.1f} {speed_kts:.1f}"
            )

            for step in range(step_interval, result.horizon_steps, step_interval):
                pos = result.trajectories[step, i, :3]
                lon_k, lat_k = proj(float(pos[0]), float(pos[1]), inverse=True)
                alt_k_ft = max(50.0, float(pos[2]) * 3.28084)
                lines.append(
                    f"00:00:00.00>ADDWPT {callsign} {lat_k:.6f} {lon_k:.6f} {alt_k_ft:.1f}"
                )

        lines.append("00:00:00.00>OP")

        with open(filepath, "w") as f:
            f.write("\n".join(lines) + "\n")
