"""Strategic 4D Multi-UAV Trajectory Deconfliction and Optimization Engine.

Implements model-based diffusion denoising (D4orm, D4orm-D) alongside sampling
baselines (MPPI, CEM) and uncoordinated benchmarks for pre-flight UTM deconfliction.
Supports exporting 4D trajectories directly to BlueSky scenario (.scn) files.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Dict, List, Literal, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from d4orm.envs.holonomic import HolonomicEnvironment
from d4orm.planners import core
from d4orm.planners.cli import PLANNERS


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


class UAVAirspace3D(HolonomicEnvironment):
    """Custom 3D Holonomic Environment with realistic UTM drone parameters."""

    def __init__(
        self,
        starts: jnp.ndarray,
        goals: jnp.ndarray,
        radius: float = 25.0,
        safe_margin: float = 5.0,
        max_speed: float = 15.0,
        max_accel: float = 4.0,
        obstacles: Optional[jnp.ndarray] = None,
    ):
        self.num_agents = starts.shape[0]
        self._custom_starts = starts
        self._custom_goals = goals
        super().__init__(self.num_agents, position_dim=3)
        self.agent_radius = radius
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
    """High-level Strategic Deconfliction Planner for Multi-UAV Operations."""

    def __init__(
        self,
        separation_radius_m: float = 25.0,
        max_speed_ms: float = 15.0,
        max_accel_ms2: float = 4.0,
        dt: float = 0.2,
        horizon_steps: int = 50,
        num_samples: int = 128,
        num_diffuse_steps: int = 20,
        max_iterations: int = 5,
    ):
        self.separation_radius_m = separation_radius_m
        self.max_speed_ms = max_speed_ms
        self.max_accel_ms2 = max_accel_ms2
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
        seed: int = 42,
    ) -> StrategicPlanResult:
        """Optimizes collision-free 4D trajectories for N UAVs using the selected method."""
        num_agents = starts.shape[0]
        t_start = time.perf_counter()

        # Uncoordinated Straight-Line baseline
        if method == "straight":
            trajectories = self._plan_straight_line(starts, goals)
            compute_time = time.perf_counter() - t_start
        else:
            starts_jnp = jnp.array(starts, dtype=jnp.float32)
            goals_jnp = jnp.array(goals, dtype=jnp.float32)

            env = UAVAirspace3D(
                starts=starts_jnp,
                goals=goals_jnp,
                radius=self.separation_radius_m,
                max_speed=self.max_speed_ms,
                max_accel=self.max_accel_ms2,
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

            # Reshape JAX state from (T, N * 6) to (T, N, 6)
            raw_states = np.array(res.states)
            trajectories = raw_states.reshape(self.horizon_steps, num_agents, 6)

        # Evaluate performance & safety metrics
        metrics = self._compute_metrics(trajectories)

        return StrategicPlanResult(
            method=method,
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
        )

    def _plan_straight_line(
        self, starts: np.ndarray, goals: np.ndarray
    ) -> np.ndarray:
        """Direct interpolation from start to goal at constant cruise velocity."""
        num_agents = starts.shape[0]
        trajectories = np.zeros((self.horizon_steps, num_agents, 6))

        for i in range(num_agents):
            p0 = starts[i, :3]
            pg = goals[i, :3]
            diff = pg - p0
            dist = np.linalg.norm(diff)
            speed = min(self.max_speed_ms, dist / max(1.0, self.horizon_steps * self.dt))
            unit_dir = diff / max(dist, 1e-6)
            vel = unit_dir * speed

            for step in range(self.horizon_steps):
                progress = min(1.0, (step * self.dt * speed) / max(dist, 1e-6))
                pos = p0 + progress * diff
                trajectories[step, i, :3] = pos
                trajectories[step, i, 3:] = vel if progress < 1.0 else 0.0

        return trajectories

    def _compute_metrics(self, trajectories: np.ndarray) -> Dict[str, float]:
        """Calculates safety, separation distance, and kinetic metrics."""
        T, N, _ = trajectories.shape
        min_sep = float("inf")
        conflicts = 0

        # Pairwise minimum distance
        for t in range(T):
            for i in range(N):
                for j in range(i + 1, N):
                    p_i = trajectories[t, i, :3]
                    p_j = trajectories[t, j, :3]
                    dist = float(np.linalg.norm(p_i - p_j))
                    if dist < min_sep:
                        min_sep = dist
                    if dist < self.separation_radius_m:
                        conflicts += 1

        # Distance, speeds, accelerations
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
            f"# Total Agents: {result.num_agents} | Horizon: {result.total_time_s:.1f}s",
            "00:00:00.00>HOLD",
            "00:00:00.00>PAN 25.2048 55.2708",
            "00:00:00.00>ZOOM 1.0",
        ]

        # Sample key waypoints (e.g. every 5 timesteps)
        step_interval = max(1, result.horizon_steps // 10)

        for i in range(result.num_agents):
            callsign = f"{callsign_prefix}{i+1:02d}"
            p0 = result.trajectories[0, i, :3]
            lon0, lat0 = proj(float(p0[0]), float(p0[1]), inverse=True)
            alt0_ft = max(100.0, float(p0[2]) * 3.28084)
            speed_kts = max(10.0, float(np.linalg.norm(result.trajectories[0, i, 3:])) * 1.94384)

            # Create aircraft in BlueSky
            lines.append(
                f"00:00:00.00>CRE {callsign} MAVIC {lat0:.6f} {lon0:.6f} 0 {alt0_ft:.1f} {speed_kts:.1f}"
            )

            # Add subsequent 4D waypoints along the deconflicted trajectory
            for step in range(step_interval, result.horizon_steps, step_interval):
                pos = result.trajectories[step, i, :3]
                lon_k, lat_k = proj(float(pos[0]), float(pos[1]), inverse=True)
                alt_k_ft = max(100.0, float(pos[2]) * 3.28084)
                lines.append(
                    f"00:00:00.00>ADDWPT {callsign} {lat_k:.6f} {lon_k:.6f} {alt_k_ft:.1f}"
                )

        lines.append("00:00:00.00>OP")

        with open(filepath, "w") as f:
            f.write("\n".join(lines) + "\n")
