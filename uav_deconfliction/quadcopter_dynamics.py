#!/usr/bin/env python3
"""
quadcopter_dynamics.py - Realistic Non-Ideal Physical Flight Dynamics Model
Simulates non-ideal multirotor movement with:
- Motor/actuator time constant (first-order lag tau)
- Acceleration limits and thrust saturation
- Dynamic attitude (roll, pitch, yaw) derived from net acceleration
- Stochastic wind gusts and steady atmospheric drift
- Velocity tracking error e_v = v_cmd - v_act
- Closed-loop feedback for tactical deconfliction
"""

import math
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class DynamicUAVState:
    id: str
    # 3D Position in Local Cartesian NED (meters): x=North, y=East, z=Down (alt = -z)
    x: float
    y: float
    z: float = -100.0  # Default 100m altitude (z = -100 in NED)
    
    # 3D Actual Physical Velocity (m/s)
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0

    # 3D Commanded Velocity from RVO2 / Autopilot (m/s)
    vx_cmd: float = 0.0
    vy_cmd: float = 0.0
    vz_cmd: float = 0.0

    # 3D Preferred Nominal Velocity (m/s)
    vx_pref: float = 0.0
    vy_pref: float = 0.0
    vz_pref: float = 0.0

    # 3D Physical Acceleration (m/s^2)
    ax: float = 0.0
    ay: float = 0.0
    az: float = 0.0

    # Attitude angles (degrees)
    roll_deg: float = 0.0
    pitch_deg: float = 0.0
    yaw_deg: float = 0.0

    # Tracking error: |v_cmd - v_act|
    tracking_error_ms: float = 0.0

    # Rotor animation angle (radians)
    rotor_phase: float = 0.0

    # Status
    in_conflict: bool = False
    min_dist_to_traffic: float = 999.0
    color: str = "#00e5ff"

    # Active Waypoint Information
    target_x: float = 0.0
    target_y: float = 0.0
    target_z: float = -100.0
    wp_index: int = 0


class QuadcopterPhysicsSimulator:
    """
    6-DOF second-order physical multirotor flight dynamics simulator.
    Converts commanded velocity targets (from RVO2 / offboard autopilot)
    into realistic physical movements subject to inertia, tilt limits,
    motor lag, and atmospheric disturbances.
    """

    def __init__(
        self,
        tau: float = 0.35,              # Motor/autopilot response lag time constant (seconds)
        max_accel: float = 5.0,         # Maximum lateral acceleration limit (m/s^2)
        max_climb_rate: float = 3.0,    # Max vertical climb rate (m/s)
        max_tilt_deg: float = 35.0,     # Max physical bank/pitch angle (degrees)
        drag_coeff: float = 0.08,       # Aerodynamic drag coefficient (1/s)
        enable_wind: bool = True,       # Enable wind & gust simulation
        steady_wind: Tuple[float, float, float] = (1.2, -0.8, 0.0), # (North, East, Down) m/s
        gust_intensity: float = 0.8,    # Standard deviation of turbulent gusts (m/s)
    ):
        self.tau = max(0.05, float(tau))
        self.max_accel = float(max_accel)
        self.max_climb_rate = float(max_climb_rate)
        self.max_tilt_deg = float(max_tilt_deg)
        self.drag_coeff = float(drag_coeff)
        self.enable_wind = enable_wind
        self.steady_wind = list(steady_wind)
        self.gust_intensity = float(gust_intensity)

        self.gravity = 9.80665

        # Gust random walk state
        self._gust_x = 0.0
        self._gust_y = 0.0
        self._gust_z = 0.0

    def update_gust(self, dt: float) -> Tuple[float, float, float]:
        """Ornstein-Uhlenbeck stochastic wind gust process."""
        if not self.enable_wind:
            return 0.0, 0.0, 0.0

        alpha = math.exp(-dt / 2.0)  # 2-second correlation time
        noise_scale = self.gust_intensity * math.sqrt(1.0 - alpha * alpha)
        
        self._gust_x = alpha * self._gust_x + random.gauss(0, noise_scale)
        self._gust_y = alpha * self._gust_y + random.gauss(0, noise_scale)
        self._gust_z = alpha * self._gust_z + random.gauss(0, noise_scale * 0.3)

        total_wind_x = self.steady_wind[0] + self._gust_x
        total_wind_y = self.steady_wind[1] + self._gust_y
        total_wind_z = self.steady_wind[2] + self._gust_z
        return total_wind_x, total_wind_y, total_wind_z

    def step_uav(
        self,
        uav: DynamicUAVState,
        v_cmd: Tuple[float, float, float],
        dt: float,
        wind: Tuple[float, float, float],
    ) -> None:
        """
        Integrates non-ideal physical multirotor equations of motion over timestep dt.
        """
        uav.vx_cmd, uav.vy_cmd, uav.vz_cmd = v_cmd

        # Relative airspeed (ground velocity minus wind)
        airspeed_x = uav.vx - wind[0]
        airspeed_y = uav.vy - wind[1]
        airspeed_z = uav.vz - wind[2]

        # Desired acceleration demanded by autopilot to track commanded velocity
        # First-order motor/controller response: a_des = (v_cmd - v_act) / tau
        des_ax = (uav.vx_cmd - uav.vx) / self.tau
        des_ay = (uav.vy_cmd - uav.vy) / self.tau
        des_az = (uav.vz_cmd - uav.vz) / (self.tau * 0.8)

        # Aerodynamic drag opposing velocity
        drag_ax = -self.drag_coeff * airspeed_x
        drag_ay = -self.drag_coeff * airspeed_y
        drag_az = -self.drag_coeff * 0.5 * airspeed_z

        # Apply motor thrust saturation / acceleration limits
        horizontal_accel = math.hypot(des_ax, des_ay)
        if horizontal_accel > self.max_accel:
            scale = self.max_accel / horizontal_accel
            des_ax *= scale
            des_ay *= scale

        # Vertical climb/descent acceleration limit
        des_az = max(-self.max_accel, min(self.max_accel, des_az))

        # Net physical acceleration = thrust acceleration + aero drag + wind drift
        uav.ax = des_ax + drag_ax
        uav.ay = des_ay + drag_ay
        uav.az = des_az + drag_az

        # Velocity integration (Euler-Cromer)
        uav.vx += uav.ax * dt
        uav.vy += uav.ay * dt
        uav.vz += uav.az * dt

        # Bound vertical climb rate
        uav.vz = max(-self.max_climb_rate, min(self.max_climb_rate, uav.vz))

        # Position integration
        uav.x += uav.vx * dt
        uav.y += uav.vy * dt
        uav.z += uav.vz * dt

        # Calculate true physical attitude angles from acceleration:
        # A quadcopter tilts forward (negative pitch) to accelerate North (+x)
        # and banks right (positive roll) to accelerate East (+y)
        pitch_rad = -math.atan2(uav.ax, self.gravity)
        roll_rad = math.atan2(uav.ay, self.gravity)

        max_tilt_rad = math.radians(self.max_tilt_deg)
        pitch_rad = max(-max_tilt_rad, min(max_tilt_rad, pitch_rad))
        roll_rad = max(-max_tilt_rad, min(max_tilt_rad, roll_rad))

        # Smooth attitude transition (filter out micro-jitter)
        alpha_att = min(1.0, dt * 15.0)
        uav.pitch_deg += alpha_att * (math.degrees(pitch_rad) - uav.pitch_deg)
        uav.roll_deg += alpha_att * (math.degrees(roll_rad) - uav.roll_deg)

        # Yaw heading aligns with velocity vector if moving > 0.3 m/s
        speed_horiz = math.hypot(uav.vx, uav.vy)
        if speed_horiz > 0.3:
            target_yaw_deg = math.degrees(math.atan2(uav.vy, uav.vx))
            # Wrap angle difference
            diff = (target_yaw_deg - uav.yaw_deg + 180.0) % 360.0 - 180.0
            uav.yaw_deg += alpha_att * 0.5 * diff

        # Rotor rotation phase (RPM increases with thrust demand)
        thrust_ratio = 1.0 + (horizontal_accel / self.max_accel) * 0.8
        uav.rotor_phase = (uav.rotor_phase + dt * 60.0 * thrust_ratio) % (2.0 * math.pi)

        # Dynamic tracking error: deviation between commanded setpoint and physical realization
        uav.tracking_error_ms = math.hypot(uav.vx_cmd - uav.vx, uav.vy_cmd - uav.vy)
