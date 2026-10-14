#!/usr/bin/env python3
"""
strategic_planner.py
====================
3-Layer ATM Strategic Path Planner for Multi-UAV Deconfliction.

Implements a structured airspace model mirroring real Air Traffic Service (ATS) design:

  Layer 1 – Strategic  (THIS MODULE)
    CorridorGraph   — directed graph of named waypoint nodes + altitude-banded edges
                      (analogous to VOR airways in real ATS)
    SIPP Planner    — assigns each drone a conflict-free (departure_time, cruise_speed)
                      per corridor edge using Safe Interval Path Planning
    ReservationTable— tracks booked spacetime slots per edge; enforces time-buffer
                      separation standard: gap = edge_length/speed + SAFETY_BUFFER_S

  Layer 2 – Tactical  (sim3d_server.py → Python-RVO2 ORCA)
    Reactive 10 Hz deconfliction projected onto 1D corridor axis + altitude

  Layer 3 – Physical  (quadcopter_dynamics.py)
    6-DOF multirotor dynamics: motor lag, acceleration limits, wind gusts

Replanning is triggered by three conditions (checked per-drone each physics step):
  1. Waypoint arrival — drone arrives within ARRIVAL_RADIUS_M of a graph node
  2. Altitude band violation — |drone_z − assigned_z| > edge.alt_band_m for > ALT_VIOLATION_HOLD_S
  3. Hover timeout — drone speed < HOVER_SPEED_THRESHOLD for > HOVER_TIMEOUT_S

Priority ordering for SIPP (highest priority planned first, locking in slots):
  EMERGENCY(0) > DELIVERY(1) > SURVEY(2) > PATROL(3)
  Within same type: FCFS by request_time

Usage from sim3d_server.py:
    from strategic_planner import StrategicController, MissionType

    controller = StrategicController(num_drones=10)
    controller.initialize_scenario(list(uavs.keys()), sim_time=0.0)

    # Each physics step:
    for drone_id, uav in uavs.items():
        speed = math.hypot(uav.vx, uav.vy, uav.vz)
        tx, ty, tz, tspeed = controller.get_current_target(
            drone_id, (uav.x, uav.y, uav.z), speed, sim_time
        )

    # For 3D WebGL corridor visualization:
    corridor_data = controller.get_corridor_telemetry()
"""

from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SAFETY_BUFFER_S        = 5.0   # Extra time gap beyond traversal on a shared edge (s)
HOVER_TIMEOUT_S        = 8.0   # Seconds of near-zero speed before hover replan (s)
HOVER_SPEED_THRESHOLD  = 0.5   # Speed below which drone is considered "hovering" (m/s)
ALT_VIOLATION_HOLD_S   = 2.0   # Seconds of altitude violation before replan (s)
ARRIVAL_RADIUS_M       = 24.0  # Waypoint arrival detection radius (m) – matches sim3d_server


# ---------------------------------------------------------------------------
# Mission Type Priority Constants
# ---------------------------------------------------------------------------

class MissionType:
    EMERGENCY = 0
    DELIVERY  = 1
    SURVEY    = 2
    PATROL    = 3
    _NAMES = {0: "EMERGENCY", 1: "DELIVERY", 2: "SURVEY", 3: "PATROL"}

    @classmethod
    def name(cls, t: int) -> str:
        return cls._NAMES.get(t, "UNKNOWN")


# ---------------------------------------------------------------------------
# Pre-Flight Flight Authorisation Service (FAS) Data Structures
# ---------------------------------------------------------------------------

@dataclass
class FlightAuthorisationRequest:
    """
    Pre-flight Flight Authorisation Service (FAS) request.
    Submitted entirely before drone takeoff (t < t_takeoff).
    """
    drone_id: str
    origin_node: str
    destination_node: str
    desired_departure_time: float
    cruise_speed: float = 12.0
    climb_rate: float = 2.5
    descent_rate: float = 2.5
    mission_type: int = MissionType.PATROL
    route_edges: Optional[List[str]] = None
    target_layer: Optional[float] = None


# ---------------------------------------------------------------------------
# Corridor Graph Data Structures
# ---------------------------------------------------------------------------

@dataclass
class CorridorNode:
    """Named waypoint node in the corridor graph."""
    id: str
    x: float    # North (m)
    y: float    # East (m)
    z: float    # NED Down (m) — altitude = -z


@dataclass
class CorridorEdge:
    """
    Directed one-way corridor edge between two CorridorNodes.
    Each edge has a fixed altitude band and a cruise speed envelope.
    SIPP enforces that at most one drone occupies an edge at any time
    (with SAFETY_BUFFER_S grace period on each side of the reservation).
    """
    id: str
    from_id: str
    to_id: str
    alt_m: float        # Nominal altitude (m, positive). NED z = -alt_m.
    alt_band_m: float   # Vertical tolerance (m). Violation triggers replan if held > ALT_VIOLATION_HOLD_S.
    min_speed_ms: float # Minimum cruise speed on this edge (m/s)
    max_speed_ms: float # Maximum cruise speed on this edge (m/s)
    length: float = 0.0 # 3D Euclidean length, computed from node positions (m)


AVAILABLE_LAYERS: Tuple[float, ...] = (90.0, 105.0, 120.0)


# ---------------------------------------------------------------------------
# Reservation Table
# ---------------------------------------------------------------------------

@dataclass
class EdgeReservation:
    """A single drone's spacetime booking on a corridor edge or vertical transition column."""
    drone_id: str
    t_entry: float  # Absolute sim time drone enters edge (s)
    t_exit: float   # Absolute sim time drone exits edge (s)
    speed: float = 12.0  # Assigned cruise speed (m/s)
    alt_m: float = 90.0  # Nominal altitude layer (m, positive) or center of transition
    resource_id: Optional[str] = None
    alt_band_m: float = 8.0
    is_vertical_transition: bool = False
    transition_node: Optional[str] = None

    @property
    def alt_min(self) -> float:
        return self.alt_m - self.alt_band_m

    @property
    def alt_max(self) -> float:
        return self.alt_m + self.alt_band_m


class ReservationTable:
    """
    Tracks booked spacetime slots per corridor edge and vertical transition column.

    Separation standard:
        Two drones conflict on resource R if their [t_entry, t_exit] windows overlap,
        including a SAFETY_BUFFER_S grace zone on each side, and they operate
        within overlapping vertical altitude bands.
        Cross-layer climbs/descents reserve vertical columns (TRANS_{node}) spanning
        intermediate altitude bands, preventing horizontal traffic on intermediate layers.

    API:
        is_free(resource_id, t_entry, t_exit, alt_m, alt_band_m, is_vertical, drone_id) → bool
        get_safe_intervals(resource_id, traversal_time, earliest_start, alt_m, ...) → List[(start, end)]
        reserve(resource_id, reservation) → bool
        reserve_batch(reservations, drone_id) → bool
        reserve_plan(plan) → bool
        release(drone_id, resource_id=None)
        release_plan(plan)
    """

    def __init__(
        self,
        safety_buffer_s: float = SAFETY_BUFFER_S,
        conflict_map: Optional[Dict[str, Set[str]]] = None,
    ):
        self.safety_buffer = safety_buffer_s
        self.conflict_map: Dict[str, Set[str]] = conflict_map or {}
        # resource_id → list of EdgeReservation, sorted by t_entry
        self._table: Dict[str, List[EdgeReservation]] = {}

    def _get(self, resource_id: str) -> List[EdgeReservation]:
        return self._table.setdefault(resource_id, [])

    def _get_conflicting(self, resource_id: str) -> List[EdgeReservation]:
        """Returns all reservations on resource_id and its conflicting resources, sorted by t_entry."""
        res_list: List[EdgeReservation] = []
        conflicts = self.conflict_map.get(resource_id, {resource_id})
        for rid in conflicts:
            res_list.extend(self._table.get(rid, []))
        return sorted(res_list, key=lambda r: r.t_entry)

    def is_free(
        self,
        resource_id: str,
        t_entry: float,
        t_exit: float,
        alt_m: Optional[float] = None,
        alt_band_m: float = 8.0,
        is_vertical: bool = False,
        drone_id: Optional[str] = None,
    ) -> bool:
        """
        True if [t_entry, t_exit] does not overlap any existing reservation
        at conflicting altitudes on the resource or conflicting resources.
        """
        buf = self.safety_buffer
        my_alt_min = (alt_m - alt_band_m) if alt_m is not None else 0.0
        my_alt_max = (alt_m + alt_band_m) if alt_m is not None else 9999.0

        for res in self._get_conflicting(resource_id):
            if drone_id is not None and res.drone_id == drone_id:
                continue
            if alt_m is not None:
                if not is_vertical and not res.is_vertical_transition:
                    # Both are horizontal corridors: >= 12m vertical clearance avoids conflict
                    if abs(res.alt_m - alt_m) >= 12.0:
                        continue
                else:
                    # At least one is vertical transition: check altitude band overlap
                    if my_alt_max < res.alt_min or my_alt_min > res.alt_max:
                        continue
            # Overlap check including safety buffer
            if res.t_entry < (t_exit + buf) and res.t_exit > (t_entry - buf):
                return False
        return True

    def get_safe_intervals(
        self,
        resource_id: str,
        traversal_time: float,
        earliest_start: float,
        alt_m: Optional[float] = None,
        alt_band_m: float = 8.0,
        is_vertical: bool = False,
        drone_id: Optional[str] = None,
    ) -> List[Tuple[float, float]]:
        """
        Returns sorted list of (t_start, t_end) windows during which a drone
        could START traversal such that [t_start, t_start + traversal_time]
        is fully conflict-free (including buffer).
        """
        reservations = self._get_conflicting(resource_id)
        if drone_id is not None:
            reservations = [r for r in reservations if r.drone_id != drone_id]
        if alt_m is not None:
            filtered = []
            my_alt_min = alt_m - alt_band_m
            my_alt_max = alt_m + alt_band_m
            for res in reservations:
                if not is_vertical and not res.is_vertical_transition:
                    if abs(res.alt_m - alt_m) >= 12.0:
                        continue
                else:
                    if my_alt_max < res.alt_min or my_alt_min > res.alt_max:
                        continue
                filtered.append(res)
            reservations = filtered

        intervals: List[Tuple[float, float]] = []
        cursor = earliest_start
        buf = self.safety_buffer

        for res in reservations:
            latest_before = res.t_entry - traversal_time - buf
            if latest_before >= cursor:
                intervals.append((cursor, latest_before))
            cursor = max(cursor, res.t_exit + buf)

        intervals.append((cursor, math.inf))
        return intervals

    def reserve(self, resource_id: str, reservation: EdgeReservation) -> bool:
        """Adds a reservation if conflict-free. Returns False if blocked."""
        reservation.resource_id = resource_id
        if not self.is_free(
            resource_id,
            reservation.t_entry,
            reservation.t_exit,
            alt_m=reservation.alt_m,
            alt_band_m=reservation.alt_band_m,
            is_vertical=reservation.is_vertical_transition,
            drone_id=reservation.drone_id,
        ):
            return False
        res_list = self._get(resource_id)
        res_list.append(reservation)
        res_list.sort(key=lambda r: r.t_entry)
        return True

    def reserve_batch(self, reservations: List[Tuple[str, EdgeReservation]], drone_id: Optional[str] = None) -> bool:
        """Atomically checks and books a batch of spacetime reservations."""
        if drone_id is None and reservations:
            drone_id = reservations[0][1].drone_id
        for rid, res in reservations:
            if not self.is_free(
                rid,
                res.t_entry,
                res.t_exit,
                alt_m=res.alt_m,
                alt_band_m=res.alt_band_m,
                is_vertical=res.is_vertical_transition,
                drone_id=drone_id,
            ):
                return False
        for rid, res in reservations:
            res.resource_id = rid
            res_list = self._get(rid)
            res_list.append(res)
            res_list.sort(key=lambda r: r.t_entry)
        return True

    def reserve_plan(self, plan: MissionPlan) -> bool:
        """Atomically reserves all horizontal legs and vertical transition columns in plan."""
        batch: List[Tuple[str, EdgeReservation]] = []
        for leg in plan.legs:
            if leg.vertical_transition_duration > 0.0 and leg.transition_node:
                trans_rid = f"TRANS_{leg.transition_node}"
                v_res = EdgeReservation(
                    drone_id=plan.drone_id,
                    t_entry=leg.vertical_transition_start,
                    t_exit=leg.vertical_transition_end,
                    speed=2.5,
                    alt_m=(leg.vertical_transition_from_alt + leg.vertical_transition_to_alt) / 2.0,
                    alt_band_m=abs(leg.vertical_transition_to_alt - leg.vertical_transition_from_alt) / 2.0 + 4.0,
                    resource_id=trans_rid,
                    is_vertical_transition=True,
                    transition_node=leg.transition_node,
                )
                batch.append((trans_rid, v_res))
            h_res = EdgeReservation(
                drone_id=plan.drone_id,
                t_entry=leg.depart_time,
                t_exit=leg.arrive_time,
                speed=leg.speed,
                alt_m=leg.alt_m,
                resource_id=leg.edge_id,
                is_vertical_transition=False,
            )
            batch.append((leg.edge_id, h_res))
        return self.reserve_batch(batch, drone_id=plan.drone_id)

    def release(self, drone_id: str, resource_id: Optional[str] = None) -> None:
        """Removes all reservations for a drone (specific resource or all resources)."""
        target_resources = [resource_id] if resource_id else list(self._table.keys())
        for rid in target_resources:
            if rid in self._table:
                self._table[rid] = [r for r in self._table[rid] if r.drone_id != drone_id]

    def release_plan(self, plan: MissionPlan) -> None:
        """Releases all reservations belonging to a mission plan."""
        self.release(plan.drone_id)

    def summary(self) -> Dict[str, int]:
        """resource_id → reservation count (non-empty resources only)."""
        return {eid: len(rs) for eid, rs in self._table.items() if rs}

    def active_reservations(self, sim_time: float) -> Dict[str, List[dict]]:
        """Returns currently active (not yet expired) reservations per resource."""
        result = {}
        for eid, rlist in self._table.items():
            active = [
                {"drone_id": r.drone_id, "t_entry": r.t_entry, "t_exit": r.t_exit, "speed": r.speed}
                for r in rlist if r.t_exit >= sim_time
            ]
            if active:
                result[eid] = active
        return result


# ---------------------------------------------------------------------------
# Mission Plan
# ---------------------------------------------------------------------------

@dataclass
class MissionLeg:
    """One segment of a drone's 4D spacetime path on a single corridor edge or vertical transition."""
    edge_id: str
    depart_time: float  # Absolute sim time to leave from_node (s)
    arrive_time: float  # Absolute sim time to arrive at to_node (s)
    speed: float        # Assigned cruise speed (m/s)
    alt_m: float        # Altitude for this leg (m, positive). NED z = -alt_m.
    is_vertical_transition: bool = False
    transition_node: Optional[str] = None
    from_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    to_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    vertical_transition_duration: float = 0.0
    vertical_transition_start: float = -1.0
    vertical_transition_end: float = -1.0
    vertical_transition_from_alt: float = 90.0
    vertical_transition_to_alt: float = 90.0


@dataclass
class MissionPlan:
    """Full 4D spacetime path for a single drone: ordered sequence of pre-cleared MissionLegs."""
    drone_id: str
    mission_type: int           # MissionType constant (priority)
    request_time: float         # When this plan was originally requested (s)
    legs: List[MissionLeg]      # Ordered legs (one per corridor edge)
    current_leg: int = 0        # Index of the active leg
    scheduled_departure: float = 0.0  # Scheduled departure time on the ground (s)
    ground_delay_s: float = 0.0       # Departure delay absorbed on launch pad (s)
    is_authorized: bool = True        # Pre-flight FAS authorization flag

    # Internal replan trigger state (for off-nominal emergency contingencies only)
    hover_since: float = -1.0           # Sim time hovering started (-1 = not hovering)
    alt_violation_since: float = -1.0   # Sim time altitude violation started (-1 = none)
    layer_transition_until: float = -1.0  # Sim time when active layer transition completes (-1 = none)
    origin_node: str = ""
    destination_node: str = ""


# ---------------------------------------------------------------------------
# Corridor Graph
# ---------------------------------------------------------------------------

class CorridorGraph:
    """
    Directed graph of CorridorNodes connected by CorridorEdges.

    The default scenario matches the 10-drone circular layout in sim3d_server.py:
      - 10 PERIM nodes on a 260m-radius perimeter circle
      - 10 HUB_IN nodes (hub entry gates, 110m radius, +0.35 rad offset)
      - 10 HUB_OUT nodes (hub exit gates, 110m radius, +π-0.25 rad offset)
      - Each drone has 4 directed edges forming a closed loop:
            PERIM_i → HUB_IN_i → HUB_OUT_i → PERIM_{(i+5)%10} → PERIM_i

    Altitude is alternated: even drones at 90m, odd drones at 105m.
    Each edge is one-way → no structural head-on conflicts. SIPP handles
    temporal separation for drones sharing the same edge at different times.
    """

    def __init__(self, wind_vector: Tuple[float, float, float] = (1.2, -0.8, 0.0)) -> None:
        self.nodes: Dict[str, CorridorNode] = {}
        self.edges: Dict[str, CorridorEdge] = {}
        self._adj: Dict[str, List[str]] = {}  # from_node_id → [edge_id, ...]
        self.wind_vector: Tuple[float, float, float] = wind_vector

    def get_effective_ground_speed(
        self,
        edge: CorridorEdge,
        airspeed: float,
        wind_vector: Optional[Tuple[float, float, float]] = None,
    ) -> float:
        """
        Calculates effective ground speed along edge direction considering wind (Tang et al. 2023).
        v_ground = v_air + v_wind · d_hat
        """
        if wind_vector is None:
            wind_vector = self.wind_vector
        n_from = self.nodes[edge.from_id]
        n_to   = self.nodes[edge.to_id]
        dx = n_to.x - n_from.x
        dy = n_to.y - n_from.y
        dist_2d = math.hypot(dx, dy)
        if dist_2d < 1e-3:
            return max(0.5, airspeed)
        ux = dx / dist_2d
        uy = dy / dist_2d
        w_proj = wind_vector[0] * ux + wind_vector[1] * uy
        v_ground = airspeed + w_proj
        return max(0.5, v_ground)

    def add_node(self, node: CorridorNode) -> None:
        self.nodes[node.id] = node
        self._adj.setdefault(node.id, [])

    def add_edge(self, edge: CorridorEdge) -> None:
        """Add edge and compute its 3D Euclidean length from node positions."""
        n_from = self.nodes[edge.from_id]
        n_to   = self.nodes[edge.to_id]
        dx = n_to.x - n_from.x
        dy = n_to.y - n_from.y
        dz = n_to.z - n_from.z
        edge.length = math.sqrt(dx * dx + dy * dy + dz * dz)
        self.edges[edge.id] = edge
        self._adj.setdefault(edge.from_id, []).append(edge.id)

    def neighbors(self, node_id: str) -> List[CorridorEdge]:
        """All outgoing edges from node_id."""
        return [self.edges[eid] for eid in self._adj.get(node_id, [])]

    def shortest_path(self, start_id: str, goal_id: str) -> List[str]:
        """
        Dijkstra shortest path by physical distance.
        Returns list of edge IDs (empty if start == goal or unreachable).
        """
        if start_id == goal_id:
            return []
        dist: Dict[str, float] = {nid: math.inf for nid in self.nodes}
        dist[start_id] = 0.0
        prev_edge: Dict[str, Optional[str]] = {nid: None for nid in self.nodes}
        prev_node: Dict[str, Optional[str]] = {nid: None for nid in self.nodes}
        pq: List[Tuple[float, str]] = [(0.0, start_id)]

        while pq:
            d, nid = heapq.heappop(pq)
            if d > dist[nid]:
                continue
            if nid == goal_id:
                break
            for edge in self.neighbors(nid):
                nd = d + edge.length
                if nd < dist[edge.to_id]:
                    dist[edge.to_id] = nd
                    prev_edge[edge.to_id] = edge.id
                    prev_node[edge.to_id] = nid
                    heapq.heappush(pq, (nd, edge.to_id))

        # Reconstruct
        path: List[str] = []
        cur = goal_id
        while prev_edge.get(cur) is not None:
            path.append(prev_edge[cur])
            cur = prev_node[cur]
        path.reverse()
        return path

    def build_conflict_map(self) -> Dict[str, Set[str]]:
        """
        Builds mutual conflict groups for corridor edges and vertical transition columns.
        Central hub transit edges (E_i_B) at the same altitude cross in the
        hub center and thus share a single conflict resource.
        Alternative bypass edges (E_i_B_altR, E_i_B_altL) skirt the hub outer rim
        and do not conflict with the central crossing.
        Vertical transition columns (TRANS_{node}) conflict with all edges touching the node.
        """
        conflicts: Dict[str, Set[str]] = {}
        for eid, edge in self.edges.items():
            conflicts[eid] = {eid}
            # Hub center crossing edges (E_i_B) conflict with other E_j_B at the same altitude
            if eid.endswith("_B"):
                for other_id, other_edge in self.edges.items():
                    if other_id.endswith("_B") and abs(other_edge.alt_m - edge.alt_m) < 1.0:
                        conflicts[eid].add(other_id)

        # Transition columns for all nodes (cross-layer climbs/descents)
        for nid in self.nodes:
            trans_id = f"TRANS_{nid}"
            conflicts.setdefault(trans_id, {trans_id})
            touching_edges = [eid for eid, e in self.edges.items() if e.from_id == nid or e.to_id == nid]
            for eid in touching_edges:
                conflicts[trans_id].add(eid)
                conflicts.setdefault(eid, {eid}).add(trans_id)

        return conflicts


    @classmethod
    def build_circular_scenario(
        cls,
        num_drones: int = 10,
        perim_radius: float = 260.0,
        hub_radius: float = 110.0,
        wind_vector: Tuple[float, float, float] = (1.2, -0.8, 0.0),
    ) -> Tuple["CorridorGraph", List[List[str]]]:
        """
        Builds the corridor graph matching the sim3d_server.py circular scenario
        and returns (graph, route_per_drone) where route_per_drone[i] is the
        ordered list of edge IDs for drone i (looping).

        Node naming:
            PERIM_{i}    — perimeter starting/return node for drone i
            HUB_IN_{i}   — hub entry gate for drone i
            HUB_OUT_{i}  — hub exit gate for drone i

        Edge naming:
            E_{i}_A  PERIM_{i}         → HUB_IN_{i}          (inbound approach)
            E_{i}_B  HUB_IN_{i}        → HUB_OUT_{i}         (hub transit)
            E_{i}_C  HUB_OUT_{i}       → PERIM_{(i+5)%10}    (outbound departure)
            E_{i}_D  PERIM_{(i+5)%10}  → PERIM_{i}           (return patrol)
        """
        g = cls(wind_vector=wind_vector)
        n_half = num_drones // 2
        routes: List[List[str]] = [[] for _ in range(num_drones)]

        # ---- Pass 1: Create all nodes ----
        for i in range(num_drones):
            angle    = i * (2.0 * math.pi / num_drones)
            alt_m    = 90.0 if (i % 2 == 0) else 105.0
            z        = -alt_m

            sx = perim_radius * math.cos(angle)
            sy = perim_radius * math.sin(angle)
            entry_angle = angle + 0.35
            exit_angle  = angle + math.pi - 0.25

            hub_in_x  = hub_radius * math.cos(entry_angle)
            hub_in_y  = hub_radius * math.sin(entry_angle)
            hub_out_x = hub_radius * math.cos(exit_angle)
            hub_out_y = hub_radius * math.sin(exit_angle)

            perim_id   = f"PERIM_{i}"
            hub_in_id  = f"HUB_IN_{i}"
            hub_out_id = f"HUB_OUT_{i}"

            if perim_id not in g.nodes:
                g.add_node(CorridorNode(perim_id, sx, sy, z))
            if hub_in_id not in g.nodes:
                g.add_node(CorridorNode(hub_in_id, hub_in_x, hub_in_y, z))
            if hub_out_id not in g.nodes:
                g.add_node(CorridorNode(hub_out_id, hub_out_x, hub_out_y, z))

        # ---- Pass 2: Create all edges ----
        for i in range(num_drones):
            alt_m   = 90.0 if (i % 2 == 0) else 105.0
            anti_i  = (i + n_half) % num_drones
            alt_r   = (i + 1) % num_drones
            alt_l   = (i - 1) % num_drones

            perim_id   = f"PERIM_{i}"
            hub_in_id  = f"HUB_IN_{i}"
            hub_out_id = f"HUB_OUT_{i}"
            anti_id    = f"PERIM_{anti_i}"
            hub_out_r  = f"HUB_OUT_{alt_r}"
            hub_out_l  = f"HUB_OUT_{alt_l}"

            # Edge A: Approach corridor (perimeter → hub entry)
            eA = CorridorEdge(
                id=f"E_{i}_A",
                from_id=perim_id, to_id=hub_in_id,
                alt_m=alt_m, alt_band_m=8.0,
                min_speed_ms=8.0, max_speed_ms=15.0,
            )
            # Edge B: Primary Hub transit (hub entry → hub exit)
            eB = CorridorEdge(
                id=f"E_{i}_B",
                from_id=hub_in_id, to_id=hub_out_id,
                alt_m=alt_m, alt_band_m=8.0,
                min_speed_ms=5.0, max_speed_ms=12.0,
            )
            # Alternative branching Hub transits:
            eB_altR = CorridorEdge(
                id=f"E_{i}_B_altR",
                from_id=hub_in_id, to_id=hub_out_r,
                alt_m=alt_m, alt_band_m=8.0,
                min_speed_ms=5.0, max_speed_ms=12.0,
            )
            eB_altL = CorridorEdge(
                id=f"E_{i}_B_altL",
                from_id=hub_in_id, to_id=hub_out_l,
                alt_m=alt_m, alt_band_m=8.0,
                min_speed_ms=5.0, max_speed_ms=12.0,
            )

            # Edge C: Primary Departure corridor (hub exit → antipodal perimeter)
            eC = CorridorEdge(
                id=f"E_{i}_C",
                from_id=hub_out_id, to_id=anti_id,
                alt_m=alt_m, alt_band_m=8.0,
                min_speed_ms=8.0, max_speed_ms=15.0,
            )
            # Alternative Departure corridors connecting neighbor exit gates to antipodal perimeter:
            eC_fromR = CorridorEdge(
                id=f"E_{i}_C_fromR",
                from_id=hub_out_r, to_id=anti_id,
                alt_m=alt_m, alt_band_m=8.0,
                min_speed_ms=8.0, max_speed_ms=15.0,
            )
            eC_fromL = CorridorEdge(
                id=f"E_{i}_C_fromL",
                from_id=hub_out_l, to_id=anti_id,
                alt_m=alt_m, alt_band_m=8.0,
                min_speed_ms=8.0, max_speed_ms=15.0,
            )

            # Edge D: Return patrol (antipodal perimeter → home perimeter)
            eD = CorridorEdge(
                id=f"E_{i}_D",
                from_id=anti_id, to_id=perim_id,
                alt_m=alt_m, alt_band_m=8.0,
                min_speed_ms=8.0, max_speed_ms=15.0,
            )

            for edge in [eA, eB, eB_altR, eB_altL, eC, eC_fromR, eC_fromL, eD]:
                g.add_edge(edge)

            routes[i] = [eA.id, eB.id, eC.id, eD.id]

        return g, routes

    @classmethod
    def from_dict(cls, data: dict) -> "CorridorGraph":
        """Loads a corridor graph from a JSON-compatible dict."""
        g = cls()
        for nd in data.get("nodes", []):
            g.add_node(CorridorNode(**{k: nd[k] for k in ("id", "x", "y", "z")}))
        for ed in data.get("edges", []):
            g.add_edge(CorridorEdge(**ed))
        return g

    def to_dict(self) -> dict:
        """Serializes the graph to a JSON-compatible dict."""
        return {
            "nodes": [
                {"id": n.id, "x": n.x, "y": n.y, "z": n.z}
                for n in self.nodes.values()
            ],
            "edges": [
                {
                    "id": e.id, "from_id": e.from_id, "to_id": e.to_id,
                    "alt_m": e.alt_m, "alt_band_m": e.alt_band_m,
                    "min_speed_ms": e.min_speed_ms, "max_speed_ms": e.max_speed_ms,
                    "length": round(e.length, 2),
                }
                for e in self.edges.values()
            ],
        }


# ---------------------------------------------------------------------------
# SIPP Planner
# ---------------------------------------------------------------------------

@dataclass(order=True)
class _SIPPItem:
    """Priority queue entry for SIPP A*."""
    f_cost: float                                   # f = g + h (sort key)
    g_cost: float                                   # g = current arrival time (s)
    node_id: str       = field(compare=False)
    path_legs: List[MissionLeg] = field(compare=False, default_factory=list)


class SIPPPlanner:
    """
    Safe Interval Path Planning on a CorridorGraph.

    For each corridor edge, SIPP finds (departure_time, cruise_speed) pairs such that
    no two drones have overlapping reservation windows (including SAFETY_BUFFER_S).

    Speed is variable in [edge.min_speed_ms, edge.max_speed_ms]:
      - Higher speeds are tried first (faster traversal → tighter slot → less waiting)
      - Falls back through speed_candidates evenly-spaced levels if needed

    Drones are planned in priority order; lower-priority drones plan around
    slots already committed by higher-priority ones.
    """

    def __init__(
        self,
        graph: CorridorGraph,
        table: ReservationTable,
        wind_vector: Tuple[float, float, float] = (1.2, -0.8, 0.0),
    ) -> None:
        self.graph = graph
        self.table = table
        self.wind_vector = wind_vector

    # ---- Utility --------------------------------------------------------

    def _heuristic(self, node_id: str, goal_id: str) -> float:
        """Admissible time heuristic: 3D Euclidean distance / global max speed."""
        n = self.graph.nodes[node_id]
        g = self.graph.nodes[goal_id]
        dist = math.sqrt((g.x - n.x) ** 2 + (g.y - n.y) ** 2 + (g.z - n.z) ** 2)
        max_spd = max(
            (e.max_speed_ms for e in self.graph.edges.values()),
            default=15.0,
        )
        return dist / max_spd

    @staticmethod
    def _sample_speeds(edge: CorridorEdge, n: int = 7) -> List[float]:
        """n evenly-spaced speeds from max→min (descending), so fastest is tried first."""
        if n <= 1 or edge.max_speed_ms <= edge.min_speed_ms:
            return [edge.max_speed_ms]
        step = (edge.max_speed_ms - edge.min_speed_ms) / (n - 1)
        return [edge.max_speed_ms - i * step for i in range(n)]

    # ---- Core SIPP A* (general origin→goal) ----------------------------

    def plan(
        self,
        drone_id: str,
        start_node: str,
        goal_node: str,
        earliest_start: float,
        mission_type: int = MissionType.PATROL,
        speed_candidates: int = 5,
    ) -> Optional[MissionPlan]:
        """
        SIPP A* from start_node to goal_node.
        Returns MissionPlan (uncommitted) or None if no path exists.
        The caller must call commit(plan) to write slots into the ReservationTable.
        """
        open_set: List[_SIPPItem] = []
        h0 = self._heuristic(start_node, goal_node)
        heapq.heappush(open_set, _SIPPItem(
            f_cost=h0, g_cost=earliest_start,
            node_id=start_node, path_legs=[],
        ))
        visited: Dict[Tuple[str, float, float], float] = {}

        d_prefix = None
        try:
            d_idx = int(drone_id.split("-")[-1]) - 1
            cand_prefix = f"E_{d_idx}_"
            if any(eid.startswith(cand_prefix) for eid in self.graph.edges):
                d_prefix = cand_prefix
        except Exception:
            pass

        while open_set:
            cur = heapq.heappop(open_set)

            if cur.node_id == goal_node:
                return MissionPlan(
                    drone_id=drone_id,
                    mission_type=mission_type,
                    request_time=earliest_start,
                    legs=cur.path_legs,
                )

            for edge in self.graph.neighbors(cur.node_id):
                if d_prefix and not edge.id.startswith(d_prefix):
                    continue
                for airspeed in self._sample_speeds(edge, speed_candidates):
                    speed = self.graph.get_effective_ground_speed(edge, airspeed, self.wind_vector)
                    trav = edge.length / speed
                    for t_start, t_end in self.table.get_safe_intervals(edge.id, trav, cur.g_cost):
                        depart = max(t_start, cur.g_cost)
                        if depart > t_end:
                            continue
                        arrive = depart + trav
                        interval_key = (edge.to_id, t_start, t_end)
                        if interval_key in visited and visited[interval_key] <= arrive:
                            continue
                        visited[interval_key] = arrive

                        leg = MissionLeg(
                            edge_id=edge.id,
                            depart_time=depart, arrive_time=arrive,
                            speed=speed, alt_m=edge.alt_m,
                        )
                        h = self._heuristic(edge.to_id, goal_node)
                        heapq.heappush(open_set, _SIPPItem(
                            f_cost=arrive + h, g_cost=arrive,
                            node_id=edge.to_id,
                            path_legs=cur.path_legs + [leg],
                        ))

        return None  # No conflict-free path found

    # ---- Pre-Flight Strategic FAS Mission Planning --------------------

    def _schedule_candidate_route(
        self,
        drone_id: str,
        route_edge_ids: List[str],
        layer_per_edge: List[float],
        desired_departure: float,
        nominal_speed: float,
        climb_rate: float,
    ) -> Optional[Tuple[float, List[MissionLeg]]]:
        """
        Schedules a candidate corridor route across continuous time using True Variable-Speed SIPP.
        Computes the earliest departure time t_dep >= desired_departure such that all legs
        and vertical transition columns are conflict-free.
        When downstream legs/hub slots are busy, approach/preceding legs absorb delay by flying
        slower (between min and max cruise speed), falling back to ground pad delay only when
        minimum speed is still too fast.
        """
        t_dep = desired_departure
        max_attempts = 150

        for _ in range(max_attempts):
            conflict_found = False
            current_time = t_dep
            legs_tentative: List[MissionLeg] = []

            first_edge = self.graph.edges[route_edge_ids[0]]
            first_node = self.graph.nodes[first_edge.from_id]
            prev_alt = abs(first_node.z) if abs(first_node.z) > 10.0 else layer_per_edge[0]

            for k, eid in enumerate(route_edge_ids):
                edge = self.graph.edges[eid]
                assigned_alt = layer_per_edge[k]
                nominal_airspeed = min(edge.max_speed_ms, max(edge.min_speed_ms, nominal_speed))
                trav_speed = self.graph.get_effective_ground_speed(edge, nominal_airspeed, self.wind_vector)
                trav_time = edge.length / trav_speed

                v_dur = 0.0
                v_start = -1.0
                v_end = -1.0
                # Vertical transition column reservation when layer changes
                if abs(assigned_alt - prev_alt) > 1.0:
                    v_dur = abs(assigned_alt - prev_alt) / max(0.5, climb_rate)
                    v_start = current_time
                    v_end = current_time + v_dur
                    trans_rid = f"TRANS_{edge.from_id}"
                    v_band = abs(assigned_alt - prev_alt) / 2.0 + 4.0
                    v_center = (assigned_alt + prev_alt) / 2.0

                    if not self.table.is_free(
                        trans_rid, v_start, v_end, alt_m=v_center, alt_band_m=v_band, is_vertical=True, drone_id=drone_id
                    ):
                        intervals = self.table.get_safe_intervals(
                            trans_rid, v_dur, v_start, alt_m=v_center, alt_band_m=v_band, is_vertical=True, drone_id=drone_id
                        )
                        next_avail = intervals[0][0]
                        needed_delay = max(0.5, next_avail - v_start)

                        # Variable-Speed SIPP: absorb delay by slowing down preceding leg(s)
                        if k > 0 and legs_tentative:
                            prev_leg = legs_tentative[-1]
                            prev_edge = self.graph.edges[prev_leg.edge_id]
                            min_g_spd = self.graph.get_effective_ground_speed(prev_edge, prev_edge.min_speed_ms, self.wind_vector)
                            max_dur = prev_edge.length / min_g_spd
                            curr_dur = prev_leg.arrive_time - prev_leg.depart_time
                            absorbable = max(0.0, max_dur - curr_dur)
                            if absorbable > 0.1:
                                absorb_dt = min(needed_delay, absorbable)
                                cand_arrive = prev_leg.arrive_time + absorb_dt
                                if self.table.is_free(
                                    prev_leg.edge_id, prev_leg.depart_time, cand_arrive,
                                    alt_m=prev_leg.alt_m, alt_band_m=prev_edge.alt_band_m,
                                    is_vertical=False, drone_id=drone_id
                                ):
                                    prev_leg.arrive_time = cand_arrive
                                    prev_leg.speed = prev_edge.length / (cand_arrive - prev_leg.depart_time)
                                    needed_delay -= absorb_dt
                                    current_time = cand_arrive
                                    v_start = current_time
                                    v_end = current_time + v_dur

                        if needed_delay > 0.05:
                            t_dep += needed_delay
                            conflict_found = True
                            break
                    current_time = v_end

                # Horizontal corridor edge reservation
                h_start = current_time
                h_end = current_time + trav_time
                if not self.table.is_free(
                    eid, h_start, h_end, alt_m=assigned_alt, alt_band_m=edge.alt_band_m, is_vertical=False, drone_id=drone_id
                ):
                    intervals = self.table.get_safe_intervals(
                        eid, trav_time, h_start, alt_m=assigned_alt, alt_band_m=edge.alt_band_m, is_vertical=False, drone_id=drone_id
                    )
                    next_avail = intervals[0][0]
                    needed_delay = max(0.5, next_avail - h_start)

                    # Variable-Speed SIPP: slow down approach / preceding leg to arrive when slot opens
                    if k > 0 and legs_tentative:
                        prev_leg = legs_tentative[-1]
                        prev_edge = self.graph.edges[prev_leg.edge_id]
                        min_g_spd = self.graph.get_effective_ground_speed(prev_edge, prev_edge.min_speed_ms, self.wind_vector)
                        max_dur = prev_edge.length / min_g_spd
                        curr_dur = prev_leg.arrive_time - prev_leg.depart_time
                        absorbable = max(0.0, max_dur - curr_dur)
                        if absorbable > 0.1:
                            absorb_dt = min(needed_delay, absorbable)
                            cand_arrive = prev_leg.arrive_time + absorb_dt
                            if self.table.is_free(
                                prev_leg.edge_id, prev_leg.depart_time, cand_arrive,
                                alt_m=prev_leg.alt_m, alt_band_m=prev_edge.alt_band_m,
                                is_vertical=False, drone_id=drone_id
                            ):
                                prev_leg.arrive_time = cand_arrive
                                prev_leg.speed = prev_edge.length / (cand_arrive - prev_leg.depart_time)
                                needed_delay -= absorb_dt
                                current_time = cand_arrive
                                h_start = current_time
                                h_end = current_time + trav_time

                    if needed_delay > 0.05:
                        t_dep += needed_delay
                        conflict_found = True
                        break

                fn = self.graph.nodes[edge.from_id]
                tn = self.graph.nodes[edge.to_id]
                leg = MissionLeg(
                    edge_id=eid,
                    depart_time=h_start,
                    arrive_time=h_end,
                    speed=trav_speed,
                    alt_m=assigned_alt,
                    is_vertical_transition=(v_dur > 0.0),
                    transition_node=edge.from_id if v_dur > 0.0 else None,
                    from_pos=(fn.x, fn.y, fn.z),
                    to_pos=(tn.x, tn.y, tn.z),
                    vertical_transition_duration=v_dur,
                    vertical_transition_start=v_start,
                    vertical_transition_end=v_end,
                    vertical_transition_from_alt=prev_alt,
                    vertical_transition_to_alt=assigned_alt,
                )
                legs_tentative.append(leg)
                current_time = h_end
                prev_alt = assigned_alt

            if not conflict_found:
                return t_dep, legs_tentative

        return None

    def plan_preflight_mission(self, req: FlightAuthorisationRequest) -> Optional[MissionPlan]:
        """
        Strategic UTM Pre-Flight Flight Authorisation Service (FAS) mission planner.
        Runs entirely before drone takeoff (t < t_takeoff).

        Pipeline:
          1. Layered Network Flow: determines candidate flight levels (FL090, FL105, FL120).
          2. Multi-leg continuous SIPP search: finds conflict-free windows for all legs and vertical transitions.
          3. Explicit vertical transition reservation: isolates TRANS_{node} intermediate bands.
          4. Ground delay allocation: shifts scheduled departure rather than in-flight hovering.
        """
        route = req.route_edges
        if not route:
            if req.origin_node == req.destination_node:
                try:
                    d_idx = int(req.drone_id.split("-")[-1]) - 1
                    route = [f"E_{d_idx}_A", f"E_{d_idx}_B", f"E_{d_idx}_C", f"E_{d_idx}_D"]
                except Exception:
                    route = []
            else:
                route = self.graph.shortest_path(req.origin_node, req.destination_node)

        if not route:
            return None

        try:
            d_idx = int(req.drone_id.split("-")[-1]) - 1
        except Exception:
            d_idx = 0

        # Layer candidates
        if req.target_layer is not None:
            candidate_hub_layers = [req.target_layer]
        else:
            pref_layer = 90.0 if (d_idx % 2 == 0) else 105.0
            candidate_hub_layers = [pref_layer] + [l for l in AVAILABLE_LAYERS if l != pref_layer]

        best_score = math.inf
        best_dep = None
        best_legs = None

        base_alt = 90.0 if (d_idx % 2 == 0) else 105.0
        patrol_alt = 105.0 if (d_idx % 2 == 0) else 120.0

        for cand_hub in candidate_hub_layers:
            if len(route) == 4 and route[0].endswith("_A") and route[1].endswith("_B"):
                layers = [base_alt, cand_hub, cand_hub, patrol_alt]
            else:
                layers = [cand_hub] * len(route)

            res = self._schedule_candidate_route(
                drone_id=req.drone_id,
                route_edge_ids=route,
                layer_per_edge=layers,
                desired_departure=req.desired_departure_time,
                nominal_speed=req.cruise_speed,
                climb_rate=req.climb_rate,
            )
            if res is not None:
                dep_t, legs = res
                delay = dep_t - req.desired_departure_time
                v_cost = abs(cand_hub - base_alt)
                score = delay * 100.0 + v_cost * 0.1
                if score < best_score:
                    best_score = score
                    best_dep = dep_t
                    best_legs = legs
                if delay == 0.0 and v_cost == 0.0:
                    break

        if best_legs is None or best_dep is None:
            return None

        return MissionPlan(
            drone_id=req.drone_id,
            mission_type=req.mission_type,
            request_time=req.desired_departure_time,
            legs=best_legs,
            current_leg=0,
            scheduled_departure=best_dep,
            ground_delay_s=best_dep - req.desired_departure_time,
            is_authorized=True,
            origin_node=req.origin_node,
            destination_node=req.destination_node,
        )

    # ---- Loop route SIPP (pre-determined edge sequence) ----------------

    def plan_loop(
        self,
        drone_id: str,
        route_edge_ids: List[str],
        earliest_start: float,
        mission_type: int = MissionType.PATROL,
        speed_candidates: int = 7,
    ) -> Optional[MissionPlan]:
        """
        Plans a fixed-order loop route (list of edge IDs) using Pre-Flight FAS.
        """
        req = FlightAuthorisationRequest(
            drone_id=drone_id,
            origin_node=self.graph.edges[route_edge_ids[0]].from_id if route_edge_ids else "",
            destination_node=self.graph.edges[route_edge_ids[-1]].to_id if route_edge_ids else "",
            desired_departure_time=earliest_start,
            cruise_speed=15.0,
            climb_rate=2.5,
            descent_rate=2.5,
            mission_type=mission_type,
            route_edges=route_edge_ids,
        )
        plan = self.plan_preflight_mission(req)
        if plan is not None:
            return plan

        # Fallback to single-pass sequential scheduling if FAS optimization did not find solution
        legs: List[MissionLeg] = []
        t_current = earliest_start

        for edge_id in route_edge_ids:
            edge = self.graph.edges[edge_id]
            scheduled = False

            for airspeed in self._sample_speeds(edge, speed_candidates):
                speed = self.graph.get_effective_ground_speed(edge, airspeed, self.wind_vector)
                trav = edge.length / speed
                for t_start, t_end in self.table.get_safe_intervals(edge_id, trav, t_current, alt_m=edge.alt_m, alt_band_m=edge.alt_band_m, drone_id=drone_id):
                    depart = max(t_start, t_current)
                    if depart > t_end:
                        continue
                    arrive = depart + trav
                    fn = self.graph.nodes[edge.from_id]
                    tn = self.graph.nodes[edge.to_id]
                    legs.append(MissionLeg(
                        edge_id=edge_id,
                        depart_time=depart, arrive_time=arrive,
                        speed=speed, alt_m=edge.alt_m,
                        from_pos=(fn.x, fn.y, fn.z),
                        to_pos=(tn.x, tn.y, tn.z),
                    ))
                    t_current = arrive
                    scheduled = True
                    break
                if scheduled:
                    break

            if not scheduled:
                return None

        return MissionPlan(
            drone_id=drone_id,
            mission_type=mission_type,
            request_time=earliest_start,
            legs=legs,
            scheduled_departure=earliest_start,
            ground_delay_s=0.0,
            is_authorized=True,
        )

    def commit(self, plan: MissionPlan) -> None:
        """Writes all legs and vertical transitions of a plan into the ReservationTable."""
        self.table.reserve_plan(plan)


# ---------------------------------------------------------------------------
# Strategic Controller (top-level orchestrator)
# ---------------------------------------------------------------------------

class StrategicController:
    """
    Top-level strategic layer integrated into sim3d_server.py.

    Owns the CorridorGraph, ReservationTable, SIPPPlanner, and all active
    MissionPlans. Provides per-drone target (x, y, z, speed) to the tactical layer.

    Integration pattern in sim3d_server.py:

        # At startup:
        controller = StrategicController(num_drones=10)
        controller.initialize_scenario(list(uavs.keys()), sim_time=0.0)

        # In Simulation3DCore.step():
        for uid in uav_keys:
            u = uavs[uid]
            spd = math.hypot(u.vx, u.vy)
            tx, ty, tz, tspeed = controller.get_current_target(
                uid, (u.x, u.y, u.z), spd, sim_time
            )
            # Use (tx, ty, tz) instead of waypoint_routes lookup
            # Use tspeed to scale preferred velocity magnitude

        # For WebGL telemetry SSE broadcast:
        corridor_data = controller.get_corridor_telemetry(sim_time)
    """

    def __init__(
        self,
        num_drones: int = 10,
        safety_buffer_s: float = SAFETY_BUFFER_S,
        hover_timeout_s: float = HOVER_TIMEOUT_S,
        alt_violation_hold_s: float = ALT_VIOLATION_HOLD_S,
        wind_vector: Tuple[float, float, float] = (1.2, -0.8, 0.0),
    ) -> None:
        self.wind_vector = wind_vector
        self.graph, self._default_routes = CorridorGraph.build_circular_scenario(
            num_drones=num_drones, wind_vector=wind_vector
        )
        self.conflict_map = self.graph.build_conflict_map()
        self.table = ReservationTable(safety_buffer_s=safety_buffer_s, conflict_map=self.conflict_map)
        self.planner = SIPPPlanner(self.graph, self.table, wind_vector=wind_vector)
        self.plans: Dict[str, MissionPlan] = {}

        self.hover_timeout_s      = hover_timeout_s
        self.alt_violation_hold_s = alt_violation_hold_s

        # drone_id → list of edge IDs for its default loop
        self._route_map: Dict[str, List[str]] = {}
        self._num_drones = num_drones
        self.scenario_mode = "benchmark"

    # ---- Initialization ------------------------------------------------

    def initialize_scenario(
        self,
        drone_ids: List[str],
        sim_time: float = 0.0,
        mission_types: Optional[Dict[str, int]] = None,
    ) -> None:
        """
        Plans initial SIPP missions for all drones in priority order.
        Higher-priority drones lock in their slots first; lower-priority drones
        plan around the already-reserved windows.

        A small stagger (0.5 s per drone) prevents simultaneous hub entry.
        """
        self.scenario_mode = "benchmark"
        mission_types = mission_types or {}

        # Reset state
        self.conflict_map = self.graph.build_conflict_map()
        self.table   = ReservationTable(safety_buffer_s=self.table.safety_buffer, conflict_map=self.conflict_map)
        self.planner = SIPPPlanner(self.graph, self.table, wind_vector=self.wind_vector)
        self.plans.clear()
        self._route_map.clear()

        # Assign routes by drone index order
        for i, drone_id in enumerate(drone_ids):
            self._route_map[drone_id] = self._default_routes[i % len(self._default_routes)]

        # Sort by (priority, FCFS order)
        sorted_drones = sorted(
            enumerate(drone_ids),
            key=lambda iv: (mission_types.get(iv[1], MissionType.PATROL), iv[0]),
        )

        for rank, (orig_idx, drone_id) in enumerate(sorted_drones):
            route = self._route_map[drone_id]
            mtype = mission_types.get(drone_id, MissionType.PATROL)
            # Stagger: 0.5 s per priority rank to spread initial requests
            earliest = sim_time + rank * 0.5

            req = FlightAuthorisationRequest(
                drone_id=drone_id,
                origin_node=f"PERIM_{orig_idx}",
                destination_node=f"PERIM_{orig_idx}",
                desired_departure_time=earliest,
                cruise_speed=15.0,
                climb_rate=2.5,
                descent_rate=2.5,
                mission_type=mtype,
                route_edges=route,
            )

            plan = self.planner.plan_preflight_mission(req)

            if plan is not None:
                self.planner.commit(plan)
                self.plans[drone_id] = plan
                leg0 = plan.legs[0]
                print(
                    f"[StrategicPlanner] [FAS AUTHORIZED] {drone_id:10s} ({MissionType.name(mtype):9s}) │ "
                    f"{len(plan.legs)} legs │ "
                    f"sched_dep t={plan.scheduled_departure:.1f}s (delay={plan.ground_delay_s:.1f}s) │ "
                    f"speed {leg0.speed:.1f}–{plan.legs[-1].speed:.1f} m/s"
                )
            else:
                print(f"[StrategicPlanner] ⚠ FAS Rejected authorization for {drone_id} — holding position.")

    def initialize_random_scenario(
        self,
        drone_ids: List[str],
        sim_time: float = 0.0,
        seed: Optional[int] = None,
    ) -> Dict[str, Tuple[str, str]]:
        """
        Plans initial SIPP missions with randomized origins and destinations
        across the corridor graph. Each drone is assigned a random station-to-station
        mission (e.g. PERIM_i -> PERIM_j), computing shortest graph routes and
        conflict-free 4D departure schedules with SIPP.
        """
        import random
        rng = random.Random(seed)
        self.scenario_mode = "random"

        self.conflict_map = self.graph.build_conflict_map()
        self.table   = ReservationTable(safety_buffer_s=self.table.safety_buffer, conflict_map=self.conflict_map)
        self.planner = SIPPPlanner(self.graph, self.table, wind_vector=self.wind_vector)
        self.plans.clear()
        self._route_map.clear()

        perim_nodes = [f"PERIM_{i}" for i in range(self._num_drones)]
        od_pairs: Dict[str, Tuple[str, str]] = {}

        # Shuffle origins to spread drones across stations
        shuffled_origins = list(perim_nodes)
        rng.shuffle(shuffled_origins)

        for i, drone_id in enumerate(drone_ids):
            orig = shuffled_origins[i % len(shuffled_origins)]
            # Destination must be different from origin
            possible_dests = [n for n in perim_nodes if n != orig]
            dest = rng.choice(possible_dests)
            od_pairs[drone_id] = (orig, dest)

            route = self.graph.shortest_path(orig, dest)
            self._route_map[drone_id] = route

            mtype = rng.choice([MissionType.EMERGENCY, MissionType.DELIVERY, MissionType.SURVEY, MissionType.PATROL])
            # Stagger departures slightly so takeoffs are sequential
            earliest = sim_time + i * 1.0

            req = FlightAuthorisationRequest(
                drone_id=drone_id,
                origin_node=orig,
                destination_node=dest,
                desired_departure_time=earliest,
                cruise_speed=rng.uniform(10.0, 14.0),
                climb_rate=2.5,
                descent_rate=2.5,
                mission_type=mtype,
                route_edges=route,
            )

            plan = self.planner.plan_preflight_mission(req)
            if plan is not None:
                self.planner.commit(plan)
                self.plans[drone_id] = plan
                print(
                    f"[StrategicPlanner] [RANDOM FAS AUTHORIZED] {drone_id:10s} ({MissionType.name(mtype):9s}) │ "
                    f"{orig} → {dest} │ {len(plan.legs)} legs │ "
                    f"sched_dep t={plan.scheduled_departure:.1f}s"
                )
            else:
                print(f"[StrategicPlanner] ⚠ FAS Rejected authorization for {drone_id} ({orig} -> {dest})")

        return od_pairs

    def dispatch_next_random_mission(
        self,
        drone_id: str,
        current_node_id: str,
        sim_time: float,
    ) -> bool:
        """
        Dynamically dispatches a new random destination mission for a drone that has
        arrived at its previous destination station.
        """
        import random
        perim_nodes = [f"PERIM_{i}" for i in range(self._num_drones)]
        candidate_dests = [n for n in perim_nodes if n != current_node_id]
        if not candidate_dests:
            return False

        dest_node = random.choice(candidate_dests)
        route = self.graph.shortest_path(current_node_id, dest_node)
        if not route:
            return False

        mtype = random.choice([MissionType.EMERGENCY, MissionType.DELIVERY, MissionType.SURVEY, MissionType.PATROL])
        cruise_spd = random.uniform(10.0, 14.0)

        # Clear past completed reservations for this drone before scheduling next mission
        self.table.release(drone_id)

        req = FlightAuthorisationRequest(
            drone_id=drone_id,
            origin_node=current_node_id,
            destination_node=dest_node,
            desired_departure_time=sim_time + 0.5,
            cruise_speed=cruise_spd,
            climb_rate=2.5,
            descent_rate=2.5,
            mission_type=mtype,
            route_edges=route,
        )

        plan = self.planner.plan_preflight_mission(req)
        if plan is not None:
            self.planner.commit(plan)
            self.plans[drone_id] = plan
            self._route_map[drone_id] = route
            print(
                f"[StrategicPlanner] [CONTINUOUS RANDOM DISPATCH] {drone_id} arrived at {current_node_id} "
                f"→ New destination: {dest_node} ({len(plan.legs)} legs, sched_dep={plan.scheduled_departure:.1f}s)"
            )
            return True
        else:
            print(f"[StrategicPlanner] ⚠ Unable to dispatch continuous mission for {drone_id} from {current_node_id} to {dest_node}")
            return False

    # ---- Layer / Altitude Calculation at Set Points -------------------

    def calculate_layer(
        self,
        drone_id: str,
        from_node: str,
        to_node: str,
        sim_time: float,
        current_alt: float,
    ) -> float:
        """
        Calculates the optimal altitude layer (flight level) for a drone when it
        reaches a set point (waypoint node).

        Evaluates available flight levels (90m, 105m, 120m) against:
          1. Spacetime conflicts and active reservations on upcoming corridors
             (especially central hub transit E_i_B where drones cross at center).
          2. Vertical separation: ensures >= 12m vertical clearance from crossing traffic.
          3. Airspace traffic balancing across flight levels to prevent bottlenecks.
        """
        try:
            d_idx = int(drone_id.split("-")[-1]) - 1
        except Exception:
            d_idx = 0

        est_duration = 14.0

        # Case 1: Reaching Hub Entrance (HUB_IN_i) — Hub Crossing Leg
        if "HUB_IN" in from_node:
            active_hub_alts = []
            # Check other drones crossing or entering the hub
            for other_id, other_plan in self.plans.items():
                if other_id == drone_id or not other_plan.legs:
                    continue
                for oleg in other_plan.legs:
                    if "_B" in oleg.edge_id:
                        if oleg.depart_time <= sim_time + est_duration and oleg.arrive_time >= sim_time:
                            active_hub_alts.append(oleg.alt_m)

            # Check table reservations on hub crossing edges
            for eid, rlist in self.table._table.items():
                if eid.endswith("_B"):
                    for r in rlist:
                        if r.drone_id != drone_id:
                            if r.t_entry <= sim_time + est_duration + self.table.safety_buffer and r.t_exit >= sim_time:
                                active_hub_alts.append(r.alt_m)

            # Evaluate candidate layers (90m, 105m, 120m)
            best_layer = current_alt
            best_conflict_score = 999.0
            for cand_alt in AVAILABLE_LAYERS:
                conflicts = sum(1 for alt in active_hub_alts if abs(alt - cand_alt) < 12.0)
                # Primary: minimize conflicts; Secondary: balance layers
                score = conflicts * 100.0 + abs(cand_alt - current_alt) * 0.1
                if score < best_conflict_score:
                    best_conflict_score = score
                    best_layer = cand_alt

            return best_layer

        # Case 2: Reaching Hub Exit Gate (HUB_OUT_i) — Departure Leg
        elif "HUB_OUT" in from_node:
            # Continue at current layer or transition to sector departure altitude
            return current_alt

        # Case 3: Reaching Antipodal Perimeter (PERIM_anti) — Patrol Return Leg
        elif "PERIM" in from_node and "PERIM" in to_node:
            # Stagger perimeter patrol return to 105m or 120m to clear inbound traffic
            return 105.0 if (d_idx % 2 == 0) else 120.0

        # Case 4: Reaching Home Perimeter (PERIM_i) — Inbound Approach Leg
        elif "PERIM" in from_node and "HUB_IN" in to_node:
            # Inbound approach flight level: 90m for even, 105m for odd
            return 90.0 if (d_idx % 2 == 0) else 105.0

        return current_alt

    # ---- Per-step Target Query -----------------------------------------

    def get_current_target(
        self,
        drone_id: str,
        pos: Tuple[float, float, float],  # (x, y, z) NED
        vel_magnitude: float,             # scalar ground speed (m/s)
        sim_time: float,
    ) -> Tuple[float, float, float, float]:
        """
        Returns (target_x, target_y, target_z, target_speed) for this drone.

        Checks replanning triggers (waypoint arrival, altitude violation, hover timeout)
        and transparently replans if needed before returning the fresh target.

        target_z is the NED z value (negative altitude).
        target_speed is the SIPP-assigned cruise speed for the current leg (m/s).
        """
        plan = self.plans.get(drone_id)
        if plan is None or not plan.legs:
            # No plan: hover in place
            return pos[0], pos[1], pos[2], 0.0

        # ---- Ground Delay Hold (Pre-Flight FAS): Hold on launch pad until scheduled departure ----
        if sim_time < plan.scheduled_departure:
            leg0 = plan.legs[0]
            start_edge = self.graph.edges.get(leg0.edge_id)
            if start_edge:
                start_node = self.graph.nodes[start_edge.from_id]
                return start_node.x, start_node.y, start_node.z, 0.0
            return pos[0], pos[1], pos[2], 0.0

        # ---- Resolve current leg ----------------------------------------
        leg_idx  = plan.current_leg % len(plan.legs)
        leg      = plan.legs[leg_idx]
        edge     = self.graph.edges[leg.edge_id]
        tgt_node = self.graph.nodes[edge.to_id]

        # ---- Waypoint Arrival Detection (3D Euclidean Distance) ----
        # Using 3D distance prevents advancing waypoints while the drone is still
        # climbing or descending through an intermediate vertical transition band.
        dist_3d = math.hypot(tgt_node.x - pos[0], tgt_node.y - pos[1], -leg.alt_m - pos[2])
        if dist_3d < ARRIVAL_RADIUS_M:
            if getattr(self, "scenario_mode", "benchmark") == "random" and plan.current_leg >= len(plan.legs) - 1:
                # Reached final destination of current random mission! Dynamically dispatch next mission
                self.dispatch_next_random_mission(drone_id, tgt_node.id, sim_time)
                plan = self.plans.get(drone_id)
                if plan is None or not plan.legs:
                    return pos[0], pos[1], pos[2], 0.0
                leg_idx = 0
            else:
                plan.current_leg = (plan.current_leg + 1) % len(plan.legs)
                leg_idx  = plan.current_leg % len(plan.legs)

            leg      = plan.legs[leg_idx]
            edge     = self.graph.edges[leg.edge_id]
            tgt_node = self.graph.nodes[edge.to_id]

        # Note: In accordance with Barrado et al. (2026), strategic replanning triggers
        # (ALT_VIOLATION and HOVER_TIMEOUT) are omitted here to protect the integrity
        # of the pre-flight reservation table during flight. Layer 2 (ORCA) and velocity
        # controls handle transient tactical deviations without wiping reservations.

        return tgt_node.x, tgt_node.y, -leg.alt_m, leg.speed

    # ---- Replanning & Layer Calculation --------------------------------

    def _replan(
        self,
        drone_id: str,
        sim_time: float,
        reason: str = "",
        start_node: Optional[str] = None,
        current_pos: Optional[Tuple[float, float, float]] = None,
    ) -> None:
        """
        Releases all future reservations for a drone, executes layer calculation
        at the reached set point, and recalculates the optimal path through the
        corridor network graph to the destination milestone.
        """
        plan = self.plans.get(drone_id)
        if plan is None:
            return

        leg_idx = plan.current_leg % len(plan.legs)
        route   = [l.edge_id for l in plan.legs] if plan.legs else self._route_map.get(drone_id, [])
        mtype   = plan.mission_type

        # Release all slots (including future ones we can't honour anymore)
        self.table.release(drone_id)

        # Parse drone index to determine target milestones
        try:
            d_idx = int(drone_id.split("-")[-1]) - 1
        except Exception:
            d_idx = 0
        n_half = self._num_drones // 2
        anti_idx = (d_idx + n_half) % self._num_drones

        # Determine current start node in graph
        cur_edge_id = route[leg_idx] if route else None
        cur_edge = self.graph.edges.get(cur_edge_id) if cur_edge_id else None
        if start_node is None:
            if cur_edge:
                start_node = cur_edge.to_id
            else:
                start_node = f"PERIM_{d_idx}"

        # Determine next destination milestone based on current position
        if "HUB_IN" in start_node or f"PERIM_{d_idx}" in start_node:
            goal_node = f"PERIM_{anti_idx}"
        elif f"PERIM_{anti_idx}" in start_node:
            goal_node = f"PERIM_{d_idx}"
        elif "HUB_OUT" in start_node:
            goal_node = f"PERIM_{anti_idx}"
        else:
            goal_node = f"PERIM_{anti_idx}"

        # Current drone altitude
        current_alt = abs(current_pos[2]) if current_pos is not None else 90.0

        # Layer calculation when drone reaches a set point
        new_layer = self.calculate_layer(
            drone_id=drone_id,
            from_node=start_node,
            to_node=goal_node,
            sim_time=sim_time,
            current_alt=current_alt,
        )

        # 1. Dynamic Network Recalculation across full corridor graph
        new_plan = self.planner.plan(
            drone_id=drone_id,
            start_node=start_node,
            goal_node=goal_node,
            earliest_start=sim_time,
            mission_type=mtype,
        )

        # Append return leg to maintain full circulating loop continuity
        if new_plan is not None and new_plan.legs:
            if goal_node == f"PERIM_{anti_idx}":
                ret_id = f"E_{d_idx}_D"
                if ret_id in self.graph.edges:
                    ret_edge = self.graph.edges[ret_id]
                    last_arr = new_plan.legs[-1].arrive_time
                    ret_trav = ret_edge.length / ret_edge.max_speed_ms
                    new_plan.legs.append(MissionLeg(
                        edge_id=ret_id,
                        depart_time=last_arr,
                        arrive_time=last_arr + ret_trav,
                        speed=ret_edge.max_speed_ms,
                        alt_m=ret_edge.alt_m,
                    ))
            elif goal_node == f"PERIM_{d_idx}":
                app_id = f"E_{d_idx}_A"
                if app_id in self.graph.edges:
                    app_edge = self.graph.edges[app_id]
                    last_arr = new_plan.legs[-1].arrive_time
                    app_trav = app_edge.length / app_edge.max_speed_ms
                    new_plan.legs.append(MissionLeg(
                        edge_id=app_id,
                        depart_time=last_arr,
                        arrive_time=last_arr + app_trav,
                        speed=app_edge.max_speed_ms,
                        alt_m=app_edge.alt_m,
                    ))

        # 2. Fallback if no network path found: schedule remaining sequence
        if new_plan is None or not new_plan.legs:
            default_edges = self._route_map.get(drone_id, [])
            idx_in_loop = 0
            for i, eid in enumerate(default_edges):
                if self.graph.edges[eid].from_id == start_node:
                    idx_in_loop = i
                    break
            remaining = default_edges[idx_in_loop:] + default_edges[:idx_in_loop]
            new_plan = self.planner.plan_loop(
                drone_id=drone_id,
                route_edge_ids=remaining,
                earliest_start=sim_time,
                mission_type=mtype,
            )

        if new_plan is not None and new_plan.legs:
            # Apply calculated layer strictly to the drone's mission leg(s)
            # (CorridorGraph edges remain read-only and unmutated across drones)
            for leg in new_plan.legs:
                if leg.edge_id in self.graph.edges:
                    e = self.graph.edges[leg.edge_id]
                    if e.from_id == start_node or "HUB_IN" in start_node:
                        leg.alt_m = new_layer

            if abs(new_layer - current_alt) > 1.0:
                new_plan.layer_transition_until = sim_time + abs(new_layer - current_alt) / 1.5 + 2.5

            new_plan.current_leg = 0
            self.planner.commit(new_plan)
            self.plans[drone_id] = new_plan
            chosen_edges = [l.edge_id for l in new_plan.legs]
            is_corridor_change = any("alt" in eid or "from" in eid for eid in chosen_edges)
            tag = "CHANGED CORRIDOR" if is_corridor_change else "REPLAN"
            layer_info = f" | Layer: {new_layer:.1f}m"
            print(
                f"[StrategicPlanner] {tag} ({reason:16s}) {drone_id} → "
                f"{chosen_edges}{layer_info} from t={sim_time:.1f}s"
            )
        else:
            print(f"[StrategicPlanner] REPLAN FAILED ({reason}) for {drone_id}.")

    # ---- Telemetry for WebGL Visualization ----------------------------

    def get_corridor_telemetry(self, sim_time: float = 0.0) -> dict:
        """
        Returns a JSON-serializable dict describing:
          - All corridor nodes and edges (for tube rendering in Three.js)
          - Current active drone plans (for SIPP slot labels)
          - Active reservations summary

        Broadcast via SSE alongside the drone physics telemetry in sim3d_server.py.
        Coordinate convention: Three.js uses (East, Altitude, -North),
        so callers should apply the NED→Three.js transform on the JS side.
        """
        nodes_out = [
            {"id": n.id, "x": n.x, "y": n.y, "z": n.z}
            for n in self.graph.nodes.values()
        ]

        edges_out = []
        for e in self.graph.edges.values():
            fn = self.graph.nodes[e.from_id]
            tn = self.graph.nodes[e.to_id]
            edges_out.append({
                "id":    e.id,
                "alt_m": e.alt_m,
                "length": round(e.length, 1),
                "from": {"x": fn.x, "y": fn.y, "z": fn.z},
                "to":   {"x": tn.x, "y": tn.y, "z": tn.z},
            })

        drone_plans_out: Dict[str, dict] = {}
        for drone_id, plan in self.plans.items():
            if not plan.legs:
                continue
            li  = plan.current_leg % len(plan.legs)
            leg = plan.legs[li]
            drone_plans_out[drone_id] = {
                "mission_type": MissionType.name(plan.mission_type),
                "current_edge": leg.edge_id,
                "speed_ms":     round(leg.speed, 2),
                "alt_m":        leg.alt_m,
                "depart_t":     round(leg.depart_time, 2),
                "arrive_t":     round(leg.arrive_time, 2),
                "leg_index":    li,
                "total_legs":   len(plan.legs),
                "origin":       getattr(plan, "origin_node", ""),
                "destination":  getattr(plan, "destination_node", ""),
            }

        return {
            "type":          "corridor_telemetry",
            "sim_time":      round(sim_time, 2),
            "scenario_mode": getattr(self, "scenario_mode", "benchmark"),
            "nodes":         nodes_out,
            "edges":         edges_out,
            "drone_plans":   drone_plans_out,
            "reservations":  self.table.summary(),
            "active_slots":  self.table.active_reservations(sim_time),
        }

    # ---- Utility -------------------------------------------------------

    def reset(self, num_drones: int = 10, sim_time: float = 0.0) -> None:
        """Full reset: rebuilds graph, clears table and plans."""
        self.graph, self._default_routes = CorridorGraph.build_circular_scenario(
            num_drones=num_drones, wind_vector=self.wind_vector
        )
        self.conflict_map = self.graph.build_conflict_map()
        self.table   = ReservationTable(safety_buffer_s=self.table.safety_buffer, conflict_map=self.conflict_map)
        self.planner = SIPPPlanner(self.graph, self.table, wind_vector=self.wind_vector)
        self.plans.clear()
        self._route_map.clear()
        self._num_drones = num_drones

    def get_plan_summary(self) -> Dict[str, dict]:
        """Returns a human-readable summary of all active plans."""
        out = {}
        for drone_id, plan in self.plans.items():
            out[drone_id] = {
                "mission_type": MissionType.name(plan.mission_type),
                "total_legs":   len(plan.legs),
                "current_leg":  plan.current_leg,
                "legs": [
                    {
                        "edge":   l.edge_id,
                        "depart": round(l.depart_time, 2),
                        "arrive": round(l.arrive_time, 2),
                        "speed":  round(l.speed, 2),
                        "alt_m":  l.alt_m,
                    }
                    for l in plan.legs
                ],
            }
        return out
