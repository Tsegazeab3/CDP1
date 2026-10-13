#!/usr/bin/env python3
"""
test_strategic_planner.py
=========================
Comprehensive test suite for the Strategic 4D Path Planner (SIPP + Airspace Corridors).

Tests:
1. CorridorGraph: nodes, edges, 3D lengths, Dijkstra routing, serialization.
2. ReservationTable: safe intervals, time buffers, overlap detection, slot release.
3. SIPPPlanner: point-to-point SIPP search, loop scheduling, variable speeds, priority ordering.
4. StrategicController: 10-drone scenario, target queries, replanning triggers (waypoint arrival,
   altitude violation, hover timeout), and WebGL corridor telemetry export.
5. ClosedLoopIntegration: 4D strategic corridor navigation coupled with 6-DOF quadcopter dynamics.
"""

import math
import unittest
from typing import List, Tuple

from strategic_planner import (
    ALT_VIOLATION_HOLD_S,
    ARRIVAL_RADIUS_M,
    HOVER_SPEED_THRESHOLD,
    HOVER_TIMEOUT_S,
    SAFETY_BUFFER_S,
    CorridorEdge,
    CorridorGraph,
    CorridorNode,
    EdgeReservation,
    FlightAuthorisationRequest,
    MissionLeg,
    MissionPlan,
    MissionType,
    ReservationTable,
    SIPPPlanner,
    StrategicController,
)


class TestCorridorGraph(unittest.TestCase):
    """Unit tests for CorridorGraph data structure and algorithms."""

    def test_build_circular_scenario(self):
        graph, routes = CorridorGraph.build_circular_scenario(num_drones=10)
        # 10 perimeter + 10 hub in + 10 hub out = 30 nodes
        self.assertEqual(len(graph.nodes), 30)
        # 8 edges per drone (4 primary + 4 alternate branching corridors) * 10 drones = 80 edges
        self.assertEqual(len(graph.edges), 80)
        self.assertEqual(len(routes), 10)
        for r in routes:
            self.assertEqual(len(r), 4)

    def test_edge_lengths_calculated(self):
        graph, _ = CorridorGraph.build_circular_scenario(num_drones=10)
        for edge_id, edge in graph.edges.items():
            self.assertGreater(edge.length, 0.0, f"Edge {edge_id} length must be > 0")

    def test_shortest_path_dijkstra(self):
        graph = CorridorGraph()
        graph.add_node(CorridorNode("N1", 0.0, 0.0, -90.0))
        graph.add_node(CorridorNode("N2", 100.0, 0.0, -90.0))
        graph.add_node(CorridorNode("N3", 200.0, 0.0, -90.0))
        graph.add_edge(CorridorEdge("E1", "N1", "N2", 90.0, 5.0, 10.0, 15.0))
        graph.add_edge(CorridorEdge("E2", "N2", "N3", 90.0, 5.0, 10.0, 15.0))

        path = graph.shortest_path("N1", "N3")
        self.assertEqual(path, ["E1", "E2"])

        # No path case
        no_path = graph.shortest_path("N3", "N1")
        self.assertEqual(no_path, [])

    def test_serialization(self):
        graph, _ = CorridorGraph.build_circular_scenario(num_drones=4)
        data = graph.to_dict()
        self.assertIn("nodes", data)
        self.assertIn("edges", data)

        reconstructed = CorridorGraph.from_dict(data)
        self.assertEqual(len(reconstructed.nodes), len(graph.nodes))
        self.assertEqual(len(reconstructed.edges), len(graph.edges))


class TestReservationTable(unittest.TestCase):
    """Unit tests for spacetime reservation table and safe intervals."""

    def setUp(self):
        self.table = ReservationTable(safety_buffer_s=5.0)

    def test_is_free_empty_table(self):
        self.assertTrue(self.table.is_free("E1", 0.0, 10.0))

    def test_reserve_and_conflict(self):
        res1 = EdgeReservation("UAV-01", t_entry=10.0, t_exit=20.0, speed=12.0)
        self.assertTrue(self.table.reserve("E1", res1))

        # Direct overlap fails
        res_overlap = EdgeReservation("UAV-02", t_entry=15.0, t_exit=25.0, speed=12.0)
        self.assertFalse(self.table.reserve("E1", res_overlap))

        # Within 5s safety buffer fails (20.0 to 23.0 is < 20.0 + 5.0)
        res_buffer_conflict = EdgeReservation("UAV-02", t_entry=23.0, t_exit=30.0, speed=12.0)
        self.assertFalse(self.table.reserve("E1", res_buffer_conflict))

        # Outside safety buffer succeeds (>= 25.0)
        res_safe = EdgeReservation("UAV-02", t_entry=26.0, t_exit=35.0, speed=12.0)
        self.assertTrue(self.table.reserve("E1", res_safe))

    def test_safe_intervals(self):
        # Reserve [20, 30]
        self.table.reserve("E1", EdgeReservation("UAV-01", t_entry=20.0, t_exit=30.0, speed=10.0))
        # With 5s buffer: blocked window is [15, 35]
        # Traversal time = 8s
        intervals = self.table.get_safe_intervals("E1", traversal_time=8.0, earliest_start=0.0)
        # Should have a window before 20.0 (needs to end by 15.0, so start <= 7.0)
        # And an open-ended window starting after 35.0
        self.assertGreaterEqual(len(intervals), 2)
        self.assertAlmostEqual(intervals[0][0], 0.0)
        self.assertAlmostEqual(intervals[0][1], 7.0)
        self.assertGreaterEqual(intervals[1][0], 35.0)
        self.assertEqual(intervals[1][1], math.inf)

    def test_release(self):
        self.table.reserve("E1", EdgeReservation("UAV-01", 10.0, 20.0, 10.0))
        self.assertFalse(self.table.is_free("E1", 12.0, 18.0))

        self.table.release("UAV-01")
        self.assertTrue(self.table.is_free("E1", 12.0, 18.0))


class TestSIPPPlanner(unittest.TestCase):
    """Unit tests for SIPP A* path search and loop scheduling."""

    def setUp(self):
        self.graph = CorridorGraph()
        self.graph.add_node(CorridorNode("A", 0.0, 0.0, -90.0))
        self.graph.add_node(CorridorNode("B", 100.0, 0.0, -90.0))
        self.graph.add_node(CorridorNode("C", 200.0, 0.0, -90.0))
        self.graph.add_edge(CorridorEdge("E_AB", "A", "B", 90.0, 5.0, 10.0, 15.0))
        self.graph.add_edge(CorridorEdge("E_BC", "B", "C", 90.0, 5.0, 10.0, 15.0))
        self.table = ReservationTable(safety_buffer_s=5.0)
        self.planner = SIPPPlanner(self.graph, self.table)

    def test_single_agent_plan(self):
        plan = self.planner.plan("UAV-01", "A", "C", earliest_start=0.0)
        self.assertIsNotNone(plan)
        self.assertEqual(len(plan.legs), 2)
        self.assertEqual(plan.legs[0].edge_id, "E_AB")
        self.assertEqual(plan.legs[1].edge_id, "E_BC")
        self.assertGreater(plan.legs[1].arrive_time, plan.legs[0].arrive_time)

    def test_variable_speed_and_commitment(self):
        plan = self.planner.plan("UAV-01", "A", "B", earliest_start=0.0)
        self.assertIsNotNone(plan)
        self.planner.commit(plan)

        # Edge E_AB should now be reserved
        self.assertFalse(self.table.is_free("E_AB", plan.legs[0].depart_time, plan.legs[0].arrive_time))

        # Second drone planning on same edge must be scheduled after UAV-01's exit + buffer
        plan2 = self.planner.plan("UAV-02", "A", "B", earliest_start=0.0)
        self.assertIsNotNone(plan2)
        self.assertGreaterEqual(plan2.legs[0].depart_time, plan.legs[0].arrive_time + self.table.safety_buffer)

    def test_priority_scheduling(self):
        # Schedule higher priority EMERGENCY first
        plan_emerg = self.planner.plan("UAV-EMERG", "A", "B", earliest_start=0.0, mission_type=MissionType.EMERGENCY)
        self.assertIsNotNone(plan_emerg)
        self.planner.commit(plan_emerg)

        # PATROL drone plans second
        plan_patrol = self.planner.plan("UAV-PATROL", "A", "B", earliest_start=0.0, mission_type=MissionType.PATROL)
        self.assertIsNotNone(plan_patrol)
        self.assertGreaterEqual(plan_patrol.legs[0].depart_time, plan_emerg.legs[0].arrive_time)


class TestStrategicController(unittest.TestCase):
    """Unit tests for top-level StrategicController and replanning triggers."""

    def setUp(self):
        self.controller = StrategicController(num_drones=10)
        self.drone_ids = [f"UAV-{i+1:02d}" for i in range(10)]
        self.controller.initialize_scenario(self.drone_ids, sim_time=0.0)

    def test_scenario_initialization(self):
        self.assertEqual(len(self.controller.plans), 10)
        for uid in self.drone_ids:
            plan = self.controller.plans[uid]
            self.assertEqual(len(plan.legs), 4)
            self.assertGreater(plan.legs[0].speed, 0.0)

    def test_target_query(self):
        # Query initial target for UAV-01
        tx, ty, tz, spd = self.controller.get_current_target("UAV-01", (260.0, 0.0, -90.0), vel_magnitude=10.0, sim_time=0.1)
        self.assertAlmostEqual(tz, -90.0)
        self.assertGreater(spd, 0.0)

    def test_waypoint_arrival_trigger(self):
        # Move drone right next to target node (< 24m)
        plan_before = self.controller.plans["UAV-01"]
        edge0 = self.controller.graph.edges[plan_before.legs[0].edge_id]
        tgt_node = self.controller.graph.nodes[edge0.to_id]

        near_pos = (tgt_node.x, tgt_node.y, tgt_node.z)
        tx, ty, tz, spd = self.controller.get_current_target("UAV-01", near_pos, vel_magnitude=10.0, sim_time=5.0)

        # In pre-flight FAS, arrival cleanly advances waypoint progression without mid-flight replan
        plan_after = self.controller.plans["UAV-01"]
        self.assertIsNotNone(plan_after)
        self.assertEqual(plan_after.current_leg, 1)

    def test_altitude_violation_trigger(self):
        plan = self.controller.plans["UAV-01"]
        # Normal query
        self.controller.get_current_target("UAV-01", (200.0, 0.0, -90.0), 10.0, sim_time=1.0)
        self.assertEqual(plan.alt_violation_since, -1.0)

        # Significant altitude deviation (e.g. at -50m instead of -90m: diff 40m > 8m band)
        self.controller.get_current_target("UAV-01", (200.0, 0.0, -50.0), 10.0, sim_time=2.0)
        self.assertAlmostEqual(plan.alt_violation_since, 2.0)

        # After holding violation > ALT_VIOLATION_HOLD_S (e.g. 2.1s later at t=4.2s)
        self.controller.get_current_target("UAV-01", (200.0, 0.0, -50.0), 10.0, sim_time=4.2)
        # Should have reset violation timer after triggering replan
        self.assertEqual(self.controller.plans["UAV-01"].alt_violation_since, -1.0)

    def test_hover_timeout_trigger(self):
        plan = self.controller.plans["UAV-01"]
        # Drone stops (< 0.5 m/s)
        self.controller.get_current_target("UAV-01", (200.0, 0.0, -90.0), vel_magnitude=0.1, sim_time=1.0)
        self.assertAlmostEqual(plan.hover_since, 1.0)

        # After holding hover > HOVER_TIMEOUT_S (8s later at t=9.5s)
        self.controller.get_current_target("UAV-01", (200.0, 0.0, -90.0), vel_magnitude=0.1, sim_time=9.5)
        self.assertEqual(self.controller.plans["UAV-01"].hover_since, -1.0)

    def test_corridor_telemetry_export(self):
        telem = self.controller.get_corridor_telemetry(sim_time=2.5)
        self.assertEqual(telem["type"], "corridor_telemetry")
        self.assertIn("nodes", telem)
        self.assertIn("edges", telem)
        self.assertIn("drone_plans", telem)
        self.assertEqual(len(telem["drone_plans"]), 10)

    def test_layer_calculation_at_set_point(self):
        # 1. Reaching antipodal perimeter set point (PERIM_5): calculates 105m patrol return layer
        patrol_layer = self.controller.calculate_layer(
            drone_id="UAV-01",
            from_node="PERIM_5",
            to_node="PERIM_0",
            sim_time=39.0,
            current_alt=90.0,
        )
        self.assertEqual(patrol_layer, 105.0)

        # 2. Reaching hub entrance set point with conflicting drone at 90m:
        self.controller.table.reserve("E_0_B", EdgeReservation("UAV-01", t_entry=10.0, t_exit=25.0, speed=12.0, alt_m=90.0))
        hub_layer = self.controller.calculate_layer(
            drone_id="UAV-03",
            from_node="HUB_IN_2",
            to_node="PERIM_7",
            sim_time=11.0,
            current_alt=90.0,
        )
        # Must deconflict away from 90.0m to 105m or 120m
        self.assertIn(hub_layer, [105.0, 120.0])
        self.assertNotEqual(hub_layer, 90.0)


class TestClosedLoopIntegration(unittest.TestCase):
    """Integration test: 4D SIPP strategic planning coupled with 3D physical flight."""

    def test_simulation_multi_step_flight(self):
        from quadcopter_dynamics import DynamicUAVState, QuadcopterPhysicsSimulator

        physics = QuadcopterPhysicsSimulator()
        controller = StrategicController(num_drones=10)
        drone_ids = [f"UAV-{i+1:02d}" for i in range(10)]
        controller.initialize_scenario(drone_ids, sim_time=0.0)

        uavs = {}
        for i, uid in enumerate(drone_ids):
            angle = i * (2.0 * math.pi / 10)
            alt_m = 90.0 if (i % 2 == 0) else 105.0
            uavs[uid] = DynamicUAVState(
                id=uid,
                x=260.0 * math.cos(angle),
                y=260.0 * math.sin(angle),
                z=-alt_m,
            )

        dt = 0.033
        sim_time = 0.0
        # Run 60 steps (~2 seconds)
        for _ in range(60):
            sim_time += dt
            wind = physics.update_gust(dt)

            for uid in drone_ids:
                u = uavs[uid]
                spd = math.hypot(u.vx, u.vy)
                tx, ty, tz, tspd = controller.get_current_target(uid, (u.x, u.y, u.z), spd, sim_time)

                dx, dy, dz = tx - u.x, ty - u.y, tz - u.z
                dist_2d = math.hypot(dx, dy)
                pref_spd = min(tspd, max(4.0, dist_2d))
                vx_cmd = (dx / dist_2d) * pref_spd if dist_2d > 1.0 else 0.0
                vy_cmd = (dy / dist_2d) * pref_spd if dist_2d > 1.0 else 0.0
                vz_cmd = max(-2.5, min(2.5, dz * 0.8))

                physics.step_uav(u, (vx_cmd, vy_cmd, vz_cmd), dt, wind)

                # Assert positions remain finite and valid
                self.assertFalse(math.isnan(u.x))
                self.assertFalse(math.isnan(u.y))
                self.assertFalse(math.isnan(u.z))

        # Check that drone has moved closer to hub
        u1 = uavs["UAV-01"]
        start_dist = 260.0
        current_dist = math.hypot(u1.x, u1.y)
        self.assertLess(current_dist, start_dist, "UAV-01 should have flown inbound along its corridor")


class TestStrategicUTMPreFlightFAS(unittest.TestCase):
    """
    Unit tests asserting strict compliance with the Strategic UTM
    Pre-Flight Flight Authorisation Service (FAS) architectural model.
    """

    def test_fas_zero_preflight_overlaps(self):
        """Assert zero spacetime overlapping reservations in ReservationTable across all authorized drones."""
        controller = StrategicController(num_drones=10)
        drone_ids = [f"UAV-{i+1:02d}" for i in range(10)]
        controller.initialize_scenario(drone_ids, sim_time=0.0)

        table = controller.table
        buf = table.safety_buffer

        for r_id, r_list in table._table.items():
            conflicts = table.conflict_map.get(r_id, {r_id})
            for c_id in conflicts:
                c_list = table._table.get(c_id, [])
                for r1 in r_list:
                    for r2 in c_list:
                        if r1.drone_id == r2.drone_id:
                            continue
                        # Check altitude overlap
                        if not r1.is_vertical_transition and not r2.is_vertical_transition:
                            alt_overlap = abs(r1.alt_m - r2.alt_m) < 12.0
                        else:
                            alt_overlap = (r1.alt_max >= r2.alt_min and r1.alt_min <= r2.alt_max)

                        if alt_overlap:
                            time_overlap = (r1.t_entry < r2.t_exit + buf) and (r1.t_exit > r2.t_entry - buf)
                            self.assertFalse(
                                time_overlap,
                                f"Pre-flight slot overlap between {r1.drone_id} and {r2.drone_id} on {r_id}/{c_id}!",
                            )

    def test_fas_ground_delay_absorption(self):
        """Assert departure delay absorption on the launch pad when corridor/hub layers are occupied."""
        controller = StrategicController(num_drones=2)
        # UAV-01 requests departure at t=0.0 on a fixed route
        req1 = FlightAuthorisationRequest(
            drone_id="UAV-01",
            origin_node="PERIM_0",
            destination_node="PERIM_0",
            desired_departure_time=0.0,
            target_layer=90.0,
            route_edges=["E_0_A", "E_0_B", "E_0_C", "E_0_D"],
        )
        plan1 = controller.planner.plan_preflight_mission(req1)
        self.assertIsNotNone(plan1)
        controller.planner.commit(plan1)
        controller.plans["UAV-01"] = plan1

        # UAV-02 requests identical departure slot and route at the same time
        req2 = FlightAuthorisationRequest(
            drone_id="UAV-02",
            origin_node="PERIM_0",
            destination_node="PERIM_0",
            desired_departure_time=0.0,
            target_layer=90.0,
            route_edges=["E_0_A", "E_0_B", "E_0_C", "E_0_D"],
        )
        plan2 = controller.planner.plan_preflight_mission(req2)
        self.assertIsNotNone(plan2)
        controller.planner.commit(plan2)
        controller.plans["UAV-02"] = plan2

        # Assert UAV-02 has been allocated a ground delay rather than in-flight hover
        self.assertGreater(plan2.scheduled_departure, 0.0)
        self.assertGreater(plan2.ground_delay_s, 0.0)

        # Before scheduled departure, UAV-02 holds velocity 0.0 on ground launch pad
        tx, ty, tz, spd = controller.get_current_target("UAV-02", (260.0, 0.0, -90.0), vel_magnitude=0.0, sim_time=0.5)
        self.assertEqual(spd, 0.0)

        # After scheduled departure, UAV-02 receives nominal cruise speed
        tx, ty, tz, spd = controller.get_current_target(
            "UAV-02", (260.0, 0.0, -90.0), vel_magnitude=0.0, sim_time=plan2.scheduled_departure + 0.1
        )
        self.assertGreater(spd, 0.0)

    def test_fas_vertical_transition_deconfliction(self):
        """Assert safe vertical layer transitions protect intermediate layers from crossing horizontal traffic."""
        controller = StrategicController(num_drones=2)
        # UAV-01 reserves a vertical climb transition at HUB_IN_0 from 90m to 120m spanning [86m, 124m]
        v_res = EdgeReservation(
            drone_id="UAV-01",
            t_entry=10.0,
            t_exit=22.0,
            speed=2.5,
            alt_m=105.0,
            alt_band_m=19.0,
            resource_id="TRANS_HUB_IN_0",
            is_vertical_transition=True,
            transition_node="HUB_IN_0",
        )
        controller.table.reserve("TRANS_HUB_IN_0", v_res)

        # UAV-02 attempts horizontal flight on corridor edge entering HUB_IN_0 at intermediate layer 105m
        free_105 = controller.table.is_free("E_0_A", t_entry=12.0, t_exit=18.0, alt_m=105.0)
        self.assertFalse(free_105, "Intermediate layer 105m must be blocked during vertical transition!")

        # Altitude outside the vertical transition column (150m) must be permitted
        free_150 = controller.table.is_free("E_0_A", t_entry=12.0, t_exit=18.0, alt_m=150.0)
        self.assertTrue(free_150, "Corridor outside vertical transition column band must remain free!")

        # After vertical transition ends plus safety buffer (> 27.0s), intermediate layer is free
        free_after = controller.table.is_free("E_0_A", t_entry=28.0, t_exit=35.0, alt_m=105.0)
        self.assertTrue(free_after, "Intermediate layer must be free after vertical transition completes!")


if __name__ == "__main__":
    unittest.main(verbosity=2)
