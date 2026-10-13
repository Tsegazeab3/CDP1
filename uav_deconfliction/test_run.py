#!/usr/bin/env python3
"""
test_run.py
===========
Multi-UAV Deconfliction Test Runner.

Supports:
1. --strategic (default): Runs the comprehensive 4D Strategic Planner (SIPP + Airspace Corridors) test suite.
2. --flight: Runs a 15-second multi-UAV closed-loop flight simulation with SIPP 4D corridors and physical dynamics.
3. --tactical: Runs the legacy tactical BlueSky bridge with real-time HUD and GIF export.
"""

import argparse
import math
import sys
import time
import unittest


def run_strategic_tests():
    """Runs all 18 unit and integration tests for strategic_planner.py."""
    print("\n" + "=" * 70)
    print("  RUNNING STRATEGIC 4D SIPP PLANNER TEST SUITE")
    print("=" * 70 + "\n")
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=".", pattern="test_strategic_planner.py")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


def run_flight_sim(duration_s: float = 15.0):
    """Runs a multi-UAV closed-loop flight simulation using the 4D strategic planner."""
    print("\n" + "=" * 70)
    print(f"  RUNNING CLOSED-LOOP 4D FLIGHT SIMULATION ({duration_s:.1f}s)")
    print("=" * 70 + "\n")

    from quadcopter_dynamics import DynamicUAVState, QuadcopterPhysicsSimulator
    from strategic_planner import StrategicController

    physics = QuadcopterPhysicsSimulator()
    controller = StrategicController(num_drones=10)
    drone_ids = [f"UAV-{i+1:02d}" for i in range(10)]
    controller.initialize_scenario(drone_ids, sim_time=0.0)

    uavs = {}
    radius = 260.0
    for i, uid in enumerate(drone_ids):
        angle = i * (2.0 * math.pi / 10)
        alt_m = 90.0 if (i % 2 == 0) else 105.0
        uavs[uid] = DynamicUAVState(
            id=uid,
            x=radius * math.cos(angle),
            y=radius * math.sin(angle),
            z=-alt_m,
            color="cyan",
        )

    dt = 0.033
    sim_time = 0.0
    total_steps = int(duration_s / dt)

    for step in range(total_steps):
        sim_time += dt
        wind = physics.update_gust(dt)

        for uid in drone_ids:
            u = uavs[uid]
            spd = math.hypot(u.vx, u.vy)
            tx, ty, tz, tspd = controller.get_current_target(uid, (u.x, u.y, u.z), spd, sim_time)

            dx, dy, dz = tx - u.x, ty - u.y, tz - u.z
            dist_2d = math.hypot(dx, dy)
            if tspd <= 0.0:
                pref_spd = 0.0
            else:
                pref_spd = min(tspd, max(4.0, dist_2d))
            vx_cmd = (dx / dist_2d) * pref_spd if (dist_2d > 1.0 and pref_spd > 0.0) else 0.0
            vy_cmd = (dy / dist_2d) * pref_spd if (dist_2d > 1.0 and pref_spd > 0.0) else 0.0
            vz_cmd = max(-2.5, min(2.5, dz * 0.8)) if pref_spd > 0.0 else 0.0

            physics.step_uav(u, (vx_cmd, vy_cmd, vz_cmd), dt, wind)

        if step % 90 == 0:
            u1 = uavs["UAV-01"]
            plan1 = controller.plans["UAV-01"]
            curr_edge = plan1.legs[plan1.current_leg % len(plan1.legs)].edge_id
            print(f"[{sim_time:4.1f}s] UAV-01 pos=({u1.x:5.1f}, {u1.y:5.1f}, {u1.z:5.1f})m | corridor={curr_edge}")

    print(f"\n[TestRunner] Flight simulation completed successfully ({duration_s:.1f}s).")
    return True


def run_tactical_node():
    """Runs the legacy tactical BlueSky bridge with real-time HUD and GIF export."""
    from traffic_bridge import TrafficBridgeNode

    print("[TestRunner] Starting TrafficBridgeNode with BlueSky stream and visual HUD...")
    node = TrafficBridgeNode(
        stream_mode="bluesky",
        waypoint_north=300.0,
        waypoint_east=0.0,
        visual=True,
        gif_output="/home/t/.gemini/antigravity-cli/brain/9de83ef3-1975-4c27-84d3-931e95175b4e/bluesky_deconfliction.gif",
    )
    start_time = time.time()

    def check_timeout():
        while time.time() - start_time < 15.0:
            time.sleep(0.5)
        print("[TestRunner] 15-second test duration reached. Stopping node...")
        node.running = False

    import threading

    t = threading.Thread(target=check_timeout, daemon=True)
    t.start()

    node.start()
    print("[TestRunner] Node halted successfully.")
    return True


def main():
    parser = argparse.ArgumentParser(description="Multi-UAV Deconfliction Test Suite")
    parser.add_argument("--flight", action="store_true", help="Run 15-second closed-loop 4D flight simulation")
    parser.add_argument("--tactical", action="store_true", help="Run tactical BlueSky bridge HUD")
    parser.add_argument("--duration", type=float, default=15.0, help="Flight sim duration in seconds (default: 15)")
    args = parser.parse_args()

    if args.tactical:
        success = run_tactical_node()
    elif args.flight:
        success = run_flight_sim(duration_s=args.duration)
    else:
        # Default: run all strategic unit tests, then a quick flight verification
        success = run_strategic_tests()
        if success:
            run_flight_sim(duration_s=5.0)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
