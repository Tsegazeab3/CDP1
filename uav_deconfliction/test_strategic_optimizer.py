"""Test and Ablation Benchmark for Strategic Deconfliction Algorithms.

Evaluates D4orm (diffusion-denoising), MPPI, CEM, and Straight-Line baselines on
a multi-UAV intersection scenario and exports BlueSky scenario files.
"""

import os
import unittest
import numpy as np

from strategic_optimizer import StrategicTrajectoryOptimizer


class TestStrategicOptimizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.optimizer = StrategicTrajectoryOptimizer(
            separation_radius_m=20.0,
            max_speed_ms=15.0,
            max_accel_ms2=4.0,
            dt=0.2,
            horizon_steps=40,
            num_samples=64,
            num_diffuse_steps=15,
            max_iterations=4,
        )

        # 4-UAV crossing scenario at identical altitude (Z=50m) to force conflict resolution
        cls.starts = np.array([
            [-100.0,    0.0, 50.0, 0.0, 0.0, 0.0],  # West -> East
            [ 100.0,    0.0, 50.0, 0.0, 0.0, 0.0],  # East -> West
            [   0.0, -100.0, 50.0, 0.0, 0.0, 0.0],  # South -> North
            [   0.0,  100.0, 50.0, 0.0, 0.0, 0.0],  # North -> South
        ])
        cls.goals = np.array([
            [ 100.0,    0.0, 50.0, 0.0, 0.0, 0.0],
            [-100.0,    0.0, 50.0, 0.0, 0.0, 0.0],
            [   0.0,  100.0, 50.0, 0.0, 0.0, 0.0],
            [   0.0, -100.0, 50.0, 0.0, 0.0, 0.0],
        ])

    def test_d4orm_planner(self):
        result = self.optimizer.plan(self.starts, self.goals, method="d4orm", seed=42)
        self.assertEqual(result.method, "d4orm")
        self.assertEqual(result.trajectories.shape, (40, 4, 6))
        self.assertGreater(result.min_separation_m, 0.0)
        print(f"\n[D4orm Test] Min separation: {result.min_separation_m:.2f} m | Compute: {result.compute_time_s:.2f} s")

    def test_mppi_planner(self):
        result = self.optimizer.plan(self.starts, self.goals, method="mppi", seed=42)
        self.assertEqual(result.method, "mppi")
        self.assertEqual(result.trajectories.shape, (40, 4, 6))
        print(f"[MPPI Test]  Min separation: {result.min_separation_m:.2f} m | Compute: {result.compute_time_s:.2f} s")

    def test_cem_planner(self):
        result = self.optimizer.plan(self.starts, self.goals, method="cem", seed=42)
        self.assertEqual(result.method, "cem")
        self.assertEqual(result.trajectories.shape, (40, 4, 6))
        print(f"[CEM Test]   Min separation: {result.min_separation_m:.2f} m | Compute: {result.compute_time_s:.2f} s")

    def test_straight_line_baseline(self):
        result = self.optimizer.plan(self.starts, self.goals, method="straight")
        self.assertEqual(result.method, "straight")
        # In an uncoordinated crossing, straight line causes near-zero separation (collision)
        self.assertLess(result.min_separation_m, 20.0)
        print(f"[Straight]   Min separation: {result.min_separation_m:.2f} m | Conflicts: {result.conflict_count}")

    def test_export_bluesky_scenario(self):
        result = self.optimizer.plan(self.starts, self.goals, method="d4orm", seed=42)
        out_scn = "/tmp/test_d4orm_scenario.scn"
        self.optimizer.export_bluesky_scenario(result, out_scn)
        self.assertTrue(os.path.exists(out_scn))
        with open(out_scn) as f:
            content = f.read()
        self.assertIn("CRE UAV01", content)
        self.assertIn("ADDWPT UAV01", content)
        print(f"[BlueSky Export] Successfully created scenario file at {out_scn}")


def run_ablation_comparison():
    """Runs a direct ablation comparison between methods and prints a formatted summary table."""
    opt = StrategicTrajectoryOptimizer(
        separation_radius_m=20.0,
        max_speed_ms=15.0,
        max_accel_ms2=4.0,
        dt=0.2,
        horizon_steps=40,
        num_samples=64,
        num_diffuse_steps=15,
        max_iterations=4,
    )

    starts = np.array([
        [-100.0,    0.0, 50.0, 0.0, 0.0, 0.0],
        [ 100.0,    0.0, 50.0, 0.0, 0.0, 0.0],
        [   0.0, -100.0, 50.0, 0.0, 0.0, 0.0],
        [   0.0,  100.0, 50.0, 0.0, 0.0, 0.0],
    ])
    goals = np.array([
        [ 100.0,    0.0, 50.0, 0.0, 0.0, 0.0],
        [-100.0,    0.0, 50.0, 0.0, 0.0, 0.0],
        [   0.0,  100.0, 50.0, 0.0, 0.0, 0.0],
        [   0.0, -100.0, 50.0, 0.0, 0.0, 0.0],
    ])

    methods = ["straight", "cem", "mppi", "d4orm"]
    print("\n" + "=" * 80)
    print("STRATEGIC DECONFLICTION ABLATION STUDY (4-UAV Intersection Benchmark)")
    print("=" * 80)
    print(f"{'Method':<12} | {'Min Sep (m)':<12} | {'Conflicts':<10} | {'Mean Speed':<12} | {'Compute Time':<12}")
    print("-" * 80)

    for m in methods:
        res = opt.plan(starts, goals, method=m, seed=42)
        print(
            f"{res.method:<12} | {res.min_separation_m:<12.2f} | {res.conflict_count:<10d} | "
            f"{res.mean_speed_ms:<10.2f} m/s | {res.compute_time_s:<10.3f} s"
        )
    print("=" * 80 + "\n")


if __name__ == "__main__":
    unittest.main(exit=False)
    run_ablation_comparison()
