"""可在无显示器环境运行的核心运动学测试。"""

import math
import unittest

from nao_right_arm_2d import GEOMETRY, LIMITS, ViaPoint, check_joint_limits, evenly_timed_via_points, forward_kinematics, inverse_kinematics, plan_trajectory, rotate_view_coordinates, unrotate_view_coordinates


START, END = (195.0, -145.0), (160.0, -35.0)


class KinematicsTests(unittest.TestCase):
    def test_view_rotation_quarters_and_inverse(self):
        point = (12.5, -34.0)
        self.assertEqual(rotate_view_coordinates(*point, 0), (12.5, -34.0))
        self.assertEqual(rotate_view_coordinates(*point, 1), (-34.0, -12.5))
        self.assertEqual(rotate_view_coordinates(*point, 2), (-12.5, 34.0))
        self.assertEqual(rotate_view_coordinates(*point, 3), (34.0, 12.5))
        self.assertEqual(rotate_view_coordinates(*point, 4), point)
        for quarter in range(8):
            displayed = rotate_view_coordinates(*point, quarter)
            recovered = unrotate_view_coordinates(*displayed, quarter)
            self.assertAlmostEqual(recovered[0], point[0])
            self.assertAlmostEqual(recovered[1], point[1])

    def test_straight_arm_matches_drawing_dimensions(self):
        pose = forward_kinematics(0.0, 0.0)
        self.assertEqual(pose.shoulder, (0.0, -113.0))
        self.assertAlmostEqual(pose.elbow[0], 105.0)
        self.assertAlmostEqual(pose.wrist[0], 160.95)
        self.assertAlmostEqual(pose.hand[0], 218.7)
        self.assertAlmostEqual(pose.hand[1], -113.0)

    def test_inverse_round_trip_both_branches(self):
        for q1, q2 in ((0.3, 0.8), (0.9, -1.2), (-1.8, 2.0)):
            with self.subTest(q1=q1, q2=q2):
                target = forward_kinematics(q1, q2).hand
                solved = inverse_kinematics(*target, elbow_sign=1 if q2 > 0 else -1)
                actual = forward_kinematics(*solved).hand
                self.assertAlmostEqual(actual[0], target[0], places=8)
                self.assertAlmostEqual(actual[1], target[1], places=8)
                self.assertEqual(math.copysign(1, solved[1]), math.copysign(1, q2))

    def test_cartesian_path_is_straight_and_stops_at_both_ends(self):
        start, end = START, END
        samples = plan_trajectory(start, end, 4.0)
        self.assertEqual((samples[0].x, samples[0].y), start)
        self.assertEqual((samples[-1].x, samples[-1].y), end)
        self.assertEqual((samples[0].vx, samples[0].vy), (0.0, 0.0))
        self.assertEqual((samples[-1].vx, samples[-1].vy), (0.0, 0.0))
        self.assertLessEqual(
            max(math.dist((a.x, a.y), (b.x, b.y)) for a, b in zip(samples, samples[1:])),
            1.00001,
        )
        dx, dy = end[0] - start[0], end[1] - start[1]
        for sample in samples:
            self.assertAlmostEqual((sample.x - start[0]) * dy - (sample.y - start[1]) * dx, 0.0, places=8)
            check_joint_limits(math.radians(sample.shoulder_roll_deg), math.radians(sample.elbow_roll_deg))
            hand = forward_kinematics(math.radians(sample.shoulder_roll_deg), math.radians(sample.elbow_roll_deg)).hand
            self.assertAlmostEqual(hand[0], sample.x, places=8)
            self.assertAlmostEqual(hand[1], sample.y, places=8)

    def test_joint_path_endpoints_and_nonstraight_midpoint(self):
        start, end = START, END
        samples = plan_trajectory(start, end, 4.0, mode="joint")
        self.assertAlmostEqual(samples[0].x, start[0])
        self.assertAlmostEqual(samples[-1].y, end[1])
        midpoint = samples[len(samples) // 2]
        cross = (midpoint.x - start[0]) * (end[1] - start[1]) - (midpoint.y - start[1]) * (end[0] - start[0])
        self.assertGreater(abs(cross), 1.0)
        for sample in samples:
            self.assertGreaterEqual(sample.shoulder_roll_deg, LIMITS.shoulder_roll_min_deg - 1e-8)
            self.assertLessEqual(sample.elbow_roll_deg, LIMITS.elbow_roll_max_deg + 1e-8)

    def test_limit_boundaries_and_outside_targets(self):
        check_joint_limits(math.radians(-70), math.radians(2))
        check_joint_limits(math.radians(18), math.radians(88.5))
        for angles, joint_name in (
            ((-71, 45), "RShoulderRoll"),
            ((19, 45), "RShoulderRoll"),
            ((-20, 1), "RElbowRoll"),
            ((-20, 89), "RElbowRoll"),
        ):
            with self.subTest(angles=angles), self.assertRaisesRegex(ValueError, joint_name):
                point = forward_kinematics(*map(math.radians, angles)).hand
                plan_trajectory(point, point, 2.0)

    def test_cartesian_segment_can_violate_limits_with_valid_endpoints(self):
        start = forward_kinematics(math.radians(10), math.radians(85)).hand
        end = forward_kinematics(math.radians(-65), math.radians(10)).hand
        with self.assertRaisesRegex(ValueError, "RElbowRoll"):
            plan_trajectory(start, end, 2.0, mode="cartesian")
        self.assertGreater(len(plan_trajectory(start, end, 2.0, mode="joint")), 2)

    def test_two_via_points_are_reached_without_stopping(self):
        vias = [ViaPoint(1.4, 190, -110), ViaPoint(2.8, 175, -70)]
        for mode in ("cartesian", "joint"):
            with self.subTest(mode=mode):
                samples = plan_trajectory(START, END, 4.0, mode=mode, via_points=vias)
                marked = [s for s in samples if s.waypoint]
                self.assertEqual([s.waypoint for s in marked], ["起", "V1", "V2", "终"])
                for sample, target in zip(marked[1:3], vias):
                    self.assertEqual(sample.t, target.t)
                    self.assertAlmostEqual(sample.x, target.x, places=8)
                    self.assertAlmostEqual(sample.y, target.y, places=8)
                    self.assertGreater(math.hypot(sample.vx, sample.vy), 5.0)
                self.assertEqual((marked[0].vx, marked[0].vy), (0.0, 0.0))
                self.assertEqual((marked[-1].vx, marked[-1].vy), (0.0, 0.0))
                self.assertLessEqual(max(math.dist((a.x, a.y), (b.x, b.y)) for a, b in zip(samples, samples[1:])), 1.00001)
                for sample in samples:
                    check_joint_limits(math.radians(sample.shoulder_roll_deg), math.radians(sample.elbow_roll_deg))

    def test_via_times_and_positions_are_validated(self):
        for points, message in (
            ([ViaPoint(0, 190, -110)], "间隔"),
            ([ViaPoint(2.5, 190, -110), ViaPoint(2.0, 175, -70)], "严格递增"),
            ([ViaPoint(4.0, 190, -110)], "间隔"),
            ([ViaPoint(2.0, float("nan"), -110)], "有限数字"),
            ([ViaPoint(2.0, 500, 500)], "V1"),
        ):
            with self.subTest(points=points), self.assertRaisesRegex(ValueError, message):
                plan_trajectory(START, END, 4.0, via_points=points)

    def test_clicked_via_points_receive_even_times(self):
        positions = [(190, -110), (175, -70), (165, -50)]
        points = evenly_timed_via_points(positions, 4.0)
        self.assertEqual(points, [
            ViaPoint(1.0, 190, -110),
            ViaPoint(2.0, 175, -70),
            ViaPoint(3.0, 165, -50),
        ])
        with self.assertRaisesRegex(ValueError, "至少需要 0.1"):
            evenly_timed_via_points(positions, 0.3)
        with self.assertRaisesRegex(ValueError, "最多支持 12"):
            evenly_timed_via_points([(180, -100)] * 13, 4.0)

    def test_reject_unreachable_point_and_line_through_inner_hole(self):
        with self.assertRaisesRegex(ValueError, "不可达"):
            inverse_kinematics(GEOMETRY.outer_radius + 1, -113.0)
        with self.assertRaisesRegex(ValueError, "不可达区域"):
            plan_trajectory((-20, -113), (20, -113), 2.0, mode="cartesian")
        with self.assertRaisesRegex(ValueError, "关节超限"):
            plan_trajectory((-20, -113), (20, -113), 2.0, mode="joint")


if __name__ == "__main__":
    unittest.main()
