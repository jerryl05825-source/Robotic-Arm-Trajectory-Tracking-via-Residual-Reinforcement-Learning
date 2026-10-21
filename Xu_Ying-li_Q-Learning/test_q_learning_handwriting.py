"""验证奖励、抬落笔、终止目标和运动学；运行 python -m unittest -v。"""
import unittest
from pathlib import Path
import numpy as np
from q_learning_handwriting import WritingEnv, QLearning, forward, inverse, load_strokes, train, rollout

ROOT = Path(__file__).parent


class HandwritingTests(unittest.TestCase):
    def setUp(self):
        self.env = WritingEnv(load_strokes(ROOT/'examples/H.json'))

    def test_kinematics(self):
        for e in np.radians([5, 30, 60, 85]):
            for s in np.radians([-70, -30, 0, 15]):
                np.testing.assert_allclose(inverse(*forward(e, s)), [e, s], atol=1e-12)
        with self.assertRaises(ValueError):
            inverse(1, 1)

    def test_correct_sequence(self):
        self.assertEqual(self.env.n_actions, 8)
        self.assertEqual(len(self.env.target), 12)
        state = self.env.reset()
        for i, (point, down) in enumerate(self.env.target):
            action = point+2 if state[1] != point else int(down)
            state, reward, done = self.env.step(action)
            self.assertEqual(reward, 1)
            self.assertEqual(state[0], i+1)
            self.assertEqual(done, i == 11)
        with self.assertRaises(RuntimeError):
            self.env.step(0)

    def test_wrong_actions_change_simulated_position_and_pen(self):
        env = self.env
        old = env.reset()
        nxt, reward, done = env.step(1)  # 在错误起点落笔：仍实际落笔
        self.assertTrue(nxt[2])
        self.assertEqual(nxt[0], 0)
        self.assertEqual(reward, -1)
        wrong_point = next(i for i in range(len(env.points))
                           if i != old[1] and i != env.target[0][0])
        nxt, reward, _ = env.step(wrong_point+2)
        self.assertEqual(nxt, (0, wrong_point, True))
        self.assertEqual(reward, -1)
        # 进度不重置，回到前一目标且抬笔，才能开始正确序列。
        env.step(0)
        env.step(env.home+2)
        nxt, reward, _ = env.step(env.target_actions[0])
        self.assertEqual((nxt[0], reward), (1, 1))

    def test_error_then_recovery_cannot_be_exported(self):
        env = self.env
        trace = []
        actions = [1, 0] + env.target_actions  # 错误落笔后恢复，随后全部正确
        for action in actions:
            old = env.state
            nxt, reward, _ = env.step(action)
            trace.append((old, action, nxt, reward))
        self.assertTrue(env.validate_trace(trace, require_complete=True))
        with self.assertRaises(ValueError):
            env.validate_execution(trace)
        # 截取部分正确动作仍然不完整，不可作为正式轨迹。
        with self.assertRaises(ValueError):
            env.validate_execution([step for step in trace if step[3] == 1][1:])

    def test_full_trace_validation(self):
        env = self.env
        trace = []
        for action in env.target_actions:
            old = env.state
            nxt, reward, _ = env.step(action)
            trace.append((old, action, nxt, reward))
        self.assertTrue(env.validate_trace(trace, require_complete=True))
        with self.assertRaises(ValueError):
            env.validate_trace(trace[:-1], require_complete=True)
        self.assertTrue(env.validate_execution(trace))
        old, action, nxt, reward = trace[2]
        bad = trace.copy()
        bad[2] = (old, 0, nxt, reward)  # 落笔中提前抬笔的伪造记录
        with self.assertRaises(ValueError):
            env.validate_trace(bad, require_complete=True)

    def test_modified_standard_template_rejected(self):
        import tempfile
        import json
        data = json.loads((ROOT/'examples/NAO.json').read_text())
        variants = []
        reversed_first = json.loads(json.dumps(data))
        reversed_first['strokes'][0].reverse()
        variants.append(reversed_first)
        swapped = json.loads(json.dumps(data))
        swapped['strokes'][0], swapped['strokes'][1] = swapped['strokes'][1], swapped['strokes'][0]
        variants.append(swapped)
        for variant in variants:
            with tempfile.TemporaryDirectory() as d:
                path = Path(d)/'bad.json'
                path.write_text(json.dumps(variant))
                with self.assertRaises(ValueError):
                    load_strokes(path)

    def test_o_is_four_corner_closed_stroke(self):
        strokes = load_strokes(ROOT/'examples/NAO.json')
        self.assertEqual(len(strokes), 6)
        circle = strokes[-1]
        np.testing.assert_equal(circle[0], circle[-1])
        self.assertEqual(len(np.unique(circle, axis=0)), 4)
        np.testing.assert_equal(circle, [[.70,.15],[.70,.85],[.93,.85],[.93,.15],[.70,.15]])
        # 屏幕 y 向下：逆时针有向面积为负。
        area = np.sum(circle[:-1,0]*circle[1:,1]-circle[1:,0]*circle[:-1,1])
        self.assertLess(area, 0)

    def test_terminal_update(self):
        a = QLearning(2)
        a.values((1,))[0] = 100
        a.update((0,), 0, 1, (1,), True)
        self.assertEqual(a.values((0,))[0], .5)

    def test_fixed_length_attempts_and_raw_reward_bounds(self):
        for name in ['H', 'NAO']:
            env = WritingEnv(load_strokes(ROOT/'examples'/f'{name}.json'))
            n = len(env.target)
            agent = QLearning(env.n_actions, seed=0)
            history = train(env, agent, 100)
            for episode, reward, steps, success, achieved in history:
                self.assertEqual(steps, n)
                self.assertTrue(-n <= reward <= n)
                self.assertEqual(reward, 2*achieved-n)
                self.assertEqual(bool(success), achieved == n)

    def test_fixed_budget_failure_resets_and_keeps_bootstrap(self):
        class AlwaysPenUp(QLearning):
            def action(self, state, training=True):
                return 0
        agent = AlwaysPenUp(self.env.n_actions)
        history = train(self.env, agent, 3)
        self.assertEqual(history, [(1, -12, 12, 0, 0), (2, -12, 12, 0, 0),
                                   (3, -12, 12, 0, 0)])
        # 时间预算耗尽不是实际终止；非终止更新保留后续价值。
        agent.values((9,))[0] = 10
        agent.update((8,), 0, -1, (9,), False)
        self.assertEqual(agent.values((8,))[0], 4)

    def test_h_learns_without_epsilon_exploration(self):
        a = QLearning(self.env.n_actions, seed=0)
        train(self.env, a, episodes=500)
        trace, reward, success = rollout(self.env, a, 100)
        self.assertTrue(success)
        self.assertEqual(reward, 12)
        self.assertEqual(len(trace), 12)


if __name__ == '__main__':
    unittest.main()
