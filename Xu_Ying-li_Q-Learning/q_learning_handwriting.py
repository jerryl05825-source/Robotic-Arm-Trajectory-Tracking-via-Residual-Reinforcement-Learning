"""论文 III 节的表格 Q-learning 仿真复现；具体补充假设见 README.md。"""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
import math
import numpy as np
from writing_rules import PROFILE, standard_strokes

L1, L2 = 0.105, 0.1137


def forward(elbow: float, shoulder: float) -> np.ndarray:
    """输入 (RER, RSR)，弧度；输出肩关节坐标系中的米制坐标。"""
    return np.array([L1 * math.cos(shoulder) + L2 * math.cos(shoulder + elbow),
                     L1 * math.sin(shoulder) + L2 * math.sin(shoulder + elbow)])


def inverse(x: float, y: float) -> tuple[float, float]:
    c = (x*x + y*y - L1*L1 - L2*L2) / (2*L1*L2)
    if not -1-1e-10 <= c <= 1+1e-10:
        raise ValueError(f'不可达坐标: {(x, y)}')
    elbow = math.acos(np.clip(c, -1, 1))
    # atan2 保留象限和符号；避免论文 arccos 写法的分支歧义。
    shoulder = math.atan2(y, x) - math.atan2(L2*math.sin(elbow), L1+L2*math.cos(elbow))
    if not (math.radians(2) < elbow < math.radians(88.5)
            and math.radians(-76) < shoulder < math.radians(18)):
        raise ValueError(f'超出论文关节范围: E={math.degrees(elbow)}, S={math.degrees(shoulder)}')
    return elbow, shoulder


def load_strokes(path: Path) -> list[np.ndarray]:
    data = json.loads(path.read_text(encoding='utf-8'))
    strokes = [np.asarray(stroke, dtype=float) for stroke in data['strokes']]
    if data.get('profile') != PROFILE:
        raise ValueError('严格模式要求已核对的规范模板 profile=uurc-print-v1；当前支持 H、NAO')
    reference = standard_strokes(data.get('text'))
    if (len(strokes) != len(reference) or any(
            s.shape != r.shape or not np.allclose(s, r, atol=1e-10, rtol=0)
            for s, r in zip(strokes, reference))):
        raise ValueError('规范模板被修改：笔顺、方向或路径与指定教材不一致')
    return strokes


class WritingEnv:
    """自由试错的书写仿真。错误动作照常改变物理状态，奖励 -1。

    已获奖励前缀保留；回到前一目标状态后才可继续得分。
    正式执行的硬约束独立由 validate_execution 检查。
    """
    def __init__(self, strokes):
        self.strokes = [np.asarray(s, dtype=float) for s in strokes]
        points, target, actions, details = [], [], [], []
        for stroke_index, stroke in enumerate(self.strokes):
            mapped = np.column_stack((.175+.020*stroke[:, 0], -.040-.030*stroke[:, 1]))
            ids = []
            for p in mapped:
                key = tuple(p)
                if key not in points:
                    points.append(key)
                ids.append(points.index(key))
            target.extend([(ids[0], False)] + [(i, True) for i in ids] + [(ids[-1], False)])
            actions.extend([ids[0]+2, 1] + [i+2 for i in ids[1:]] + [0])
            details.extend([(stroke_index+1, 'move_to_start', 0),
                            (stroke_index+1, 'pen_down', 0)] +
                           [(stroke_index+1, 'draw', i) for i in range(1, len(ids))] +
                           [(stroke_index+1, 'pen_up', len(ids)-1)])
        self.points = np.array(points)
        self.joints = np.array([inverse(*p) for p in points])
        self.target, self.target_actions, self.details = target, actions, details
        self.n_actions = 2 + len(points)
        self.home = target[-1][0]
        self.reset()

    def reset(self):
        self.state = (0, self.home, False)
        return self.state

    def transition(self, state, action):
        k, point, down = state
        if k == len(self.target):
            raise RuntimeError('终止后必须 reset')
        if not 0 <= action < self.n_actions:
            raise ValueError('无效动作')
        previous = self.target[k-1] if k else (self.home, False)
        new_point = point if action < 2 else action-2
        new_down = bool(action) if action < 2 else down
        correct = ((point, down) == previous
                   and action == self.target_actions[k]
                   and (new_point, new_down) == self.target[k])
        nxt = (k+int(correct), new_point, new_down)
        return nxt, (1 if correct else -1), nxt[0] == len(self.target)

    def step(self, action):
        self.state, reward, done = self.transition(self.state, action)
        return self.state, reward, done

    def validate_trace(self, trace, require_complete=False):
        state = (0, self.home, False)
        for old, action, nxt, reward in trace:
            if old != state:
                raise ValueError('轨迹状态不连续')
            expected, expected_reward, _ = self.transition(state, action)
            if nxt != expected or reward != expected_reward:
                raise ValueError('轨迹违反笔顺、落笔方向或抬落笔约束')
            state = nxt
        complete = state[0] == len(self.target)
        if require_complete and not complete:
            raise ValueError('轨迹未完成全部笔画，不得导出为正式书写轨迹')
        return complete

    def validate_execution(self, trace):
        """完整模拟轨迹逐步检查，不可丢弃错误动作后拼出合规轨迹。"""
        self.validate_trace(trace, require_complete=True)
        if len(trace) != len(self.target):
            raise ValueError('正式书写轨迹包含额外尝试或错误动作')
        for k, (old, action, nxt, reward) in enumerate(trace):
            if (action != self.target_actions[k] or nxt[0] != k+1
                    or (nxt[1], nxt[2]) != self.target[k] or reward != 1):
                raise ValueError('正式书写轨迹违反规定笔顺、方向或抬落笔顺序')
        return True


class QLearning:
    def __init__(self, n_actions, alpha=0.5, gamma=0.9, epsilon=0.0, seed=0):
        self.n_actions, self.alpha, self.gamma, self.epsilon = n_actions, alpha, gamma, epsilon
        self.rng = np.random.default_rng(seed)
        self.q = {}

    def values(self, state):
        if state not in self.q:
            self.q[state] = np.zeros(self.n_actions)
        return self.q[state]

    def action(self, state, training=True):
        values = self.values(state)
        if training and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        # ε=0 仍需要处理并列；随机打破并列而不添加额外 ε 探索。
        return int(self.rng.choice(np.flatnonzero(values == values.max())))

    def update(self, state, action, reward, next_state, terminated):
        bootstrap = 0.0 if terminated else self.values(next_state).max()
        values = self.values(state)
        values[action] += self.alpha * (reward + self.gamma*bootstrap-values[action])


def rollout(env, agent, max_steps):
    state = env.reset()
    trace, total = [], 0
    for _ in range(max_steps):
        action = agent.action(state, training=False)
        next_state, reward, done = env.step(action)
        trace.append((state, action, next_state, reward))
        total += reward
        state = next_state
        if done:
            return trace, total, True
    return trace, total, False


def train(env, agent, episodes):
    """每轮恰好尝试 len(target) 次动作，不裁剪或缩放奖励。"""
    horizon = len(env.target)
    history = []
    for episode in range(episodes):
        state, total, done = env.reset(), 0, False
        for step in range(horizon):
            action = agent.action(state)
            next_state, reward, done = env.step(action)
            # 步数限制是 truncation，不是 MDP terminal：仍然 bootstrap。
            agent.update(state, action, reward, next_state, done)
            total += reward
            state = next_state
            if done:
                break
        history.append((episode+1, total, step+1, int(done), state[0]))
    return history


def pyplot(interactive=False):
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/handwriting-matplotlib')
    import matplotlib
    if not interactive:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def save_results(out, env, agent, history, before, after, summary):
    out.mkdir(parents=True, exist_ok=True)
    with (out/'training.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'return', 'steps', 'success', 'achieved'])
        writer.writerows(history)
    env.validate_trace(after)
    try:
        env.validate_execution(after)
        exportable = True
    except ValueError:
        exportable = False
    # 全部评估尝试单独留档，禁止把错误动作过滤后冒充正确书写。
    attempts = [dict(state=list(old), action=int(action), next_state=list(nxt), reward=reward)
                for old, action, nxt, reward in after]
    (out/'evaluation_attempts.json').write_text(json.dumps(attempts, indent=2), encoding='utf-8')
    records = []
    for state, action, next_state, reward in after:
        if not exportable:
            break
        stroke, phase, path_index = env.details[state[0]]
        e, s = env.joints[next_state[1]]
        records.append(dict(action=int(action), point=int(next_state[1]),
                            pen_down=bool(next_state[2]), stroke=stroke, phase=phase,
                            path_index=path_index, x_m=float(env.points[next_state[1], 0]),
                            y_m=float(env.points[next_state[1], 1]), elbow_rad=float(e),
                            shoulder_rad=float(s), reward=reward, achieved=next_state[0]))
    if exportable:
        env.validate_execution(after)
        (out/'trajectory.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    else:
        (out/'trajectory.json').unlink(missing_ok=True)
    (out/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    keys = list(agent.q)
    np.savez_compressed(out/'q_table.npz', states=np.array(keys, dtype=int),
                        q_values=np.stack([agent.q[k] for k in keys]))
    plt = pyplot()
    fig, grid = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    axes = [grid[0, 0], grid[1, 0], grid[1, 1]]
    h = np.array(history)
    axes[0].plot(h[:, 0], h[:, 1], lw=1, label='Episode return')
    axes[0].axhline(len(env.target), color='green', ls='--', label='Optimal return')
    axes[0].set(xlabel='Episode', ylabel='Return', title='Tabular Q-learning')
    axes[0].legend()
    early = h[:min(50, len(h))]
    grid[0, 1].plot(early[:, 0], early[:, 1], 'o-', ms=3)
    grid[0, 1].axhline(len(env.target), color='green', ls='--')
    grid[0, 1].set(xlabel='Episode', ylabel='Return', title='First 50 episodes (unclipped)')
    for ax, trace, title in zip(axes[1:], [before, after], ['Before training (simulation)', 'After training (evaluation)']):
        for stroke in env.strokes:
            p = np.column_stack((0.175+0.020*stroke[:, 0], -0.040-0.030*stroke[:, 1]))
            ax.plot(*p.T, color='lightgray', lw=6)
        if title == 'After training (evaluation)':
            for i, stroke in enumerate(env.strokes):
                p = np.column_stack((.175+.020*stroke[:, 0], -.040-.030*stroke[:, 1]))
                ax.annotate(str(i+1), p[0], xytext=(-7 if i % 2 == 0 else 7, 7),
                            ha='right' if i % 2 == 0 else 'left',
                            textcoords='offset points', color='darkred')
                for j in range(0, len(p)-1, max(1, (len(p)-1)//4)):
                    ax.annotate('', xy=p[j+1], xytext=p[j],
                                arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
        # 绘制所有实际尝试，包括错误落笔移动；不筛掉失败动作。
        for state, action, nxt, reward in trace:
            if state[1] != nxt[1]:
                p = env.points[[state[1], nxt[1]]]
                ax.plot(*p.T, color='navy' if state[2] else 'orange',
                        ls='-' if state[2] else '--', lw=1.5, alpha=0.65)
        ax.set(title=title, xlabel='x (m)', ylabel='y (m)', xlim=(.17, .20), ylim=(-.075, -.035))
        ax.set_aspect('equal')
    fig.savefig(out/'results.png', dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--strokes', type=Path, default=Path(__file__).parent/'examples/NAO.json')
    parser.add_argument('--output', type=Path, default=Path('outputs/NAO'))
    parser.add_argument('--record', type=Path, help='鼠标录入并保存 JSON，然后退出')
    parser.add_argument('--episodes', type=int, default=1500)
    parser.add_argument('--alpha', type=float, default=.5)
    parser.add_argument('--gamma', type=float, default=.9)
    parser.add_argument('--epsilon', type=float, default=0.)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if args.record:
        parser.error('严格模式不接受未经核对的鼠标示范；请使用 examples/H.json 或 examples/NAO.json')
    if args.episodes < 1 or not (0 < args.alpha <= 1 and 0 <= args.gamma < 1 and 0 <= args.epsilon <= 1):
        parser.error('episodes 必须为正；alpha∈(0,1]，gamma∈[0,1)，epsilon∈[0,1]')
    env = WritingEnv(load_strokes(args.strokes))
    baseline = QLearning(env.n_actions, seed=args.seed)
    before, before_return, _ = rollout(env, baseline, len(env.target))
    agent = QLearning(env.n_actions, args.alpha, args.gamma, args.epsilon, args.seed)
    history = train(env, agent, args.episodes)
    after, total, success = rollout(env, agent, len(env.target))
    optimal_episodes = [row[0] for row in history if row[1] == len(env.target) and row[3]]
    try:
        env.validate_execution(after)
        execution_valid, execution_error = True, None
    except ValueError as error:
        execution_valid, execution_error = False, str(error)
    nonoptimal = [row[0] for row in history if row[1] != len(env.target) or not row[3]]
    stable_from = nonoptimal[-1]+1 if nonoptimal else 1
    summary = dict(seed=args.seed, alpha=args.alpha, gamma=args.gamma, epsilon=args.epsilon,
                   episodes=args.episodes, episode_length=len(env.target),
                   reward_min=-len(env.target), reward_max=len(env.target), strokes=len(env.strokes),
                   training_mode='free_exploration', episode_mode='fixed_length', execution_validation='strict',
                   trajectory_complete=env.validate_trace(after),
                   actions=env.n_actions, optimal_return=len(env.target), baseline_return=before_return,
                   baseline_steps=len(before), evaluation_return=total, evaluation_steps=len(after),
                   evaluation_success=success, evaluation_optimal=success and total == len(env.target),
                   trajectory_exported=execution_valid, execution_validation_error=execution_error,
                   observed_optimal_tail_start=stable_from if stable_from <= args.episodes else None,
                   final_training_return=history[-1][1],
                   first_optimal_episode=optimal_episodes[0] if optimal_episodes else None,
                   truncated_episodes=sum(not row[3] for row in history), q_states=len(agent.q))
    save_results(args.output, env, agent, history, before, after, summary)
    print(json.dumps(summary, indent=2))
    print(f'Results saved to {args.output.resolve()}')


if __name__ == '__main__':
    main()
