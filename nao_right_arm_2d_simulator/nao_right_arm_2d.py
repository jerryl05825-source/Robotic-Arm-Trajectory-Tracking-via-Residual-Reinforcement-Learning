"""NAO 右臂俯视图：双转动关节的运动学与轨迹规划模拟器。

运行：python nao_right_arm_2d.py
单位：位置 mm，角度 deg，时间 s。仅需 Python 标准库。
"""

from __future__ import annotations

import csv
import math
import tkinter as tk
import tkinter.font as tkfont
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk


@dataclass(frozen=True)
class ArmGeometry:
    # 图 2-1：O0 到肩关节 98 + 15 mm；肩-肘 105 mm。
    shoulder_x: float = 0.0
    shoulder_y: float = -113.0
    upper_arm: float = 105.0
    elbow_to_wrist: float = 55.95
    wrist_to_hand: float = 57.75

    @property
    def forearm_and_hand(self) -> float:
        return self.elbow_to_wrist + self.wrist_to_hand

    @property
    def inner_radius(self) -> float:
        return abs(self.upper_arm - self.forearm_and_hand)

    @property
    def outer_radius(self) -> float:
        return self.upper_arm + self.forearm_and_hand


GEOMETRY = ArmGeometry()


@dataclass(frozen=True)
class JointLimits:
    # 用户所附表 3.1；图 3-2 的肩部下限另标 -76°，这里采用较保守的 -70°。
    shoulder_roll_min_deg: float = -70.0
    shoulder_roll_max_deg: float = 18.0
    elbow_roll_min_deg: float = 2.0
    elbow_roll_max_deg: float = 88.5


LIMITS = JointLimits()
UI_FONT_CANDIDATES = ("Noto Sans CJK SC", "Microsoft YaHei", "WenQuanYi Micro Hei", "PingFang SC")
UI_FONT_SIZE = 13


@dataclass(frozen=True)
class Pose:
    shoulder: tuple[float, float]
    elbow: tuple[float, float]
    wrist: tuple[float, float]
    hand: tuple[float, float]


@dataclass(frozen=True)
class Sample:
    t: float
    x: float
    y: float
    shoulder_roll_deg: float
    elbow_roll_deg: float
    vx: float = 0.0
    vy: float = 0.0
    waypoint: str = ""


@dataclass(frozen=True)
class ViaPoint:
    t: float
    x: float
    y: float


def evenly_timed_via_points(
    positions: tuple[tuple[float, float], ...] | list[tuple[float, float]], duration: float
) -> list[ViaPoint]:
    """按点击顺序将 VIA 点等间隔分配在起止时间之间。"""
    if len(positions) > 12:
        raise ValueError("最多支持 12 个 VIA POINTS")
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("运动时间必须为正数")
    interval = duration / (len(positions) + 1)
    if interval < 0.1 - 1e-9:
        raise ValueError("VIA POINTS 过多或运动时间过短；自动分配后每段至少需要 0.1 s")
    return [ViaPoint(interval * index, x, y) for index, (x, y) in enumerate(positions, 1)]


def rotate_view_coordinates(x: float, y: float, clockwise_quarters: int) -> tuple[float, float]:
    """绕 O₀ 将世界坐标顺时针旋转若干个 90°，只用于显示。"""
    quarter = clockwise_quarters % 4
    if quarter == 0:
        return x, y
    if quarter == 1:
        return y, -x
    if quarter == 2:
        return -x, -y
    return -y, x


def unrotate_view_coordinates(x: float, y: float, clockwise_quarters: int) -> tuple[float, float]:
    """撤销显示旋转，用于把鼠标像素位置还原为机器人坐标。"""
    return rotate_view_coordinates(x, y, -clockwise_quarters)


def forward_kinematics(q1: float, q2: float, geom: ArmGeometry = GEOMETRY) -> Pose:
    """q1=RShoulderRoll，q2=RElbowRoll；在俯视图中正角均逆时针。"""
    c1, s1 = math.cos(q1), math.sin(q1)
    c12, s12 = math.cos(q1 + q2), math.sin(q1 + q2)
    sx, sy = geom.shoulder_x, geom.shoulder_y
    elbow = (sx + geom.upper_arm * c1, sy + geom.upper_arm * s1)
    wrist = (
        elbow[0] + geom.elbow_to_wrist * c12,
        elbow[1] + geom.elbow_to_wrist * s12,
    )
    hand = (
        wrist[0] + geom.wrist_to_hand * c12,
        wrist[1] + geom.wrist_to_hand * s12,
    )
    return Pose((sx, sy), elbow, wrist, hand)


def inverse_kinematics(
    x: float, y: float, elbow_sign: int = 1, geom: ArmGeometry = GEOMETRY
) -> tuple[float, float]:
    """返回两组数学解之一；右臂实物限位只允许 q2 正分支。"""
    if elbow_sign not in (-1, 1):
        raise ValueError("肘部分支必须是 +1 或 -1")
    dx, dy = x - geom.shoulder_x, y - geom.shoulder_y
    radius = math.hypot(dx, dy)
    tolerance = 1e-7
    if radius < geom.inner_radius - tolerance or radius > geom.outer_radius + tolerance:
        raise ValueError(
            f"目标 ({x:.1f}, {y:.1f}) mm 不可达；距肩 {radius:.1f} mm，"
            f"允许范围 {geom.inner_radius:.1f}–{geom.outer_radius:.1f} mm"
        )
    l1, l2 = geom.upper_arm, geom.forearm_and_hand
    c2 = (dx * dx + dy * dy - l1 * l1 - l2 * l2) / (2 * l1 * l2)
    q2 = elbow_sign * math.acos(max(-1.0, min(1.0, c2)))
    q1 = math.atan2(dy, dx) - math.atan2(l2 * math.sin(q2), l1 + l2 * math.cos(q2))
    return q1, q2


def check_joint_limits(q1: float, q2: float, where: str = "", limits: JointLimits = LIMITS) -> None:
    """按附表校验两个关节角；输入为弧度，超限时说明哪个关节不合规。"""
    shoulder, elbow = math.degrees(q1), math.degrees(q2)
    tolerance = 1e-7
    violations = []
    if not limits.shoulder_roll_min_deg - tolerance <= shoulder <= limits.shoulder_roll_max_deg + tolerance:
        violations.append(
            f"RShoulderRoll={shoulder:.2f}°（允许 {limits.shoulder_roll_min_deg:g}°～{limits.shoulder_roll_max_deg:g}°）"
        )
    if not limits.elbow_roll_min_deg - tolerance <= elbow <= limits.elbow_roll_max_deg + tolerance:
        violations.append(
            f"RElbowRoll={elbow:.2f}°（允许 {limits.elbow_roll_min_deg:g}°～{limits.elbow_roll_max_deg:g}°）"
        )
    if violations:
        prefix = f"{where}：" if where else ""
        raise ValueError(prefix + "关节超限：" + "；".join(violations))


def _quintic_hermite(p0: float, p1: float, v0: float, v1: float, dt: float, u: float) -> float:
    """端点位置和速度给定、两端加速度为零的五次插值。"""
    delta = p1 - p0 - v0 * dt
    velocity_delta = (v1 - v0) * dt
    return (
        p0 + v0 * dt * u
        + (10 * delta - 4 * velocity_delta) * u**3
        + (7 * velocity_delta - 15 * delta) * u**4
        + (6 * delta - 3 * velocity_delta) * u**5
    )


def _check_cartesian_segment(start: tuple[float, float], end: tuple[float, float], geom: ArmGeometry) -> None:
    """线段完整落在可达圆环内的解析判定。"""
    sx, sy = geom.shoulder_x, geom.shoulder_y
    for x, y in (start, end):
        inverse_kinematics(x, y, geom=geom)
    dx, dy = end[0] - start[0], end[1] - start[1]
    length_squared = dx * dx + dy * dy
    if length_squared:
        u = max(0.0, min(1.0, -((start[0] - sx) * dx + (start[1] - sy) * dy) / length_squared))
        nearest = math.hypot(start[0] + u * dx - sx, start[1] + u * dy - sy)
        if nearest < geom.inner_radius - 1e-7:
            raise ValueError("直线路径穿过肩部附近的不可达区域；请改用关节空间规划或调整端点")


def plan_trajectory(
    start: tuple[float, float],
    end: tuple[float, float],
    duration: float,
    mode: str = "cartesian",
    via_points: tuple[ViaPoint, ...] | list[ViaPoint] = (),
    geom: ArmGeometry = GEOMETRY,
) -> list[Sample]:
    """经由零个或多个 VIA 点的五次轨迹；中间点连续通过，两端速度为零。"""
    if not math.isfinite(duration) or not 0.1 <= duration <= 30.0:
        raise ValueError("运动时间须在 0.1–30 秒之间")
    if not all(math.isfinite(v) for v in (*start, *end)):
        raise ValueError("坐标必须是有限数字")
    if mode not in ("cartesian", "joint"):
        raise ValueError("规划方式必须是 cartesian 或 joint")
    if len(via_points) > 12:
        raise ValueError("最多支持 12 个 VIA POINTS")
    knots = [ViaPoint(0.0, *start), *via_points, ViaPoint(duration, *end)]
    for index, point in enumerate(knots):
        if not all(math.isfinite(v) for v in (point.t, point.x, point.y)):
            raise ValueError(f"VIA POINT {index} 的时间和坐标必须是有限数字")
        if index and point.t - knots[index - 1].t < 0.1 - 1e-9:
            raise ValueError("VIA POINTS 的时间须严格递增，且每段至少间隔 0.1 s")
    if mode == "cartesian" and not via_points:
        _check_cartesian_segment(start, end, geom)

    joint_knots = []
    for index, point in enumerate(knots):
        label = "起点" if index == 0 else "终点" if index == len(knots) - 1 else f"V{index}"
        try:
            angles = inverse_kinematics(point.x, point.y, elbow_sign=1, geom=geom)
            check_joint_limits(*angles, where=label)
        except ValueError as exc:
            raise ValueError(f"{label} (t={point.t:g} s)：{exc}") from exc
        joint_knots.append(angles)

    times = [point.t for point in knots]
    values = [(point.x, point.y) for point in knots] if mode == "cartesian" else joint_knots
    velocities = [(0.0, 0.0)]
    for index in range(1, len(knots) - 1):
        dt = times[index + 1] - times[index - 1]
        velocities.append(tuple((values[index + 1][axis] - values[index - 1][axis]) / dt for axis in (0, 1)))
    velocities.append((0.0, 0.0))

    def value_at(segment: int, u: float) -> tuple[float, float]:
        if u == 0.0:
            return values[segment]
        if u == 1.0:
            return values[segment + 1]
        dt = times[segment + 1] - times[segment]
        return tuple(
            _quintic_hermite(values[segment][axis], values[segment + 1][axis],
                             velocities[segment][axis], velocities[segment + 1][axis], dt, u)
            for axis in (0, 1)
        )

    samples: list[Sample] = []
    for segment in range(len(knots) - 1):
        segment_duration = times[segment + 1] - times[segment]
        distance = math.dist((knots[segment].x, knots[segment].y),
                             (knots[segment + 1].x, knots[segment + 1].y))
        steps = max(2, math.ceil(segment_duration / 0.02), math.ceil(1.875 * distance))
        # VIA 点处可保持非零速度；必要时再细分，控制手端相邻采样间距。
        while True:
            segment_values = [value_at(segment, j / steps) for j in range(steps + 1)]
            positions = (segment_values if mode == "cartesian" else
                         [forward_kinematics(*angles, geom=geom).hand for angles in segment_values])
            largest_gap = max(math.dist(a, b) for a, b in zip(positions, positions[1:]))
            if largest_gap <= 1.0 + 1e-9:
                break
            if steps >= 8192:
                raise ValueError(f"第 {segment + 1} 段速度过大；请增大 VIA POINTS 的时间间隔")
            steps *= 2
        for j in range(0 if segment == 0 else 1, steps + 1):
            t = times[segment] + segment_duration * j / steps
            if j == steps:
                t = times[segment + 1]
            if mode == "cartesian":
                x, y = segment_values[j]
                try:
                    q = inverse_kinematics(x, y, elbow_sign=1, geom=geom)
                except ValueError as exc:
                    raise ValueError(f"第 {segment + 1} 段 t={t:.2f} s：{exc}") from exc
            else:
                q = segment_values[j]
                x, y = positions[j]
            check_joint_limits(*q, where=f"第 {segment + 1} 段 t={t:.2f} s")
            label = ("起" if segment == 0 and j == 0 else
                     f"V{segment + 1}" if j == steps and segment < len(knots) - 2 else
                     "终" if j == steps else "")
            samples.append(Sample(t, x, y, math.degrees(q[0]), math.degrees(q[1]), waypoint=label))

    # 中心差分速度仅供显示/CSV；首尾端点由轨迹条件明确为零。
    result: list[Sample] = []
    for i, sample in enumerate(samples):
        if i in (0, len(samples) - 1):
            vx = vy = 0.0
        else:
            prev, nxt = samples[i - 1], samples[i + 1]
            vx = (nxt.x - prev.x) / (nxt.t - prev.t)
            vy = (nxt.y - prev.y) / (nxt.t - prev.t)
        result.append(Sample(sample.t, sample.x, sample.y, sample.shoulder_roll_deg,
                             sample.elbow_roll_deg, vx, vy, sample.waypoint))
    return result


class ArmSimulator(tk.Tk):
    WIDTH, HEIGHT = 850, 720
    VIEW_PADDING_MM = 24.0
    PLOT_MARGIN_PX = 24.0

    def __init__(self) -> None:
        super().__init__()
        self.title("NAO 右臂双自由度轨迹规划模拟器")
        self.geometry("1370x850")
        self.minsize(1050, 650)
        self.configure(bg="#f5f7fa")
        self._configure_fonts()
        self.start_x = tk.StringVar(value="195")
        self.start_y = tk.StringVar(value="-145")
        self.end_x = tk.StringVar(value="160")
        self.end_y = tk.StringVar(value="-35")
        self.duration = tk.StringVar(value="4")
        self.mode = tk.StringVar(value="cartesian")
        self.click_target = tk.StringVar(value="end")
        self.view_rotation_quarters = 0
        self.rotation_button_text = tk.StringVar(value="旋转画布 90° ↻（当前 0°）")
        self.rotation_status_text = tk.StringVar(value="画布：顺时针 0°")
        self.via_positions: list[tuple[float, float]] = []
        self.via_count = tk.StringVar(value="已添加 0 个 VIA POINTS")
        self.progress = tk.DoubleVar(value=0.0)
        self.status = tk.StringVar()
        self.samples: list[Sample] = []
        self.frame_index = 0
        self.playing = False
        self.after_id: str | None = None
        self._setting_progress = False
        self._build_ui()
        self.plan()
        self.protocol("WM_DELETE_WINDOW", self._close)

    def _configure_fonts(self) -> None:
        """从 Tk 可用字体中选中文字体，供控件和画布共用。"""
        available = {name.casefold(): name for name in tkfont.families(self)}
        ui_font = next((available[name.casefold()] for name in UI_FONT_CANDIDATES
                        if name.casefold() in available), None)
        if ui_font is None:
            raise RuntimeError("Tkinter 未找到可用的中文字体；请安装 Noto Sans CJK SC 或微软雅黑")
        for name in tkfont.names(self):
            if name.startswith("Tk"):
                tkfont.nametofont(name, root=self).configure(family=ui_font, size=UI_FONT_SIZE)
        self.body_font = tkfont.Font(root=self, family=ui_font, size=UI_FONT_SIZE)
        self.bold_font = tkfont.Font(root=self, family=ui_font, size=UI_FONT_SIZE, weight="bold")
        self.title_font = tkfont.Font(root=self, family=ui_font, size=20, weight="bold")
        self.tick_font = tkfont.Font(root=self, family=ui_font, size=12)
        self.axis_font = tkfont.Font(root=self, family=ui_font, size=14, weight="bold")

    def _build_ui(self) -> None:
        style = ttk.Style(self)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("TLabel", font=self.body_font)
        style.configure("TButton", font=self.body_font, padding=(10, 7))
        style.configure("TRadiobutton", font=self.body_font, padding=(2, 3))
        style.configure("TEntry", font=self.body_font)
        style.configure("TLabelframe", padding=10)
        style.configure("TLabelframe.Label", font=self.bold_font)

        sidebar = tk.Canvas(self, width=365, bg="#f5f7fa", highlightthickness=0)
        sidebar.pack(side="left", fill="y")
        sidebar_scroll = ttk.Scrollbar(self, orient="vertical", command=sidebar.yview)
        sidebar_scroll.pack(side="left", fill="y")
        sidebar.configure(yscrollcommand=sidebar_scroll.set)
        left = ttk.Frame(sidebar, padding=(14, 12))
        sidebar_window = sidebar.create_window((0, 0), window=left, anchor="nw")
        left.bind("<Configure>", lambda _event: sidebar.configure(scrollregion=sidebar.bbox("all")))
        sidebar.bind("<Configure>", lambda event: sidebar.itemconfigure(sidebar_window, width=event.width))
        right = ttk.Frame(self, padding=(0, 12, 12, 12))
        right.pack(side="left", fill="both", expand=True)
        ttk.Label(left, text="右臂俯视运动学", font=self.title_font).pack(anchor="w", pady=(0, 12))

        geometry = ttk.LabelFrame(left, text="模型尺寸 · mm")
        geometry.pack(fill="x", pady=(0, 10))
        ttk.Label(
            geometry,
            text="躯干原点 O₀ = (0, 0)\n肩关节 S = (0, -113)  [98 + 15]\n肩→肘 = 105\n肘→腕 = 55.95\n腕→手端 = 57.75\nθ₁ = RShoulderRoll\nθ₂ = RElbowRoll",
            justify="left",
        ).pack(anchor="w")

        target = ttk.LabelFrame(left, text="手端目标 · 相对 O₀ · mm")
        target.pack(fill="x", pady=(0, 10))
        for row, (label, var) in enumerate((
            ("起点 X", self.start_x), ("起点 Y", self.start_y),
            ("终点 X", self.end_x), ("终点 Y", self.end_y),
            ("时长 s", self.duration),
        )):
            ttk.Label(target, text=label).grid(row=row, column=0, sticky="w", pady=3)
            ttk.Entry(target, textvariable=var, width=12).grid(row=row, column=1, sticky="e", padx=(12, 0), pady=3)
        target.columnconfigure(1, weight=1)

        choice = ttk.LabelFrame(left, text="规划设置")
        choice.pack(fill="x", pady=(0, 10))
        ttk.Radiobutton(choice, text="手端轨迹（无 VIA 时直线）", value="cartesian", variable=self.mode).pack(anchor="w")
        ttk.Radiobutton(choice, text="关节平滑插值", value="joint", variable=self.mode).pack(anchor="w")
        ttk.Separator(choice).pack(fill="x", pady=6)
        ttk.Label(choice, text="RShoulderRoll：−70°～18°\nRElbowRoll：2°～88.5°", justify="left").pack(anchor="w")

        actions = ttk.Frame(left)
        actions.pack(fill="x", pady=(0, 10))
        ttk.Button(actions, text="规划轨迹", command=self.plan).grid(row=0, column=0, sticky="ew", padx=(0, 4), pady=3)
        self.play_button = ttk.Button(actions, text="播放", command=self.toggle_play)
        self.play_button.grid(row=0, column=1, sticky="ew", pady=3)
        ttk.Button(actions, text="重置", command=self.reset).grid(row=1, column=0, sticky="ew", padx=(0, 4), pady=3)
        ttk.Button(actions, text="导出 CSV", command=self.export_csv).grid(row=1, column=1, sticky="ew", pady=3)
        ttk.Button(actions, textvariable=self.rotation_button_text, command=self._rotate_canvas).grid(
            row=2, column=0, columnspan=2, sticky="ew", pady=(5, 3)
        )
        actions.columnconfigure(0, weight=1)
        actions.columnconfigure(1, weight=1)

        click = ttk.LabelFrame(left, text="画布点选")
        click.pack(fill="x", pady=(0, 10))
        ttk.Radiobutton(click, text="点击设置起点", value="start", variable=self.click_target).pack(anchor="w")
        ttk.Radiobutton(click, text="点击设置终点", value="end", variable=self.click_target).pack(anchor="w")
        ttk.Radiobutton(click, text="点击添加 VIA POINT", value="via", variable=self.click_target).pack(anchor="w")
        ttk.Label(click, textvariable=self.via_count, foreground="#70408a").pack(anchor="w", pady=(5, 2))
        via_actions = ttk.Frame(click)
        via_actions.pack(fill="x", pady=(2, 0))
        ttk.Button(via_actions, text="撤销最后一个", command=self._undo_via).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        ttk.Button(via_actions, text="清空全部", command=self._clear_vias).grid(row=0, column=1, sticky="ew")
        via_actions.columnconfigure(0, weight=1)
        via_actions.columnconfigure(1, weight=1)
        ttk.Label(click, text="VIA 到达时间按点击顺序自动均分；点选后按“规划轨迹”更新。", foreground="#526377", wraplength=315, justify="left").pack(anchor="w", pady=(5, 0))

        ttk.Label(left, text="说明：已检查起终点和全部轨迹采样点的关节限位。仍是二维理想模型，未考虑碰撞、速度/力矩和其他关节耦合。", foreground="#35485e", justify="left", wraplength=325).pack(anchor="w", pady=(6, 0))

        canvas_header = ttk.Frame(right, padding=(8, 4))
        canvas_header.pack(fill="x")
        ttk.Label(
            canvas_header,
            text="浅绿：关节限位内    蓝：计划轨迹    橙：已走轨迹    紫：VIA POINTS",
            foreground="#2d455e",
        ).pack(side="left")
        ttk.Label(
            canvas_header, textvariable=self.rotation_status_text,
            foreground="#7b4b20", font=self.bold_font,
        ).pack(side="right", padx=(12, 0))

        self.canvas = tk.Canvas(right, width=self.WIDTH, height=self.HEIGHT, bg="white", highlightthickness=1, highlightbackground="#d6dee8")
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Button-1>", self._canvas_click)
        self.canvas.bind("<Configure>", lambda _event: self.draw())
        scrub = ttk.Frame(right)
        scrub.pack(fill="x", pady=(8, 2))
        ttk.Label(scrub, text="时间").pack(side="left")
        self.scale = ttk.Scale(scrub, from_=0, to=100, variable=self.progress, command=self._scrub)
        self.scale.pack(side="left", fill="x", expand=True, padx=8)
        self.time_label = ttk.Label(scrub, text="0.00 s", width=9)
        self.time_label.pack(side="right")
        ttk.Label(right, textvariable=self.status, font=self.body_font, foreground="#203449").pack(anchor="w")

    def _canvas_size(self) -> tuple[float, float]:
        width = float(self.canvas.winfo_width())
        height = float(self.canvas.winfo_height())
        if width <= 2:
            width = float(self.canvas.cget("width"))
        if height <= 2:
            height = float(self.canvas.cget("height"))
        return max(width, 100.0), max(height, 100.0)

    def _view_parameters(self) -> tuple[float, float, float]:
        """返回响应式比例和 O₀ 的像素位置，确保完整工作区居中且尽量铺满。"""
        width, height = self._canvas_size()
        radius = GEOMETRY.outer_radius + self.VIEW_PADDING_MM
        available_width = max(1.0, width - 2 * self.PLOT_MARGIN_PX)
        available_height = max(1.0, height - 2 * self.PLOT_MARGIN_PX)
        scale = min(available_width, available_height) / (2 * radius)
        center_x, center_y = rotate_view_coordinates(
            GEOMETRY.shoulder_x, GEOMETRY.shoulder_y, self.view_rotation_quarters
        )
        origin_x = width / 2 - center_x * scale
        origin_y = height / 2 + center_y * scale
        return scale, origin_x, origin_y

    def _xy_to_canvas(self, x: float, y: float) -> tuple[float, float]:
        view_x, view_y = rotate_view_coordinates(x, y, self.view_rotation_quarters)
        scale, origin_x, origin_y = self._view_parameters()
        return (origin_x + view_x * scale, origin_y - view_y * scale)

    def _canvas_to_xy(self, px: float, py: float) -> tuple[float, float]:
        scale, origin_x, origin_y = self._view_parameters()
        view_x = (px - origin_x) / scale
        view_y = (origin_y - py) / scale
        return unrotate_view_coordinates(view_x, view_y, self.view_rotation_quarters)

    def _visible_world_bounds(self) -> tuple[float, float, float, float]:
        width, height = self._canvas_size()
        corners = [
            self._canvas_to_xy(0, 0), self._canvas_to_xy(width, 0),
            self._canvas_to_xy(0, height), self._canvas_to_xy(width, height),
        ]
        return (
            min(point[0] for point in corners), max(point[0] for point in corners),
            min(point[1] for point in corners), max(point[1] for point in corners),
        )

    def _rotate_canvas(self) -> None:
        self.view_rotation_quarters = (self.view_rotation_quarters + 1) % 4
        degrees = self.view_rotation_quarters * 90
        self.rotation_button_text.set(f"旋转画布 90° ↻（当前 {degrees}°）")
        self.rotation_status_text.set(f"画布：顺时针 {degrees}°")
        self.draw()
        self.status.set(f"画布已顺时针旋转到 {degrees}°；机器人坐标和轨迹数据未改变")

    def _canvas_click(self, event: tk.Event) -> None:
        x, y = self._canvas_to_xy(event.x, event.y)
        target = self.click_target.get()
        if target == "start":
            self.start_x.set(f"{x:.1f}")
            self.start_y.set(f"{y:.1f}")
            label = "起点"
        elif target == "end":
            self.end_x.set(f"{x:.1f}")
            self.end_y.set(f"{y:.1f}")
            label = "终点"
        else:
            if len(self.via_positions) >= 12:
                messagebox.showerror("无法添加", "最多支持 12 个 VIA POINTS。", parent=self)
                return
            try:
                angles = inverse_kinematics(x, y, elbow_sign=1)
                check_joint_limits(*angles, where=f"V{len(self.via_positions) + 1}")
            except ValueError as exc:
                messagebox.showerror("VIA POINT 不可用", str(exc), parent=self)
                return
            self.via_positions.append((x, y))
            self._update_via_count()
            label = f"V{len(self.via_positions)}"
        self._invalidate_plan()
        self.status.set(f"已设置 {label} ({x:.1f}, {y:.1f}) mm；点击“规划轨迹”更新")

    def _update_via_count(self) -> None:
        text = f"已添加 {len(self.via_positions)} 个 VIA POINTS"
        if self.via_positions:
            try:
                points = evenly_timed_via_points(self.via_positions, float(self.duration.get()))
                text += "\n自动时间：" + ", ".join(f"V{i}={point.t:.2f}s" for i, point in enumerate(points, 1))
            except ValueError:
                text += "\n当前时长无法自动分配时间"
        self.via_count.set(text)

    def _invalidate_plan(self) -> None:
        self._stop()
        self.samples = []
        self.frame_index = 0
        self._set_progress(0)
        self.draw()

    def _undo_via(self) -> None:
        if self.via_positions:
            self.via_positions.pop()
        self._update_via_count()
        self._invalidate_plan()
        self.status.set("已撤销最后一个 VIA POINT；点击“规划轨迹”更新")

    def _clear_vias(self) -> None:
        self.via_positions.clear()
        self._update_via_count()
        self._invalidate_plan()
        self.status.set("已清空 VIA POINTS；点击“规划轨迹”生成无中间点轨迹")

    def _stop(self) -> None:
        self.playing = False
        self.play_button.configure(text="播放")
        if self.after_id is not None:
            self.after_cancel(self.after_id)
            self.after_id = None

    def _set_progress(self, value: float) -> None:
        # 部分 Tk 版本在程序更新 Scale 时也会调用 command；避免它中断播放。
        self._setting_progress = True
        try:
            self.progress.set(value)
        finally:
            self._setting_progress = False

    def plan(self) -> None:
        self._stop()
        try:
            start = (float(self.start_x.get()), float(self.start_y.get()))
            end = (float(self.end_x.get()), float(self.end_y.get()))
            duration = float(self.duration.get())
            via_points = evenly_timed_via_points(self.via_positions, duration)
            new_samples = plan_trajectory(start, end, duration, self.mode.get(), via_points)
        except ValueError as exc:
            messagebox.showerror("规划失败", str(exc), parent=self)
            return
        self.samples = new_samples
        self._update_via_count()
        self.frame_index = 0
        self._set_progress(0)
        self.draw()

    def toggle_play(self) -> None:
        if not self.samples:
            return
        if self.playing:
            self._stop()
            return
        if self.frame_index >= len(self.samples) - 1:
            self.frame_index = 0
        self.playing = True
        self.play_button.configure(text="暂停")
        self._advance()

    def _advance(self) -> None:
        self.after_id = None
        if not self.playing:
            return
        self.draw()
        if self.frame_index >= len(self.samples) - 1:
            self._stop()
            return
        current = self.samples[self.frame_index]
        self.frame_index += 1
        next_sample = self.samples[self.frame_index]
        delay = max(1, round(1000 * (next_sample.t - current.t)))
        self.after_id = self.after(delay, self._advance)

    def _scrub(self, _value: str) -> None:
        if self._setting_progress or not self.samples:
            return
        self._stop()
        self.frame_index = min(len(self.samples) - 1, round(self.progress.get() / 100 * (len(self.samples) - 1)))
        self.draw()

    def reset(self) -> None:
        self._stop()
        self.frame_index = 0
        self._set_progress(0)
        self.draw()

    def export_csv(self) -> None:
        if not self.samples:
            messagebox.showinfo("无轨迹", "请先规划轨迹。", parent=self)
            return
        output = filedialog.asksaveasfilename(
            parent=self, title="导出轨迹", initialfile="nao_right_arm_trajectory.csv",
            defaultextension=".csv", filetypes=[("CSV 文件", "*.csv")],
        )
        if not output:
            return
        try:
            with Path(output).open("w", newline="", encoding="utf-8-sig") as file:
                writer = csv.writer(file)
                writer.writerow(["time_s", "hand_x_mm", "hand_y_mm", "RShoulderRoll_deg", "RElbowRoll_deg", "vx_mm_s", "vy_mm_s", "waypoint"])
                for s in self.samples:
                    writer.writerow([f"{v:.6f}" for v in (s.t, s.x, s.y, s.shoulder_roll_deg, s.elbow_roll_deg, s.vx, s.vy)] + [s.waypoint])
        except OSError as exc:
            messagebox.showerror("导出失败", str(exc), parent=self)
            return
        self.status.set(f"已导出 {len(self.samples)} 个采样点：{output}")

    def _line(self, points: list[tuple[float, float]], **kwargs: object) -> None:
        coordinates = [coordinate for x, y in points for coordinate in self._xy_to_canvas(x, y)]
        self.canvas.create_line(*coordinates, **kwargs)

    def _dot(self, point: tuple[float, float], radius: float, color: str, label: str = "", label_offset: tuple[int, int] = (8, -10)) -> None:
        px, py = self._xy_to_canvas(*point)
        self.canvas.create_oval(px - radius, py - radius, px + radius, py + radius, fill=color, outline="white", width=1.5)
        if label:
            self.canvas.create_text(px + label_offset[0], py + label_offset[1], text=label, anchor="w", fill="#25374a", font=self.bold_font)

    def _draw_limited_workspace(self) -> None:
        """将两个关节的允许角度矩形映射为画布上的浅绿色区域。"""
        low_s, high_s = LIMITS.shoulder_roll_min_deg, LIMITS.shoulder_roll_max_deg
        low_e, high_e = LIMITS.elbow_roll_min_deg, LIMITS.elbow_roll_max_deg
        steps = 44
        edges = (
            [(low_s, low_e + (high_e - low_e) * i / steps) for i in range(steps + 1)]
            + [(low_s + (high_s - low_s) * i / steps, high_e) for i in range(1, steps + 1)]
            + [(high_s, high_e - (high_e - low_e) * i / steps) for i in range(1, steps + 1)]
            + [(high_s - (high_s - low_s) * i / steps, low_e) for i in range(1, steps)]
        )
        coordinates = []
        for shoulder, elbow in edges:
            point = forward_kinematics(math.radians(shoulder), math.radians(elbow)).hand
            coordinates.extend(self._xy_to_canvas(*point))
        self.canvas.create_polygon(*coordinates, fill="#edf8ef", outline="#a1cbaa", width=2)

    def draw(self) -> None:
        if not hasattr(self, "canvas"):
            return
        c = self.canvas
        c.delete("all")
        w, h = c.winfo_width(), c.winfo_height()
        c.create_rectangle(0, 0, w, h, fill="white", outline="")
        self._draw_limited_workspace()
        # 按当前窗口尺寸计算可见范围，使网格铺满画布；文字保持正向便于阅读。
        x_min, x_max, y_min, y_max = self._visible_world_bounds()
        first_x = math.ceil(x_min / 20) * 20
        last_x = math.floor(x_max / 20) * 20
        first_y = math.ceil(y_min / 20) * 20
        last_y = math.floor(y_max / 20) * 20
        for x in range(first_x, last_x + 1, 20):
            self._line([(x, y_min), (x, y_max)], fill="#eef2f6")
        for y in range(first_y, last_y + 1, 20):
            self._line([(x_min, y), (x_max, y)], fill="#eef2f6")

        self._line([(x_min, 0), (x_max, 0)], fill="#788ba0", width=1.5, arrow="last")
        self._line([(0, y_min), (0, y_max)], fill="#788ba0", width=1.5, arrow="last")
        x_offsets = ((0, 16), (-24, 0), (0, -16), (24, 0))
        y_offsets = ((24, 0), (0, 16), (-24, 0), (0, -16))
        x_offset = x_offsets[self.view_rotation_quarters]
        y_offset = y_offsets[self.view_rotation_quarters]
        for x in range(math.ceil(x_min / 40) * 40, math.floor(x_max / 40) * 40 + 1, 40):
            if x == 0:
                continue
            px, py = self._xy_to_canvas(x, 0)
            if 14 <= px <= w - 14 and 14 <= py <= h - 14:
                c.create_text(px + x_offset[0], py + x_offset[1], text=str(x), anchor="center", fill="#34485e", font=self.tick_font)
        for y in range(math.ceil(y_min / 40) * 40, math.floor(y_max / 40) * 40 + 1, 40):
            if y == 0:
                continue
            px, py = self._xy_to_canvas(0, y)
            if 14 <= px <= w - 14 and 14 <= py <= h - 14:
                c.create_text(px + y_offset[0], py + y_offset[1], text=str(y), anchor="center", fill="#34485e", font=self.tick_font)

        x_label = self._xy_to_canvas(x_max - 18, 0)
        y_label = self._xy_to_canvas(0, y_max - 18)
        c.create_text(*x_label, text="x₀", fill="#354e68", anchor="center", font=self.axis_font)
        c.create_text(*y_label, text="y₀", fill="#354e68", anchor="center", font=self.axis_font)
        self._dot((0, 0), 5, "#364d66", "O₀", (9, 15))

        cx, cy = self._xy_to_canvas(GEOMETRY.shoulder_x, GEOMETRY.shoulder_y)
        scale, _, _ = self._view_parameters()
        for radius in (GEOMETRY.outer_radius, GEOMETRY.inner_radius):
            r = radius * scale
            c.create_oval(cx - r, cy - r, cx + r, cy + r, outline="#d6e5f0", width=1.5, dash=(5, 5))

        for index, point in enumerate(self.via_positions, 1):
            self._dot(point, 6, "#944db4", f"V{index}", (9, 15))

        if not self.samples:
            try:
                start = (float(self.start_x.get()), float(self.start_y.get()))
                end = (float(self.end_x.get()), float(self.end_y.get()))
                self._dot(start, 4, "#2a9c75", "起")
                self._dot(end, 4, "#b65ba4", "终")
            except ValueError:
                pass
            return
        sample = self.samples[self.frame_index]
        pose = forward_kinematics(math.radians(sample.shoulder_roll_deg), math.radians(sample.elbow_roll_deg))
        if len(self.samples) > 1:
            stride = max(1, len(self.samples) // 400)
            planned = [(s.x, s.y) for s in self.samples[::stride]]
            if planned[-1] != (self.samples[-1].x, self.samples[-1].y):
                planned.append((self.samples[-1].x, self.samples[-1].y))
            self._line(planned, fill="#51a9d4", width=2.5, smooth=False)
            actual = [(s.x, s.y) for s in self.samples[: self.frame_index + 1 : stride]]
            actual.append((sample.x, sample.y))
            if len(actual) >= 2:
                self._line(actual, fill="#ee9851", width=3)
        self._line([pose.shoulder, pose.elbow], fill="#345477", width=13, capstyle="round")
        self._line([pose.elbow, pose.wrist], fill="#4f789b", width=10, capstyle="round")
        self._line([pose.wrist, pose.hand], fill="#7d9fbc", width=9, capstyle="round")
        self._dot(pose.shoulder, 9, "#234f79", "S")
        self._dot(pose.elbow, 8, "#316b9b", "E")
        self._dot(pose.wrist, 6, "#5b8bb2", "W")
        self._dot(pose.hand, 8, "#ea7d32", "H")
        self._dot((self.samples[0].x, self.samples[0].y), 4, "#2a9c75", "起")
        self._dot((self.samples[-1].x, self.samples[-1].y), 4, "#b65ba4", "终")
        self._set_progress(100 * self.frame_index / (len(self.samples) - 1))
        self.time_label.configure(text=f"{sample.t:.2f} s")
        speed = math.hypot(sample.vx, sample.vy)
        self.status.set(
            f"{sample.waypoint + '  ' if sample.waypoint else ''}手端 ({sample.x:.1f}, {sample.y:.1f}) mm   "
            f"RShoulderRoll {sample.shoulder_roll_deg:.1f}°   RElbowRoll {sample.elbow_roll_deg:.1f}°   "
            f"速度 {speed:.1f} mm/s"
        )

    def _close(self) -> None:
        self._stop()
        self.destroy()


def main() -> None:
    ArmSimulator().mainloop()


if __name__ == "__main__":
    main()
